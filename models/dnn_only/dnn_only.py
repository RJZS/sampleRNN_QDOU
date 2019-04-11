"""
RNN Audio Generation Model

Three-tier model, Quantized input
For more info:
$ python three_tier.py -h

How-to-run example:
sampleRNN$ pwd
/u/mehris/sampleRNN


sampleRNN$ \
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u \
models/three_tier/three_tier.py --exp AXIS1 --seq_len 512 --big_frame_size 8 \
--frame_size 2 --weight_norm True --emb_size 64 --skip_conn False --dim 32 \
--n_rnn 2 --rnn_type LSTM --learn_h0 False --q_levels 16 --q_type linear \
--batch_size 128 --which_set MUSIC

To resume add ` --resume` to the END of the EXACTLY above line. You can run the
resume code as many time as possible, depending on the TRAIN_MODE.
(folder name, file name, flags, their order, and the values are important)
"""
from time import time
from datetime import datetime
print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
exp_start = time()

import os, sys, glob
sys.path.insert(1, os.getcwd())
# import argparse
import itertools

import numpy
numpy.random.seed(123)
np = numpy
import random
random.seed(123)

import theano
import theano.tensor as T
import theano.ifelse
import lasagne
import scipy.io.wavfile

import lib

import sys
sys.path.append('/home/dawna/tts/qd212/lib_QDOU')
from HRNN import get_args_3t as get_args
from HRNN import get_flag_dict

import pdb

#LEARNING_RATE = 0.005*10
#LEARNING_RATE = 0.0002/10
#LEARNING_RATE = 0.001 #default

### Parsing passed args/hyperparameters ###
args, tag = get_args()
# pdb.set_trace()

SEQ_LEN = args.seq_len # How many samples to include in each truncated BPTT pass
#print "------------------previous SEQ_LEN:", SEQ_LEN
# TODO: test incremental training
#SEQ_LEN = 512 + 256
#print "---------------------------new SEQ_LEN:", SEQ_LEN
BIG_FRAME_SIZE = args.big_frame_size # how many samples per big frame
FRAME_SIZE = args.frame_size # How many samples per frame
OVERLAP = BIG_FRAME_SIZE
WEIGHT_NORM = args.weight_norm
EMB_SIZE = args.emb_size
SKIP_CONN = args.skip_conn
DIM = args.dim # Model dimensionality.
BIG_DIM = DIM # Dimensionality for the slowest level.
N_RNN = args.n_rnn # How many RNNs to stack in the frame-level model

FRAME_SIZE_DNN = args.frame_size_dnn # How many previous samples per setp for DNN
if FRAME_SIZE_DNN==0: FRAME_SIZE_DNN = FRAME_SIZE

if args.n_big_rnn==0:
    N_BIG_RNN = N_RNN # how many RNNs to stack in the big-frame-level model
else:
    N_BIG_RNN = args.n_big_rnn
#pdb.set_trace()

RNN_TYPE = args.rnn_type
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1
LEARN_H0 = args.learn_h0
Q_LEVELS = args.q_levels # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
Q_TYPE = args.q_type # log- or linear-scale
WHICH_SET = args.which_set
BATCH_SIZE = args.batch_size
RESUME = args.resume
assert SEQ_LEN % BIG_FRAME_SIZE == 0,\
    'seq_len should be divisible by big_frame_size'
assert BIG_FRAME_SIZE % FRAME_SIZE == 0,\
    'big_frame_size should be divisible by frame_size'
N_FRAMES = SEQ_LEN / FRAME_SIZE # Number of frames in each truncated BPTT pass

if Q_TYPE == 'mu-law' and Q_LEVELS != 256:
    raise ValueError('For mu-law Quantization levels should be exactly 256!')


LEARNING_RATE = float(args.lr)
UCINIT_DIRFILE = args.uc

GEN_DIRFILE = args.gen
FLAG_GEN = (GEN_DIRFILE!='not_gen')

###set FLAGS for options
flag_dict = get_flag_dict(args)

FLAG_QUANTLAB = flag_dict['QUANTLAB']

# Fixed hyperparams
GRAD_CLIP = 1 # Elementwise grad clip threshold
BITRATE = 16000

# Other constants
#TRAIN_MODE = 'iters' # To use PRINT_ITERS and STOP_ITERS
# TRAIN_MODE = 'time' # To use PRINT_TIME and STOP_TIME
#TRAIN_MODE = 'time-iters'
# To use PRINT_TIME for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
TRAIN_MODE = 'iters'
# To use PRINT_ITERS for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
PRINT_ITERS = 10000 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 200000 # Stop after this many iterations
PRINT_TIME = 60*60*24*3 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*24*3.5 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
N_SEQS = 5  # Number of samples to generate every time monitoring.
RESULTS_DIR = '/home/dawna/tts/rjzs2/noise_results_dnn'
if WHICH_SET != 'SPEECH': RESULTS_DIR = os.path.join(RESULTS_DIR, WHICH_SET)

if FLAG_GEN:
    # N_SEQS = 10
    # N_SECS = 8 #LENGTH = 8*BITRATE #640*80
    N_SEQS = BATCH_SIZE #60
    # N_SECS = 8 #LENGTH = 8*BITRATE #640*80
    RESULTS_DIR = os.path.join(RESULTS_DIR,'gen')

FOLDER_PREFIX = os.path.join(RESULTS_DIR, tag)
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

LAB_SIZE = 80 #one label covers 80 points on waveform
LAB_PERIOD = float(0.005) #one label covers 0.005s ~ 200Hz
LAB_DIM = 601
if flag_dict['ACOUSTIC']:
    if WHICH_SET in ['SPEECH','NANCY']: LAB_DIM = 163
    elif WHICH_SET=='LESLEY': LAB_DIM = 85
    elif WHICH_SET=='VCBK': LAB_DIM = 86
UP_RATE = LAB_SIZE/FRAME_SIZE

epoch_str = 'epoch'
iter_str = 'iter'
lowest_valid_str = 'lowest valid cost'
corresp_test_str = 'correponding test cost'
train_nll_str, valid_nll_str, test_nll_str = \
    'train NLL (bits)', 'valid NLL (bits)', 'test NLL (bits)'

if args.debug:
    import warnings
    warnings.warn('----------RUNNING IN DEBUG MODE----------')
    TRAIN_MODE = 'time'
    PRINT_TIME = 100
    STOP_TIME = 3000
    STOP_ITERS = 1000

### Create directories ###
#   FOLDER_PREFIX: root, contains:
#       log.txt, __note.txt, train_log.pkl, train_log.png [, model_settings.txt]
#   FOLDER_PREFIX/params: saves all checkpoint params as pkl
#   FOLDER_PREFIX/samples: keeps all checkpoint samples as wav
#   FOLDER_PREFIX/best: keeps the best parameters, samples, ...
if not os.path.exists(FOLDER_PREFIX):
    os.makedirs(FOLDER_PREFIX)
PARAMS_PATH = os.path.join(FOLDER_PREFIX, 'params')
if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)
SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'samples')
if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)
BEST_PATH = os.path.join(FOLDER_PREFIX, 'best')
if not os.path.exists(BEST_PATH):
    os.makedirs(BEST_PATH)

lib.print_model_settings(locals(), path=FOLDER_PREFIX, sys_arg=True)


### Import the data_feeder ###
# Handling WHICH_SET
if WHICH_SET == 'ONOM':
    from datasets.dataset import onom_train_feed_epoch as train_feeder
    from datasets.dataset import onom_valid_feed_epoch as valid_feeder
    from datasets.dataset import onom_test_feed_epoch  as test_feeder
elif WHICH_SET == 'BLIZZ':
    from datasets.dataset import blizz_train_feed_epoch as train_feeder
    from datasets.dataset import blizz_valid_feed_epoch as valid_feeder
    from datasets.dataset import blizz_test_feed_epoch  as test_feeder
elif WHICH_SET == 'MUSIC':
    from datasets.dataset import music_train_feed_epoch as train_feeder
    from datasets.dataset import music_valid_feed_epoch as valid_feeder
    from datasets.dataset import music_test_feed_epoch  as test_feeder
elif WHICH_SET == 'HUCK':
    from datasets.dataset import huck_train_feed_epoch as train_feeder
    from datasets.dataset import huck_valid_feed_epoch as valid_feeder
    from datasets.dataset import huck_test_feed_epoch  as test_feeder
elif WHICH_SET == 'SPEECH' or 'LESLEY' or 'NANCY':
    from datasets.dataset_con import speech_train_feed_epoch as train_feeder
    from datasets.dataset_con import speech_valid_feed_epoch as valid_feeder
    from datasets.dataset_con import speech_test_feed_epoch  as test_feeder


def get_lab_big(seqs_lab):
    seqs_lab_big = seqs_lab[:,::BIG_FRAME_SIZE/FRAME_SIZE,:]
    return seqs_lab_big


def load_data(data_feeder):
    """
    Helper function to deal with interface of different datasets.
    `data_feeder` should be `train_feeder`, `valid_feeder`, or `test_feeder`.
    """
    return data_feeder(FRAME_SIZE,
                       BATCH_SIZE,
                       SEQ_LEN,
                       OVERLAP,
                       Q_LEVELS,
                       Q_ZERO,
                       Q_TYPE)
def load_data_gen(data_feeder,SEQ_LEN_gen):
    return data_feeder(FRAME_SIZE,
                       BATCH_SIZE,
                       SEQ_LEN_gen,
                       OVERLAP,
                       Q_LEVELS,
                       Q_ZERO,
                       Q_TYPE)
print('----got to def---')
### Creating computation graph ###

# the DNNs!! Taking the outputs from the RNN (frame_level_outputs)
# prev_samples is a bunch of samples from previous timesteps (FRAME_SIZE timesteps).
# Want to remove this as an input so not conditioning on previous steps anymore!
def sample_level_predictor(frame_level_outputs, prev_samples):
    """
    frame_level_outputs.shape: (batch size, DIM) -> (BATCH_SIZE * SEQ_LEN, DIM)
    prev_samples.shape:        (batch size, FRAME_SIZE) -> (BATCH_SIZE * SEQ_LEN, FRAME_SIZE_DNN)
    output.shape:              (batch size, Q_LEVELS)
    """
    # Handling EMB_SIZE
    if EMB_SIZE == 0:  # no support for one-hot in three_tier and one_tier.
        prev_samples = lib.ops.T_one_hot(prev_samples, Q_LEVELS)
        # (BATCH_SIZE*N_FRAMES*FRAME_SIZE, FRAME_SIZE_DNN, Q_LEVELS)
        last_out_shape = Q_LEVELS
    elif EMB_SIZE > 0:
        prev_samples = lib.ops.Embedding(
            'SampleLevel.Embedding',
            Q_LEVELS,
            EMB_SIZE,
            prev_samples)
        # (BATCH_SIZE*N_FRAMES*FRAME_SIZE, FRAME_SIZE_DNN, EMB_SIZE), f32
        last_out_shape = EMB_SIZE
    else:
        raise ValueError('EMB_SIZE cannot be negative.')

    prev_samples = prev_samples.reshape((-1, FRAME_SIZE_DNN * last_out_shape))

    # comment out this block
    out = lib.ops.Linear(
        'SampleLevel.L1_PrevSamples',
        FRAME_SIZE_DNN * last_out_shape,
        DIM,
        prev_samples,
        biases=False,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    out += frame_level_outputs
    # change to out = frame_level_outputs, as frame_level_outputs is from tier2.
    # out = T.nnet.relu(out)  # commented out to be similar to two_tier

    out = lib.ops.Linear('SampleLevel.L2',
                         DIM,
                         DIM,
                         out,
                         initialization='he',
                         weightnorm=WEIGHT_NORM)
    out = T.nnet.relu(out)

    # L3
    out = lib.ops.Linear('SampleLevel.L3',
                         DIM,
                         DIM,
                         out,
                         initialization='he',
                         weightnorm=WEIGHT_NORM)
    out = T.nnet.relu(out)

    # Output
    # We apply the softmax later
    out = lib.ops.Linear('SampleLevel.Output',
                         DIM,
                         Q_LEVELS,
                         out,
                         weightnorm=WEIGHT_NORM)
    return out

print('----got to T var---')
# After defined graph, need to define theano variables!
sequences   = T.imatrix('sequences')
noise       = T.imatrix('noise')
reset       = T.iscalar('reset')
mask        = T.matrix('mask')
frame_level_outputs = T.imatrix('frame_level_outputs')
if FLAG_QUANTLAB:
    print('REMINDER: lab is quantized')
    sequences_lab      = T.itensor3('sequences_lab')
    sequences_lab_big      = T.itensor3('sequences_lab_big')
else:
    print('REMINDER: lab is NOT quantized')
    sequences_lab      = T.tensor3('sequences_lab')
    sequences_lab_big      = T.tensor3('sequences_lab_big')

if args.debug:
    # Solely for debugging purposes.
    # Maybe I should set the compute_test_value=warn from here.
    # theano.config.compute_test_value = 'warn'
    sequences.tag.test_value = numpy.zeros((BATCH_SIZE, SEQ_LEN+OVERLAP), dtype='int32')
    reset.tag.test_value = numpy.array(1, dtype='int32')
    mask.tag.test_value = numpy.ones((BATCH_SIZE, SEQ_LEN+OVERLAP), dtype='float32')


input_sequences = noise[:, BIG_FRAME_SIZE-FRAME_SIZE:-FRAME_SIZE] # Condition target sequence on this input sequence.
big_input_sequences = noise[:, :-BIG_FRAME_SIZE]
target_sequences = sequences[:, BIG_FRAME_SIZE:]

target_mask = mask[:, BIG_FRAME_SIZE:]

#---debug---
#pdb.set_trace()
#---debug---


frame_level_outputs = np.zeros((BATCH_SIZE, SEQ_LEN, DIM))
frame_level_outputs[:, , :FRAME_SIZE] = noise[:, BIG_FRAME_SIZE-FRAME_SIZE:-FRAME_SIZE]
prev_samples = noise[:, BIG_FRAME_SIZE-FRAME_SIZE_DNN:-1]
prev_samples = prev_samples.reshape((1, BATCH_SIZE, 1, -1))
prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (1, FRAME_SIZE_DNN), neib_step=(1, 1), mode='valid')
prev_samples = prev_samples.reshape((BATCH_SIZE * SEQ_LEN, FRAME_SIZE_DNN))

sample_level_outputs = sample_level_predictor(
    frame_level_outputs.reshape((BATCH_SIZE * SEQ_LEN, DIM)),
    prev_samples
)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(sample_level_outputs),
    target_sequences.flatten()
)
cost = cost.reshape(target_sequences.shape)
cost = cost * target_mask
# Don't use these lines; could end up with NaN
# Specially at the end of audio files where mask is
# all zero for some of the shorter files in mini-batch.
#cost = cost.sum(axis=1) / target_mask.sum(axis=1)
#cost = cost.mean(axis=0)

# Use this one instead.
cost = cost.sum()
cost = cost / target_mask.sum()

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
# log_2(e) = 1.44269504089
cost = cost * lib.floatX(numpy.log2(numpy.e))

### Getting the params, grads, updates, and Theano functions ###
#params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
#ip_params = lib.get_params(ip_cost, lambda x: hasattr(x, 'param') and x.param==True\
#    and 'BigFrameLevel' in x.name)
#other_params = [p for p in params if p not in ip_params]
#params = ip_params + other_params
#lib.print_params_info(params, path=FOLDER_PREFIX)
#
#grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
#grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]
#
#updates = lasagne.updates.adam(grads, params, learning_rate=LEARNING_RATE)

###########
all_params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
other_params = [p for p in all_params]
all_params = other_params
lib.print_params_info(other_params, path=FOLDER_PREFIX)
lib.print_params_info(all_params, path=FOLDER_PREFIX)

other_grads = T.grad(cost, wrt=other_params, disconnected_inputs='warn')
other_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in other_grads]

grads = T.grad(cost, wrt=all_params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

#---debug---
#pdb.set_trace()
#---debug---

other_updates = lasagne.updates.adam(other_grads, other_params)
updates = lasagne.updates.adam(grads, all_params)

print('----got to fn---')
# Training function(s)
train_fn = theano.function(
    [sequences, sequences_lab, sequences_lab_big, reset, mask, noise], # Add noise as input here.
    [cost],
    updates=updates,
    on_unused_input='warn'
)

# Validation and Test function, hence no updates
test_fn = theano.function(
    [sequences, sequences_lab, sequences_lab_big, reset, mask, noise],
    [cost],
    on_unused_input='warn'
)

# Sampling at audio sample level
frame_level_outputs = T.matrix('frame_level_outputs')
prev_samples        = T.imatrix('prev_samples')
sample_level_generate_fn = theano.function(
    [frame_level_outputs, noise],
    lib.ops.softmax_and_sample(
        sample_level_predictor(
            frame_level_outputs,
            noise
        )
    ),
    on_unused_input='warn'
)

# Uniform [-0.5, 0.5) for half of initial state for generated samples
# to study the behaviour of the model and also to introduce some diversity
# to samples in a simple way. [it's disabled]

FLAG_USETRAIN_WHENTEST = False
# Used for monitoring model progress.
# When ready to actually generate samples properly, use --gen.
def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= numpy.mean(data)
        data /= numpy.absolute(data).max() # [-1,1]
        data *= 32768
        data = data.astype('int16')
        scipy.io.wavfile.write(
            os.path.join(SAMPLES_PATH, name+'.wav'),
            BITRATE,
            data)

    total_time = time()
    # Generate N_SEQS' sample files, each 5 seconds long
    N_SECS = 5
    LENGTH = N_SECS*BITRATE if not args.debug else 160 #before it was 100, but 160 was better as it should be divisible by 80
    if FLAG_GEN: LENGTH = 785*80

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples_noise = numpy.zeros((N_SEQS, LENGTH), dtype='int32')

    if FLAG_USETRAIN_WHENTEST:
        print('')
        print('REMINDER: using training data for test')
        print('')
        testData_feeder = load_data_gen(train_feeder,LENGTH)
    else:
        testData_feeder = load_data_gen(test_feeder,LENGTH)
    mini_batch = testData_feeder.next()
    tmp, _, _, seqs_lab, seqs_noise = mini_batch
    samples_lab = seqs_lab[:N_SEQS]
    seqs_noise = seqs_noise.astype('int32')
    samples_noise = seqs_noise[:N_SEQS, :LENGTH]

    if flag_dict['RMZERO']:
        samples[:, :BIG_FRAME_SIZE] = tmp[:N_SEQS, :BIG_FRAME_SIZE]
        samples_noise[:, :BIG_FRAME_SIZE] = seqs_noise[:N_SEQS, :BIG_FRAME_SIZE]
    else:
        samples[:, :BIG_FRAME_SIZE] = Q_ZERO
        samples_noise[:, :BIG_FRAME_SIZE] = Q_ZERO

    samples_lab_big = get_lab_big(samples_lab)

    big_frame_level_outputs = None
    frame_level_outputs = None

    # LENGTH is length of utterance to generate.
    # Take one frame of silence, then start at index BIG_FRAME_SIZE.
    # Do this for training and debugging.
    # As the RNN needs initial state.
    # Once model is good enough, actually use 20 frames.
    for t in xrange(BIG_FRAME_SIZE, LENGTH): # for loop going sample by sample
        if t % FRAME_SIZE == 0:
            frame_level_outputs = numpy.zeros(N_SEQS, DIM)
            frame_level_outputs[:, :FRAME_SIZE] = samples_noise[:, t-FRAME_SIZE:t]

        samples[:, t] = sample_level_generate_fn(
            frame_level_outputs[:, t % FRAME_SIZE],
            samples_noise[:, t-FRAME_SIZE_DNN:t]
        )

    total_time = time() - total_time
    log = "{} samples of {} seconds length generated in {} seconds."
    log = log.format(N_SEQS, N_SECS, total_time)
    print log,

    samps = np.zeros((N_SEQS, LENGTH))
    samps_q = np.zeros((N_SEQS, LENGTH))
    for i in xrange(N_SEQS):
        samp = samples[i]
        samps[i,:] = samp
        if Q_TYPE == 'mu-law':
            from datasets.dataset import mu2linear
            samp = mu2linear(samp)
        elif Q_TYPE == 'a-law':
            raise NotImplementedError('a-law is not implemented')
        samps_q[i, :] = samp
        # write_audio_file("sample_{}_{}".format(tag, i), samp)

    name = "samples_{}".format(tag)
    numpy.save(os.path.join(SAMPLES_PATH, name), samps)
    name = "samples_{}_q".format(tag)
    numpy.save(os.path.join(SAMPLES_PATH, name), samps_q)

def monitor(data_feeder):
    """
    Cost and time of test_fn on a given dataset section.
    Pass only one of `valid_feeder` or `test_feeder`.
    Don't pass `train_feed`.

    :returns:
        Mean cost over the input dataset (data_feeder)
        Total time spent
    """
    _total_time = time()
    _costs = []
    _data_feeder = load_data(data_feeder)
    for _seqs, _reset, _mask, _seqs_lab, _seqs_noise in _data_feeder:
        _seqs_lab_big = get_lab_big(_seqs_lab)
        _seqs_noise = _seqs_noise.astype(numpy.int32)
        _cost = test_fn(_seqs, _seqs_lab, _seqs_lab_big, _reset, _mask, _seqs_noise)
        _costs.append(_cost)

    return numpy.mean(_costs), time() - _total_time

print "Wall clock time spent before training started: {:.2f}h"\
        .format((time()-exp_start)/3600.)
print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
costs = []
lowest_valid_cost = numpy.finfo(numpy.float32).max
corresponding_test_cost = numpy.finfo(numpy.float32).max
new_lowest_cost = False
end_of_batch = False
epoch = 0

cost_log_list = []

# Initial load train dataset
tr_feeder = load_data(train_feeder)

if FLAG_GEN:
    print('---loading gen_para.pkl---')
    lib.load_params(GEN_DIRFILE)
    print('---loading complete---')
    print('sampling')
    tmp = GEN_DIRFILE.split('/')[-1]
    generate_and_save_samples(tmp)
    print('ok')
    sys.exit()

### start from uncon
if UCINIT_DIRFILE == 'flat_start': FLAG_UCINIT = False
else: FLAG_UCINIT = True
if (FLAG_UCINIT and not RESUME):
    print('---loading uncon_para_expand_3t.pkl---')
    uncon_para_expand_path = UCINIT_DIRFILE
    lib.load_params(uncon_para_expand_path)
    print('---loading complete---')

### Handling the resume option:
if RESUME:
    # Check if checkpoint from previous run is not corrupted.
    # Then overwrite some of the variables above.
    iters_to_consume, res_path, epoch, total_iters,\
        [lowest_valid_cost, corresponding_test_cost, test_cost] = \
        lib.resumable(path=FOLDER_PREFIX,
                      iter_key=iter_str,
                      epoch_key=epoch_str,
                      add_resume_counter=True,
                      other_keys=[lowest_valid_str,
                                  corresp_test_str,
                                  test_nll_str])
    # At this point we saved the pkl file.
    last_print_iters = total_iters
    print "### RESUMING JOB FROM EPOCH {}, ITER {}".format(epoch, total_iters)
    # Consumes this much iters to get to the last point in training data.
    consume_time = time()
    for i in xrange(iters_to_consume):
        tr_feeder.next()
    consume_time = time() - consume_time
    print "Train data ready in {:.2f}secs after consuming {} minibatches.".\
            format(consume_time, iters_to_consume)

    lib.load_params(res_path)
    print "Parameters from last available checkpoint loaded."

    lib.load_updates(res_path,updates)
    cost_log_list = lib.load_costs(PARAMS_PATH)
    print "Updates from last available checkpoint loaded."

FLAG_DEBUG_SAMPLE = False
if FLAG_DEBUG_SAMPLE:
    print('debug: sampling')
    generate_and_save_samples(tag)
    print('debug: ok')

while True:
    # THIS IS ONE ITERATION
    if total_iters % 500 == 0:
        print total_iters,

    total_iters += 1

    try:
        # Take as many mini-batches as possible from train set
        mini_batch = tr_feeder.next()
    except StopIteration:
        # Mini-batches are finished. Load it again.
        # Basically, one epoch.
        tr_feeder = load_data(train_feeder)

        # and start taking new mini-batches again.
        mini_batch = tr_feeder.next()
        epoch += 1
        end_of_batch = True
        print "[Another epoch]",

    seqs, reset, mask, seqs_lab, seqs_noise = mini_batch
    seqs_lab_big = get_lab_big(seqs_lab)

    start_time = time()
    # pdb.set_trace()
    seqs_noise = seqs_noise.astype(numpy.int32)
    cost = train_fn(seqs, seqs_lab, seqs_lab_big, reset, mask, seqs_noise)
    total_time += time() - start_time

    costs.append(cost)

    # Monitoring step
    if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
        (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='time-iters' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='iters-time' and total_iters-last_print_iters >= PRINT_ITERS):
        # 0. Validation
        print "\nValidation!",
        valid_cost, valid_time = monitor(valid_feeder)
        print "Done!"

        # 1. Test
        test_time = 0.
        # Only when the validation cost is improved get the cost for test set.
        if valid_cost < lowest_valid_cost:
            lowest_valid_cost = valid_cost
            print "\n>>> Best validation cost of {} reached. Testing!"\
                    .format(valid_cost),
            test_cost, test_time = monitor(test_feeder)
            print "Done!"
            # Report last one which is the lowest on validation set:
            print ">>> test cost:{}\ttotal time:{}".format(test_cost, test_time)
            corresponding_test_cost = test_cost
            new_lowest_cost = True

        # 2. Stdout the training progress
        print_info = "epoch:{}\ttotal iters:{}\twall clock time:{:.2f}h\n"
        print_info += ">>> Lowest valid cost:{}\t Corresponding test cost:{}\n"
        print_info += "\ttrain cost:{:.4f}\ttotal time:{:.2f}h\tper iter:{:.3f}s\n"
        print_info += "\tvalid cost:{:.4f}\ttotal time:{:.2f}h\n"
        print_info += "\ttest  cost:{:.4f}\ttotal time:{:.2f}h"
        print_info = print_info.format(epoch,
                                       total_iters,
                                       (time()-exp_start)/3600,
                                       lowest_valid_cost,
                                       corresponding_test_cost,
                                       numpy.mean(costs),
                                       total_time/3600,
                                       total_time/total_iters,
                                       valid_cost,
                                       valid_time/3600,
                                       test_cost,
                                       test_time/3600)
        print print_info

        tag = "e{}_i{}_t{:.2f}_tr{:.4f}_v{:.4f}"
        tag = tag.format(epoch,
                         total_iters,
                         total_time/3600,
                         numpy.mean(cost),
                         valid_cost)
        tag += ("_best" if new_lowest_cost else "")

        # 3. Save params of model (IO bound, time consuming)
        # If saving params is not successful, there shouldn't be any trace of
        # successful monitoring step in train_log as well.
        print "Saving params!",
        lib.save_params(
                os.path.join(PARAMS_PATH, 'params_{}.pkl'.format(tag))
        )
        print "Done!"
        #save updates
        print 'saving updates, costs'
        lib.save_updates(PARAMS_PATH,tag, updates)
        lib.save_costs(PARAMS_PATH,cost_log_list)
        print 'complete!'

        # 4. Save and graph training progress (fast)
        training_info = {epoch_str : epoch,
                         iter_str : total_iters,
                         train_nll_str : numpy.mean(costs),
                         valid_nll_str : valid_cost,
                         test_nll_str : test_cost,
                         lowest_valid_str : lowest_valid_cost,
                         corresp_test_str : corresponding_test_cost,
                         'train time' : total_time,
                         'valid time' : valid_time,
                         'test time' : test_time,
                         'wall clock time' : time()-exp_start}
        lib.save_training_info(training_info, FOLDER_PREFIX)
        print "Train info saved!",

        y_axis_strs = [train_nll_str, valid_nll_str, test_nll_str]
        lib.plot_traing_info(iter_str, y_axis_strs, FOLDER_PREFIX)
        print "And plotted!"

        # 5. Generate and save samples (time consuming)
        # If not successful, we still have the params to sample afterward
        print "Sampling!",
        # Generate samples
        generate_and_save_samples(tag)
        print "Done!"

        if total_iters-last_print_iters == PRINT_ITERS \
            or total_time-last_print_time >= PRINT_TIME:
                # If we are here b/c of onom_end_of_batch, we shouldn't mess
                # with costs and last_print_iters
            costs = []
            last_print_time += PRINT_TIME
            last_print_iters += PRINT_ITERS

        end_of_batch = False
        new_lowest_cost = False

        print "Validation Done!\nBack to Training..."

    if (TRAIN_MODE=='iters' and total_iters == STOP_ITERS) or \
       (TRAIN_MODE=='time' and total_time >= STOP_TIME) or \
       ((TRAIN_MODE=='time-iters' or TRAIN_MODE=='iters-time') and \
            (total_iters == STOP_ITERS or total_time >= STOP_TIME)):

        print "Done! Total iters:", total_iters, "Total time: ", total_time
        print "Experiment ended at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
        print "Wall clock time spent: {:.2f}h"\
                    .format((time()-exp_start)/3600)

        sys.exit()
