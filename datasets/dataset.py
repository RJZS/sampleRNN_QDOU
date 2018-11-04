"""
RNN Vocal Generation Model
Blizzard, Music, and Huckleberry Finn, and speech, data feeders.
"""

import numpy
np = numpy
#import scikits.audiolab

import random
import time
import os
import glob

from __main__ import flag_dict
FLAG_RMZERO = flag_dict['RMZERO']
FLAG_NORMED_ALRDY = flag_dict['NORMED_ALRDY']
# FLAG_NORMED_UTT = flag_dict['NORMED_UTT']
#try code is temporary, when all old exps are ok, the first line is enough
try:
    FLAG_NORMED_UTT = flag_dict['NORMED_UTT']
except KeyError:
    print "KeyError: maybe NORMED_UTT was not defined in main code"
    FLAG_NORMED_UTT = False
# else:
#     print "other error"
#     FLAG_NORMED_UTT = False

WHICH_SET = flag_dict['WHICH_SET']

FLAG_GRID = flag_dict['GRID']
if FLAG_GRID:
    __base = [('Grid', '/scratch/qd212/datasets/')]
else:
    __base = [('Local', 'datasets/')]
    
    
__blizz_file = 'Blizzard/Blizzard9k_{}.npy'  # in float16 8secs*16000samples/sec
__music_file = 'music/music_{}.npy'  # in float16 8secs*16000samples/sec
__huck_file = 'Huckleberry/Huckleberry_{}.npy'  # in float16 8secs*16000samples/sec

if FLAG_NORMED_ALRDY:
    if WHICH_SET == 'SPEECH':
        if FLAG_NORMED_UTT:
            __speech_file = 'speech/ln_MA_f32_CE_8s_norm_utt/speech_{}.npy'  # normed on utt level: zero mean, increased volume
            __speech_file_noise = 'speech/ln_MA_f32_CE_8s_norm_utt/speech_{}_noise.npy'
        else:
            __speech_file = 'speech/manuCutAlign_f32_norm_rmDC/speech_{}.npy'  # normed on cps level: zero mean
            __speech_file_noise = 'speech/manuCutAlign_f32_norm_rmDC/speech_{}_noise.npy'
            #__speech_file = 'speech/manuAlign_float32_cutEnd_norm/speech_{}.npy'  # in float16 8secs*16000samples/sec
    if WHICH_SET == 'LESLEY':
        __speech_file = 'speech/ln_16k_resil_Lesley_norm_utt/speech_{}.npy'  # lesley data
        __speech_file_noise = 'speech/ln_16k_resil_Lesley_norm_utt/speech_{}_noise.npy'
else:
    if WHICH_SET == 'SPEECH':
        # __speech_file = 'speech/MA_f32_CE_5s/speech_{}.npy'
        #__speech_file = 'speech/MA_f32_CE_8s_rpt3/speech_{}.npy'
        __speech_file = 'speech/manuAlign_float32_cutEnd/speech_{}.npy'  # in float16 8secs*16000samples/sec
        __speech_file_noise = 'speech/manuAlign_float32_cutEnd/speech_{}_noise.npy'
    if WHICH_SET == 'LESLEY':
        # __speech_file = 'speech/16k_resil_Lesley/speech_{}.npy'  # lesley data, changed dir, should be replaced by the following line
        __speech_file = 'speech/16k_resil_Lesley_full/speech_{}.npy'  # lesley data
        __speech_file_noise = 'speech/16k_resil_Lesley_full/speech_{}_noise.npy'
        
        
import sys
sys.path.append('/home/dawna/tts/qd212/lib_QDOU')
from HRNN import get_file_lab_str

if WHICH_SET in ['NANCY','VCBK']:
    __speech_file,_ = get_file_lab_str(flag_dict,WHICH_SET)
    
__blizz_train_mean_std = np.array([0.0008558356760380169,
                                   0.098437514304141299],
                                   dtype='float64')
__music_train_mean_std = np.array([-2.7492260671334582e-05,
                                   0.056233098718291352],
                                   dtype='float64')

__speech_train_mean_std = np.array([6.6694896289095623e-07,
                                   0.042258811348676345],
                                   dtype='float64')

# print 'dir for wav:'
# print __speech_file

# TODO:
#__huck_train_mean_std = ...

__train = lambda s: s.format('train')
__valid = lambda s: s.format('valid')
__test = lambda s: s.format('test')

def find_dataset(filename):
    for (k, v) in __base:
        tmp_path = os.path.join(v, filename)
        if os.path.exists(tmp_path):
            #print "Path on {}:".format(k)
            #print tmp_path
            return tmp_path
        #print "not found on {}".format(k)
    raise Exception('{} NOT FOUND!'.format(filename))

### Basic utils ###
def __round_to(x, y):
    """round x up to the nearest y"""
    return int(numpy.ceil(x / float(y))) * y

def __normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def __linear_quantize(data, q_levels):
    """
    floats in (0, 1) to ints in [0, q_levels-1]
    scales normalized across axis 1
    """
    # Normalization is on mini-batch not whole file
    #eps = numpy.float64(1e-5)
    #data -= data.min(axis=1)[:, None]
    #data *= ((q_levels - eps) / data.max(axis=1)[:, None])
    #data += eps/2
    #data = data.astype('int32')

    eps = numpy.float64(1e-5)
    data *= (q_levels - eps)
    data += eps/2
    data = data.astype('int32')
    return data

def __a_law_quantize(data):
    """
    :todo:
    """
    raise NotImplementedError

def linear2mu(x, mu=255):
    """
    From Joao
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it
    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.
    :usage:
        >>> bitrate, samples = scipy.io.wavfile.read('orig.wav')
        >>> norm = __normalize(samples)[None, :]  # It takes 2D as inp
        >>> mu_encoded = linear2mu(2.*norm-1.)  # From [0, 1] to [-1, 1]
        >>> print mu_encoded.min(), mu_encoded.max(), mu_encoded.dtype
        0, 255, dtype('int16')
        >>> mu_decoded = mu2linear(mu_encoded)  # Back to linear
        >>> print mu_decoded.min(), mu_decoded.max(), mu_decoded.dtype
        -1, 0.9574371, dtype('float32')
    """
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')

def mu2linear(x, mu=255):
    """
    From Joao with modifications
    Converts an integer array from mu to linear
    For important notes and usage see: linear2mu
    """
    mu = float(mu)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)

def __mu_law_quantize(data):
    return linear2mu(data)

def __batch_quantize(data, q_levels, q_type):
    """
    One of 'linear', 'a-law', 'mu-law' for q_type.
    """
    data = data.astype('float64')
    if not FLAG_NORMED_ALRDY:
        data = __normalize(data)
    if q_type == 'linear':
        return __linear_quantize(data, q_levels)
    if q_type == 'a-law':
        return __a_law_quantize(data)
    if q_type == 'mu-law':
        # from [0, 1] to [-1, 1]
        data = 2.*data-1.
        # Automatically quantized to 256 bins.
        return __mu_law_quantize(data)
    raise NotImplementedError

__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return
    #import collections
    #if isinstance(inp_list, (collections.Sequence)):
    if isinstance(inp_list, numpy.ndarray):
        numpy.random.seed(__RAND_SEED)
        numpy.random.shuffle(inp_list)
        return
    # destructive operations; in place; no need to return
    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

def __make_random_batches(inp_list, batch_size):
    batches = []
    for i in xrange(len(inp_list) / batch_size):
        batches.append(inp_list[i*batch_size:(i+1)*batch_size])

    __fixed_shuffle(batches)
    return batches


def get_files_init(batch,overlap):
    tmp = batch[:-1,-overlap:]
    #row1 = numpy.full((1, overlap), 0, dtype='float32')
    row1 = batch[0:1,:overlap]
    tmp = numpy.concatenate((row1,tmp),axis=0)
    return tmp
### SPEECH DATASET LOADER ###
def __speech_feed_epoch(files,
                        files_noise,
                       batch_size,
                       seq_len,
                       overlap,
                       q_levels,
                       q_zero,
                       q_type,
                       real_valued=False):
    """
    Helper function to load speech dataset.
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).
    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.
    Assumes all flac files have the same length.
    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """
    print('')
    if FLAG_RMZERO: print('REMINDER: starting from real data')
    else: print('REMINDER: starting from q_zeros')
    if FLAG_NORMED_ALRDY: print('REMINDER: normalized already')
    else: print('REMINDER: normalize when running the exp, on sentence level')
    print 'wav dir: '+__speech_file
    if FLAG_GRID: print('REMINDER: using data on air')
    else: print('REMINDER: using local data')
    print('')
    
    if FLAG_RMZERO:
        ##build files_init to init with real data##
        files_init = get_files_init(files,overlap)
        files = numpy.concatenate((files_init,files),axis=1)
    
    batches = __make_random_batches(files, batch_size)
    batches_noise = __make_random_batches(files_noise, batch_size)

    for bch, bch_noise in zip(batches, batches_noise):
        # batch_seq_len = length of longest sequence in the batch, rounded up to
        # the nearest SEQ_LEN.
        batch_seq_len = len(bch[0])  # should be 8*16000
        #batch_seq_len = __round_to(batch_seq_len, seq_len)

        batch = numpy.zeros(
            (batch_size, batch_seq_len),
            dtype='float64'
        )

        batch_noise = numpy.zeros(
            (batch_size, batch_seq_len),
            dtype='float64'
        )

        mask = numpy.ones(batch.shape, dtype='float32')

        for i, data in enumerate(bch):
            #data, fs, enc = scikits.audiolab.flacread(path)
            # data is float16 from reading the npy file
            batch[i, :len(data)] = data
            # This shouldn't change anything. All the flac files for Speech
            # are the same length and the mask should be 1 every where.
            # mask[i, len(data):] = numpy.float32(0)

        for i, data in enumerate(bch_noise):
            batch_noise[i, :len(data)] = data

        if not real_valued:
            batch = __batch_quantize(batch, q_levels, q_type)

            if not FLAG_RMZERO:
                batch = numpy.concatenate([
                    numpy.full((batch_size, overlap), q_zero, dtype='int32'),
                    batch
                ], axis=1)
                batch_noise = numpy.concatenate([
                    numpy.full((batch_size, overlap), q_zero, dtype='int32'),
                    batch
                ], axis=1)
        else:
            batch -= __speech_train_mean_std[0]
            batch /= __speech_train_mean_std[1]
            if not FLAG_RMZERO:
                batch = numpy.concatenate([
                    numpy.full((batch_size, overlap), 0, dtype='float32'),
                    batch
                ], axis=1).astype('float32')
                batch_noise = numpy.concatenate([
                    numpy.full((batch_size, overlap), 0, dtype='float32'),
                    batch
                ], axis=1).astype('float32')

        mask = numpy.concatenate([
            numpy.full((batch_size, overlap), 1, dtype='float32'),
            mask
        ], axis=1)

        for i in xrange(batch_seq_len // seq_len):
            reset = numpy.int32(i==0)
            subbatch = batch[:, i*seq_len : (i+1)*seq_len + overlap]
            subbatch_noise = batch_noise[:, i * seq_len: (i + 1) * seq_len + overlap]
            submask = mask[:, i*seq_len : (i+1)*seq_len + overlap]
            yield (subbatch, reset, submask, subbatch_noise)

def speech_train_feed_epoch(*args):
    """
    :parameters:
        batch_size: int
        seq_len:
        overlap:
        q_levels:
        q_zero:
        q_type: One the following 'linear', 'a-law', or 'mu-law'
    4,340 (9.65 hours) in total
    With batch_size = 128:
        4,224 (9.39 hours) in total
        3,712 (88%, 8.25 hours)for training set
        256 (6%, .57 hours) for validation set
        256 (6%, .57 hours) for test set
    Note:
        32 of Beethoven's piano sonatas available on archive.org (Public Domain)
    :returns:
        A generator yielding (subbatch, reset, submask)
    """
    # Just check if valid/test sets are also available. If not, raise.
    find_dataset(__valid(__speech_file))
    find_dataset(__test(__speech_file))
    # Load train set
    tmp = __train(__speech_file)
    tmp_noise = __train(__speech_file_noise)
    # if WHICH_SET=='VCBK':
    if flag_dict['SPLIT']:
        # idx = random.randint(0, 9)
        idx = int(time.time())%10
        tmp = tmp.replace('train','train_'+str(idx))
        tmp_noise = tmp_noise.replace('train', 'train_' + str(idx))
        print('')
        print('REMINDER: using split {} for training'.format(idx))
        print('')
    data_path = find_dataset(tmp)
    files = numpy.load(data_path)
    data_path = find_dataset(tmp_noise)
    files_noise = numpy.load(data_path)
    
    # #asup
    # tmp_ft = 10
    # nb_row = files.shape[0]
    # files = files[:int(nb_row*tmp_ft/100)]
    # print('')
    # print('REMINDER: fine tuning using {}/100 of training data'.format(tmp_ft))
    # print('')
        
    generator = __speech_feed_epoch(files, files_noise, *args)
    return generator

def speech_valid_feed_epoch(*args):
    """
    See:
        speech_train_feed_epoch
    """
    data_path = find_dataset(__valid(__speech_file))
    files = numpy.load(data_path)
    data_path = find_dataset(__valid(__speech_file_noise))
    files_noise = numpy.load(data_path)
    generator = __speech_feed_epoch(files, files_noise, *args)
    return generator

def speech_test_feed_epoch(*args):
    """
    See:
        speech_train_feed_epoch
    """
    data_path = find_dataset(__test(__speech_file))
    files = numpy.load(data_path)
    data_path = find_dataset(__valid(__speech_file_noise))
    files_noise = numpy.load(data_path)
    generator = __speech_feed_epoch(files, files_noise, *args)
    return generator