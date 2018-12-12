import numpy

def uniform(stdev, size):
    """
    uniform distribution with the given stdev and size

    From Ishaan's code:
        https://github.com/igul222/speech
    """
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype('float32')

def get_init_weight_values(inp_dim,output_dim,initialization,rdm_lv=1):
    if initialization == 'lecun' or (initialization == None and inp_dim != output_dim):
        weight_values = uniform(rdm_lv*numpy.sqrt(1.0 / inp_dim), (inp_dim, output_dim))
    if initialization == 'he':
        weight_values = uniform(rdm_lv*numpy.sqrt(2.0 / inp_dim), (inp_dim, output_dim))
    return weight_values

def get_pad_weight_values(inp_dim,output_dim,initialization,lab_dim=601,rdm_lv=1.0):
    weight_values = get_init_weight_values(inp_dim,output_dim,initialization,rdm_lv)
    return weight_values[-lab_dim:]

import pickle
def mk_uncon_para_expand_3t_rdm(src_dirFile,tgt_dirFile,big_fr_sz=80,fr_sz=20,dim=1024,lab_dim=601,rdm_lv=1):    
    path = src_dirFile
    with open(path, 'rb') as f:
        param_vals_uncon = pickle.load(f)

    #2 modify weight
    import numpy
    w_uncon = param_vals_uncon['BigFrameLevel.GRU1.Step.Input.W0']
    print(w_uncon.shape)
    w_pad = get_pad_weight_values(big_fr_sz+lab_dim,3*dim,'lecun',lab_dim=lab_dim,rdm_lv=rdm_lv)
#     print 'debug', w_pad.shape
    w_uncon_3 = numpy.concatenate((w_uncon,w_pad))
    print(w_uncon_3.shape)
    
    w_uncon = param_vals_uncon['FrameLevel.InputExpand.W0']
    print(w_uncon.shape)
    w_pad = get_pad_weight_values(fr_sz+lab_dim,dim,'he',lab_dim=lab_dim,rdm_lv=rdm_lv)
#     print 'debug', w_pad.shape
    w_uncon_2 = numpy.concatenate((w_uncon,w_pad))
    print(w_uncon_2.shape)

    #3 save into pickle
    uncon_para_expand = param_vals_uncon
    uncon_para_expand['BigFrameLevel.GRU1.Step.Input.W0'] = w_uncon_3
    uncon_para_expand['FrameLevel.InputExpand.W0'] = w_uncon_2
    
    path = tgt_dirFile
    with open(path, 'wb') as f:
        pickle.dump(uncon_para_expand, f)
    return

src_dirFile='results_3t/models-three_tier-three_tier.py-expuncondition_noise-seq_len800-big_fr_sz80-fr_sz20-emb_sz256-skip_connF-dim1024-n_rnn2-rnn_typeGRU-q_levels256-q_typemu-law-bch_sz20-weight_normT-learn_h0T-n_big_rnn2-normed-utt-rmzero-acoustic/params/params_e9_i60000_t11.38_tr7.6896_v7.5518.pkl'
tgt_dirFile='bkup/uc_expand_3t_test_run.pkl'
mk_uncon_para_expand_3t_rdm(src_dirFile,tgt_dirFile,big_fr_sz=80,fr_sz=20,dim=1024,lab_dim=163,rdm_lv=1)
