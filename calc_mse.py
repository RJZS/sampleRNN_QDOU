import numpy as np


def load_binary_file(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data'%(dimension)
    features = features[:(dimension * (features.size / dimension))]
    features = features.reshape((-1, dimension))
    return features

ref_dir = "/home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/datasets/REF_TEST_RMDC/5s/features"

noise_dir = "/home/dawna/tts/rjzs2/noise_results_3t/models-three_tier-three_tier_con.py-expnoise_con_from_original_uncon-seq_len800-big_fr_sz80-fr_sz20-emb_sz256-skip_connF-dim1024-n_rnn2-rnn_typeGRU-q_levels256-q_typemu-law-bch_sz20-weight_normT-learn_h0T-n_big_rnn2-normed-utt-rmzero-acoustic/features/ep31"
noise_file = "sample_e31_i200000_t66.64_tr6.6497_v6.6667"

ar_dir = "/home/dawna/tts/rjzs2/original_model_3t/models-three_tier-three_tier_con.py-expconditional_original-seq_len800-big_fr_sz80-fr_sz20-emb_sz256-skip_connF-dim1024-n_rnn2-rnn_typeGRU-q_levels256-q_typemu-law-bch_sz20-weight_normT-learn_h0T-n_big_rnn2-normed-utt-rmzero-acoustic/features/ep31"
ar_file = "sample_e31_i200000_t35.66_tr2.7205_v2.9522"

specdim = 60
fdim = 1
noisedim = 21

ref_spec = load_binary_file("{}/Spec/ref_test_4.mcep".format(ref_dir), specdim)
noise_spec = load_binary_file("{}/Spec/{}_4.mcep".format(noise_dir, noise_file), specdim)
ar_spec = load_binary_file("{}/Spec/{}_4.mcep".format(ar_dir, ar_file), specdim)

noise_spec = noise_spec[:-2]
ref_spec = ref_spec[:-2]

noise_spec_mse = np.mean((ref_spec - noise_spec)**2, 0)
ar_spec_mse = np.mean((ref_spec - ar_spec)**2, 0)

print(noise_spec_mse.shape)
print(noise_spec_mse)
print(ar_spec_mse)
