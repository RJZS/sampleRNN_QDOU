import numpy as np
from scipy.io import wavfile
import os

# Script to take the output waveform and turn it into a .npy file with the appropriate form to pass it back in.
wav_dir = '/home/dawna/tts/rjzs2/noise_results_3t/models-three_tier-three_tier_con.py-expstacked_test_l1-seq_len800-big_fr_sz80-fr_sz20-emb_sz256-skip_connF-dim1024-n_rnn2-rnn_typeGRU-q_levels256-q_typemu-law-bch_sz20-weight_normT-learn_h0T-n_big_rnn2-normed-utt-rmzero-acoustic/samples'
wav_file_root = 'sample_e15_i100000_t44.18_tr6.4393_v6.5313_best'

out_dir = 'speech/ln_MA_f32_CE_8s_norm_utt'

utts = os.listdir('{}'.format(wav_dir))
noise = np.zeros((len(utts), 128000))
for utt in range(len(utts)):
    try:
        fs, audio = wavfile.read('{}/{}_{}.wav'.format(wav_dir, wav_file_root, utt))
        audio = audio.astype(np.float32)
        audio = (audio / np.amax(np.abs(audio))) + 1
        audio = (audio * 255) / 2
        audio = np.round(audio)
        audio = audio.astype(np.int32)
        noise[utt, :] = audio
    except IOError:
        print("Could not find utterance {}. Skipping...".format(utt))
        continue

np.save('{}/generated_noise.npy'.format(out_dir), noise)
