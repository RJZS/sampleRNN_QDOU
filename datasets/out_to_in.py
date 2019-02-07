import numpy as np
from scipy.io import wavfile
import os

# Script to take the output waveform and turn it into a .npy file with the appropriate form to pass it back in.
wav_dir = '/home/dawna/tts/rjzs2/noise_results_3t/gen/models-three_tier-three_tier_con.py-expgen_test-seq_len800-big_fr_sz80-fr_sz20-emb_sz256-skip_connF-dim1024-n_rnn2-rnn_typeGRU-q_levels256-q_typemu-law-bch_sz27-weight_normT-learn_h0T-n_big_rnn2-normed-utt-rmzero-acoustic-gen/samples'
wav_file_root = 'sample_c_noise_random_start.pkl'

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
        print(utt)
    except IOError:
        print("Could not find utterance {}. Skipping...".format(utt))
        continue

np.save('{}/speech_test_noise_1.npy'.format(out_dir), noise)
