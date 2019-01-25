import numpy as np
from scipy.io import wavfile

# Script to take the output waveform and turn it into a .npy file with the appropriate form to pass it back in.
wav_dir = '/home/dawna/tts/rjzs2/noise_results_3t/c_noise_random_start'
wav_file_root = 'sample_e31_i200000_t39.03_tr6.8481_v6.5334'

for utt in range(5):
    fs, audio = wavfile.read('{}/{}_{}.wav'.format(wav_dir, wav_file_root, utt))
    print(np.mean(audio))
    print(np.max(audio))
    print(np.min(audio))
