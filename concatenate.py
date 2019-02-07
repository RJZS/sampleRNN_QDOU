import numpy as np
from scipy.io import wavfile

gen_dir = '../gen'
wav_file_root = 'sample_conditional_original.pkl'


for utt in range(20):
    fs, one = wavfile.read('{}/baseline_1_16000/{}_{}.wav'.format(gen_dir, wav_file_root, utt))
    fs, two = wavfile.read('{}/baseline_2_16000/{}_{}.wav'.format(gen_dir, wav_file_root, utt))
    fs, three = wavfile.read('{}/baseline_3_16000/{}_{}.wav'.format(gen_dir, wav_file_root, utt))
    fs, four = wavfile.read('{}/baseline_4_16000/{}_{}.wav'.format(gen_dir, wav_file_root, utt))

    audio = np.concatenate((one, two[1600:], three[1600:], four[1600:]))
    wavfile.write('{}/baseline/irregular_concat_{}.wav'.format(gen_dir, utt), fs, audio)
