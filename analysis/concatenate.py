import numpy as np
from scipy.io import wavfile

gen_dir = '../gen'
wav_file_root = 'sample_params_e2_i118721_t53.40_tr2.4280_v2.3452_best.pkl'


for utt in range(10):
    fs, one = wavfile.read('{}/nancy_baseline_1/{}_{}.wav'.format(gen_dir, wav_file_root, utt))
    fs, two = wavfile.read('{}/nancy_baseline_2/{}_{}.wav'.format(gen_dir, wav_file_root, utt))
    fs, three = wavfile.read('{}/nancy_baseline_3/{}_{}.wav'.format(gen_dir, wav_file_root, utt))
    # fs, four = wavfile.read('{}/nancy_baseline_4/{}_{}.wav'.format(gen_dir, wav_file_root, utt))

    audio = np.concatenate((one, two[8000:], three[8000:]))# , four[1600:]))
    wavfile.write('{}/nancy_concat/irregular_nancy_concat_{}.wav'.format(gen_dir, utt), fs, audio)
