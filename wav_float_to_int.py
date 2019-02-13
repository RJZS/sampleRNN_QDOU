import numpy as np
from scipy.io import wavfile

utts = range(505, 525)

in_root = 'data/wav_16_kHz_orig'
out_root = 'data/wav_16kHz_proc'
for utt in utts:
    fs, ref = wavfile.read('{}/hvd_{}.wav'.format(in_root, utt))
    ref = ref.copy()
    ref *= 32768
    ref = ref.astype('int16')
    wavfile.write('{}/hvd_{}.wav'.format(out_root, utt), fs, ref)
