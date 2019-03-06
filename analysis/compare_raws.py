import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.io.wavfile

# fs, audio = wavfile.read("../results/test4.wav")
# fs, raw = wavfile.read("../results/c_noise_random_start/sample_e31_i200000_t39.03_tr6.8481_v6.5334_4.wav")

# Extract raw audio.
# data = np.load("../results/speech_train.npy")  # change if appropriate
# data = data[4]  # change
# data = data.astype('float32')
# data -= np.mean(data)
# data /= np.absolute(data).max()  # [-1,1]
# data *= 32768
# data = data.astype('int16')
# scipy.io.wavfile.write(
#     '../results/train4.wav',  # change
#     16000,
#     data)

# fs, audio = wavfile.read("../results/test4.wav")  # change
# fs, raw = wavfile.read("../results/usetrain_whentest_double_iters_con/second_attempt/sample_e51_i330000_t63.13_tr6.6525_v6.6154_4.wav")  # change if appropriate
fs, stacked = wavfile.read("../results/stacked_test_l1/sample_e20_i130000_t52.19_tr6.5348_v6.5461_3.wav")
fs, raw = wavfile.read("../results/c_noise_random_start/sample_e31_i200000_t39.03_tr6.8481_v6.5334_3.wav")
fs, orig = wavfile.read("../results/conditional_original/sample_e31_i200000_t35.66_tr2.7205_v2.9522_3.wav")
fs, a = wavfile.read("../speech_valid.wav")
fs, b = wavfile.read("../speech_valid_noise.wav")

raw = raw[:len(stacked)]

stacked = stacked.copy()
raw = raw.copy()

# audio_voiced = np.abs(audio) < 500
# audio[audio_voiced] = 0
#
# raw_voiced = np.abs(raw) < 500
# raw[raw_voiced] = 0

print(np.mean(np.abs(stacked)))

ratio = np.mean(np.abs(raw)) / np.mean(np.abs(stacked))

stacked = stacked.astype('float32')
# raw *= ratio

a = a.copy()
a = np.divide(a,np.max(np.abs(a)))
a -= np.min(a)
a *= 255

plt.figure(1)
xaxis = np.linspace(1, len(a), len(a))
xaxis = xaxis / 16000
plt.plot(xaxis, a, xaxis, b)
plt.legend(['Ref', 'Noise'])
plt.xlabel('Seconds / s')

plt.show()
