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

fs, audio = wavfile.read("../results/test4.wav")  # change
fs, raw = wavfile.read("../results/usetrain_whentest_double_iters_con/second_attempt/sample_e51_i330000_t63.13_tr6.6525_v6.6154_4.wav")  # change if appropriate

audio = audio[:len(raw)]

raw = raw.copy()
audio = audio.copy()

# audio_voiced = np.abs(audio) < 500
# audio[audio_voiced] = 0
#
# raw_voiced = np.abs(raw) < 500
# raw[raw_voiced] = 0

print(np.mean(np.abs(raw)))

ratio = np.mean(np.abs(audio)) / np.mean(np.abs(raw))

raw = raw.astype('float32')
# raw *= ratio

plt.figure(1)
xaxis = np.linspace(1, len(raw), len(raw))
xaxis = xaxis / 16000
plt.plot(xaxis, raw, xaxis, audio)
plt.legend(['Model Output', 'Original Audio'])
plt.xlabel('Seconds / s')

plt.show()
