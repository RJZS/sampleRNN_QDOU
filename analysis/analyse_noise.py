import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

sample_dir = "C:/Users/Rafi/IIBProject/gen/baseline"
sample_file = "irregular_concat_7.wav"

# sample_dir = "C:/Users/Rafi/Desktop/samples/uncondition_noise"
# sample_file = "sample_e9_i60000_t11.38_tr7.6896_v7.5518_2.wav"

fs, data = wavfile.read("{}/{}".format(sample_dir, sample_file))
print("Mean: {}".format(np.mean(data)))
print("Variance: {}".format(np.var(data)))
data = data.astype('float32')
# autocorr = np.correlate(data, data, mode='full')
# print(int((autocorr.size+1)/2))
# norm = data - np.mean(data)
# norm = np.sum(norm ** 2)
# autocorr = autocorr / np.var(data)
# autocorr = autocorr[int((autocorr.size+1)/2):]

f_0 = 16000  # Sampling frequency.
T = len(data) / f_0
fft = np.fft.fft(data)
fft = fft[0:int((len(data)/2)+1)]  # # 'data' is real, so the fft is symmetric.
xaxis = np.linspace(0, f_0/2, len(fft))

# Commented out as phase information is not relevant.
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(xaxis, np.abs(fft))
# axs[0].set_xlabel('Frequency')
# axs[0].set_ylabel('Magnitude')
# axs[1].plot(xaxis, np.angle(fft))
# axs[1].set_xlabel('Frequency')
# axs[1].set_ylabel('Phase')

# FIGURE 1
fig, axs = plt.subplots(1, 2)
axs[0].plot(xaxis, np.abs(fft))
axs[0].set_xlabel('Frequency')
axs[0].set_ylabel('Magnitude')
axs[0].set_title('FFT')
axs[1].acorr(data, maxlags=None)
axs[1].set_xlabel('Lag')
axs[1].set_title('Autocorrelation')

# FIGURE 2
plt.figure(2)
# mu = np.mean(data)
# variance = np.var(data)
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# hist = np.histogram(data, 1000)
# print(mlab.normpdf(x, mu, sigma).max())
# plt.plot(x,mlab.normpdf(x, mu, sigma)*80000000, hist[1][:-1], hist[0])
plt.hist(data, bins=200)

# FIGURE 3
plt.figure(3)
raw_xaxis = np.linspace(1, len(data), len(data))
raw_xaxis = raw_xaxis / f_0
plt.plot(raw_xaxis, data)
plt.xlabel("Time / s")

plt.show()
print(np.histogram(data))
