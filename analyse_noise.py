import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

sample_dir = "C:/Users/Rafi/Desktop/samples/c_noise_random_start"
sample_file = "sample_e31_i200000_t39.03_tr6.8481_v6.5334_2.wav"

# sample_dir = "C:/Users/Rafi/Desktop/samples/uncondition_noise"
# sample_file = "sample_e9_i60000_t11.38_tr7.6896_v7.5518_2.wav"

fs, data = wavfile.read("{}/{}".format(sample_dir, sample_file))
print("Mean: {}".format(np.mean(data)))
print("Variance: {}".format(np.var(data)))
autocorr = np.correlate(data, data, mode='full')
print(int((autocorr.size+1)/2))
norm = data - np.mean(data)
norm = np.sum(norm **2)
autocorr = autocorr / norm
autocorr = autocorr[int((autocorr.size+1)/2):]

# plt.plot(autocorr[0:1000])
plt.hist(data, bins=100)
plt.show()
print(np.histogram(data))
