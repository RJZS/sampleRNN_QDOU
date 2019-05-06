import numpy as np
from scipy.io import wavfile
import os
import re

def numpy_from_file(data_folder, filename):
        fs, audio = wavfile.read('{}/{}.wav'.format(data_folder, filename))
        return audio

data_folder = '/home/dawna/tts/qd212/data/data_central/nick/merlinData/all_model_synthesis_cmp163/wav'
output_folder = 'speech/ln_MA_f32_CE_8s_norm_utt'
ids_file = 'file_id_list.scp'

r = 128000
mu = 255

train = []
valid = []
test = []

# Get names of wav files in each dataset.
with open(ids_file) as f:
	l = f.readline()
	i = 1
	while l:
		if i <= 2254:
			train.append(l.strip())
		elif i <= 2324:
			valid.append(l.strip())
		elif i <= 2396:
			test.append(l.strip())
		l = f.readline()
		i += 1


def create_noise_file(file_list, num_rows, output_name):
    # Now read the files.
    oA = np.zeros(num_rows*128000)  # outputArray
    idx = 0
    for i, t in enumerate(file_list):
        row = numpy_from_file(data_folder,t)
        oA[idx:idx+len(row)] = row
        idx += len(row)

    print(idx)
    oA = np.reshape(oA, (num_rows, 128000))
    oA = (oA/np.amax(np.abs(oA)))
    x_mu = np.sign(oA) * np.log(1 + mu * np.abs(oA)) / np.log(1 + mu)
    x_mu = ((x_mu + 1) / 2 * mu).astype('int16')
    np.save('{}/{}.npy'.format(output_folder, output_name), x_mu)

print("Train")
create_noise_file(train, 900, 'speech_train_noise')
print("Test")
create_noise_file(test, 27, 'speech_test_noise')
print("Valid")
create_noise_file(valid, 26, 'speech_valid_noise')

#for the_folder in dataset_folders:
#    # Generate a list of the datasets in that folder, ie ['speech_test', 'speech_valid', etc...]
#    datasets = os.listdir('nancy/{}'.format(the_folder))
#
#    for the_file in datasets:
     #   if ("noise" not in the_file) and ("lab" not in the_file) and ("trj" not in the_file) and (".npy" in the_file):
    #        try:
   #             data = np.load('nancy/{}/{}'.format(the_folder, the_file))

#            # Generate the noise, normalise and quantise to 256 levels.
#                noise = np.random.normal(size=np.shape(data))
#                noise = (noise / np.amax(np.abs(noise))) + 1
  #              noise = (noise * 255) / 2
##                noise = np.round(noise)
#                noise = noise.astype(np.int32)
#                np.save('nancy/{}/{}_noise.npy'.format(the_folder, the_file[:-4]), noise)
#                print "{}/{}_noise.npy".format(the_folder, the_file[:-4])
  #          except MemoryError:
 #               print("Memory error for {}/{}".format(the_folder, the_file[:-4]))
#                continue
