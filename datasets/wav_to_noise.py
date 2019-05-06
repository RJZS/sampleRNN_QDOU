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

# Now read the files.
tA = np.zeros(900*128000)  # trainArray
print(128000*27)
idx = 0
for i, t in enumerate(train):
	row = numpy_from_file(data_folder,t)
	tA[idx:idx+len(row)] = row
	idx += len(row)
	print i

print(idx)
tA = np.reshape(tA, (900,128000))
tA = (tA/np.amax(np.abs(tA)))+1  # [0,2]
tA = (tA * 255)/2
tA = np.round(tA)
tA = tA.astype(np.int32)
np.save('speech_train.npy', tA)

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
