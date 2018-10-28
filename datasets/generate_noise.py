import numpy as np
import os
import re

dataset_folders = os.listdir('speech')

for the_folder in dataset_folders:
    # Generate a list of the datasets in that folder, ie ['speech_test', 'speech_valid', etc...]
    datasets = os.listdir('speech/{}'.format(the_folder))

    for the_file in datasets:
        if ("noise" not in the_file) and ("lab" not in the_file) and (".npy" in the_file):
            try:
                data = np.load('speech/{}/{}'.format(the_folder, the_file))

            # Generate the noise, normalise and quantise to 256 levels.
                noise = np.random.normal(size=np.shape(data))
                noise = (noise / np.amax(np.abs(noise))) + 1
                noise = (noise * 255) / 2
                noise = np.round(noise)
                noise = noise.astype(np.int32)
                np.save('speech/{}/{}_noise.npy'.format(the_folder, the_file[:-4]), noise)
                print "{}/{}_noise.npy".format(the_folder, the_file[:-4])
            except MemoryError:
                print("Memory error for {}/{}".format(the_folder, the_file[:-4]))
                continue
