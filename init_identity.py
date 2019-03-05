import numpy as np
import pickle

bfs = 80  # Big frame size
fs = 20  # Frame size
h = 1024  # Length of hidden state
c = 0.1

example_pkl_file = 'bkup/c_noise_random_start.pkl'  # Use this file to get keys of dictionary and dimensions of each matrix.
save_file = 'bkup/identity_transform.pkl'
with open(example_pkl_file, 'rb') as f:
    t = pickle.load(f)  # Parameters are of type 'float32'.

print("Loaded file!")

for key in t.keys():
    t[key] = np.zeros(t[key].shape)


def top_left_ci(t, key, c, n):
    for i in range(n):
        t["{}.W0".format(key)][i, i] = c
    t["{}.g0".format(key)] = np.apply_along_axis(np.linalg.norm,0,t["{}.W0".format(key)])
    return t


def concat_top_left_ci(t, key, n, c, start):
    # When multiple weight matrices are concatenated together, and you don't want the top one to be cI.
    for i in range(n):
        t["{}.W0".format(key)][i, start+i] = c
    t["{}.g0".format(key)] = np.apply_along_axis(np.linalg.norm,0,t["{}.W0".format(key)])
    return t


def split_top_left_ci(t, key, n, c, h, ratio):
    # For when you expand from higher tier down to a lower one.i
    i = 0
    for r in range(ratio):
        for j in range(h*r, h*r+n):
            t["{}.W0".format(key)][i, j] = c
            i += 1
    t["{}.g0".format(key)] = np.apply_along_axis(np.linalg.norm,0,t["{}.W0".format(key)])
    return t

t = concat_top_left_ci(t, 'BigFrameLevel.GRU1.Step.Input', bfs, c, 2*h)
t = concat_top_left_ci(t, 'BigFrameLevel.GRU2.Step.Input', bfs, c, 2*h)

t = concat_top_left_ci(t, 'FrameLevel.GRU1.Step.Input', fs, c, 2*h)
t = concat_top_left_ci(t, 'FrameLevel.GRU2.Step.Input', fs, c, 2*h)

t = top_left_ci(t, 'FrameLevel.Output', c, fs)

t = split_top_left_ci(t, 'BigFrameLevel.Output', fs, c, h, int(bfs/fs))

for key in t.keys():
    t[key] = np.array(t[key]).astype('float32')

print("Saving to file...")
with open(save_file, 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
