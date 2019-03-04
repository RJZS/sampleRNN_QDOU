import numpy as np
import pickle

bfs = 80  # Big frame size
fs = 20  # Frame size
h = 1024  # Length of hidden state
c = 0.1

example_pkl_file = 'c_noise_random_start.pkl'  # Use this file to get keys of dictionary and dimensions of each matrix.
with open(example_pkl_file, 'rb') as f:
    t = pickle.load(f)  # Parameters are of type 'float32'.

for key in t.keys():
    t[key] = np.zeros(t[key].shape)


def top_left_ci(t, key, c, n):
    for i in range(n):
        t[key][i, i] = c
    return t


def concat_top_left_ci(t, key, n, c, start):
    # When multiple weight matrices are concatenated together, and you don't want the top one to be cI.
    for i in range(n):
        t[key][i, start+i] = c
    return t

t = concat_top_left_ci(t, 'BigFrameLevel.GRU1.Step.Input.W0', bfs, c, 2*h)
t = concat_top_left_ci(t, 'BigFrameLevel.GRU2.Step.Input.W0', bfs, c, 2*h)

t = concat_top_left_ci(t, 'FrameLevel.GRU1.Step.Input.W0', fs, c, 2*h)
t = concat_top_left_ci(t, 'FrameLevel.GRU2.Step.Input.W0', fs, c, 2*h)

t = top_left_ci(t, 'BigFrameLevel.Output.W0', c, bfs)
t = top_left_ci(t, 'FrameLevel.InputExpand.W0', c, fs)
t = top_left_ci(t, 'FrameLevel.Output.W0', c, fs)
