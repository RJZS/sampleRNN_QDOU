import numpy as np
import pickle

bfs = 80  # Big frame size
fs = 20  # Frame size
h = 1024  # Length of hidden state
q_levels = 256  # Defined in run script (.sh script to run job)
emb_size = 256  # Defiend in run script(.sh script to run job)
c = 1

example_pkl_file = 'bkup/identity_transform.pkl'  # Use this file to get keys of dictionary and dimensions of each matrix.
save_file = 'bkup/identity_and_dnn.pkl'
with open(example_pkl_file, 'rb') as f:
    s = pickle.load(f)  # Parameters are of type 'float32'.

print("Loaded file!")

t = {}
for key in s.keys():
    t[key] = np.zeros(s[key].shape)


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


def init_he(t, key, inp_dim, output_dim):
    t["{}.W0".format(key)] = uniform(np.sqrt(2. / inp_dim), (inp_dim, output_dim))
    return t


def init_lecun(t, key, inp_dim, output_dim):
    t["{}.W0".format(key)] = uniform(np.sqrt(1. / inp_dim), (inp_dim, output_dim))
    return t


def init_embedding(t, key, q_levels, emb_size):
    t[key] = np.random.randn(
            q_levels,
            emb_size
        ).astype('float32')

    return t

def uniform(stdev, size):
    """
    uniform distribution with the given stdev and size

    From Ishaan's code:
        https://github.com/igul222/speech
    """
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype('float32')

t = concat_top_left_ci(t, 'BigFrameLevel.GRU1.Step.Input', bfs, c, 2*h)
t = concat_top_left_ci(t, 'BigFrameLevel.GRU2.Step.Input', bfs, c, 2*h)

t = concat_top_left_ci(t, 'FrameLevel.GRU1.Step.Input', fs, c, 2*h)
t = concat_top_left_ci(t, 'FrameLevel.GRU2.Step.Input', fs, c, 2*h)

t = split_top_left_ci(t, 'FrameLevel.Output', 1, c, h, fs)
t = split_top_left_ci(t, 'BigFrameLevel.Output', fs, c, h, int(bfs/fs))

t = init_he(t, 'SampleLevel.L1_PrevSamples', h, h)
t = init_he(t, 'SampleLevel.L2', h, h)
t = init_he(t, 'SampleLevel.L3', h, h)

t = init_lecun(t, 'SampleLevel.Output', h, q_levels)
t = init_embedding(t, 'SampleLevel.Embedding', q_levels, emb_size)

bias = np.zeros(2*h)
bias[:h] = 4
t['BigFrameLevel.GRU1.Step.Recurrent_Gates.b'] = bias
t['BigFrameLevel.GRU2.Step.Recurrent_Gates.b'] = bias
t['FrameLevel.GRU1.Step.Recurrent_Gates.b'] = bias
t['FrameLevel.GRU2.Step.Recurrent_Gates.b'] = bias

for key in t.keys():
    t[key] = np.array(t[key]).astype('float32')
    if key.endswith('.g0'):
        t.pop(key)

print("Saving to file...")
with open(save_file, 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
