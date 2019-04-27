import numpy as np

folder = "speech/ln_MA_f32_CE_8s_norm_utt"
mu = 255
files = {"speech_test": "speech_test_noise", "speech_valid": "speech_valid_noise", "speech_train": "speech_train_noise",
         "speech_test_gen": "speech_test_noise_gen"}

for k, v in files.iteritems():
    print(k)
    inp = np.load("{}/{}.npy".format(folder, k))
    inp = 2*inp - 1
    x_mu = np.sign(inp) * np.log(1 + mu * np.abs(inp)) / np.log(1 + mu)
    x_mu = ((x_mu + 1)/2 * mu).astype('int16')
    np.save("{}/{}.npy".format(folder, v), x_mu)
