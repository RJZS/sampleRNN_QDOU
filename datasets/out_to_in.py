import numpy as np
from scipy.io import wavfile

# Script to take the output waveform and turn it into a .npy file with the appropriate form to pass it back in.
wav_dir = '/home/dawna/tts/rjzs2/noise_results_3t/c_noise_random_start'
wav_file_root = 'sample_e31_i200000_t39.03_tr6.8481_v6.5334'

out_dir = 'speech/ln_MA_f32_CE_8s_norm_utt'

noise = np.load('{}/speech_test_noise.npy'.format(out_dir))  # Noise to (partially) overwrite.
for utt in range(5):
    try:
        fs, audio = wavfile.read('{}/{}_{}.wav'.format(wav_dir, wav_file_root, utt))
        audio = audio.astype(np.float32)
        audio = (audio / np.amax(np.abs(audio))) + 1
        audio = (audio * 255) / 2
        audio = np.round(audio)
        audio = audio.astype(np.int32)
        noise[utt, :len(audio)] = audio
    except IOError:
        print("Could not find utterance {}. Skipping...".format(utt))
        continue

np.save('{}/speech_test_noise_1.npy'.format(out_dir), noise)
