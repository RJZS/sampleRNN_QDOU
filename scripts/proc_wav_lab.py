import numpy
np = numpy
import os
import scipy.io.wavfile
import pdb

import sys

sys.path.insert(0, '/home/dawna/tts/qd212/lib_QDOU')

from CMD_bash import *
from IO_wav_lab import *

# pdb.set_trace()

class Configuration:
    def __init__(self,flag_mk_utt=True,
                flag_norm_wav=False,
                flag_save_lab_mtx=True):
        self.work_dir = '/home/dawna/tts/qd212/mphilproj/data'
        self.data_dir = os.path.join(self.work_dir, 'speech')
        self.wav_dir = os.path.join(self.data_dir, 'speechRawWavData')
        self.lab_dir = os.path.join(self.data_dir, 'speechGenCmpData')
        # self.lab_dir = '/home/dawna/tts/gad27/exp/char2wav/upcls_Nick_3xBLSTM1024_16kHz_LSE/out/model-gen/' #alternative
        self.file_id_list = os.path.join(self.data_dir, 'file_id_list.scp')
        
        self.output_dir_wav = os.path.join(self.data_dir, 'speechNpyData/wav')
        self.output_dir_lab = os.path.join(self.data_dir, 'speechNpyData/lab')
        self.output_dir_wav_utt = os.path.join(self.output_dir_wav, 'utt_float32')
        self.output_dir_lab_utt = os.path.join(self.output_dir_lab, 'utt_traj') #utt_lab
        
        self.wav_fmt = '.used.wav' #'.wav'
        self.lab_fmt = '.cmp' #'.lab'
        self.lab_dim = 163
        
        self.wav_utt_name_pattern = 'speech_{}_utt.npy'
        self.lab_utt_name_pattern = 'speech_{}_utt_traj.npy' #speech_{}_utt_lab.npy
        
        self.file_dict = {}
        self.file_dict['train_file'] = {'start_file_index': 0, 'end_file_index': 2253, 'nb_repeat':0}
        self.file_dict['valid_file'] = {'start_file_index': 2254, 'end_file_index': 2323, 'nb_repeat':0}
        self.file_dict['test_file'] = {'start_file_index': 2324, 'end_file_index': 2395, 'nb_repeat':0}
        
        self.flag_mk_utt = flag_mk_utt
        self.flag_norm_wav = flag_norm_wav
        self.flag_save_lab_mtx = flag_save_lab_mtx
        
        if self.flag_norm_wav:
            self.output_dir_wav_mtx = os.path.join(self.output_dir_wav,'MA_f32_8s_norm_utt')
        else:
            self.output_dir_wav_mtx = os.path.join(self.output_dir_wav,'MA_f32_8s')
            
        # self.output_dir_lab_mtx = os.path.join(self.output_dir_lab,'MA_lab_8s_norm')
        self.output_dir_lab_mtx = os.path.join(self.output_dir_lab,'MA_traj_8s')
        self.output_dir_lab_mtx_norm = os.path.join(self.output_dir_lab,'MA_traj_8s_norm')
        

#20180228
def mk_utt_files(fileList,cfg):
    # wav_dir = 'lesley/16k_resil_LesleySimsFixedPauses/'
    # lab_dir = '/home/dawna/tts/zm273/LesleySimsFixedPauses_baseline/data/nn_no_silence_lab_norm_601'
    output_dir_wav_utt = cfg.output_dir_wav_utt
    output_dir_lab_utt = cfg.output_dir_lab_utt
    checkMakeDir(output_dir_wav_utt)
    checkMakeDir(output_dir_lab_utt)
    
    speech_utt = []
    speech_utt_lab = []
    cnt = 0
    print 'making utt npy files: '
    for f in fileList:
        cnt += 1
        if cnt%500==0: print cnt,
        wav = readWav(os.path.join(cfg.wav_dir,f+cfg.wav_fmt))
        lab = load_binary_file(os.path.join(cfg.lab_dir,f+cfg.lab_fmt),cfg.lab_dim)
        speech_utt.append(wav)
        speech_utt_lab.append(lab)
        
    wavDict,labDict = {},{}
    for x in ['train','valid','test']:
        x_file = '{}_file'.format(x)
        wavDict[x] = speech_utt[cfg.file_dict[x_file]['start_file_index']:cfg.file_dict[x_file]['end_file_index']+1]
        labDict[x] = speech_utt_lab[cfg.file_dict[x_file]['start_file_index']:cfg.file_dict[x_file]['end_file_index']+1]

    #remove one utt, whose wav is much shorter than lab
    wavDict['train'] = numpy.delete(wavDict['train'],[1073])
    labDict['train'] = numpy.delete(labDict['train'],[1073])

    print 'ok, saving...'
    for x in ['train','valid','test']:
        print x,
        # tmp = os.path.join(output_dir_wav_utt,cfg.wav_utt_name_pattern.format(x))
        # numpy.save(tmp,wavDict[x])
        tmp = os.path.join(output_dir_lab_utt,cfg.lab_utt_name_pattern.format(x))
        print ' saved to:',tmp
        numpy.save(tmp,labDict[x])
    print 'ok!'
    return

#20180228
def mk_utt_files_align(fileList,cfg):
    # wav_dir = 'lesley/16k_resil_LesleySimsFixedPauses/'
    # lab_dir = '/home/dawna/tts/zm273/LesleySimsFixedPauses_baseline/data/nn_no_silence_lab_norm_601'
    output_dir_wav_utt = cfg.output_dir_wav_utt
    output_dir_lab_utt = cfg.output_dir_lab_utt
    checkMakeDir(output_dir_wav_utt)
    checkMakeDir(output_dir_lab_utt)
    
    speech_utt = []
    speech_utt_lab = []
    cnt = 0
    lab_ref_dir = os.path.join(cfg.data_dir, 'speechRawLabData/binary_label_601')
    print 'making utt npy files: '
    for f in fileList:
        cnt += 1
        if cnt%500==0: print cnt,
        wav = readWav(os.path.join(cfg.wav_dir,f+cfg.wav_fmt))
        lab = load_binary_file(os.path.join(cfg.lab_dir,f+cfg.lab_fmt),cfg.lab_dim)
        lab_ref = load_binary_file(os.path.join(lab_ref_dir,f+'.lab'),601)
        if len(lab)>=len(lab_ref):
            lab = lab[:len(lab_ref)]
        else:
            tmp = len(lab_ref)-len(lab)
            lab = np.concatenate([lab,lab[-tmp:]])
        speech_utt.append(wav)
        speech_utt_lab.append(lab)
        
    wavDict,labDict = {},{}
    for x in ['train','valid','test']:
        x_file = '{}_file'.format(x)
        wavDict[x] = speech_utt[cfg.file_dict[x_file]['start_file_index']:cfg.file_dict[x_file]['end_file_index']+1]
        labDict[x] = speech_utt_lab[cfg.file_dict[x_file]['start_file_index']:cfg.file_dict[x_file]['end_file_index']+1]

    print 'ok, saving...'
    for x in ['train','valid','test']:
        print x,
        # tmp = os.path.join(output_dir_wav_utt,cfg.wav_utt_name_pattern.format(x))
        # numpy.save(tmp,wavDict[x])
        tmp = os.path.join(output_dir_lab_utt,cfg.lab_utt_name_pattern.format(x))
        print ' saved to:',tmp
        numpy.save(tmp,labDict[x])
    print 'ok!'
    return


#pre 2018
def printItemShape(d):
    for k,v in d.items():
        print(k),
        print(v.shape)

def checkIfMoreWav(wavs,labs):
    cnt = 0
    cntPb = 0
    for wav,lab in zip(wavs,labs):
        cnt += 1
        alignLen = lab.shape[0]*80
        if wav.shape[0]<alignLen:
            cntPb += 1
            print('wav less than lab*80')
            print(cnt)
            print(wav.shape[0],alignLen)
    if cntPb==0:
        print('all is well, wav>lab')
    return

def concatAll(wavs,labs):
    #input: list, speech_{}_utt.npy, speech_{}_utt_lab/traj.npy
    #output: array, wav_all_array,lab_all_array
    wav_all_array = np.array([])
    lab_all_array = np.array([])

    for wav,lab in zip(wavs,labs):
        alignLen = lab.shape[0]*80
        #wav_all_array = np.concatenate((wav_all_array,wav[-alignLen:]))
        wav_all_array = np.concatenate((wav_all_array,wav[:alignLen]))
        if len(lab_all_array)==0:
            lab_all_array = lab
        else:
            lab_all_array = np.concatenate((lab_all_array,lab))
    
    return wav_all_array,lab_all_array

def cutEqLen(wav_all_array,lab_all_array,lab_dim=601):
    nb_sec = 8
    # cut ending, reshape into 8-sec rows
    allLen = len(wav_all_array)
    rowLen = nb_sec*16000
    rowNb = allLen//rowLen
    wav_all_array_save = wav_all_array[:rowNb*rowLen].reshape(rowNb,rowLen)
    print('wav_all_array_save.shape:'),
    print(wav_all_array_save.shape)
    
    # cut ending, reshape into 8-sec rows
    allLen = len(lab_all_array)
    rowLen = nb_sec*16000/80
    rowNb = allLen//rowLen
    lab_all_array_save = lab_all_array[:rowNb*rowLen].reshape(rowNb,rowLen,lab_dim)
    print('lab_all_array_save.shape:'),
    print(lab_all_array_save.shape)

    return wav_all_array_save,lab_all_array_save

def norm_utt(wav_utt):
    wav_utt_norm = wav_utt
    for utt in wav_utt_norm:
        utt -= utt.mean() #[-1,1], zero mean
        utt /= abs(utt).max()
        utt /= 2
        utt += 0.5 #[0,1],0.5 mean as if zero mean
    return wav_utt_norm


def mk_mtx_files(cfg, file_list):
    print('1 -------------- prepare utt & utt_lab')
    wavDict,labDict = {},{}
    for x in ['train','valid','test']:
        dirFile = os.path.join(cfg.output_dir_wav_utt,cfg.wav_utt_name_pattern.format(x))
        wavDict[x] = numpy.load(dirFile)
        dirFile = os.path.join(cfg.output_dir_lab_utt,cfg.lab_utt_name_pattern.format(x))
        labDict[x] = numpy.load(dirFile)

    #remove one utt, whose wav is much shorter than lab
    wavDict['train'] = numpy.delete(wavDict['train'],[1073])
    labDict['train'] = numpy.delete(labDict['train'],[1073])
    
    printItemShape(wavDict)
    printItemShape(labDict)
    
    for k in wavDict:
        print(k+': '),
        checkIfMoreWav(wavDict[k],labDict[k])

    # pdb.set_trace()
    
    
    if cfg.flag_norm_wav:
        print('1.5 -------------- normalize on utt level: rm mean, increase volume')
        for k in wavDict:
            print(k+': '),
            wavDict[k] = norm_utt(wavDict[k])
            print 'ok'

    print('2 -------------- manually align utt & utt_lab, get wav_all_array_save & lab_all_array_save')
    wavSaveDict = {}
    labSaveDict = {}
    for k in wavDict:
        print(k+': ')
        wav_all_array,lab_all_array = concatAll(wavDict[k],labDict[k])
        if k in ['valid','test']:
            tmp_wav,tmp_lab = wav_all_array,lab_all_array
            nb_repeat = cfg.file_dict['{}_file'.format(k)]['nb_repeat']#should be 0 normally
            print('nb_repeat: '+str(nb_repeat))
            for i in range(nb_repeat):
                wav_all_array = np.concatenate((wav_all_array,tmp_wav))
                lab_all_array = np.concatenate((lab_all_array,tmp_lab))
        wav_all_array_save,lab_all_array_save = cutEqLen(wav_all_array,lab_all_array,lab_dim=cfg.lab_dim)
        wavSaveDict[k],labSaveDict[k] = wav_all_array_save,lab_all_array_save

    print('3 -------------- save wav_all_array_save & lab_all_array_save')
    tgt_dir_wav = cfg.output_dir_wav_mtx
    tgt_dir_lab = cfg.output_dir_lab_mtx
    checkMakeDir(tgt_dir_wav)
    checkMakeDir(tgt_dir_lab)

    for k in wavDict:
        print(k+': ')
        # numpy.save(tgt_dir_wav+'/speech_{}.npy'.format(k),wavSaveDict[k])
        if cfg.flag_save_lab_mtx:
            numpy.save(tgt_dir_lab+'/speech_{}_traj.npy'.format(k),labSaveDict[k])
            # numpy.save(tgt_dir_lab+'/speech_{}_lab.npy'.format(k),labSaveDict[k])
            
            
    
    # print('00 -------------- complete, lab yet to be normed')
    print('4 -------------- norm lab/traj')
    #0 read data
    #lab is in labSaveDict

    #1 normalize
    labNormDict = {}
    
    #option 2-2: fit normalizer on training data
    lab = labSaveDict['train']
    rowNb,rowLen,featNb = lab.shape
    lab = lab.reshape(rowNb*rowLen,featNb)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(lab)
    for k in labSaveDict:
        lab = labSaveDict[k]
        lab_normed = getMappedLabData_uni(lab,min_max_scaler)
        labNormDict[k] = lab_normed

    #2 save
    tgt_dir = cfg.output_dir_lab_mtx_norm
    checkMakeDir(tgt_dir)
    for k in labNormDict:
        lab_normed = labNormDict[k]
        numpy.save(tgt_dir + '/speech_{}_traj.npy'.format(k),lab_normed)
    print('00 -------------- generated normed lab')
    return

def mk_utt_files_norm(cfg):
    #0 load mtx.npy for train, load utt.npy for test
    
    
    #1 wav
    #1.1 compute MinMaxScaler with train_mtx.npy
    dirFile = os.path.join(cfg.output_dir_wav,'manuAlign_float32_cutEnd/speech_train.npy')
    # dirFile = '/home/dawna/tts/qd212/mphilproj/data/speech/speechNpyData/wav/manuAlign_float32_cutEnd/speech_train.npy'
    wav = numpy.load(dirFile)
    wavMax = abs(wav).max()*2
    
    #1.2 normalize utt.npy
    dirFile = os.path.join(cfg.output_dir_wav, 'utt_float32/speech_test_utt.npy')
    wav_utt = numpy.load(dirFile)
    for i,w in enumerate(wav_utt):
        w = w-w.mean() #[-1,1] zero-mean
        w /= wavMax #[-0.5,0.5] zero-mean
        w += 0.5 #[0,1] 0.5-mean, corresponding to zero-mean when using [0,255] quantization
        wav_utt[i] = w

    #1.3 save
    tgt_dir = os.path.join(cfg.output_dir_wav, 'utt_float32_norm')
    checkMakeDir(tgt_dir)
    # pdb.set_trace()
    np.save(tgt_dir + '/speech_{}_utt.npy'.format('test'),wav_utt)
    
    
    #2 lab
    #2.1 compute MinMaxScaler with train_mtx.npy
    dirFile = os.path.join(cfg.output_dir_lab,'manuAlign_lab/speech_train_lab.npy')
    # dirFile = '/home/dawna/tts/qd212/mphilproj/data/speech/speechNpyData/lab/manuAlign_lab/speech_train_lab.npy'
    lab = np.load(dirFile)
    
    rowNb,rowLen,featNb = lab.shape
    lab = lab.reshape(rowNb*rowLen,featNb)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(lab)
    
    #2.2 normalize utt.npy
    dirFile = os.path.join(cfg.output_dir_lab,'utt_lab/speech_test_utt_lab.npy')
    # dirFile = '/home/dawna/tts/qd212/mphilproj/data/speech/speechNpyData/lab/utt_lab/speech_test_utt_lab.npy'
    lab_utt = np.load(dirFile)
    for i,l in enumerate(lab_utt):
        l = min_max_scaler.transform(l)
        lab_utt[i]=l
    
    #2.3 save
    tgt_dir = os.path.join(cfg.output_dir_lab, 'utt_lab_norm')
    checkMakeDir(tgt_dir)
    # pdb.set_trace()
    np.save(tgt_dir + '/speech_{}_utt_lab.npy'.format('test'),lab_utt)
    
    
    #3 traj
    #3.1 compute MinMaxScaler with train_mtx.npy
    dirFile = os.path.join(cfg.output_dir_lab_mtx,'speech_{}_traj.npy'.format('train'))
    lab = np.load(dirFile)
    
    rowNb,rowLen,featNb = lab.shape
    lab = lab.reshape(rowNb*rowLen,featNb)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(lab)
    
    #3.2 normalize utt.npy
    dirFile = os.path.join(cfg.output_dir_lab_utt,'speech_{}_utt_traj.npy'.format('test'))
    lab_utt = np.load(dirFile)
    for i,l in enumerate(lab_utt):
        l = min_max_scaler.transform(l)
        lab_utt[i]=l
    
    #3.3 save
    tgt_dir = os.path.join(cfg.output_dir_lab, 'utt_traj_norm')
    checkMakeDir(tgt_dir)
    # pdb.set_trace()
    np.save(tgt_dir + '/speech_{}_utt_traj.npy'.format('test'),lab_utt)
    return

def mk_mtx_files_NCY(cfg):
    print('1 -------------- prepare utt & utt_lab')
    print 'skipped'
            
    print('1.5 -------------- '),
    # if cfg.flag_wav_norm: print 'normalize on utt level: rm mean, increase volume'
    if cfg.flag_norm_wav: print 'normalize on utt level: rm mean, increase volume'
    else: print 'normalize on cps level: rm mean here'
    print 'skipped'

    print('2 -------------- manually align utt & utt_lab, get wav_all_array_save & lab_all_array_save')
    print 'skipped'

    print('3 -------------- save wav_all_array_save & lab_all_array_save')
    print 'skipped'
    
    print('4 -------------- norm with Nancy training data')
    print('4.1 -------------- norm wav')
    #1 wav
    #1.1 compute MinMaxScaler with Nancy train_mtx.npy
    wav = numpy.load('/home/dawna/tts/qd212/data/nancy/npyData/wav/MA_8s/train.npy')
    wavMax = abs(wav).max()*2
    
    #1.2 normalize Nick train_mtx.npy
    wavDict = {}
    tmp = '/home/dawna/tts/qd212/mphilproj/data/'
    wavDict['train'] = numpy.load(tmp + 'speech/speechNpyData/wav/manuAlign_float32_cutEnd/speech_train.npy')
    wavDict['valid'] = numpy.load(tmp + 'speech/speechNpyData/wav/manuAlign_float32_cutEnd/speech_valid.npy')
    wavDict['test'] = numpy.load(tmp + 'speech/speechNpyData/wav/manuAlign_float32_cutEnd/speech_test.npy')
    
    wavNormDict = {}
    for k in wavDict:
        wav = wavDict[k]
        wav_normed = wav #[-1,1] zero-mean
        wav_normed /= wavMax #[-0.5,0.5]ish zero-mean
        wav_normed += 0.5 #[0,1]ish 0.5-mean, corresponding to zero-mean when using [0,255] quantization
        wavNormDict[k] = wav_normed

    #1.3 save
    tgt_dir = os.path.join(cfg.output_dir_wav, 'MA_8s_norm_NCY')
    checkMakeDir(tgt_dir)
    for k in wavNormDict:
        print k
        wav_normed = wavNormDict[k]
        numpy.save(tgt_dir + '/speech_{}.npy'.format(k),wav_normed)
    
    
    print('4.2 -------------- norm lab/traj')
    #0 read data
    labDict = {}
    tmp = '/home/dawna/tts/qd212/mphilproj/data/'
    labDict['train'] = numpy.load(tmp + 'speech/speechNpyData/lab/MA_traj_8s/speech_train_traj.npy')
    labDict['valid'] = numpy.load(tmp + 'speech/speechNpyData/lab/MA_traj_8s/speech_valid_traj.npy')
    labDict['test'] = numpy.load(tmp + 'speech/speechNpyData/lab/MA_traj_8s/speech_test_traj.npy')

    #1 normalize
    #option 2-2: fit normalizer on training data
    lab = numpy.load('/home/dawna/tts/qd212/data/nancy/npyData/trj/MA_8s/train_trj.npy')
    rowNb,rowLen,featNb = lab.shape
    lab = lab.reshape(rowNb*rowLen,featNb)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(lab)
    
    labNormDict = {}
    for k in labDict:
        lab = labDict[k]
        lab_normed = getMappedLabData_uni(lab,min_max_scaler)
        labNormDict[k] = lab_normed

    #2 save
    tgt_dir = os.path.join(cfg.output_dir_lab, 'MA_traj_8s_norm_NCY')
    checkMakeDir(tgt_dir)
    for k in labNormDict:
        print k
        lab_normed = labNormDict[k]
        numpy.save(tgt_dir + '/speech_{}_traj.npy'.format(k),lab_normed)
    print('00 -------------- generated normed lab')
    return

def mk_utt_files_norm_NCY(cfg):
    #0 load mtx.npy for train, load utt.npy for test
    #1 wav
    #1.1 compute MinMaxScaler with train_mtx.npy
    dirFile = '/home/dawna/tts/qd212/data/nancy/npyData/wav/MA_8s/train.npy'
    wav = numpy.load(dirFile)
    wavMax = abs(wav).max()*2
    
    #1.2 normalize utt.npy
    dirFile = os.path.join(cfg.output_dir_wav, 'utt_float32/speech_test_utt.npy')
    wav_utt = numpy.load(dirFile)
    for i,w in enumerate(wav_utt):
        w = w-w.mean() #[-1,1] zero-mean
        w /= wavMax #[-0.5,0.5] zero-mean
        w += 0.5 #[0,1] 0.5-mean, corresponding to zero-mean when using [0,255] quantization
        wav_utt[i] = w

    #1.3 save
    tgt_dir = os.path.join(cfg.output_dir_wav, 'utt_float32_norm_NCY')
    checkMakeDir(tgt_dir)
    # pdb.set_trace()
    np.save(tgt_dir + '/speech_{}_utt.npy'.format('test'),wav_utt)
    
    
#     #2 lab
#     #2.1 compute MinMaxScaler with train_mtx.npy
#     dirFile = os.path.join(cfg.output_dir_lab,'manuAlign_lab/speech_train_lab.npy')
#     # dirFile = '/home/dawna/tts/qd212/mphilproj/data/speech/speechNpyData/lab/manuAlign_lab/speech_train_lab.npy'
#     lab = np.load(dirFile)
    
#     rowNb,rowLen,featNb = lab.shape
#     lab = lab.reshape(rowNb*rowLen,featNb)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler.fit_transform(lab)
    
#     #2.2 normalize utt.npy
#     dirFile = os.path.join(cfg.output_dir_lab,'utt_lab/speech_test_utt_lab.npy')
#     # dirFile = '/home/dawna/tts/qd212/mphilproj/data/speech/speechNpyData/lab/utt_lab/speech_test_utt_lab.npy'
#     lab_utt = np.load(dirFile)
#     for i,l in enumerate(lab_utt):
#         l = min_max_scaler.transform(l)
#         lab_utt[i]=l
    
#     #2.3 save
#     tgt_dir = os.path.join(cfg.output_dir_lab, 'utt_lab_norm')
#     checkMakeDir(tgt_dir)
#     # pdb.set_trace()
#     np.save(tgt_dir + '/speech_{}_utt_lab.npy'.format('test'),lab_utt)
    
    
    #3 traj
    #3.1 compute MinMaxScaler with train_mtx.npy
    dirFile = '/home/dawna/tts/qd212/data/nancy/npyData/trj/MA_8s/train_trj.npy'
    lab = np.load(dirFile)
    
    rowNb,rowLen,featNb = lab.shape
    lab = lab.reshape(rowNb*rowLen,featNb)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(lab)
    
    #3.2 normalize utt.npy
    dirFile = os.path.join(cfg.output_dir_lab_utt,'speech_{}_utt_traj.npy'.format('test'))
    lab_utt = np.load(dirFile)
    for i,l in enumerate(lab_utt):
        l = min_max_scaler.transform(l)
        lab_utt[i]=l
    
    #3.3 save
    tgt_dir = os.path.join(cfg.output_dir_lab, 'utt_traj_norm_NCY')
    checkMakeDir(tgt_dir)
    # pdb.set_trace()
    np.save(tgt_dir + '/speech_{}_utt_traj.npy'.format('test'),lab_utt)
    return

    
if __name__ == '__main__': 
    
    cfg = Configuration()
    file_list = get_file_list(cfg.file_id_list)
    
    # if cfg.flag_mk_utt:
        # mk_utt_files(file_list,cfg)
        # mk_utt_files_align(file_list,cfg)
        
    # mk_mtx_files(cfg, file_list)
    
    # mk_utt_files_norm(cfg)
    # mk_utt_files_norm_NCY(cfg)
    
    mk_mtx_files_NCY(cfg)
    
    
    

