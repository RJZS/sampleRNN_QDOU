#tool for frequency analysis
#input: experiment directory, epoch index
#epoch: save RMSE array, for further analysis

#1 find .wav under exp directory
#2 convert to int (from float)
#3 extract feature
#4 compute RMSE

import numpy
np = numpy
import os
import scipy.io.wavfile
import sys

from sklearn.metrics import mean_squared_error
from math import sqrt
import pdb

import subprocess
#------------------------------------
#subfunc: execute CMD, version 1
def exeCMD(cmd):
    print("running CMD: "+cmd)
    try:
        info=subprocess.check_output(cmd, shell=True,stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    print(info)
    return

def checkMakeDir(directory):
    if os.path.exists(directory):
        return True
    else:
        os.makedirs(directory)
        return False
#------------------------------------

#------------------------------------
BITRATE = 16000
class Configuration:
    def __init__(self,exp_dir,ep=20,
                 max_utt_nb=5,
                 ref_feature_dir='/home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/datasets/REF_TEST_RMDC/5s/features/',
                 flag_cvt_int=True,
                 flag_ext_ft=True,
                 flag_relative=False,
                 save_name='distMin_array',
                 find_by='ep'):
        self.exp_dir = exp_dir
        self.ep = ep
        self.wav_dir = os.path.join(self.exp_dir,'samples')
        self.wav_int_dir = os.path.join(self.wav_dir,'wav_int')
        self.feature_dir = os.path.join(self.exp_dir,'features')
        self.find_by = find_by
        if find_by=='ep':
            self.wav_int_dir_ep = os.path.join(self.wav_int_dir,'ep%s'%(self.ep))
            self.feature_dir_ep = os.path.join(self.feature_dir,'ep%s'%(self.ep))
        elif find_by=='iter':
            self.wav_int_dir_ep = os.path.join(self.wav_int_dir,'iter%s'%(self.ep))
            self.feature_dir_ep = os.path.join(self.feature_dir,'iter%s'%(self.ep))
        elif find_by=='name':
            self.wav_int_dir_ep = self.wav_int_dir
            self.feature_dir_ep = self.feature_dir
        self.max_utt_nb=max_utt_nb
        
        self.feature_dir_ep_meta = os.path.join(self.feature_dir_ep,'Meta')
        self.feature_dir_ep_usedwav = os.path.join(self.feature_dir_ep,'UsedWav')
        self.feature_dir_ep_spec = os.path.join(self.feature_dir_ep,'Spec')
        self.feature_dir_ep_f0 = os.path.join(self.feature_dir_ep,'F0')
        self.feature_dir_ep_noise = os.path.join(self.feature_dir_ep,'Noise')
        
        self.ref_feature_dir = ref_feature_dir
        self.ref_feature_dir_meta = os.path.join(self.ref_feature_dir,'Meta')
        self.ref_feature_dir_usedwav = os.path.join(self.ref_feature_dir,'UsedWav')
        self.ref_feature_dir_spec = os.path.join(self.ref_feature_dir,'Spec')
        self.ref_feature_dir_f0 = os.path.join(self.ref_feature_dir,'F0')
        self.ref_feature_dir_noise = os.path.join(self.ref_feature_dir,'Noise')
        
        self.flag_cvt_int = flag_cvt_int
        self.flag_ext_ft = flag_ext_ft
        self.flag_relative = flag_relative
        
        self.save_name = save_name
        
    
def convert_int_ep(cfg):
    #find .wav files to convert
    wav_list = os.listdir(cfg.wav_dir)
    if cfg.find_by=='ep':
        wav_list = [wav for wav in wav_list if '.wav' in wav and 'e%s'%(cfg.ep) in wav]
        if len(wav_list)>cfg.max_utt_nb:
            wav_list.sort()
            wav_list = wav_list[:cfg.max_utt_nb]
    elif cfg.find_by=='iter':
        wav_list = [wav for wav in wav_list if '.wav' in wav and 'i%s'%(cfg.ep) in wav]
        
    elif cfg.find_by=='name':
        wav_list = [wav for wav in wav_list if '.wav' in wav and cfg.ep in wav] #ep can be: hvd, YOYT...      
    
    #convert
    out_dir = cfg.wav_int_dir_ep
    checkMakeDir(out_dir)
    print 'converting files to',out_dir
    for wav in wav_list:
        print wav,
        #read
        dirFile = os.path.join(cfg.wav_dir,wav)
        data = scipy.io.wavfile.read(dirFile)[1]
        #convert if needed
        if data.dtype=='float32':
            data -= numpy.mean(data)
            data /= numpy.absolute(data).max() #(-1,1)
            data *= 32768.0
            data = data.astype('int16')
        else:
            assert data.dtype=='int16', 'data.dtype %s should be float32 or int16'%(data.dtype)
        #save
        dirFile = os.path.join(out_dir,wav)
        scipy.io.wavfile.write(dirFile,BITRATE,data)
    print 'OK!'
    return

def extract_feature_setup():
    exeCMD('source activate feature')
    exeCMD('export PYTHONPATH=${PYTHONPATH}:${PWD}:/home/dawna/tts/qd212/tools/cp_tools_GD/:/home/dawna/tts/mw545/tools/straight')
    return

def extract_feature(cfg):
    cmd = '/home/dawna/tts/qd212/tools/cp_tools_GD/common_tts/wavs2features.sh /home/dawna/tts/qd212/tools/cp_tools_GD/common_tts/common_tts.conf {inp} {out_meta} {out_usedwav} {out_spec} {out_f0} {out_noise}'
    cmd = cmd.format(inp=cfg.wav_int_dir_ep,
                     out_meta=cfg.feature_dir_ep_meta,
                     out_usedwav=cfg.feature_dir_ep_usedwav,
                     out_spec=cfg.feature_dir_ep_spec,
                     out_f0=cfg.feature_dir_ep_f0,
                     out_noise=cfg.feature_dir_ep_noise,
                     )
    exeCMD(cmd)
    return
#------------------------------------

#------------------------------------
def load_binary_file(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = numpy.fromfile(fid_lab, dtype=numpy.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    features = features[:(dimension * (features.size / dimension))]
    features = features.reshape((-1, dimension))

    return features
#------------------------------------

#------------------------------------
def proc_lfo(features):
    for i,f in enumerate(features):
        if f<0: features[i]=0
    return features

# def getRMSE(ref,rec):
#     return sqrt(mean_squared_error(ref, rec))

# #20180322 relative RMSE
def getRMSE(ref,rec,flag_relative=False):
    if flag_relative:
        tmp = np.zeros(ref.shape)
        norm = sqrt(mean_squared_error(ref, tmp))
        return sqrt(mean_squared_error(ref, rec))/norm
    else:
        return sqrt(mean_squared_error(ref, rec))

def getRecOKlen_addEnd(ref,rec):
    lref,lrec = len(ref),len(rec)
    if lref > lrec:
        zeroPad = np.zeros(lref - lrec)
        recOKlen = np.concatenate((rec,zeroPad))
    else:
        recOKlen = rec[:lref]
    return recOKlen

def getRecOKlen_addHead(ref,rec):
    lref,lrec = len(ref),len(rec)
    if lref > lrec:
        zeroPad = np.zeros(lref - lrec)
        recOKlen = np.concatenate((zeroPad,rec))
    else:
        recOKlen = rec[:lref]
    return recOKlen

#v2: slightly better sync
def getRecOKlen(ref,rec):
    return getRecOKlen_addHead(ref,rec)

#version2, bug fixed, 201801
def getDistIdxMin(ref,rec,flag_relative=False):
    lref,lrec = len(ref),len(rec)
    recOKlen = getRecOKlen(ref,rec)
    distMin = getRMSE(ref,recOKlen,flag_relative=flag_relative)
    idxMin = 0
    for idx in range(lref): # all start positions
        #build new rec
        tmp = getRecOKlen_addEnd(ref,recOKlen[idx:])
        #comp rmse with ref
        dist = getRMSE(ref,tmp,flag_relative=flag_relative)
        #if smaller than dist, replace dist
        if dist<distMin: distMin,idxMin = dist,idx
    return distMin,idxMin

# #version3, align in two ways, 201804
# def getDistIdxMin(ref,rec,flag_relative=False):
#     lref,lrec = len(ref),len(rec)
#     recOKlen = getRecOKlen(ref,rec)
#     distMin = getRMSE(ref,recOKlen,flag_relative=flag_relative)
#     idxMin = 0
#     flag_direction = 'forward'
#     for idx in range(lref): # all start positions
#         #build new rec, shift forward by adding at end
#         tmp = getRecOKlen_addEnd(ref,recOKlen[idx:])
#         #comp rmse with ref
#         dist = getRMSE(ref,tmp,flag_relative=flag_relative)
#         #if smaller than dist, replace dist
#         if dist<distMin: distMin,idxMin,flag_direction = dist,idx,'forward'

#         #build new rec, shift backward by adding at head
#         tmp = getRecOKlen_addHead(ref,recOKlen[:idx])
#         #comp rmse with ref
#         dist = getRMSE(ref,tmp,flag_relative=flag_relative)
#         #if smaller than dist, replace dist
#         if dist<distMin: distMin,idxMin,flag_direction = dist,idx,'backward'
#     return distMin,idxMin,flag_direction

#201801, speed up
#only one axis, for dimension
def getRMSE_list(features_ref,features_rec,flag_relative=False):
    distMin_list = []
    dimIdx = 0
    ref = features_ref[:,dimIdx]
    rec = features_rec[:,dimIdx]
    distMin,idxMin = getDistIdxMin(ref,rec,flag_relative=flag_relative)
    distMin_list.append(distMin)
    
    for dimIdx in range(1,features_ref.shape[1]):
        ref = features_ref[:,dimIdx]
        rec = features_rec[:,dimIdx]
        recOKlen = getRecOKlen(ref,rec)
        tmp = getRecOKlen_addEnd(ref,recOKlen[idxMin:])
        distMin = getRMSE(ref,tmp,flag_relative=flag_relative)
        distMin_list.append(distMin)
    return distMin_list

# #version2, align in two ways, 201804
# def getRMSE_list(features_ref,features_rec,flag_relative=False):
#     distMin_list = []
#     dimIdx = 0
#     ref = features_ref[:,dimIdx]
#     rec = features_rec[:,dimIdx]
#     distMin,idxMin,flag_direction = getDistIdxMin(ref,rec,flag_relative=flag_relative)
#     distMin_list.append(distMin)
    
#     for dimIdx in range(1,features_ref.shape[1]):
#         ref = features_ref[:,dimIdx]
#         rec = features_rec[:,dimIdx]
#         recOKlen = getRecOKlen(ref,rec)
#         if flag_direction=='forward': tmp = getRecOKlen_addEnd(ref,recOKlen[idxMin:])
#         if flag_direction=='backward': tmp = getRecOKlen_addHead(ref,recOKlen[:idxMin])
#         distMin = getRMSE(ref,tmp,flag_relative=flag_relative)
#         distMin_list.append(distMin)
#     return distMin_list


#axis0 is for utt, axis1 is for dimension
def getRMSE_distMin_array(src_dir_ref,src_dir_rec,nbDim=60,name_pattern='.mcep',flag_relative=False):
    ref_name_list = sorted(os.listdir(src_dir_ref))
    ref_name_list = [n for n in ref_name_list if name_pattern in n and '.txt' not in n]
    rec_name_list = sorted(os.listdir(src_dir_rec))
    rec_name_list = [n for n in rec_name_list if name_pattern in n and '.txt' not in n]

    distMin_array = np.array([])
    cnt = 0
    print 'processing utt:',
    for ref_name,rec_name in zip(ref_name_list,rec_name_list):
        cnt+=1
        print cnt,
    #     if cnt>2: break
        dirFile = os.path.join(src_dir_ref,ref_name)
        features_ref = load_binary_file(dirFile, nbDim)
        dirFile = os.path.join(src_dir_rec,rec_name)
        features_rec = load_binary_file(dirFile, nbDim)
        if name_pattern=='.lf0':
            features_ref = proc_lfo(features_ref)
            features_rec = proc_lfo(features_rec)
        distMin_utt_array = np.array(getRMSE_list(features_ref,features_rec,flag_relative=flag_relative))
        distMin_utt_array = distMin_utt_array.reshape([1,-1])
        if len(distMin_array) == 0:
            distMin_array = distMin_utt_array
        else:
            distMin_array = np.concatenate((distMin_array,distMin_utt_array))
    print 'processing complete:)'

    return distMin_array

def compute_rmse(cfg):
    name = cfg.save_name
    if cfg.flag_relative:
        name += '_relative'
        print 'relative RMSE'
    else:
        print 'absolute RMSE'
    
    print '1) spec - mcep'
    distMin_array = getRMSE_distMin_array(cfg.ref_feature_dir_spec,cfg.feature_dir_ep_spec,flag_relative=cfg.flag_relative)
    np.save(os.path.join(cfg.feature_dir_ep_spec,name),distMin_array)
    
    print '2) noise - bndap'
    distMin_array = getRMSE_distMin_array(cfg.ref_feature_dir_noise,cfg.feature_dir_ep_noise,nbDim=21,name_pattern='.bndap',flag_relative=cfg.flag_relative)
    np.save(os.path.join(cfg.feature_dir_ep_noise,name),distMin_array)
    
    print '3) f0 - lf0'
    distMin_array = getRMSE_distMin_array(cfg.ref_feature_dir_f0,cfg.feature_dir_ep_f0,nbDim=1,name_pattern='.lf0',flag_relative=cfg.flag_relative)
    np.save(os.path.join(cfg.feature_dir_ep_f0,name),distMin_array)
    return

def compute_rmse_PML(cfg):
    name = cfg.save_name
    if cfg.flag_relative:
        name += '_relative'
        print 'relative RMSE'
    else:
        print 'absolute RMSE'
    
    print '1) spec - mcep'
    distMin_array = getRMSE_distMin_array(cfg.ref_feature_dir_spec,cfg.feature_dir_ep_spec,flag_relative=cfg.flag_relative)
    np.save(os.path.join(cfg.feature_dir_ep_spec,name),distMin_array)
    print '2) noise - bndnm'
    distMin_array = getRMSE_distMin_array(cfg.ref_feature_dir_noise,cfg.feature_dir_ep_noise,nbDim=21,name_pattern='.bndnm',flag_relative=cfg.flag_relative)
    np.save(os.path.join(cfg.feature_dir_ep_noise,name),distMin_array)
    print '3) f0 - lf0'
    distMin_array = getRMSE_distMin_array(cfg.ref_feature_dir_f0,cfg.feature_dir_ep_f0,nbDim=1,name_pattern='.lf0',flag_relative=cfg.flag_relative)
    np.save(os.path.join(cfg.feature_dir_ep_f0,name),distMin_array)
    return
#------------------------------------

#------------------------------------
if __name__ == '__main__':
    #0 parse arguments
    assert len(sys.argv)>=2, 'should pass arguments to python: exp_dir, ep'
    exp_dir = sys.argv[1]
    ep = sys.argv[2]
    find_by = sys.argv[3] if len(sys.argv)>3 else 'ep'
    assert find_by in ['ep','iter','name'], 'find_by should be: iter / ep / name'
    
    tmp='/home/dawna/tts/qd212/mphilproj/sampleRNN_QDOU/results_4t/gen/ref/features/'
    ref_feature_dir = sys.argv[4] if len(sys.argv)>4 else tmp
    cfg = Configuration(exp_dir,ep,ref_feature_dir=ref_feature_dir,
                        flag_cvt_int=True,flag_ext_ft=True,flag_relative=False,
                        save_name='distMin_array',find_by=find_by)
    
    
    
    #1 find .wav under exp directory
    #2 convert to int (from float)
    print 'STEP 1&2 converting to int waveform'
    if cfg.flag_cvt_int: convert_int_ep(cfg)
    else: print 'already ok!'
    
    #3 extract feature
    print 'STEP 3 extracting features'
    if cfg.flag_ext_ft: extract_feature(cfg)
    else: print 'already ok!'
    
#     #4 compute RMSE
    print 'STEP 4 computing RMSE'
    compute_rmse(cfg)

#------------------------------------
#     cfg.flag_relative = not cfg.flag_relative
#     compute_rmse(cfg)


    
    