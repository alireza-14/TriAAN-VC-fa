import warnings
warnings.filterwarnings(action='ignore')
import os
from os.path import join as opj
import json
from pathlib import Path
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import *
from src.wavlm import *
from preprocess.audio import *

def _load_wav(path):
    
    wav, fs = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=60)
    if fs != 32000:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=32000, axis=0)

    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak

    wav  = torch.from_numpy(wav).unsqueeze(0).float()
    
    return wav


def load_wavlm(checkpoint_path):
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg)
    cmodel.load_state_dict(checkpoint['model'])
    return cmodel


def main(cfg):
    
    data_path = Path(cfg.data_path)
    wavlm       = load_wavlm(f'{cfg.cpc_path}/WavLM-Large.pt').cuda()
    wavlm.eval()
    with torch.no_grad():
        modes = ['train', 'valid', 'test']
        for mode in modes:
            metadata = Read_json(data_path/f'{mode}.json')

            for i in tqdm(range(len(metadata))):
                wav       = _load_wav(metadata[i]['wav_path']).cuda()
                feat      =  wavlm.extract_features(wav)[0].squeeze().detach().cpu().numpy()
                save_path = metadata[i]['mel_path'].replace('mels', 'wavlm')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, feat)
                metadata[i]['wavlm_path'] = save_path
                
            Write_json(metadata, data_path/f'{mode}.json')
                                        

if __name__ == '__main__':
    
    cfg = Config('./config/base.yaml')
    main(cfg)