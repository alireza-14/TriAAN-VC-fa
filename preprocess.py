import warnings
warnings.filterwarnings(action='ignore')
import os
from os.path import join as opj
import json
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import *
from preprocess.audio import *


def main(cfg):
    
    seed_init()
    MakeDir(cfg.output_path)
    if cfg.use_hf:
        all_spks, gen2spk = GetSpeakerInfoHF(cfg)
    else:
        all_spks, gen2spk = GetSpeakerInfo(cfg)

    print('---Split dataset---')
    if cfg.use_hf:
        all_wavs, train_wavs_names, valid_wavs_names, test_wavs_names = SplitDatasetHF(all_spks, cfg)
    else:
        all_wavs, train_wavs_names, valid_wavs_names, test_wavs_names = SplitDataset(all_spks, cfg)

    split_results = {}
    for split, data in [('train', train_wavs_names), ('valid', valid_wavs_names), ('test', test_wavs_names)]:
        print('---Feature extraction---')
        if cfg.use_hf:
            results = Parallel(n_jobs=-1)(delayed(ProcessingTrainDataHF)(wav_path, cfg) for wav_path in tqdm(data))
        else:
            results = Parallel(n_jobs=-1)(delayed(ProcessingTrainData)(wav_path, cfg) for wav_path in tqdm(data))
        
        wn2info = {}
        for r in results:
            wav_name, mel, lf0, mel_len, speaker, text, path = r
            wn2info[wav_name] = [mel, lf0, mel_len, speaker, text, path]
        
        mean, std = ExtractMelstats(wn2info, train_wavs_names, cfg) # only use train wav for normalizing stats

        print('---Write Features---')
        split_results[split] = Parallel(n_jobs=-1)(delayed(SaveFeatures)(wav_name, wn2info[wav_name], split, cfg) for wav_name in tqdm(data))

    if cfg.use_hf:
        train_results, valid_results, test_results = GetMetaResultsHF(split_results['train'], split_results['valid'], split_results['test'], cfg)
    else:
        train_results, valid_results, test_results = GetMetaResults(split_results['train'], split_results['valid'], split_results['test'], cfg)
    
    print('---Write Infos---')
    Write_json(train_results, f'{cfg.output_path}/train.json')
    Write_json(valid_results, f'{cfg.output_path}/valid.json')
    Write_json(test_results, f'{cfg.output_path}/test.json')
    print('---Done---')
    
if __name__ == '__main__':
    
    cfg = Config('./config/preprocess.yaml')
    main(cfg)