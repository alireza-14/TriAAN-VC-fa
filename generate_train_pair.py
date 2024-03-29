from src.utils import Config, Read_json, Write_json
from tqdm import tqdm
from itertools import combinations
import random



def get_speaker_samples(info):
  spk2samples = {}
  for i, sample in enumerate(tqdm(info)):
    spk_id = sample['speaker']
    samples_list = spk2samples.get(spk_id, [])
    samples_list.append(sample)
    spk2samples[spk_id] = samples_list
  return spk2samples


def generate_train_pairs(cfg, spk_samples, shuffle_func):
  train_pairs = list(combinations(spk_samples, 2))
  shuffle_func(train_pairs)
  if len(train_pairs) > cfg.num_train_samples:
    selected_pairs = train_pairs[:cfg.num_train_samples]
  else:
    selected_pairs = train_pairs
  annotated_pairs = [{'src_sample':pair[0], 'trg_sample':pair[1]} for pair in selected_pairs]
  return annotated_pairs
    


def main(cfg):
  shuffle = random.Random(cfg.seed).shuffle
  print("--- Read Train Samples---")
  train_info = Read_json(f"{cfg.output_path}/{cfg.split}.json")
  print(f"   Number of Total Samples: {len(train_info)}")
  spk2samples = get_speaker_samples(train_info)
  train_samples = []
  for spk, samples in tqdm(spk2samples.items()):
    train_samples += generate_train_pairs(cfg, samples, shuffle)
  print('---Write Infos---')
  Write_json(train_samples, f'{cfg.output_path}/{cfg.split}_pairs.json')
  print('---Done---')


if __name__ == '__main__':
  cfg = Config("./config/preprocess.yaml")
  main(cfg)