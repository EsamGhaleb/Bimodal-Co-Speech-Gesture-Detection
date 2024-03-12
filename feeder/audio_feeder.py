from typing import Any
import sys
import random
import warnings
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio import functional as F
from torch.functional import F as F2
from torchaudio.utils import download_asset
from torch.utils.data import DataLoader

sys.path.extend(['../'])

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
sample_rate1 = 16000

# Define effects
effects = [
    ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
    ["speed", "0.8"],  # reduce the speed
    # This only changes sample rate, so it is necessary to
    # add `rate` effect with original sample rate after this.
    ["rate", f"{sample_rate1}"],
    ["reverb", "-w"],  # Reverbration gives some dramatic feeling
]

configs = [
    {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    {"format": "gsm"},
    {"format": "vorbis", "compression": -1},
]

class AudioFeeder(Dataset):
   def __init__(self, audio_path, label_path, normalize=False, debug=False, apply_augmentation=True):
      """
      Args:
            data_path (string): Path to the npy file with the data.
            label_path (string): Path to the pkl file with the labels.
            normalize (bool): If True, normalize the data.
      """
      self.label_path = label_path
      self.audio_path = audio_path
      self.normalize = normalize
      self.debug = debug
      self.load_data()
      self.label_map = {'non-gesture':0, 'gesture':1, 'undecided': 0}
      
      SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
      SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
      self.apply_augmentation = True

      rir_raw, rir_sample_rate = torchaudio.load(SAMPLE_RIR)
      self.rir_sample_rate = rir_sample_rate
      self.noise, noise_sample_rate = torchaudio.load(SAMPLE_NOISE)
      # target sample rate is 16000
      self.noise = F.resample(self.noise, orig_freq=noise_sample_rate, new_freq=sample_rate1)
      rir = rir_raw[:, int(rir_sample_rate * 1.01) : int(rir_sample_rate * 1.3)]
      self.rir = rir / torch.linalg.vector_norm(rir, ord=2)
   def augment_data(self, apply=True):
      self.apply_augmentation = apply
   def load_data(self):
      # for pickle file from python2
      with open(self.label_path, 'rb') as f:
         self.sample_names, self.lengths, self.labels = pickle.load(f)
      # load data
      self.audio = np.load(self.audio_path).squeeze()

      if self.debug:
         # choose randomly 1000 samples
         random.seed(0)
         idx = random.sample(range(len(self.label)), 100)
         self.label = [self.label[i] for i in idx]
         self.audio = np.array([self.audio[i] for i in idx])
         self.sample_name = [self.sample_name[i] for i in idx]
         # self.pair_speaker_referent = [self.pair_speaker_referent[i] for i in idx]
         self.lengths = [self.lengths[i] for i in idx]

   def __len__(self):
      return len(self.labels)

   def __iter__(self):
      return self
   def apply_codec(self, waveform, orig_sample_rate, **kwargs):
    if orig_sample_rate != 8000:
        waveform = F.resample(waveform, orig_sample_rate, 8000)
        sample_rate = 8000
    augmented = F.apply_codec(waveform, sample_rate, **kwargs)
    # resample to original sample rate
    augmented = F.resample(augmented, sample_rate, orig_sample_rate)
    return augmented
   def __getitem__(self, index):
      audio = self.audio[index]
      labels = self.label_map[self.labels[index]]
      lengths = int(self.lengths[index])
      augmented_length = audio.shape[0] - lengths
      audio = audio[:lengths]
      # apply effects
      audio = torch.from_numpy(audio).float().unsqueeze(0)
      
      augemntation_apply = self.apply_augmentation
      # apply effects with 50% probability
      if random.random() > 0.5 and augemntation_apply:
         audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sample_rate1, effects)
         # choose randomly one augmented speech from the two augmented speech
         idx = random.randint(0, 1)
         audio = audio[idx].unsqueeze(0)
         augemntation_apply = False

      
      
      # apply rir with 50% probability
      if random.random() > 0.5 and augemntation_apply:
         if self.rir_sample_rate != sample_rate1:
            audio = F.resample(audio, sample_rate1, self.rir_sample_rate)
         audio = F.fftconvolve(audio, self.rir)
         # resample to original sample rate
         audio = F.resample(audio, self.rir_sample_rate, sample_rate1)
         augemntation_apply = False
         
      # apply noise with 50% probability
      if random.random() > 0.5 and augemntation_apply:
         if self.noise.shape[1] < audio.shape[1]:
            noise = self.noise.repeat(1, 2)[:,:audio.shape[1]]
         else:
            noise = self.noise[:, : audio.shape[1]]
         snr_dbs = torch.tensor([20, 10, 3])
         audio = F.add_noise(audio, noise, snr_dbs)
         # choose randomly one noisy speech
         idx = random.randint(0, 2)
         audio = audio[idx].unsqueeze(0)
         augemntation_apply = False
      # apply codec with 50% probability
      if random.random() > 0.5 and augemntation_apply:   
         waveforms = []
         for param in configs:
            augmented = self.apply_codec(audio, sample_rate1, **param)
            waveforms.append(augmented)
         # choose randomly one codec
         idx = random.randint(0, 2)
         audio = waveforms[idx]
         augemntation_apply = False
      lengths = audio.shape[1]
      # each audio sample has a unique length, write a function to pad them to the same length
      return audio, labels, lengths, index

def collate_fn(batch):
   audio, labels, lengths, index = zip(*batch)
   # pad audio to the same length, using the maximum length in the batch
   max_length = max(lengths)
   audio = [F2.pad(audio[i], (0, max_length - lengths[i])) for i in range(len(audio))]
   audio = torch.cat(audio, dim=0)
   labels = torch.tensor(labels)
   lengths = torch.tensor(lengths)
   index = torch.tensor(index)
   return audio, labels, lengths, index
if __name__ == '__main__':
   fold = 0
   train_labels_path ='/home/eghaleb/Projects/MMCSGD/data/CABB/{}/train_label.pkl'.format(fold)
   train_audio_path = '/home/eghaleb/Projects/MMCSGD/data/CABB/{}/train_audio.npy'.format(fold)
   test_audio_path = '/home/eghaleb/Projects/MMCSGD/data/CABB/{}/test_audio.npy'.format(fold)
   test_labels_path = '/home/eghaleb/Projects/MMCSGD/data/CABB/{}/test_label.pkl'.format(fold)
   train_dataset = AudioFeeder(train_audio_path, train_labels_path, normalize=False, apply_augmentation=True)
   test_dataset = AudioFeeder(test_audio_path, test_labels_path, normalize=False, apply_augmentation=False)


   test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
   train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
   
   # apply few iteration to see the effect of augmentation
   for i, (audio, labels, lengths, index) in enumerate(test_loader):
      print(audio.shape)
      print(labels)
      print(lengths)
      print(index)
      # if i == 4:
      #    break
   for i, (audio, labels, lengths, index) in enumerate(train_loader):
      print(audio.shape)
      print(labels)
      print(lengths)
      print(index)
      # if i == 4:
      #    break
      
