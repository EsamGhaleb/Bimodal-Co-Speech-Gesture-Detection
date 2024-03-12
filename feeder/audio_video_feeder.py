import pickle
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import librosa
from tqdm import tqdm
from . import tools
from torchaudio import functional as F
from torch.functional import F as F2
from torchaudio.utils import download_asset
import torchaudio

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

mmpose_flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 

# Define effects
effects = [
    ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
   #  ["speed", "0.8"],  # reduce the speed
    # This only changes sample rate, so it is necessary to
    # add `rate` effect with original sample rate after this.
   #  ["rate", f"{sample_rate1}"],
    ["reverb", "-w"],  # Reverbration gives some dramatic feeling
]

configs = [
    {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    {"format": "gsm"},
   #  {"format": "vorbis", "compression": -1},
]

class SequentialAudioSkeletonFeeder(Dataset):
   def __init__(
         self, 
         data_path, 
         label_path,
         random_choose=False, 
         random_shift=False, 
         random_move=True,
         window_size=-1, 
         normalization=True, 
         debug=False, 
         random_mirror=False, 
         random_mirror_p=0.5, 
         is_vector=False, 
         max_length=40, 
         gesture_unit=True, 
         vggish=True, 
         all_audio_path='', 
         subject_joint=True, 
         audio_path='', 
         offset=2, 
         original_dataset=True, 
         speech_buffer=0.0, 
         sample_rate1 = 16000, 
         sanity_check = False,
         **kwargs
      ):

      """
      Args:
            data_path (string): Path to the npy file with the data.
            label_path (string): Path to the pkl file with the labels.
            normalize (bool): If True, normalize the data.
      """

      self.normalization = normalization
      self.debug = debug
      self.sanity_check = sanity_check
      self.audio_path = audio_path
      self.data_path = data_path
      self.label_path = label_path
      self.speech_buffer = speech_buffer
      self.all_audio_path = all_audio_path
      self.vggish = vggish
      self.audio_path = audio_path.format(self.speech_buffer)
      self.data_path = data_path.format(self.speech_buffer)
      self.label_path = label_path.format(self.speech_buffer)
      self.random_choose = random_choose
      self.random_shift = random_shift
      self.random_move = random_move
      self.window_size = window_size
      self.normalization = normalization
      self.random_mirror = random_mirror
      self.random_mirror_p = random_mirror_p
      self.debug = debug
      self.original_dataset = original_dataset
      self.load_data()
      self.label_map = {'non-gesture':0, 'gesture':1}
      self.fps = 29.97
      self.audio_sample_rate = sample_rate1
      self.audio_sample_per_frame = int(self.audio_sample_rate/self.fps)
      if gesture_unit:
         self.labels_dict = {'outside left': 0, 'starting': 0, 'early': 1, 'middle': 1, 'full': 1, 'outside right': 0, 'ending': 0, 'late': 1, 'outside': 0}
      else:
            self.labels_dict = {'outside left': 0, 'starting': 0, 'early': 0, 'middle': 1, 'full': 1, 'outside right': 0, 'ending': 0, 'late': 0, 'outside': 0}
      self.is_vector = is_vector
      SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
      SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
      rir_raw, rir_sample_rate = torchaudio.load(SAMPLE_RIR)
      self.rir_sample_rate = rir_sample_rate
      self.noise, noise_sample_rate = torchaudio.load(SAMPLE_NOISE)
      # target sample rate is 16000
      self.noise = F.resample(self.noise, orig_freq=noise_sample_rate, new_freq=sample_rate1)
      rir = rir_raw[:, int(rir_sample_rate * 1.01) : int(rir_sample_rate * 1.3)]
      self.rir = rir / torch.linalg.vector_norm(rir, ord=2)
      self.augmentation_apply = False
      if normalization:
         self.get_mean_map()

   def get_mean_map(self):
      data = self.data
      N, S, C, T, V, M = data.shape # --> N, S, C, T, V, M , so we need to reshape it to N * S, C, T, V, M
      data = data.reshape((N * S, C, T, V, M))
      self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
      self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * S* T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
       
   def load_data(self):
      with open(self.label_path, 'rb') as f:
         self.sample_name, self.pair_speaker_referent, self.labels, self.lengths = pickle.load(f)
      # load data
      self.data = np.load(self.data_path)
      print(self.data_path)

      speakers = np.unique([speaker_frames[0].split('_')[0]+'_'+speaker_frames[0].split('_')[1] for speaker_frames in self.sample_name])
      if self.vggish and not self.sanity_check:
         self.audio = np.load(self.audio_path)
         print(self.audio_path)
      elif not self.sanity_check:
         self.audio_dict = {}
         print('Loading audio files...')
         for speaker in tqdm(speakers):
            pair = speaker.split('_')[0]
            speaker = speaker.split('_')[1]
            pair_speaker = f"{pair}_{speaker}"
            if self.original_dataset:
               audio_path = self.all_audio_path.format(pair, pair, speaker)
            else:
               audio_path = self.all_audio_path.format(pair, speaker)
            input_audio, sample_rate = librosa.load(audio_path, sr=16000)
            self.audio_dict[pair_speaker] = {'audio': input_audio, 'sample_rate': sample_rate}
      # check if the number of dimensions is 5, if so, add an extra dimension
      if len(self.data.shape) == 5:
         self.data = np.expand_dims(self.data, -1)
      if self.debug:
         # choose randomly 100 samples
         random.seed(0)
         idx = random.sample(range(len(self.label)), 100)
         self.label = [self.label[i] for i in idx]
         self.data = np.array([self.data[i] for i in idx])
         self.audio = np.array([self.audio[i] for i in idx])
         self.sample_name = [self.sample_name[i] for i in idx]
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
   
   def augment_audio(self, audio, augemntation_apply=True):
      if not augemntation_apply:
         return audio
      # apply effects
      lengths = audio.shape[0]
      audio = torch.from_numpy(audio).float().unsqueeze(0)
      # apply effects with 50% probability
      coin_toss = random.random()
      if coin_toss < 0.33:
         audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.audio_sample_rate, effects)
         # choose randomly one augmented speech from the two augmented speech
         idx = random.randint(0, audio.shape[0] - 1)
         audio = audio[idx].unsqueeze(0)
         augemntation_apply = False
      elif coin_toss < 0.66:
         if self.noise.shape[1] < audio.shape[1]:
            noise = self.noise.repeat(1, 2)[:,:audio.shape[1]]
         else:
            noise = self.noise[:, : audio.shape[1]]
         snr_dbs = torch.tensor([20, 10, 3])
         audio = F.add_noise(audio, noise, snr_dbs)
         # choose randomly one noisy speech
         idx = random.randint(0, audio.shape[0] - 1)
         audio = audio[idx].unsqueeze(0)
         augemntation_apply = False
      else:
         waveforms = []
         for param in configs:
            augmented = self.apply_codec(audio, self.audio_sample_rate, **param)
            waveforms.append(augmented)
         # choose randomly one codec
         idx = random.randint(0, len(waveforms) - 1)
         audio = waveforms[idx]
         augemntation_apply = False
         if audio.shape[1] > lengths: # TODO: check the validity of this operation: if the augmented speech is longer than the original speech, truncate it
            audio = audio[:, :lengths]
      audio = audio.squeeze(0).numpy()
      return audio
   
   def augment_skeleton(self, data_numpy):
      S, C, V, T, M = data_numpy.shape
      if self.random_mirror:
         if random.random() > self.random_mirror_p:
            if data_numpy.shape[3] == 27:
               flip_index = mmpose_flip_index
            data_numpy = data_numpy[:,:,:,flip_index,:]
            if self.is_vector:
               data_numpy[:, 0,:,:,:] = - data_numpy[:, 0,:,:,:]
            else: 
               data_numpy[:, 0,:,:,:] = 512 - data_numpy[:, 0,:,:,:]
      if self.random_shift:
         if self.is_vector:
               data_numpy[:, 0,:,0,:] += random.random() * 20 - 10.0
               data_numpy[:, 1,:,0,:] += random.random() * 20 - 10.0
         else:
               data_numpy[:, 0,:,:,:] += random.random() * 20 - 10.0
               data_numpy[:, 1,:,:,:] += random.random() * 20 - 10.0
      for i in range(S):
         # TODO, if you want to apply preprocessing, do it here taken into account the number of segments
         if self.random_choose:
               data_numpy[i] = tools.random_choose(data_numpy[i], self.window_size)
         # if self.random_shift:
         #     data_numpy[i] = tools.random_shift(data_numpy[i])
         # elif self.window_size > 0:
         #     data_numpy[i] = tools.auto_pading(data_numpy[i], self.window_size)
         if self.random_move:
               data_numpy[i] = tools.random_move(data_numpy[i])
      return data_numpy
   
   def __getitem__(self, index):
      data_numpy = np.array(self.data[index])
      S, C, V, T, M = data_numpy.shape      
      speaker_frames = self.sample_name[index]
      pair_speaker = speaker_frames[0].split('_')[0]+'_'+speaker_frames[0].split('_')[1]
      if self.normalization:
         # data_numpy = (data_numpy - self.mean_map) / self.std_map
         assert data_numpy.shape[1] == 3
         assert data_numpy.shape[1] == 3
         if self.is_vector:
               data_numpy[:, 0,:,0,:] -= data_numpy[:, 0,:,0,0].mean(axis=1, keepdims=True)
               data_numpy[:, 1,:,0,:] -= data_numpy[:, 1,:,0,0].mean(axis=1, keepdims=True)
         else:
               mean_0 = data_numpy[:, 0,:,0,0].mean(axis=1, keepdims=True)
               mean_1 = data_numpy[:, 1,:,0,0].mean(axis=1, keepdims=True)
               data_numpy[:, 0,:,:,:] -= mean_0.reshape(mean_0.shape[0], 1, 1, 1)
               data_numpy[:, 1,:,:,:] -= mean_1.reshape(mean_1.shape[0], 1, 1, 1)

      if self.sanity_check:
         audio_data = torch.zeros(1, 16000)
      else:
         if self.vggish:
            audio_data = self.audio[index]
         else:
            buffer_frames = round(self.speech_buffer * self.fps)
            start_frame = int(speaker_frames[0].split('_')[2]) * self.audio_sample_per_frame
            end_frame = round(int(int(speaker_frames[-1].split('_')[3])+buffer_frames) * self.audio_sample_per_frame)
            if end_frame > len(self.audio_dict[pair_speaker]['audio']):
               padding = np.zeros(end_frame - len(self.audio_dict[pair_speaker]['audio']))
               end_frame = len(self.audio_dict[pair_speaker]['audio'])
               audio_data = np.concatenate((self.audio_dict[pair_speaker]['audio'][start_frame:end_frame], padding))
            else:
               audio_data = self.audio_dict[pair_speaker]['audio'][start_frame:end_frame]
            audio_data = np.array(audio_data)
            # audio_data = self.augment_audio(audio_data, augemntation_apply=self.augmentation_apply)
            # lengths = np.array(lengths, dtype=np.int32)
      start_frames = np.array([int(speaker_frame.split('_')[2]) for speaker_frame in speaker_frames])
      end_frames = np.array([int(speaker_frame.split('_')[3]) for speaker_frame in speaker_frames])
      pair_speaker = np.array([pair_speaker for speaker_frame in speaker_frames])
      # extract the pair numerical ID
      pair_speaker = pair_speaker[0]
      pair = int(pair_speaker.split('_')[0].split('pair')[1])
      # extract the speaker numerical ID
      speaker_ID = {'A': 0, 'B': 1}[pair_speaker.split('_')[1]]
      speaker_ID = np.array([speaker_ID for speaker_frame in speaker_frames])
      pair_ID = np.array([pair for speaker_frame in speaker_frames])
      speaker_frames = {'start_frames': start_frames, 'end_frames': end_frames, 'speaker_ID': speaker_ID, 'pair_ID': pair_ID}
      lengths = self.lengths[index]
      labels = self.labels[index]
      labels = np.array([self.labels_dict[l.split('_')[4].strip()] for l in labels]) 
      return audio_data, data_numpy, labels, lengths, speaker_frames

class WrapperDataset(Dataset):
   def __init__(self, base_dataset, augmentation_apply=True):
      super(WrapperDataset, self).__init__()
      self.base = base_dataset
      self.augmentation_apply = augmentation_apply

   def __len__(self):
      return len(self.base)

   def __getitem__(self, idx):
      audio_data, data_numpy, labels, lengths, speaker_frames = self.base[idx]
      if self.augmentation_apply:
         if not self.base.dataset.vggish and not self.base.dataset.sanity_check:
            audio_data = self.base.dataset.augment_audio(audio_data)
         data_numpy = self.base.dataset.augment_skeleton(data_numpy)
      return audio_data, data_numpy, labels, lengths, speaker_frames