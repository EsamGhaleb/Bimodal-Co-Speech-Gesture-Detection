#!/usr/bin/env python
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as td
import torch.nn.functional as F
import math
from model.multimodal_sequence_fusion import PositionalEncoding, MultiModalAttention, Decoder, Encoder


from itertools import chain



class Labeler(nn.Module):
    def __init__(self, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, audio_embed_dim=512, embed_dim=512, decoder='transformer', label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        super().__init__()
        self._labelset_size = labelset_size
        self._pad = pad_id
        # _bos is the id of the begin of sentence token, it is the index of the bos token in the tagset
        self._bos = labels.index('bos')
        self._eos = len(labels)
        self._labels = labels
        self._classes = classes
        self._max_seq_len = max_seq_len
        self._ngram_size = ngram_size
        self._label_embed_dim = label_embed_dim
        self._include_embeddings = include_embeddings
        self._labeler_name = labeler_name
        self._recurrent_encoder = recurrent_encoder
        self._training_seq = training_seq
        self._embed_dim= embed_dim
        self._num_of_classes = len(classes)
        self._decoder = decoder
        self._joints_embed_dim = joints_embed_dim
        self._audio_embed_dim = audio_embed_dim

    # Python properties allow client code to access the property 
    # without the risk of modifying it.

    @property
    def labelset_size(self):
        return self._labelset_size

    @property
    def pad(self):
        return self._pad

    @property
    def bos(self):
        return self._bos
    @property
    def eos(self):
        return self._eos
    @property
    def labels(self):
        return self._labels
    @property
    def classes(self):
        return self._classes
    @property
    def max_seq_len(self):
        return self._max_seq_len
    @property
    def ngram_size(self):
        return self._ngram_size
    @property
    def label_embed_dim(self):
        return self._label_embed_dim
    @property
    def include_embeddings(self):
        return self._include_embeddings
    @property
    def labeler_name(self):
        return self._labeler_name
    @property
    def recurrent_encoder(self):
        return self._recurrent_encoder
    @property
    def training_seq(self):
        return self._training_seq
    @property
    def num_of_classes(self):
        return self._num_of_classes
    @property
    def embed_dim(self):
        return self._embed_dim
    @property
    def decoder(self):
        return self._decoder
    @property
    def joints_embed_dim(self):
        return self._joints_embed_dim
    @property
    def audio_embed_dim(self):
        return self._audio_embed_dim
    @property
    def num_of_labels(self):
        return self._num_of_labels
        
    def num_parameters(self):
        return sum(np.prod(theta.shape) for theta in self.parameters())
        
    def forward(self, x, y):
        raise NotImplementedError("Each type of tagger will have a different implementation here")

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)   

    def greedy(self, x):
        """
        For each cpd Y[i]|X=x, predicts the mode of the cpd.
        x: [batch_size, max_length]

        Return: tag sequences [batch_size, max_length]
        """
        raise NotImplementedError("Each type of tagger differs here")

    def sample(self, x, sample_size=None):
        """
        Per snippet sequence in the batch, draws a number of samples from the model, each sample is a complete tag sequence.

        x: [batch_size, max_len]

        Return: tag sequences with shape [batch_size, max_len] if sample_size is None
            else with shape [sample_size, batch_size, max_len]
        """
        raise NotImplementedError("Each type of tagger differs here")

    def loss(self, x, y):
        """
        Compute a scalar loss from a batch of sentences.
        The loss is the negative log likelihood of the model estimated on a single batch:
            - 1/batch_size * \sum_{s} log P(y[s]|x[s], theta)

        x: snippet sequences [batch_size, max_length] 
        y: tag sequences [batch_size, max_length] 
        """
        return -self.log_prob(x=x, y=y).mean(0)
class MFS(Labeler):
    def __init__(self, gcns_model, labeler_name: str, recurrent_encoder: bool, 
                 training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], 
                 classes=['outside', 'begin', 'inside', 'end'],
                 max_seq_len=40, ngram_size=3, joints_embed_dim=256, audio_embed_dim=512, 
                 embed_dim=512, decoder='transformer', label_embed_dim=32, include_embeddings=False, hidden_size=128,  **kwargs):
        """        
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        hidden_size: dimensionality of hidden layers
        recurrent_encoder: enable recurrent encoder
        bidirectional_encoder: for a recurrent encoder, make it bidirectional
        """
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, audio_embed_dim, embed_dim, decoder, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        self.num_of_labels = len(labels) - 1 # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        # we need to embed tags in the history 
        # we need to encode snippet sequences
        gesture_dim = len(classes)
        if gesture_dim == 2:
            gesture_dim = 1 # binary classification
        
        num_heads = 8
        num_layers = 6
        self.audio_encoder = Encoder(audio_embed_dim, embed_dim, num_heads, num_layers)
        self.video_encoder = Encoder(joints_embed_dim, embed_dim, num_heads, num_layers)
        self.multi_modal_attention = MultiModalAttention(embed_dim)  # New line
        if self.decoder == 'transformer':
            self.decoder = Decoder(gesture_dim, embed_dim, num_heads, num_layers)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, gesture_dim),
            )
        
    def forward(self, x, audio_embedings, gesture_labels, eval=False, get_predictions=False, get_memory=False, combined_memory=None, lengths=None, keep_prob=0.9):
        """
        Parameterise the conditional distributions over Y[i] given history y[:i] and all of x.

        This procedure takes care that the ith output distribution conditions only on the n-1 observations before y[i].
        It also takes care of padding to the left with BOS symbols.

        x: snippet sequences [batch_size, max_length]
        y: tag sequences  [batch_size, max_length]

        Return: a batch of V-dimensional Categorical distributions, one per step of the sequence.
        """
        # Let's start by encoding the snippet sequences
        # 1. we embed the snippets independently
        # [batch_size, max_length, embed_dim]
        
        # We begin by embedding the tokens        
        # [batch_size, max_length, embed_dim]
        if not get_predictions:
            N, S, C, F, V, M = x.shape
            x = x.view(N*S, C, F, V, M)
            # actual lengths of each sequence in the batch is needed for packing
            e = self.joints_embed(x, keep_prob=keep_prob)
            # [batch_size, max_length, embed_dim]
            e = e.view(N, S, -1)
            # remove dimension with size 1
            audio_embedings = audio_embedings.squeeze(2)
            audio_embedings = self.audio_positional_encoding(audio_embedings.permute(1, 0, 2))
            # 2. and then encode them in their left-to-right and right-to-left context
            # [batch_size, max_length, 2*hidden_size] 
            e = self.positional_encoding(e.permute(1, 0, 2))
            src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
            for i in range(N):
                src_mask[i, lengths[i]:] = True
            video_h = self.video_encoder(e, src_key_padding_mask=src_mask)
            audio_h = self.audio_encoder(audio_embedings, src_key_padding_mask=src_mask)
            # 3. we combine the two modalities 
            combined_memory = self.multi_modal_attention(audio_h, video_h)  # New line replacing simple addition
            try:
                assert video_h.shape == (S, N, self.joints_embed_dim)
            except AssertionError as error:
                print('the input shape is ' + str(e.shape))
                print('the output shape is ' + str(video_h.shape))
                print('The mask shape is ' + str(src_mask.shape))
                print('The lengths array is ' + str(lengths))
                print(S, N, self.joints_embed_dim)
                print(error)
                # halt execution
                raise error
            if get_memory:
                return combined_memory
        
        gesture_labels = gesture_labels.permute(1, 0, 2)
        gesture_predictions = self.decoder(gesture_labels, combined_memory)
        gesture_predictions = gesture_predictions.permute(1, 0, 2)
       
       
        return gesture_predictions