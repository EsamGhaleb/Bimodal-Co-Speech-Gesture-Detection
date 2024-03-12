#!/usr/bin/env python
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as td
import torch.nn.functional as F
import torch.nn.functional as FN

import math


from itertools import chain


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, num_layers):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Linear(output_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1, max_len=100)
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)
        
    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.fc_out(output)
        return output
    
class Encoder(nn.Module):
   def __init__(self, input_dim, embed_dim, num_heads, num_layers):
      super(Encoder, self).__init__()
      dim_feedforward = embed_dim * 4
      self.embedding = nn.Linear(input_dim, embed_dim)
      self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1, max_len=100)
      encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=False)
      self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
      self.encoder.apply(self.initialize_weights)
      
   #   self.embedding = nn.Linear(input_dim, embed_dim)
   #   encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
   #   self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
   
   def forward(self, x, src_key_padding_mask=None):
      x = self.embedding(x)
      x = self.positional_encoding(x)
      x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
      return x
   def initialize_weights(self, m):
      if isinstance(m, nn.Linear):
         nn.init.xavier_uniform_(m.weight)
         if m.bias is not None:
               nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
         nn.init.constant_(m.weight, 1)
         nn.init.constant_(m.bias, 0) 

 
class MultiModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(MultiModalAttention, self).__init__()
        
        self.embed_dim = embed_dim  # Initialize embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, audio_memory, video_memory):
        # Computing Query, Key and Value
        Q = self.query(audio_memory)
        K = self.key(video_memory)
        V = self.value(video_memory)
        
        # Attention Score Calculation
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attention_distribution = self.softmax(attention_score)
        
        # Compute the combined memory
        combined_memory = torch.matmul(attention_distribution, V)
        
        return combined_memory
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Labeler(nn.Module):
    def __init__(self, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, audio_embed_dim=512, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
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
        self._num_of_classes = len(classes)

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
class IndependentLabeler(Labeler):
    def __init__(self, gcns_model, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, audio_embed_dim=512, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        """        
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        hidden_size: dimensionality of hidden layers
        recurrent_encoder: enable recurrent encoder
        bidirectional_encoder: for a recurrent encoder, make it bidirectional
        """
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, audio_embed_dim, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        self.num_of_labels = len(labels) - 1 # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        # we need to embed tags in the history 
        # we need to encode snippet sequences
        if include_embeddings:
            first_fc_dim =  2 * joints_embed_dim
            first_audio_fc_dim = 2 * audio_embed_dim
        else:
            first_fc_dim = joints_embed_dim
            first_audio_fc_dim = audio_embed_dim
        if recurrent_encoder:
            # use transformer
            nhead = 8
            num_layers = 4
            dim_feedforward = hidden_size * 4
            self.positional_encoding = PositionalEncoding(d_model=joints_embed_dim, dropout=0.1, max_len=100)
            encoder_layers = nn.TransformerEncoderLayer(d_model=joints_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.encoder.apply(self.initialize_weights)
            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
            
            # audio transformer
            nhead = 8
            num_layers = 4
            dim_feedforward = hidden_size * 4

            self.audio_positional_encoding = PositionalEncoding(d_model=audio_embed_dim, dropout=0.1, max_len=100)
            audio_encoder_layers = nn.TransformerEncoderLayer(d_model=audio_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=False)
            self.audio_encoder = nn.TransformerEncoder(audio_encoder_layers, num_layers=num_layers)
            self.audio_encoder.apply(self.initialize_weights)
        else:
            self.encoder = None
            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
        self.logits_predictor = nn.Sequential(
            nn.Linear(int(first_fc_dim), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        self.audio_logits_predictor = nn.Sequential(
            nn.Linear(int(first_audio_fc_dim), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        for layer in self.logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))
        for layer in self.audio_logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))
                      
    def forward(self, x, audio_embedings, y, eval=False, get_embeddings=False, get_predictions=False, e=None, lengths=None, keep_prob=0.9):
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
            if self.encoder is not None:
                audio_embedings = self.audio_positional_encoding(audio_embedings.permute(1, 0, 2))
                # 2. and then encode them in their left-to-right and right-to-left context
                # [batch_size, max_length, 2*hidden_size] 
                e = self.positional_encoding(e.permute(1, 0, 2))
                src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
                for i in range(N):
                    src_mask[i, lengths[i]:] = True
                h = self.encoder(e, src_key_padding_mask=src_mask)
                audio_h = self.audio_encoder(audio_embedings, src_key_padding_mask=src_mask)
                try:
                    assert h.shape == (S, N, self.joints_embed_dim)
                except AssertionError as error:
                    print('the input shape is ' + str(e.shape))
                    print('the output shape is ' + str(h.shape))
                    print('The mask shape is ' + str(src_mask.shape))
                    print('The lengths array is ' + str(lengths))
                    print(S, N, self.joints_embed_dim)
                    print(error)
                    # halt execution
                    raise error
                audio_h = audio_h.permute(1, 0, 2)
                h = h.permute(1, 0, 2)
                 # concatenate h and e, in the embedding dimension
                # [batch_size, max_length, 2*hidden_size + embed_dim]
                if self.include_embeddings:
                    e = torch.cat([e, h], 2)
                    audio_h = torch.cat([audio_embedings, audio_h], 2)
                else:
                    e = h
                    audio_e = audio_h
        if get_embeddings:
            return e, audio_e
        # We are now ready to map the state of each step of the sequence to a C-dimensional vector of logits
        # we do so using our FFNN
        # [batch_size, max_length, num_of_labels]
        
        s = self.logits_predictor(e)
        audio_s = self.audio_logits_predictor(audio_e)
        # sum the logits from the two modalities
        
        if not eval:
            return s + audio_s
        else:
            cat_log_probs = td.Categorical(logits=s + audio_s)
            return cat_log_probs.probs



    
# ---------- MFS Labelers --------------
class MFSLabeler(Labeler):
    def __init__(self, gcns_model, speech_model, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, audio_embed_dim=128, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        """        
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        hidden_size: dimensionality of hidden layers
        recurrent_encoder: enable recurrent encoder
        bidirectional_encoder: for a recurrent encoder, make it bidirectional
        """
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, audio_embed_dim, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        self.num_of_labels = len(labels) - 1 # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        self.speech_embed = speech_model
        # we need to embed tags in the history 
        # we need to encode snippet sequences

        nhead = 8
        num_layers = 6
        embed_dim = 512
        
        # self.joints_encoder = Encoder(input_dim=joints_embed_dim, embed_dim=embed_dim, num_heads=nhead, num_layers=num_layers)
        # self.audio_encoder = Encoder(input_dim=audio_embed_dim, embed_dim=embed_dim, num_heads=nhead, num_layers=num_layers)

        self.logits_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        self.audio_logits_predictor = nn.Sequential(
            nn.Linear(audio_embed_dim, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        for layer in self.logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))         
        for layer in self.audio_logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))
    def forward(self, x, audio_data, y, eval=False, get_embeddings=False, get_predictions=False, e=None, lengths=None, keep_prob=0.9):
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
            # x = x.view(N*S, C, F, V, M)
            # # actual lengths of each sequence in the batch is needed for packing
            # e = self.joints_embed(x, keep_prob=keep_prob)
            # # [batch_size, max_length, embed_dim]
            # e = e.view(N, S, -1)
            # remove dimension with size 1
            audio_data = audio_data.squeeze()
            N, S, audio_C, audio_F = audio_data.shape
            audio_data = audio_data.view(N*S, 1, audio_C, audio_F)
            audio_embedings = self.speech_embed(audio_data)
            audio_embedings = audio_embedings.view(N, S, -1)
            # audio_embedings = audio_embedings.permute(1, 0, 2)
            # audio_embedings = self.audio_embedding(audio_embedings)
            # 2. and then encode them in their left-to-right and right-to-left context
            # [batch_size, max_length, 2*hidden_size] 
            # e = e.permute(1, 0, 2)
            # # concatenate the audio embeddings with the video embeddings

            # # Normalizing along the feature dimensions (last dimension)
            # # e_norm = FN.normalize(e, p=2, dim=2)
            # # audio_embeddings_norm = FN.normalize(audio_embedings, p=2, dim=2)

            # # Concatenation
            # src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
            # for i in range(N):
            #     src_mask[i, lengths[i]:] = True
            # h = self.joints_encoder(e)#, src_key_padding_mask=src_mask)
            # h_audio = self.audio_encoder(audio_embedings)#, src_key_padding_mask=src_mask)
            
            # h = h.permute(1, 0, 2)
            # h_audio = h_audio.permute(1, 0, 2)
       
       
        # We are now ready to map the state of each step of the sequence to a C-dimensional vector of logits
        # we do so using our FFNN
        # [batch_size, max_length, num_of_labels]
        # s = self.logits_predictor(h)
        s_audio = self.audio_logits_predictor(audio_embedings)
        # sum the logits from the two modalities
        
        if not eval:
            return s_audio
        else:
            cat_log_probs = td.Categorical(logits=s_audio)
            return cat_log_probs.probs
class AutoregressiveLabeler(Labeler):
    def __init__(self, gcns_model, speech_model, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, audio_embed_dim=128, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        """        
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        hidden_size: dimensionality of hidden layers
        recurrent_encoder: enable recurrent encoder
        bidirectional_encoder: for a recurrent encoder, make it bidirectional
        """
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, audio_embed_dim, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        self.num_of_labels = len(labels) - 1 # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        self.speech_embed = speech_model
        # we need to embed tags in the history 
        # we need to encode snippet sequences

        nhead = 8
        num_layers = 6
        embed_dim = 512
        
        self.joints_encoder = Encoder(input_dim=joints_embed_dim, embed_dim=embed_dim, num_heads=nhead, num_layers=num_layers)
        self.audio_encoder = Encoder(input_dim=audio_embed_dim, embed_dim=embed_dim, num_heads=nhead, num_layers=num_layers)
        
        concatenated_embed_dim = embed_dim + embed_dim

        # self.multi_modal_attention = MultiModalAttention(embed_dim)  # New line
        gesture_dim = len(classes)
        # if self.decoder == 'transformer':
        self.decoder = Decoder(gesture_dim, concatenated_embed_dim, nhead, num_layers)
        # else:
        #     self.decoder = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_size*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, self.num_of_classes),
        #     )

        self.logits_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        self.audio_logits_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        for layer in self.logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))         
        for layer in self.audio_logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))
    def forward(self, x, audio_data, y, eval=False, get_embeddings=False, get_predictions=False, e=None, lengths=None, keep_prob=0.9):
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
            audio_data = audio_data.squeeze()
            N, S, audio_C, audio_F = audio_data.shape
            audio_data = audio_data.view(N*S, 1, audio_C, audio_F)
            audio_embedings = self.speech_embed(audio_data)
            audio_embedings = audio_embedings.view(N, S, -1)
            audio_embedings = audio_embedings.permute(1, 0, 2)
            # audio_embedings = self.audio_embedding(audio_embedings)
            # 2. and then encode them in their left-to-right and right-to-left context
            # [batch_size, max_length, 2*hidden_size] 
            e = e.permute(1, 0, 2)
            # concatenate the audio embeddings with the video embeddings

            # Normalizing along the feature dimensions (last dimension)
            # e_norm = FN.normalize(e, p=2, dim=2)
            # audio_embeddings_norm = FN.normalize(audio_embedings, p=2, dim=2)

            # Concatenation
            src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
            for i in range(N):
                src_mask[i, lengths[i]:] = True
            h = self.joints_encoder(e)#, src_key_padding_mask=src_mask)
            h_audio = self.audio_encoder(audio_embedings)#, src_key_padding_mask=src_mask)
            
            h = h.permute(1, 0, 2)
            h_audio = h_audio.permute(1, 0, 2)
       
       
       
        # concatenate h and e, in the embedding dimension
        audio_video_h = torch.cat([h, h_audio], 2)
        
        # call the decoder 
        # [batch_size, max_length, num_of_labels]
        # if get_predictions:
        #     tgt_mask = 
            
        s = self.decoder(tgt=y, memory=audio_video_h, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        
        if not eval:
            return (s+s_audio)/2
        else:
            cat_log_probs = td.Categorical(logits=(s+s_audio)/2)
            return cat_log_probs.probs
