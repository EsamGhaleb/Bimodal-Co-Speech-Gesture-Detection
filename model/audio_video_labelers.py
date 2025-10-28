import math
from os import path

import torch
import torch.nn as nn
import torchaudio
from typing import Tuple
from torch.nn import Transformer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from model.decouple_gcn_attn_sequential import Model as STGCN
from model.torchvggish import vggish
from model.audio_video_cross_attn import LXRTEncoder

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
    
def load_STGCN(weights_path, device='cuda'):
    weights_path = path.join(weights_path, 'pt_models/joint_finetuned.pt')
    gcn_model = STGCN(device=device)
    gcn_model.load_state_dict(torch.load(weights_path))
    return gcn_model

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
class Vggish(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=6, buffer=0.0, postprocess=False):
        super(Vggish, self).__init__()
        self.audio_model = vggish(postprocess=False, buffer=buffer)
        
    def forward(self, audio_data, src_key_padding_mask=None):
        audio_data = audio_data.float()
        N, S, audio_C, audio_F = audio_data.shape
        audio_data = audio_data.view(N*S, 1, audio_C, audio_F)
        audio_embedings = self.audio_model(audio_data)
        audio_embedings = audio_embedings.view(N, S, -1)
        return audio_embedings

class AnEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, device='cuda', dropout=0.1, max_len=100, hidden_size=128):
        super(AnEncoder, self).__init__()
        dim_feedforward = hidden_size * 4
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1, max_len=100)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.encoder.apply(initialize_weights)
        
    def forward(self, audio_embedings, src_key_padding_mask=None):
        audio_embedings = audio_embedings.permute(1, 0, 2)
        audio_embedings = self.positional_encoding(audio_embedings)
        audio_embedings = self.encoder(audio_embedings, src_key_padding_mask=src_key_padding_mask)
        audio_embedings = audio_embedings.permute(1, 0, 2)
        return audio_embedings

class Fusion(nn.Module):
    def __init__(self, 
                vggish: bool = True, 
                weights_path: str = "",
                speech_embed_dim: int = 256, 
                skeleton_embed_dim: int = 256, 
                use_lstm: bool = False, 
                fine_tuned_audio_model: bool = False, 
                speech_buffer: float = 0.0, 
                offset: int = 2, 
                encoder_for_audio: bool = False,
                align_speech_windows: bool = True,
                encoder_for_skeleton: bool = True,
                sanity_check: bool = False,
                skeleton_alone: bool = False,
                speech_alone: bool = False,
                audio_model_name: str = 'pretrained') -> None:

        
        super(Fusion, self).__init__()
        
        self.audio_model_name = audio_model_name
        self.sanity_check = sanity_check
        self.speech_alone = speech_alone
        self.skeleton_alone = skeleton_alone
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        if vggish:
            speech_embed_dim = 256
        else:
            speech_embed_dim = 1024
        self.init_hyperparameters(offset, speech_buffer, use_lstm, vggish, 
                                    skeleton_embed_dim, speech_embed_dim, fine_tuned_audio_model, encoder_for_audio, align_speech_windows, encoder_for_skeleton)
        if not self.speech_alone:
            self.joints_embed = self.load_stgcn_model(weights_path, device)
        if not self.skeleton_alone and not self.sanity_check:
            self.audio_model, self.speech_embed_dim = self.load_audio_model(device, speech_embed_dim, skeleton_embed_dim)
        else:
            self.speech_embed_dim = skeleton_embed_dim
        if self.encoder_for_audio:
            self.init_audio_encoder(device)
        if self.use_lstm and not self.vggish:
            self.init_lstm(self.speech_embed_dim)
        if self.encoder_for_skeleton:
            self.skeleton_encoder = self.init_skeleton_encoder(self.skeleton_embed_dim)
            
    def init_skeleton_encoder(self, skeleton_embed_dim: int) -> nn.Module:
        # Initialize the skeleton encoder
        return AnEncoder(embed_dim=skeleton_embed_dim, num_heads=8, num_layers=6, 
                            device=self.device, dropout=0.1, max_len=100, hidden_size=128).to(self.device) 
    
    def init_lstm(self, speech_embed_dim: int) -> None:
        self.lstm = nn.LSTM(speech_embed_dim, speech_embed_dim, batch_first=True)
        self.lstm.apply(initialize_weights)  

    def init_hyperparameters(self, offset, speech_buffer, use_lstm, vggish, 
                            skeleton_embed_dim, speech_embed_dim, fine_tuned_audio_model, encoder_for_audio, align_speech_windows, encoder_for_skeleton):
        # Initialize hyperparameters
        self.offset = offset
        self.speech_buffer = speech_buffer
        self.use_lstm = use_lstm
        self.vggish = vggish
        self.skeleton_embed_dim = skeleton_embed_dim
        self.speech_embed_dim = speech_embed_dim
        self.fine_tuned_audio_model = fine_tuned_audio_model
        self.encoder_for_audio = encoder_for_audio
        self.align_speech_windows = align_speech_windows
        self.encoder_for_skeleton = encoder_for_skeleton
        
    def load_stgcn_model(self, weights_path: str, device: str) -> nn.Module:
        # Load the STGCN model
        return load_STGCN(weights_path=weights_path, device=device).to(device)
    
    def load_audio_model(self, device: str, speech_embed_dim: int, skeleton_embed_dim: int) -> Tuple[nn.Module, int]:
        # Load the appropriate audio model and adjust the speech embedding dimension
        if self.vggish:
            audio_model = Vggish(buffer=self.speech_buffer).to(device)
            return audio_model, 256
        if self.audio_model_name == 'pretrained': 
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
            audio_model = bundle.get_model().to(device)
            # freeze the audio model
            if not self.fine_tuned_audio_model:
                for param in audio_model.model.parameters():
                    param.requires_grad = False
            else:
                for param in audio_model.model.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            model_name = "GroNLP/wav2vec2-dutch-large-ft-cgn"
            audio_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
            audio_model.freeze_feature_extractor()          
    
        self.linear_project = nn.Linear(speech_embed_dim, skeleton_embed_dim)
        self.linear_project.apply(initialize_weights)
        return audio_model, skeleton_embed_dim
    
    def init_audio_encoder(self, device: str) -> None:
        # Initialize the audio encoder
        self.audio_encoder = AnEncoder(
            embed_dim=self.speech_embed_dim, num_heads=8, num_layers=4, device=device, 
            dropout=0.1, max_len=100, hidden_size=128
        ).to(device)
        self.audio_encoder.apply(initialize_weights)
        
    def generate_square_subsequent_mask(self, sz, DEVICE='cuda'):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def process_audio(self, skeleton_data, audio_data: torch.Tensor) -> torch.Tensor:
        if self.sanity_check:
            # create a dummy audio data: N, S, C
            N = skeleton_data.shape[0]
            S = skeleton_data.shape[1]
            C = self.skeleton_embed_dim
            audio_data = torch.randn(N, S, C).to(self.device)
            return audio_data
        if self.vggish:
            return self.audio_model(audio_data)
        if self.audio_model_name == 'pretrained':
            audio_data = self.audio_model(audio_data.to(self.device))[0]
        else:
            audio_data = self.audio_model(audio_data.to(self.device)).last_hidden_state
                
        audio_data = self.linear_project(audio_data)
        
        if self.align_speech_windows:
            return self.get_wav2vec2_per_window(skeleton_data, audio_data)
        else:
            return audio_data
    
    def process_skeleton(self, skeleton_data: torch.Tensor) -> torch.Tensor:
        N, S, C, F, V, M = skeleton_data.shape
        skeleton_data = skeleton_data.view(N * S, C, F, V, M).float()
        skeleton_data = self.joints_embed(skeleton_data)
        return skeleton_data.view(N, S, -1)
        
    def load_pretrained_audio_model(self):
        if not self.vggish:
            if self.fine_tuned_audio_model:
                # TODO, give the path of the pre-trained model
                model_path = 'save_models/sequential_speech_model_fold_{}_lr_{}_weighted_sampler_audio_vggish_with_encoder_audio_model_only.pt'.format(arg.feeder_args['fold'], arg.lr)
                weights = torch.load(model_path)
                for name, param in self.audio_model.named_parameters():
                # load the weights of the pre-trained model
                    param.data = weights[name]
        else:
            if self.fine_tuned_audio_model:
                # load the audio_model weights from weights of the pre-trained model
                model_path = 'save_models/sequential_speech_model_fold_{}_lr_{}_weighted_sampler_audio_frozen_cnn.pt'.format(arg.feeder_args['fold'], arg.lr)
                weights = torch.load(model_path)
                for name, param in self.audio_model.named_parameters():
                # load the weights of the pre-trained model
                    param.data = weights['model.audio_model.'+name]
                    param.requires_grad = True
            else:
                # freeze the audio model
                for param in self.audio_model.model.parameters():
                    param.requires_grad = False

    def get_wav2vec2_per_window(self, skeleton_data, audio_data):
        N, S, C, F, V, M = skeleton_data.shape
        window_size = F
        offset = self.offset
        num_time_windows = S
        #TODO check the FPS of the audio and video, per dataset and per video
        FPS = 29.97
        speech_buffer_frames = FPS * self.speech_buffer
        total_num_frames = offset * num_time_windows + window_size - offset + speech_buffer_frames
        total_number_of_audio_frames = audio_data.shape[1]
        audio_to_video_ratio = total_number_of_audio_frames / total_num_frames
        # obtain num_time_windows number of frames from the audio data
        audio_window_size = round(window_size * audio_to_video_ratio)
        # for loop with offset of 2
        audio_window_size += round(speech_buffer_frames * audio_to_video_ratio)
        # TODO take into account the change in the audio window size, with varying speech buffer
        init_audio_windows = torch.zeros(N, S, audio_window_size, audio_data.shape[-1]).to(self.device)
        for i in range(num_time_windows): 
            start_audio_idx = round(i*offset * audio_to_video_ratio)
            end_audio_idx = round((i*offset + window_size+speech_buffer_frames) * audio_to_video_ratio)
            if end_audio_idx - start_audio_idx < audio_window_size:
                end_audio_idx = start_audio_idx + audio_window_size
            elif end_audio_idx - start_audio_idx > audio_window_size:
                end_audio_idx = start_audio_idx + audio_window_size
            init_audio_windows[:, i, :, :] = audio_data[:, start_audio_idx:end_audio_idx, :]

        if self.use_lstm:
            N, S, T, C = init_audio_windows.shape
            init_audio_windows = init_audio_windows.view(N*S, T, C)
            _, (h_n, _) = self.lstm(init_audio_windows)
            # h_n is the last hidden state of the LSTM
            init_audio_windows = h_n.unsqueeze(0).view(N, S, -1)
            return init_audio_windows
        else:
            return torch.mean(init_audio_windows, dim=2)
    
class EarlyFusion(Fusion):
    def __init__(self, 
                vggish: bool = True, 
                weights_path: str = "",
                speech_embed_dim: int = 256, 
                skeleton_embed_dim: int = 256, 
                use_lstm: bool = False, 
                fine_tuned_audio_model: bool = False, 
                speech_buffer: float = 0.0, 
                offset: int = 2, 
                encoder_for_audio: bool = False,
                align_speech_windows: bool = True,
                encoder_for_skeleton: bool = False,
                sanity_check: bool = False, 
                audio_model_name: str = 'pretrained') -> None:

        
        super().__init__(
            vggish, 
            weights_path,
            speech_embed_dim, 
            skeleton_embed_dim, 
            use_lstm, 
            fine_tuned_audio_model, 
            speech_buffer, 
            offset, 
            encoder_for_audio=encoder_for_audio, 
            encoder_for_skeleton=encoder_for_skeleton, 
            align_speech_windows=align_speech_windows, 
            sanity_check=sanity_check, 
            audio_model_name=audio_model_name
            )
        
        self.mm_encoder = self.init_mm_encoder(self.skeleton_embed_dim, self.speech_embed_dim)
        self.mm_classifier = self.init_mm_classifier()


    def init_mm_encoder(self, skeleton_embed_dim: int, speech_embed_dim: int) -> nn.Module:
        return AnEncoder(embed_dim=skeleton_embed_dim + speech_embed_dim, 
                        num_heads=8, num_layers=6, 
                        device=self.device, 
                        dropout=0.1, 
                        max_len=100, 
                        hidden_size=128).to(self.device)

    def init_mm_classifier(self) -> nn.Module:
        mm_classifier = nn.Sequential(
            nn.Linear(self.skeleton_embed_dim + self.speech_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        mm_classifier.apply(initialize_weights)
        return mm_classifier.to(self.device)

    def forward(self, audio_data: torch.Tensor, skeleton_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_data = self.process_audio(skeleton_data, audio_data)
        skeleton_data = self.process_skeleton(skeleton_data)
        
        mm_data = self.mm_encoder(torch.cat((skeleton_data, audio_data), dim=-1))
        final_preds = self.mm_classifier(mm_data)
        
        return final_preds, final_preds, final_preds

class LateFusion(Fusion):
    def __init__(self,
                vggish: bool = True,
                weights_path: str = "",
                speech_embed_dim: int = 256,
                skeleton_embed_dim: int = 256,
                use_lstm: bool = False,
                fine_tuned_audio_model: bool = False,
                speech_buffer: float = 0.0,
                offset: int = 2,
                encoder_for_audio: bool = False,
                align_speech_windows: bool = True,
                encoder_for_skeleton: bool = True,
                sanity_check: bool = False,
                skeleton_alone: bool = False,
                speech_alone: bool = False,
                audio_model_name: str = 'pretrained') -> None:

        super().__init__(
            vggish, 
            weights_path,
            speech_embed_dim, 
            skeleton_embed_dim, 
            use_lstm, 
            fine_tuned_audio_model, 
            speech_buffer, 
            offset,
            encoder_for_audio=encoder_for_audio, 
            encoder_for_skeleton=encoder_for_skeleton, 
            align_speech_windows=align_speech_windows, 
            sanity_check=sanity_check, 
            skeleton_alone=skeleton_alone, 
            speech_alone=speech_alone, 
            audio_model_name=audio_model_name
            )
                
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_classifiers(self.speech_embed_dim, self.skeleton_embed_dim)
        
        
    def init_classifiers(self, speech_embed_dim: int, skeleton_embed_dim: int) -> None:
        # Initialize the audio and skeleton classifiers
        if not self.speech_alone:   
            self.skeleton_classifier = self.create_classifier(skeleton_embed_dim)
        if not self.skeleton_alone:
            self.audio_classifier = self.create_classifier(speech_embed_dim)
    
    def create_classifier(self, embed_dim: int) -> nn.Module:
        # Create a classifier model
        classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        classifier.apply(initialize_weights)
        return classifier.to(self.device)

    
    def forward(self, audio_data: torch.Tensor, skeleton_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.skeleton_alone:
            audio_data = self.process_audio(skeleton_data, audio_data)
            # speech and skeleton encoders
            if self.encoder_for_audio:
                audio_data = self.audio_encoder(audio_data)
            audio_preds = self.audio_classifier(audio_data)
            
        # Forward pass through the LateFusion model
        if not self.speech_alone:
            skeleton_data = self.process_skeleton(skeleton_data)
            if self.encoder_for_skeleton:
                skeleton_data = self.skeleton_encoder(skeleton_data)
            skeleton_preds = self.skeleton_classifier(skeleton_data)
        
        if not self.skeleton_alone and not self.speech_alone:
            final_preds = (audio_preds + skeleton_preds) / 2
        elif self.skeleton_alone:
            final_preds = skeleton_preds
            audio_preds = skeleton_preds
        elif self.speech_alone:
            final_preds = audio_preds
            skeleton_preds = audio_preds
        
        return audio_preds, skeleton_preds, final_preds
    
class CrossAttn(Fusion):
    def __init__(self, 
                vggish: bool = True,
                weights_path: str = "",
                speech_embed_dim: int = 256,
                skeleton_embed_dim: int = 256,
                use_lstm: bool = False,
                fine_tuned_audio_model: bool = False,
                speech_buffer: float = 0.0,
                offset: int = 2,
                encoder_for_audio: bool = False,
                align_speech_windows: bool = False,
                encoder_for_skeleton: bool = False,
                sanity_check: bool = False,
                audio_model_name: str = 'pretrained') -> None:

        
        super().__init__(
            vggish, 
            weights_path,
            speech_embed_dim, 
            skeleton_embed_dim, 
            use_lstm,
            fine_tuned_audio_model, 
            speech_buffer, 
            offset, 
            encoder_for_audio=encoder_for_audio, 
            align_speech_windows=align_speech_windows, 
            encoder_for_skeleton=encoder_for_skeleton, 
            sanity_check=sanity_check, 
            audio_model_name=audio_model_name
            )

        # self.init_classifiers(speech_embed_dim, skeleton_embed_dim)
        self.lxrt_encoder = self.init_lxrt_encoder()
        self.mm_classifier = self.init_mm_classifier()
    
    def init_classifiers(self, speech_embed_dim: int, skeleton_embed_dim: int) -> None:
        self.audio_classifier = self.create_classifier(speech_embed_dim)
        self.skeleton_classifier = self.create_classifier(skeleton_embed_dim)
        
    def create_classifier(self, embed_dim: int) -> nn.Module:
        classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        classifier.apply(initialize_weights)
        return classifier.to(self.device)
    
    def init_mm_classifier(self) -> nn.Module:
        mm_classifier = nn.Sequential(
            nn.Linear(self.skeleton_embed_dim + self.speech_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        mm_classifier.apply(initialize_weights)
        return mm_classifier.to(self.device)
    
    def init_lxrt_encoder(self) -> nn.Module:
        self.skeleton_positional_encoding = PositionalEncoding(d_model=self.skeleton_embed_dim, dropout=0.1, max_len=100)
        self.speech_positional_encoding = PositionalEncoding(d_model=self.speech_embed_dim, dropout=0.1, max_len=500)
        lxrt_encoder = LXRTEncoder()
        lxrt_encoder.apply(initialize_weights)
        return lxrt_encoder
    
    def forward(self, audio_data: torch.Tensor, orig_skeleton_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_data = self.process_audio(orig_skeleton_data, audio_data)
        skeleton_data = self.process_skeleton(orig_skeleton_data)    
        
        if self.encoder_for_audio:
            audio_data = self.audio_encoder(audio_data)
        if self.encoder_for_skeleton:
            skeleton_data = self.skeleton_encoder(skeleton_data)
        
        skeleton_data, audio_data = self.cross_attention(skeleton_data, audio_data)

        # even if we do not align speech windows, we still need to get the mean of the audio data per window for classification
        if not self.vggish and not self.align_speech_windows:
            audio_data = self.get_wav2vec2_per_window(orig_skeleton_data, audio_data) 

        mm_data = torch.cat((skeleton_data, audio_data), dim=-1)
        final_preds = self.mm_classifier(mm_data)
        return final_preds, final_preds, final_preds
    
    def cross_attention(self, skeleton_data: torch.Tensor, audio_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_attention_mask = None
        skeleton_attention_mask = None
        skeleton_data = self.skeleton_positional_encoding(skeleton_data.permute(1, 0, 2))
        audio_data = self.speech_positional_encoding(audio_data.permute(1, 0, 2))
        skeleton_data, audio_data = self.lxrt_encoder(skeleton_data.permute(1, 0, 2), skeleton_attention_mask, audio_data.permute(1, 0, 2), speech_attention_mask)
        return skeleton_data, audio_data

class EarlyFusionDecoder(Fusion):
    def __init__(self, 
                vggish: bool = True, 
                weights_path: str = "",
                speech_embed_dim: int = 256, 
                skeleton_embed_dim: int = 256, 
                use_lstm: bool = False, 
                fine_tuned_audio_model: bool = False, 
                speech_buffer: float = 0.0, 
                offset: int = 2, 
                encoder_for_audio: bool = False,
                align_speech_windows: bool = True,
                encoder_for_skeleton: bool = False,
                sanity_check: bool = False) -> None:
        
        super().__init__(
            vggish, 
            weights_path,
            speech_embed_dim, 
            skeleton_embed_dim, 
            use_lstm, 
            fine_tuned_audio_model, 
            speech_buffer, 
            offset, 
            encoder_for_audio=encoder_for_audio, 
            encoder_for_skeleton=encoder_for_skeleton, 
            align_speech_windows=align_speech_windows, 
            sanity_check=sanity_check
            )
        
        self.use_decoder = True
        
        if not self.use_decoder:
            self.mm_encoder = self.init_mm_encoder(self.skeleton_embed_dim, self.speech_embed_dim)
            self.mm_classifier = self.init_mm_classifier()
        else:
            self.transformer = Transformer(d_model=self.skeleton_embed_dim + self.speech_embed_dim,
                                        nhead=8,
                                        num_encoder_layers=6,
                                        num_decoder_layers=4,
                                        dim_feedforward= 2048,
                                        dropout=0.1,
                                        batch_first=False)
            self.positional_encoding = PositionalEncoding(d_model=self.skeleton_embed_dim + self.speech_embed_dim, dropout=0.1, max_len=100)
            self.transformer.apply(initialize_weights)
            # TODO: check if the generator is needed
            self.generator = nn.Linear(self.skeleton_embed_dim + self.speech_embed_dim, 2)
            self.generator.apply(initialize_weights)
            self.tgt_embed = nn.Embedding(3, self.skeleton_embed_dim + self.speech_embed_dim)
            self.tgt_embed.apply(initialize_weights)

    def init_mm_encoder(self, skeleton_embed_dim: int, speech_embed_dim: int) -> nn.Module:
        return AnEncoder(embed_dim=skeleton_embed_dim + speech_embed_dim, 
                        num_heads=8, num_layers=6, 
                        device=self.device, 
                        dropout=0.1, 
                        max_len=100, 
                        hidden_size=128).to(self.device)

    def init_mm_classifier(self, output_dim: int = 2) -> nn.Module:
        mm_classifier = nn.Sequential(
            nn.Linear(self.skeleton_embed_dim + self.speech_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        mm_classifier.apply(initialize_weights)
        return mm_classifier.to(self.device)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(
                        self.tgt_embed(tgt)), memory,
                        tgt_mask)
    def forward(self, audio_data: torch.Tensor, skeleton_data: torch.Tensor, tgt: torch.Tensor = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass through the EarlyFusion model
        IMPORTANT: the batch is always converted to second dimension
        '''
        # add start token to the tgt tensor
        start_symbol = 2
        audio_data = self.process_audio(skeleton_data, audio_data)
        skeleton_data = self.process_skeleton(skeleton_data)
        mm_data = torch.cat((skeleton_data, audio_data), dim=-1)
        if self.use_decoder:
            mm_data = self.positional_encoding(mm_data.permute(1, 0, 2))
            max_len = mm_data.shape[1]
            tgt = torch.cat((torch.ones(tgt.shape[0], 1).to(self.device)*start_symbol, tgt), dim=1)
            tgt = tgt.transpose(0, 1)
            tgt = self.tgt_embed(tgt.long())
            tgt = self.positional_encoding(tgt)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0), DEVICE=self.device).type(torch.bool).to(self.device)
            mm_data = self.transformer(mm_data, tgt, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=tgt_mask) 
            final_preds = self.generator(mm_data)
            final_preds = final_preds[1:, :, :].permute(0, 1, 2)
            return final_preds, final_preds, final_preds
        else:
            mm_data = self.mm_encoder(mm_data)
            final_preds = self.mm_classifier(mm_data)
            return final_preds, final_preds, final_preds


class Skeleton(Fusion):
    def __init__(self,
                vggish: bool = True,
                weights_path: str = "",
                speech_embed_dim: int = 256,
                skeleton_embed_dim: int = 256,
                use_lstm: bool = False,
                fine_tuned_audio_model: bool = False,
                speech_buffer: float = 0.0,
                offset: int = 2,
                encoder_for_audio: bool = False,
                align_speech_windows: bool = False,
                encoder_for_skeleton: bool = True,
                sanity_check: bool = False,
                skeleton_alone: bool = True,
                speech_alone: bool = False) -> None:
        super().__init__(
            vggish, 
            weights_path,
            speech_embed_dim, 
            skeleton_embed_dim, 
            use_lstm, 
            fine_tuned_audio_model, 
            speech_buffer, 
            offset, 
            encoder_for_audio=encoder_for_audio, 
            encoder_for_skeleton=encoder_for_skeleton, 
            align_speech_windows=align_speech_windows, 
            sanity_check=sanity_check, 
            skeleton_alone=skeleton_alone, 
            speech_alone=speech_alone
            )
                
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_classifiers(self.speech_embed_dim, self.skeleton_embed_dim)
        
        
    def init_classifiers(self, speech_embed_dim: int, skeleton_embed_dim: int) -> None:
        # Initialize the audio and skeleton classifiers
        self.skeleton_classifier = self.create_classifier(skeleton_embed_dim)
        
    def create_classifier(self, embed_dim: int) -> nn.Module:
        # Create a classifier model
        classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        classifier.apply(initialize_weights)
        return classifier.to(self.device)

    
    def forward(self, audio_data: torch.Tensor, skeleton_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the LateFusion model
        skeleton_data = self.process_skeleton(skeleton_data)
        if self.encoder_for_skeleton:
            skeleton_data = self.skeleton_encoder(skeleton_data)
        skeleton_preds = self.skeleton_classifier(skeleton_data)
        final_preds = skeleton_preds
        audio_preds = skeleton_preds
    
        return audio_preds, skeleton_preds, final_preds


class Speech(Fusion):
    def __init__(self,
                vggish: bool = True,
                weights_path: str = "",
                speech_embed_dim: int = 256,
                skeleton_embed_dim: int = 256,
                use_lstm: bool = False,
                fine_tuned_audio_model: bool = False,
                speech_buffer: float = 0.0,
                offset: int = 2,
                encoder_for_audio: bool = False,
                align_speech_windows: bool = True,
                encoder_for_skeleton: bool = False,
                sanity_check: bool = False,
                skeleton_alone: bool = False,
                speech_alone: bool = True,
                audio_model_name: str = 'pretrained') -> None:

        super().__init__(
            vggish, 
            weights_path,
            speech_embed_dim, 
            skeleton_embed_dim, 
            use_lstm, 
            fine_tuned_audio_model, 
            speech_buffer, 
            offset, 
            encoder_for_audio=encoder_for_audio, 
            encoder_for_skeleton=encoder_for_skeleton, 
            align_speech_windows=align_speech_windows, 
            sanity_check=sanity_check, 
            skeleton_alone=skeleton_alone, 
            speech_alone=speech_alone, 
            audio_model_name=audio_model_name
            )
                
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_classifiers(self.speech_embed_dim, self.skeleton_embed_dim)
        
        
    def init_classifiers(self, speech_embed_dim: int, skeleton_embed_dim: int) -> None:
        self.audio_classifier = self.create_classifier(speech_embed_dim)
    
    def create_classifier(self, embed_dim: int) -> nn.Module:
        # Create a classifier model
        classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        classifier.apply(initialize_weights)
        return classifier.to(self.device)

    
    def forward(self, audio_data: torch.Tensor, skeleton_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_data = self.process_audio(skeleton_data, audio_data)
        # speech and skeleton encoders
        if self.encoder_for_audio:
            audio_data = self.audio_encoder(audio_data)
        audio_preds = self.audio_classifier(audio_data)
        final_preds = audio_preds
        skeleton_preds = audio_preds
    
        return audio_preds, skeleton_preds, final_preds