import torch
import torch.nn as nn
import torch.optim as optim
import math

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
   
   def forward(self, x):
      x = self.embedding(x)
      x = self.positional_encoding(x)
      x = self.encoder(x)
      return x
   def initialize_weights(self, m):
      if isinstance(m, nn.Linear):
         nn.init.xavier_uniform_(m.weight)
         if m.bias is not None:
               nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
         nn.init.constant_(m.weight, 1)
         nn.init.constant_(m.bias, 0)  

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, num_layers):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Linear(output_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1, max_len=100)
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)
        
    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory)
        output = self.fc_out(output)
        return output

import math  # Don't forget to import math

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


# Existing CoSpeechGestureModel class
class CoSpeechGestureModel(nn.Module):
    def __init__(self, audio_dim=512, video_dim=256, gesture_dim=1, embed_dim=512, num_heads=8, num_layers=6):
        super(CoSpeechGestureModel, self).__init__()
        
        self.audio_encoder = Encoder(audio_dim, embed_dim, num_heads, num_layers)
        self.video_encoder = Encoder(video_dim, embed_dim, num_heads, num_layers)
        
        self.multi_modal_attention = MultiModalAttention(embed_dim)  # New line
        
        self.decoder = Decoder(gesture_dim, embed_dim, num_heads, num_layers)
        
    def forward(self, audio_data, video_data, gesture_data):
        audio_memory = self.audio_encoder(audio_data)
        video_memory = self.video_encoder(video_data)
        
        combined_memory = self.multi_modal_attention(audio_memory, video_memory)  # New line replacing simple addition
        
        gesture_output = self.decoder(gesture_data, combined_memory)
        return gesture_output

if __name__ == '__main__':

   # Instantiate the model
   audio_dim = 512
   video_dim = 256
   gesture_dim = 1  # Changed to 1 since output is binary
   embed_dim = 512
   num_heads = 8
   num_layers = 6

   model = CoSpeechGestureModel(audio_dim, video_dim, gesture_dim, embed_dim, num_heads, num_layers)
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.BCEWithLogitsLoss()  # Changed to Binary Cross-Entropy Loss with Logits

   # Your data and training loop should remain similar, but do ensure that `gesture_data` is binary.

   # Simulate some data for demonstration
   # Replace these with your actual data
   n_samples = 100  # Number of data samples
   seq_length = 50  # Sequence length
   batch_size = 32  # Batch size

   audio_data = torch.rand((n_samples, seq_length, batch_size, audio_dim))
   video_data = torch.rand((n_samples, seq_length, batch_size, video_dim))
   gesture_data = torch.rand((n_samples, seq_length, batch_size, gesture_dim))

   # Training loop
   n_epochs = 10  # Number of epochs
   for epoch in range(n_epochs):
      for i in range(n_samples):
         # Zero the gradients
         optimizer.zero_grad()
         
         # Forward pass
         output = model(audio_data[i], video_data[i], gesture_data[i])
         
         # Compute the loss
         loss = criterion(output, gesture_data[i])
         
         # Backward pass
         loss.backward()
         
         # Update the parameters
         optimizer.step()
         
      print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
      
      # Turn off gradient calculations
   with torch.no_grad():
      # Prepare your new audio and video data
      n_new_samples = 10  # Replace this with your actual number of new samples
      new_audio_data = torch.rand((n_new_samples, seq_length, batch_size, audio_dim))
      new_video_data = torch.rand((n_new_samples, seq_length, batch_size, video_dim))

      # Initialize an empty tensor to store the predicted gesture sequences
      predicted_gesture_data = torch.zeros((n_new_samples, seq_length, batch_size, gesture_dim))

      # Loop over new samples
      for i in range(n_new_samples):
         # Initialize the first token for the decoder. This could be a tensor of zeros.
         gesture_init = torch.zeros((1, batch_size, gesture_dim))

         # Initialize a tensor to hold the full sequence prediction for current sample
         full_seq_pred = torch.zeros((seq_length, batch_size, gesture_dim))

         # Loop over sequence length
         for t in range(seq_length):
               # Forward pass to get the predicted gesture for this time step
               if t == 0:
                  output = model(new_audio_data[i], new_video_data[i], gesture_init)
               else:
                  output = model(new_audio_data[i], new_video_data[i], full_seq_pred[:t])

               # Store the output in the full sequence
               full_seq_pred[t] = output[-1]  # Assuming output is in shape [seq_len, batch, dim]

         # Store the full sequence prediction for this sample
         predicted_gesture_data[i] = full_seq_pred

   # Now, `predicted_gesture_data` contains the predicted gesture sequences.
