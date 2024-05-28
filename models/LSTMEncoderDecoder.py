import torch
import torch.nn as nn
import random

class LSTMDecoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_rnn_layers):
    super(LSTMDecoder, self).__init__()
    self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_rnn_layers)
    self.linear = nn.Linear(hidden_size, output_size)
  def forward(self, input, hidden):
    decoder_output, decoder_hidden = self.decoder(input.unsqueeze(1), hidden)
    linear_output = self.linear(decoder_output.squeeze(1))
    return linear_output, decoder_hidden

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, training_prediction, Max_len_out, device, teacher_forcing_ratio = 0.5, num_rnn_layers=1):
        super(LSTMEncoderDecoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size, hidden_size,num_layers=num_rnn_layers, batch_first=True)
        self.decoder = LSTMDecoder(input_size, hidden_size, output_size, num_rnn_layers = num_rnn_layers)
        self.training_prediction = training_prediction
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Max_len_out = Max_len_out
        self.device = device

    def forward(self, x, target=None):
        # Encoder
        _, encoder_hidden = self.encoder(x)
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size,self.Max_len_out, self.output_size, device=self.device)

        decoder_input = x[:,-1, :]   # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        # Decoder
        if self.training_prediction == 'recursive':
          # predict recursively
          for t in range(self.Max_len_out):
              decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
              outputs[:, t,:] = decoder_output
              decoder_input = decoder_output

        if self.training_prediction == 'teacher_forcing':
            # use teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                for t in range(self.Max_len_out):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t,:] = decoder_output
                    decoder_input = target[:, t, :]

            # predict recursively
            else:
                for t in range(self.Max_len_out):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t,:] = decoder_output
                    decoder_input = decoder_output

        if self.training_prediction == 'mixed_teacher_forcing':
            # predict using mixed teacher forcing
            for t in range(self.Max_len_out):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t,:] = decoder_output

                # predict with teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    decoder_input = target[:, t, :]

                # predict recursively
                else:
                    decoder_input = decoder_output
        return outputs
