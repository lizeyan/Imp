import random
from typing import Callable, Union
import torch
from torch import nn, optim
import torch.nn.functional as func


class EncoderSeq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *,
                 n_layers: int = 1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)

    def forward(self, x, hidden=None):
        """
        :param x: A Tensor in shape (seq_length, 1, ) or (1, )
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :return:
        """
        if len(x.size()) == 2:
            seq_length = x.size(0)
        elif len(x.size()) <= 1:
            seq_length = 1
        else:
            raise ValueError(f"Input x has invalid shape: {x.size()}")
        embedded = self.embedding(x).view(seq_length, 1, -1)
        output, hidden = self.rnn(embedded, hidden)

        output = output.squeeze(1)
        if len(x.size()) == 1:
            output = output.squeeze(0)
        return output, hidden


class DecoderSeq(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, *,
                 n_layers: int = 1):
        super().__init__()
        self.n_layers = n_layers
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None):
        """
        :param x: A Tensor in shape (seq_length, 1, ) (a sequence) or (1, ) (an element in a sequence)
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :return:
        """
        if len(x.size()) == 2:
            seq_length = x.size(0)
        elif len(x.size()) <= 1:
            seq_length = 1
        else:
            raise ValueError(f"Input x has invalid shape: {x.size()}")
        embedded = self.embedding(x).view(seq_length, 1, -1)
        output = func.relu(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        output = output.squeeze(1)
        if len(x.size()) == 1:
            output = output.squeeze(0)
        return output, hidden

    def forward_n(self, x, hidden=None, *, n_steps: int, stop_token: int = None):
        """
        :param x: A Tensor in shape (1,), representing an element in a sequence
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :param n_steps:
        :param stop_token:
        :return:
        """
        decoder_input = x
        decoder_outputs = torch.zeros(n_steps, self.output_size, device=x.device)
        for di in range(n_steps):
            decoder_output, decoder_hidden = self.forward(decoder_input, hidden)
            decoder_outputs[di] = decoder_output
            _, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach()
            if stop_token is not None and decoder_input.item() == stop_token:
                break
        return decoder_outputs


class Seq2SeqTrainer(object):
    def __init__(self, *, encoder: nn.Module, decoder: nn.Module,
                 optim_e: optim.Optimizer, optim_d: optim.Optimizer,
                 criterion: Callable,
                 device: Union[int, str] = "cpu",
                 eos_token: int = 1, sos_token: int = 0,
                 teach_forcing_prob: float = 0.5):
        self.teach_forcing_prob = teach_forcing_prob
        self.device = device
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.criterion = criterion
        self.optim_d = optim_d
        self.optim_e = optim_e
        self.decoder = decoder
        self.encoder = encoder

    def step(self, input_tensor, target_tensor):
        encoder_hidden = None

        self.optim_e.zero_grad()
        self.optim_d.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(input_length, self.encoder.hidden_size, device=input_tensor.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output

        sos_tensor = torch.tensor([[self.sos_token]], device=self.device)
        decoder_hidden = encoder_hidden

        use_teach_forcing = True if random.random() < self.teach_forcing_prob else False
        decoder_input = sos_tensor
        if use_teach_forcing:
            decoder_input = torch.cat([sos_tensor, target_tensor[:-1]])
            decoder_outputs, _ = self.decoder(decoder_input, decoder_hidden)
            loss = self.criterion(decoder_outputs, target_tensor[:, 0])
        else:
            decoder_outputs = self.decoder.forward_n(decoder_input, decoder_hidden,
                                                     n_steps=target_length,
                                                     stop_token=self.eos_token)
            loss = self.criterion(decoder_outputs, target_tensor[:, 0])
        loss.backward()
        self.optim_d.step(closure=None)
        self.optim_e.step(closure=None)
        return loss.item() / target_length
