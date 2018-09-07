import random
from typing import Callable, Union
import torch
from torch import nn, optim
import torch.nn.functional as func

MAX_LENGTH = 25


class EncoderSeq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int, *,
                 n_layers: int = 1, reverse_input=False):
        super().__init__()
        self.reverse_input = reverse_input
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=n_layers)

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
        if self.reverse_input:
            x = torch.flip(x, dims=(0,))
        embedded = self.embedding(x).view(seq_length, 1, -1)
        output, hidden = self.rnn(embedded, hidden)

        output = output.squeeze(1)
        if len(x.size()) == 1:
            output = output.squeeze(0)
        return output, hidden


class DecoderSeq(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, embedding_size: int, *,
                 n_layers: int = 1, dropout_p: float = 0.5):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None, **kwargs):
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
        embedded = self.input_dropout(embedded)
        output = func.relu(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        output = output.squeeze(1)
        if len(x.size()) == 1:
            output = output.squeeze(0)
        return output, hidden

    def forward_n(self, x, hidden=None, *, n_steps: int, stop_token: int = None, trim=False,
                  **kwargs):
        """
        :param trim: trim output to actual length or keep max_length
        :param x: A Tensor in shape (1,), representing an element in a sequence
        :param hidden: Corresponding hidden state, keeping it None will use new hidden state
        :param n_steps:
        :param stop_token:
        :return:
        """
        decoder_input = x
        decoder_outputs = torch.zeros(n_steps, self.output_size, device=x.device)
        output_length = 0
        for di in range(n_steps):
            decoder_output, decoder_hidden = self.forward(decoder_input, hidden, **kwargs)
            _, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach()
            decoder_outputs[di] = decoder_output
            if stop_token is not None and decoder_input.item() == stop_token:
                break
            output_length += 1
        if not trim:
            return decoder_outputs
        else:
            return decoder_outputs[:output_length]


class LuongGlobalAttentionDecoderSeq(DecoderSeq):
    """
    Effective approaches to attention-based neural machine translation
    Global Attention
    """

    def forward(self, x, hidden=None, **kwargs):
        """
        :param x: in shape [target_seq_length(st), 1] or [1, ]
        :param hidden:
        :param kwargs:
            encoder_outputs: in shape [input_seq_length(ss), hidden_size]
        :return:
        """
        assert "encoder_outputs" in kwargs, \
            f"encoder_outputs are missing for {self.__class__}"
        encoder_outputs = kwargs["encoder_outputs"]
        if len(x.size()) == 2:
            seq_length = x.size(0)
        elif len(x.size()) <= 1:
            seq_length = 1
        else:
            raise ValueError(f"Input x has invalid shape: {x.size()}")
        assert len(encoder_outputs.size()) == 2 and encoder_outputs.size(1) == self.hidden_size, \
            f"encoder_outputs should in shape (input_sequence_length, hidden_size, \
            but in {encoder_outputs.size()} now)"
        batch_size = 1
        embedded = self.embedding(x).view(seq_length, batch_size, -1)
        embedded = self.input_dropout(embedded)
        embedded = func.relu(embedded)
        assert embedded.size() == (seq_length, batch_size, self.embedding_size)
        gru_output, hidden = self.rnn(embedded, hidden)
        st = gru_output.size(0)  # sequence length of target
        ss = encoder_outputs.size(0)  # sequence length of source
        assert gru_output.size() == (seq_length, batch_size, self.hidden_size)
        assert hidden.size() == (self.n_layers, batch_size, self.hidden_size)
        cat = torch.cat([gru_output.expand(-1, ss, -1),
                         encoder_outputs.unsqueeze(1).transpose(0, 1).expand(st, -1, -1)],
                        dim=-1)
        assert cat.size() == torch.Size((st, ss, self.hidden_size * 2))
        attn_prod = self.attn(cat)
        attn_prod = attn_prod.transpose(1, 2)
        # attn_prod in shape [st, 1, ss]
        attn_weight = func.softmax(attn_prod, dim=2)
        assert attn_weight.size() == (st, 1, ss)
        context_vector = attn_weight @ encoder_outputs
        # context vector(c) in shape(st, 1, h)

        # gru_output: h_t, in shape [st, 1, h]
        # context_vector: c_t, in shape [st, 1, h]
        assert context_vector.size() == torch.Size((st, 1, self.hidden_size))
        output = self.out(torch.cat([gru_output, context_vector], dim=-1))
        output = output.squeeze(1)  # in shape [seq_length, output_size]
        if len(x.size()) == 1:
            output = output.squeeze(0)  # in shape [output_size]
        return output, hidden

    def __init__(self, hidden_size: int, output_size: int, embedding_size: int, *,
                 n_layers: int = 1, dropout_p: float = 0.5):
        super().__init__(hidden_size, output_size, embedding_size,
                         n_layers=n_layers)
        self.attn = nn.Linear(2 * hidden_size, 1, bias=False)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_size, output_size),
            # nn.Tanh(),
            nn.LogSoftmax(dim=-1),
        )
        self.input_dropout = nn.Dropout(dropout_p)


class BahdanauAttentionDecoderSeq(DecoderSeq):
    def __init__(self, hidden_size: int, output_size: int, embedding_size: int,
                 n_layers: int = 1, dropout_p: float = 0.5):
        super().__init__(hidden_size=hidden_size, output_size=output_size,
                         embedding_size=embedding_size,
                         n_layers=n_layers, dropout_p=dropout_p)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.attn_combine = nn.Linear(2 * hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None, **kwargs):
        assert "encoder_outputs" in kwargs, \
            f"encoder_outputs are missing for {self.__class__}"
        encoder_outputs = kwargs["encoder_outputs"]
        if len(x.size()) == 2:
            seq_length = x.size(0)
        elif len(x.size()) <= 1:
            seq_length = 1
        else:
            raise ValueError(f"Input x has invalid shape: {x.size()}")
        assert len(encoder_outputs.size()) == 2 and encoder_outputs.size(1) == self.hidden_size, \
            f"encoder_outputs should in shape (input_sequence_length, hidden_size, \
            but in {encoder_outputs.size()} now)"
        batch_size = 1
        source_sequence_length = encoder_outputs.size(0)
        target_sequence_length = seq_length
        embedded = self.embedding(x).view(seq_length, batch_size, -1)
        embedded = self.input_dropout(embedded)  # (st, 1, hidden_size)
        assert hidden.size() == (self.n_layers, batch_size, self.hidden_size)

        cat = torch.cat([
            hidden[0].unsqueeze(0).expand(target_sequence_length, source_sequence_length, -1),
            encoder_outputs.unsqueeze(0).expand(target_sequence_length, -1, -1)
        ], dim=-1)
        assert cat.size() == (target_sequence_length, source_sequence_length, self.hidden_size * 2)
        attn_weights = self.attn(cat)
        attn_weights = func.softmax(attn_weights, dim=-1)
        assert attn_weights.size() == (target_sequence_length, source_sequence_length, 1)
        context_vector = attn_weights.transpose(1, 2) @ encoder_outputs
        assert context_vector.size() == (target_sequence_length, 1, self.hidden_size)
        assert context_vector.size() == embedded.size()

        rnn_input = self.attn_combine(torch.cat([context_vector, embedded], dim=-1))
        rnn_outputs, hidden = self.rnn(rnn_input, hidden)
        output = self.out(rnn_outputs)
        output = output.squeeze(1)  # in shape [seq_length, output_size]
        if len(x.size()) == 1:
            output = output.squeeze(0)  # in shape [output_size]
        return output, hidden


class AttentionDecoderSeq(DecoderSeq):
    """
    Attention Mechanism following
    [a pytorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
    """
    def __init__(self, hidden_size: int, output_size: int, embedding_size: int,
                 n_layers: int = 1, dropout_p: float = 0.5,
                 max_length=MAX_LENGTH):
        super().__init__(hidden_size=hidden_size, output_size=output_size,
                         embedding_size=embedding_size,
                         n_layers=n_layers, dropout_p=dropout_p)
        self.max_length = max_length
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(2 * hidden_size, max_length)
        self.attn_combine = nn.Linear(2 * hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None, **kwargs):
        assert "encoder_outputs" in kwargs, \
            f"encoder_outputs are missing for {self.__class__}"
        encoder_outputs = kwargs["encoder_outputs"]
        if len(x.size()) == 2:
            seq_length = x.size(0)
        elif len(x.size()) <= 1:
            seq_length = 1
        else:
            raise ValueError(f"Input x has invalid shape: {x.size()}")
        assert len(encoder_outputs.size()) == 2 and encoder_outputs.size(1) == self.hidden_size, \
            f"encoder_outputs should in shape (input_sequence_length, hidden_size, \
            but in {encoder_outputs.size()} now)"
        batch_size = 1
        source_sequence_length = encoder_outputs.size(0)
        assert source_sequence_length <= self.max_length
        encoder_outputs = func.pad(encoder_outputs,
                                   (0, 0, 0, self.max_length - source_sequence_length))
        source_sequence_length = self.max_length
        assert encoder_outputs.size() == (self.max_length, self.hidden_size)
        target_sequence_length = seq_length
        embedded = self.embedding(x).view(seq_length, batch_size, -1)
        embedded = self.input_dropout(embedded)  # (st, 1, hidden_size)
        assert hidden.size() == (self.n_layers, batch_size, self.hidden_size)
        cat = torch.cat([
            embedded,
            hidden[0].unsqueeze(0).expand(target_sequence_length, -1, -1),
        ], dim=-1)
        assert cat.size() == (target_sequence_length, batch_size, self.hidden_size * 2)
        attn_weights = self.attn(cat)
        attn_weights = func.softmax(attn_weights, dim=-1)
        assert attn_weights.size() == \
            (target_sequence_length, batch_size, source_sequence_length)
        context_vector = attn_weights @ encoder_outputs
        assert context_vector.size() == (target_sequence_length, batch_size, self.hidden_size)
        assert context_vector.size() == embedded.size()

        rnn_input = self.attn_combine(torch.cat([context_vector, embedded], dim=-1))
        rnn_input = func.relu(rnn_input)
        rnn_outputs, hidden = self.rnn(rnn_input, hidden)
        output = self.out(rnn_outputs)
        output = output.squeeze(1)  # in shape [seq_length, output_size]
        if len(x.size()) == 1:
            output = output.squeeze(0)  # in shape [output_size]
        return output, hidden


class Seq2SeqTrainer(object):
    def __init__(self, *, encoder: nn.Module, decoder: nn.Module,
                 optim_e: optim.Optimizer, optim_d: optim.Optimizer,
                 criterion: Callable,
                 device: Union[int, str] = "cpu",
                 eos_token: int = 1, sos_token: int = 0,
                 teach_forcing_prob: float = 0.5,
                 clip_grad_norm: float = 5.0):
        self.clip_grad_norm = clip_grad_norm
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

        # noinspection PyCallingNonCallable
        sos_tensor = torch.tensor([[self.sos_token]], device=self.device)
        decoder_hidden = encoder_hidden

        use_teach_forcing = True if random.random() < self.teach_forcing_prob else False
        decoder_input = sos_tensor
        if use_teach_forcing:
            # loss = 0
            # for di in range(target_length):
            #     decoder_output, decoder_hidden, = self.decoder(
            #         decoder_input, decoder_hidden, )
            #     loss += self.criterion(decoder_output, target_tensor[di:di+1, 0])
            #     decoder_input = target_tensor[di:di+1]  # Teacher forcing
            # print(f"iter_loss={loss}")
            decoder_input = torch.cat([sos_tensor, target_tensor[:-1]])
            decoder_outputs, _ = self.decoder(decoder_input, decoder_hidden,
                                              encoder_outputs=encoder_outputs)
            loss = self.criterion(decoder_outputs, target_tensor[:, 0])
        else:
            # loss = 0
            # for di in range(target_length):
            #     decoder_output, decoder_hidden, = self.decoder(
            #         decoder_input, decoder_hidden, )
            #     topv, topi = decoder_output.topk(1)
            #     decoder_input = topi.detach()  # detach from history as input
            #
            #     loss += self.criterion(decoder_output, target_tensor[di:di+1, 0])
            #     if decoder_input.item() == self.eos_token:
            #         break
            decoder_outputs = self.decoder.forward_n(decoder_input, decoder_hidden,
                                                     n_steps=target_length,
                                                     stop_token=self.eos_token,
                                                     encoder_outputs=encoder_outputs)
            loss = self.criterion(decoder_outputs, target_tensor[:, 0])
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad_norm)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_grad_norm)
        loss.backward()
        self.optim_d.step(closure=None)
        self.optim_e.step(closure=None)
        return loss.item() / target_length
