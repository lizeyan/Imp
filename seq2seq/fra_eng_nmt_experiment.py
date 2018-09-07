import os

import numpy as np
import visdom
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from config import VISDOM_SERVER, VISDOM_PORT
from prepare_fra_eng_nmt_data import *
from seq2seq import *
from snippets.scaffold import TrainLoop, TestLoop


def fra_eng_nmt(hidden_size, embedding_size, n_layers,
                learning_rate, name,
                reverse_input,
                max_steps,
                dropout_p,
                decoder_cls, load=False,
                encoder_save_file=None, decoder_save_file=None, *args, **kwargs):
    encoder = EncoderSeq(input_size=input_lang.n_words,
                         hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         n_layers=n_layers,
                         reverse_input=reverse_input).cuda()
    decoder = eval(decoder_cls)(output_size=output_lang.n_words,
                                hidden_size=hidden_size,
                                embedding_size=embedding_size,
                                n_layers=n_layers, dropout_p=dropout_p).cuda()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    e_scheduler = CosineAnnealingLR(encoder_optimizer, T_max=5)
    d_scheduler = CosineAnnealingLR(decoder_optimizer, T_max=5)

    train_set_split = 0.5
    training_pairs = [tensorsFromPair(pairs[i])
                      for i in range(int(len(pairs) * train_set_split))]
    train_data_loader = DataLoader(dataset=training_pairs, batch_size=1, shuffle=True, )
    trainer = Seq2SeqTrainer(encoder=encoder, decoder=decoder,
                             optim_e=encoder_optimizer, optim_d=decoder_optimizer,
                             device=torch.cuda.current_device(),
                             eos_token=EOS_token, sos_token=SOS_token,
                             teach_forcing_prob=0.5,
                             criterion=nn.NLLLoss(reduction="sum"),
                             clip_grad_norm=10.)
    vis = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env="Eng-Fra NMT")

    encoder_path = os.path.expanduser("~/experiments/Imp/seq2seq/seq2seq_encoder_state_dict.pkl")
    decoder_path = os.path.expanduser("~/experiments/Imp/seq2seq/seq2seq_decoder_state_dict.pkl")
    if not load:
        with TrainLoop(max_steps=max_steps, disp_step_freq=500).with_context() as loop:
            for epoch in loop.iter_epochs():
                # e_scheduler.step()
                # d_scheduler.step()
                for step, (input_seq, target_seq) in loop.iter_steps(train_data_loader):
                    loss = trainer.step(input_seq.squeeze(0), target_seq.squeeze(0))
                    loop.submit_metric("train_loss", loss)
                train_loss = loop.get_metric("train_loss", epoch=epoch)

                # noinspection PyArgumentList
                vis.line(X=np.asarray([epoch]), Y=np.asarray([np.mean(train_loss)]),
                         opts=dict(
                             legend=["Train Loss"],
                             title=f"{name} Loss",
                         ),
                         win=f"{name} Training", update="append" if epoch > 1 else False, )
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
    else:
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
    if encoder_save_file is not None:
        torch.save(encoder.state_dict(), encoder_save_file)
    if decoder_save_file is not None:
        torch.save(decoder.state_dict(), decoder_save_file)

    test_pairs = pairs[len(training_pairs):]
    reference_list = []
    hypotheses_list = []
    with TestLoop(disp_step_freq=1000, max_steps=len(test_pairs)).with_context() as test_loop:
        for _, (x, y) in test_loop.iter_steps(test_pairs):
            # print('>', x)
            # print('=', y)
            output_words = evaluate(encoder, decoder, x)
            reference_list.append([y.split(" ")])
            hypotheses_list.append(output_words)
            # output_sentence = " ".join(output_words)
            # print('<', output_sentence)
            # print('')
    # smooth = SmoothingFunction()
    bleu_score = corpus_bleu(reference_list, hypotheses_list,
                             emulate_multibleu=True,)
    print("BLEU", bleu_score)
    return bleu_score
