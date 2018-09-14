from nltk.translate.bleu_score import corpus_bleu
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

from explib.prepare_fra_eng_nmt_data import prepareData
from snippets.modules import *
from snippets.scaffold import TrainLoop, TestLoop


def fra_eng_nmt(*,
                hidden_size: int = 256, embedding_size: int = 256, n_layers: int = 1,
                reverse_input: bool = False, device="cuda",
                decoder_cls: str = "DecoderSeq",
                dropout_p: float = 0.5, learning_rate: float = 1e-3,
                batch_size: int = 64, max_steps: int = 75000,
                encoder_load_file=None, decoder_load_file=None,
                encoder_save_file=None, decoder_save_file=None,
                grad_norm: float = 10., **kwargs):
    # read data
    (train_dataset, validation_dataset, test_dataset), (source_lang, target_lang) = \
        prepareData("eng", "fra", reverse=True, split_radio=[0.5, 0.01, 0.49])

    # define model
    encoder = EncoderSeq(input_size=source_lang.n_words,
                         hidden_size=hidden_size,
                         embedding_size=embedding_size,
                         n_layers=n_layers,
                         reverse_input=reverse_input) \
        .to(device)
    decoder = eval(decoder_cls)(output_size=target_lang.n_words,
                                hidden_size=hidden_size,
                                embedding_size=embedding_size,
                                n_layers=n_layers, dropout_p=dropout_p) \
        .to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    e_scheduler = StepLR(encoder_optimizer, step_size=5, gamma=0.75)
    d_scheduler = StepLR(decoder_optimizer, step_size=5, gamma=0.75)

    train_data_loader = DataLoader(dataset=train_dataset, pin_memory=True,
                                   batch_size=batch_size, shuffle=True,
                                   collate_fn=lambda batch: tuple(zip(*batch)))
    trainer = Seq2SeqTrainer(encoder=encoder, decoder=decoder,
                             eos_index=target_lang.EOS_INDEX,
                             sos_index=target_lang.SOS_INDEX,
                             teach_forcing_prob=0.5, )
    if encoder_load_file:
        encoder.load_state_dict(torch.load(encoder_load_file))
    if decoder_load_file:
        decoder.load_state_dict(torch.load(decoder_load_file))
    with TrainLoop(max_steps=max_steps, disp_step_freq=500).with_context() as loop:
        for _ in loop.iter_epochs():
            # e_scheduler.step()
            # d_scheduler.step()
            for step, (input_seqs, target_seqs) in loop.iter_steps(train_data_loader):
                input_seqs = list(map(lambda _: _.to(device, non_blocking=True), input_seqs))
                target_seqs = list(map(lambda _: _.to(device, non_blocking=True), target_seqs))
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = trainer.step(input_seqs, target_seqs)
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), grad_norm)
                nn.utils.clip_grad_norm_(decoder.parameters(), grad_norm)
                encoder_optimizer.step()
                decoder_optimizer.step()
                loop.submit_metric("train_loss", loss.item())
    if encoder_save_file is not None:
        torch.save(encoder.state_dict(), encoder_save_file)
    if decoder_save_file is not None:
        torch.save(decoder.state_dict(), decoder_save_file)

    reference_list = []
    hypotheses_list = []
    inference = Seq2SeqInference(encoder, decoder, target_lang)
    with TestLoop(disp_step_freq=1000, max_steps=len(test_dataset)).with_context() as test_loop:
        for _, (x, y) in test_loop.iter_steps(test_dataset):
            output_words = inference(x.to(device), len(y))
            target_words = target_lang.tensor_to_tokens(y)
            reference_list.append([target_words])
            hypotheses_list.append(output_words)
            print(">>", " ".join(source_lang.tensor_to_tokens(x)))
            print("==", " ".join(target_words))
            print("<<", " ".join(output_words))
    # smooth = SmoothingFunction()
    bleu_score = corpus_bleu(reference_list, hypotheses_list,
                             emulate_multibleu=True, )
    return bleu_score
