from __future__ import unicode_literals, print_function, division

import tempfile

from config import init_sacred
from fra_eng_nmt_experiment import fra_eng_nmt

ex = init_sacred(name="Fra-Eng NMT by Attention Seq2Seq")


# noinspection PyUnusedLocal
@ex.config
def config():
    hidden_size = 256
    learning_rate = 0.01
    decoder_cls = "AttentionDecoderSeq"
    embedding_size = 256
    n_layers = 1
    load = False
    reverse_input = False
    max_steps = 75000
    dropout_p = 0.1


@ex.automain
def main():
    encoder_save_file = tempfile.NamedTemporaryFile()
    decoder_save_file = tempfile.NamedTemporaryFile()
    bleu = fra_eng_nmt(name=ex.path,
                       encoder_save_file=encoder_save_file,
                       decoder_save_file=decoder_save_file,
                       **ex.current_run.config)
    ex.add_artifact(encoder_save_file.name, name="encoder")
    ex.add_artifact(decoder_save_file.name, name="decoder")
    return bleu


