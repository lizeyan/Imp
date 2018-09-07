from __future__ import unicode_literals, print_function, division
import inspect
from config import init_sacred
from fra_eng_nmt_experiment import fra_eng_nmt
import tempfile

ex = init_sacred(name="Fra-Eng NMT by Seq2Seq")


# noinspection PyUnusedLocal
@ex.config
def config():
    hidden_size = 256
    learning_rate = 0.01
    decoder_cls = "DecoderSeq"
    embedding_size = 256
    n_layers = 1
    load = False
    reverse_input = True
    max_steps = 150000


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
