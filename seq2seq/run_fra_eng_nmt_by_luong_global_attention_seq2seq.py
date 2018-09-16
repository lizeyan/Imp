from config import init_sacred
from explib.fra_eng_nmt_experiment import fra_eng_nmt, EncoderSeq
import tempfile

ex = init_sacred(name="Fra-Eng NMT by Luong Global Attention Seq2Seq")


# noinspection PyUnusedLocal
@ex.config
def experiment_config():
    hidden_size = 256
    learning_rate = 0.1
    decoder_cls = "LuongGlobalAttentionDecoderSeq"
    embedding_size = 256
    n_layers = 1
    reverse_input = True
    max_steps = 500000
    dropout_p = 0.1
    batch_size = 1
    device = "cuda"


@ex.automain
def main():
    encoder_save_file = tempfile.NamedTemporaryFile()
    decoder_save_file = tempfile.NamedTemporaryFile()
    bleu = fra_eng_nmt(
                       encoder_save_file=encoder_save_file,
                       decoder_save_file=decoder_save_file,
                       **ex.current_run.config
    )
    ex.add_artifact(encoder_save_file.name, name="encoder")
    ex.add_artifact(decoder_save_file.name, name="decoder")
    return bleu

