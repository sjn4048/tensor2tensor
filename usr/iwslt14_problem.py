from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate, text_encoder
from tensor2tensor.utils import registry
import os
import tensorflow as tf

def _get_iwslt_deen_bpe_dataset(directory, filename):
  """Extract the iwslt de-en corpus `filename` to directory."""
  train_path = os.path.join(directory, filename)
  if not (tf.gfile.Exists(train_path + ".de") and
          tf.gfile.Exists(train_path + ".en")):
    raise NotImplementedError()
  return train_path

@registry.register_problem
class TranslateEndeWmtBpe32k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def oov_token(self):
    return "UNK"

  @property
  def vocab_filename(self):
    return "vocab."

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the WMT en->de task, training set."""
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset_path = ("train"
                    if train else "test")
    train_path = _get_iwslt_deen_bpe_dataset(tmp_dir, dataset_path)

    # Vocab
    vocab_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_path):
      bpe_vocab = os.path.join(tmp_dir, "vocab.bpe.32000")
      with tf.gfile.Open(bpe_vocab) as f:
        vocab_list = f.read().split("\n")
      vocab_list.append(self.oov_token)
      text_encoder.TokenTextEncoder(
          None, vocab_list=vocab_list).store_to_file(vocab_path)

    return text_problems.text2text_txt_iterator(train_path + ".de",
                                                train_path + ".en")