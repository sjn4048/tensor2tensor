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
class TranslateDeEnIwsltBpe32k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def approx_vocab_size(self):
    return 2**15

  @property
  def oov_token(self):
    return "UNK"

  @property
  def vocab_filename(self):
    return "vocab.31719.subwords"

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the IWSLT de->en task, training set."""
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset_path = ("train"
                    if train else "test")
    train_path = _get_iwslt_deen_bpe_dataset(data_dir, dataset_path)

    # Vocab
    vocab_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_path):
      raise NotImplementedError()

    return text_problems.text2text_txt_iterator(train_path + ".de",
                                                train_path + ".en")

@registry.register_problem
class TranslateDeEnIwslt(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, subword version."""
  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    return 2**15

  @property
  def oov_token(self):
    return "UNK"

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    def generate_lines_for_vocab_fn(data_dir, source, file_byte_budget=1e6):
      tf.logging.info("Generating vocab from: %s", str(source))
      for lang_file in source:
        tf.logging.info("Reading file: %s" % lang_file)
        filepath = os.path.join(data_dir, lang_file)

        with tf.gfile.GFile(filepath, mode="r") as source_file:
          file_byte_budget_ = file_byte_budget
          counter = 0
          countermax = int(source_file.size() / file_byte_budget_ / 2)
          for line in source_file:
            if counter < countermax:
              counter += 1
            else:
              if file_byte_budget_ <= 0:
                break
              line = line.strip()
              file_byte_budget_ -= len(line)
              counter = 0
              yield line

    return generate_lines_for_vocab_fn(data_dir,
                                       ('train.en', 'train.de'))
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the IWSLT de->en task, training set."""
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset_path = ("train"
                    if train else "test")
    train_path = _get_iwslt_deen_bpe_dataset(data_dir, dataset_path)

    return text_problems.text2text_txt_iterator(train_path + ".de",
                                                train_path + ".en")