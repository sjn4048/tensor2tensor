export PYTHONPATH=.:${PYTHONPATH}
binFile=./tensor2tensor/bin
PROBLEM=translate_de_en_iwslt_bpe32k
DATA_DIR=./data/iwslt14.tokenized.de-en
python ${binFile}/t2t-datagen \
  --t2t_usr_dir=usr \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR \
  --tmp_dir=$DATA_DIR/tmp

