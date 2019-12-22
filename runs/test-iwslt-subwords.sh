export PYTHONPATH=./:${PYTHONPATH}
binFile=./tensor2tensor/bin
problem=${problem:-translate_de_en_iwslt}
model=${model:-transformer}
hparams_set=${hparams_set:-transformer_small}
exp_name=${exp_name}
hparams=${hparams:-}
decode_hparams=${decode_hparams:-"beam_size=4,alpha=1,batch_size=450,log_results=False"}
gpu=${gpu:-1}

data=${data:-iwslt14.tokenized.de-en.raw}
DATA_DIR=./data/${data}
TRAIN_DIR=./checkpoints/${exp_name}


python ${binFile}/t2t-decoder \
  --t2t_usr_dir=./usr \
  --data_dir=${DATA_DIR} \
  --problem=${problem} \
  --hparams=${hparams} \
  --model=${model} \
  --hparams_set=${hparams_set} \
  --output_dir=${TRAIN_DIR} \
  --decode_hparams=${decode_hparams} \
  --decode_to_file="./checkpoints/${exp_name}/output.en" \
  --worker_gpu=1
