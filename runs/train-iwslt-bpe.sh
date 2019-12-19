export PYTHONPATH=./:${PYTHONPATH}
binFile=./tensor2tensor/bin
problem=translate_de_en_iwslt_bpe32k
model=transformer
hparams_set=${hparams_set}
exp_name=${exp_name}
hparams=${hparams}
gpu=${gpu:-4}
if [[ $CUDA_VISIBLE_DEVICES != "" ]]; then
  t=(${CUDA_VISIBLE_DEVICES//,/ })
  gpu=${#t[@]}
fi

echo "Using #gpu=$gpu..."

data=${data:-iwslt14.tokenized.de-en}
steps=${steps:-80000}
DATA_DIR=./data/iwslt14.tokenized.de-en
TRAIN_DIR=./checkpoints/${exp_name}
mkdir -p $TRAIN_DIR

python ${binFile}/t2t-trainer \
  --schedule=train \
  --t2t_usr_dir=./usr \
  --data_dir=${DATA_DIR} \
  --output_dir=${TRAIN_DIR} \
  --problem=${problem} \
  --model=${model} \
  --hparams_set=${hparams_set} \
  --worker_gpu=$gpu \
  --hparams=$hparams \
  --keep_checkpoint_max=1000 \
  --train_steps=$steps \
  --save_checkpoints_secs=3600  2>&1 | tee $TRAIN_DIR/log.txt
