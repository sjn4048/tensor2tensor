echo 'Start Training&Evaluating IWSLT14-De-En'
export PYTHONPATH=./:${PYTHONPATH}
binFile=./tensor2tensor/bin
problem=${problem:-translate_de_en_iwslt_bpe32k}
model=transformer
hparams_set=${hparams_set}
exp_name=${exp_name}
hparams=${hparams}
gpu=${gpu:-4}
decode_hparams=${decode_hparams:-"beam_size=4,alpha=1,batch_size=800,log_results=False"}
if [[ $CUDA_VISIBLE_DEVICES != "" ]]; then
  t=(${CUDA_VISIBLE_DEVICES//,/ })
  gpu=${#t[@]}
fi

echo "Using #gpu=$gpu..."

data=${data:-iwslt14.tokenized.de-en}
max_steps=${max_steps:-100000}
eval_every=${eval_every:-2000}
DATA_DIR=./data/${data}
TRAIN_DIR=./checkpoints/${exp_name}
EVAL_DIR=${TRAIN_DIR}/evals
LOG_DIR=${TRAIN_DIR}/logs
mkdir -p $EVAL_DIR
mkdir -p $LOG_DIR

for ((steps = eval_every; steps <= max_steps; steps += eval_every))
do
  fl="model_checkpoint_path: \"model.ckpt-${steps}\""
  sed -i "1s/.*/${fl}/" ${TRAIN_DIR}/checkpoint

  python ${binFile}/t2t-decoder \
    --t2t_usr_dir=./usr \
    --data_dir=${DATA_DIR} \
    --problem=${problem} \
    --hparams=${hparams} \
    --model=${model} \
    --hparams_set=${hparams_set} \
    --output_dir=${TRAIN_DIR} \
    --decode_hparams=${decode_hparams} \
    --decode_to_file="./checkpoints/${exp_name}/evals/${steps}" \
    --decode_from_file=${DATA_DIR}/test.de
    --worker_gpu=1

#  perl multi-bleu.perl "${DATA_DIR}/target.en" <  "${EVAL_DIR}/output_${steps}.txt" 2>&1 | tee ${EVAL_DIR}/bleu_${steps}.txt
done
