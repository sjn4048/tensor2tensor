exp_name=${exp_name}
max_steps=${max_steps:-100000}
eval_every=${eval_every:-2000}
EVAL_DIR=./checkpoints/${exp_name}/evals

for ((steps = eval_every; steps <= max_steps; steps += eval_every))
do
  output_filename=${EVAL_DIR}/${steps}.*.decodes
  target_filename=${EVAL_DIR}/${steps}.*.targets
  echo ${steps}:
  if [ -f "$output_filename" ]; then
     perl multi-bleu.perl target_filename < target_filename 2>&1 | head -n 1
  fi
done