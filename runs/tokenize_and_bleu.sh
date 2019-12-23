max_steps=${max_steps:-200000}
eval_every=${eval_every:-2000}
eval_start=${eval_start:-2000}

exp_name=${exp_name}

for ((steps = eval_start; steps <= max_steps; steps += eval_every))
do
  target=$(ls -1L ./checkpoints/${exp_name}/evals/${steps}.*.targets)
  output=$(ls -1L ./checkpoints/${exp_name}/evals/${steps}.*.decodes)
  if [ -z "$target" ]; then
    echo "${steps}: No output"
    continue
  fi
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $output >  "${output}.tok"
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $target >  "${target}.tok"
  echo "${steps}: $(perl mosesdecoder/scripts/generic/multi-bleu.perl ${target}.tok < ${output}.tok | tail -n 1)"
done
