max_steps=${max_steps:-100000}
eval_every=${eval_every:-2000}

exp_name=${exp_name}

for ((steps = eval_every; steps <= max_steps; steps += eval_every))
do
  target=${ls -1L "./checkpoints/${exp_name}/evals/${steps}.*.targets"}
  target=${ls -1L "./checkpoints/${exp_name}/evals/${steps}.*.decodes"}
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $output >  "${output}.tok" 2>&1 > /dev/null
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $target >  "${target}.tok" 2>&1 > /dev/null
  echo "${step}: ${perl mosesdecoder/scripts/generic/multi-bleu.perl ${target}.tok < ${output}.tok | tail -n 1}"
done