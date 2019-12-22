max_steps=${max_steps:-100000}
eval_every=${eval_every:-2000}

exp_name=${exp_name}

for ((steps = eval_every; steps <= max_steps; steps += eval_every))
do
  target=./checkpoints/${exp_name}/evals/${steps}.transformer.transformer_small.translate_de_en_iwslt.beam4.alpha1.0.targets
  output=./checkpoints/${exp_name}/evals/${steps}.transformer.transformer_small.translate_de_en_iwslt.beam4.alpha1.0.decodes
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $output >  "${output}.tok"
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $target >  "${target}.tok"

  perl mosesdecoder/scripts/generic/multi-bleu.perl ${target}.tok < ${output}.tok