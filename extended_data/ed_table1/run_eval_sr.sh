#!/usr/bin/env bash
# Evaluate HiCFoundation super-resolved .hic files with the Evo2HiC evaluation
# pipeline (same commands that produced result/supp17 and Fig 6).
#   human : chr9/10 (split=test), 10 kb, tasks diagonal (PCC/SPC) + chrom (MSE/PSNR/SSIM)
#   mouse : chr1-4  (split=test)
#   multi : DNA Zoo first scaffold
# Usage: run_eval_sr.sh <human|mouse|multi> [diagonal|chrom]
set -u
SP=$1; TASKS=${2:-"diagonal chrom"}
. "$(dirname "$0")/../../paths.sh"
W=$RA_WORK_ROOT/extended_data/ed_table1             # outputs
R=$REPO                                   # java (>= 1.8) must be on PATH
mkdir -p $W/logs
cd $R
case $SP in
  human|mouse) DIR=$W/$SP; SPARG=$SP;;
  multi)       DIR=$W/multi_10000; SPARG=multi;;
esac
for t in $TASKS; do
  $PY -m evaluate.batch_evaluate.general_eval_batch -d0 $DIR --species $SPARG -t $t -r 10000 --split test \
      > $W/logs/eval_${SP}_${t}.log 2>&1
  echo "[eval] $SP $t -> $DIR/eval_results_${t}_10000/results.tsv ($(tail -n +2 $DIR/eval_results_${t}_10000/results.tsv | wc -l) rows)"
done
