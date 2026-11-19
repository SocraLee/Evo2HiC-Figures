#!/usr/bin/env bash
# Per-map evaluation for human/mouse HiCFoundation outputs (bypasses the xlsx index).
# Usage: run_eval_hm.sh <human|mouse>
set -u
SP=$1
. "$(dirname "$0")/../../paths.sh"
HERE=$(cd "$(dirname "$0")" && pwd)       # scripts (hm_25.txt)
W=$RA_WORK_ROOT/extended_data/ed_table1             # outputs
R=$REPO
D=$HIC_DATA_ROOT
cd $R
for t in diagonal chrom; do
  OUT=$W/$SP/eval_results_${t}_10000; mkdir -p $OUT; : > $OUT/results.tsv
  first=1
  grep "^$SP " $HERE/hm_25.txt | while read -r _ A; do
    $PY -m evaluate.general_eval -f0 $W/$SP/${A}_enhanced_test.hic -f1 $D/data/hic/raw_hic/$A.hic \
        --species $SP --split test -t $t -r 10000 --save-dir $OUT/$A > $OUT/$A.log 2>&1
    F=$OUT/$A/${t}_10000_test.tsv
    if [ -s "$F" ]; then
      if [ $first = 1 ]; then head -1 $F | sed 's/^[^\t]*/hic/' >> $OUT/results.tsv; first=0; fi
      tail -1 $F | sed "s/^[^\t]*/$A/" >> $OUT/results.tsv
    else echo "[fail] $A $t"; fi
  done
  echo "[eval] $SP $t: $(tail -n +2 $OUT/results.tsv | wc -l) rows"
done
