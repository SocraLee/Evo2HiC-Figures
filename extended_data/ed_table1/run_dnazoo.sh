#!/usr/bin/env bash
# HiCFoundation resolution enhancement on DNA Zoo species (1/16 downsampled, 10 kb).
# Usage: run_dnazoo.sh <species_list_file> <gpu_id>
# Output layout mirrors the Evo2HiC multi_10000 dir so evaluate.general_eval_batch can consume it:
#   $W/multi_10000/<species>_enhanced.hic
set -u
LIST=$1; GPU=$2
. "$(dirname "$0")/../../paths.sh"
HERE=$(cd "$(dirname "$0")" && pwd)       # scripts
W=$RA_WORK_ROOT/extended_data/ed_table1             # outputs
HF=$HICFOUNDATION_DIR
PY=$HICFOUNDATION_PY
D=$HIC_DATA_ROOT                          # java (>= 1.8) must be on PATH for .hic writing
OUTDIR=$W/multi_10000; TMP=$W/tmp_multi; mkdir -p $OUTDIR $TMP $W/logs
cd $HF
while read -r S; do
  [ -z "$S" ] && continue
  FINAL=$OUTDIR/${S}_enhanced.hic
  if [ -s "$FINAL" ]; then echo "[skip] $S"; continue; fi
  IN=$D/data/hic/downsample_multi/$S.16.seed0.hic
  CS=$D/data/hic/downsample_multi/$S.16.seed0.chrom.size
  if [ ! -f "$IN" ] || [ ! -f "$CS" ]; then echo "[missing-input] $S"; continue; fi
  O=$TMP/$S; rm -rf $O; mkdir -p $O
  t0=$(date +%s)
  $PY -u inference.py --input $IN --batch_size 16 --resolution 10000 --task 3 \
      --input_row_size 224 --input_col_size 224 --stride 20 --bound 0 \
      --model_path hicfoundation_model/hicfoundation_resolution.pth.tar \
      --output $O --gpu "$GPU" --genome_id $CS > $O/infer.log 2>&1
  if [ ! -s $O/HiCFoundation_enhanced.pkl ]; then echo "[infer-fail] $S"; continue; fi
  $PY $HERE/pkl2hic.py $O 10000 $CS $FINAL > $O/convert.log 2>&1
  if [ -s "$FINAL" ]; then
    cp $CS $OUTDIR/${S}_enhanced.chrom.size
    rm -f $O/HiCFoundation_enhanced.pkl $O/input.pkl $O/HiCFoundation_enhanced.hic
    echo "[ok] $S $(( $(date +%s) - t0 ))s"
  else
    echo "[convert-fail] $S"
  fi
done < "$LIST"
echo "[done] $LIST"
