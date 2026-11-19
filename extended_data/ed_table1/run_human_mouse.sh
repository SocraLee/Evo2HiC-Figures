#!/usr/bin/env bash
# HiCFoundation resolution enhancement on the Evo2HiC human/mouse test maps
# (1/16 downsampled, 10 kb, test chromosomes only: human chr9/10, mouse chr1-4).
# Usage: run_human_mouse.sh <list_file: lines "<species> <accession>"> <gpu_id>
# Output: $W/<species>/<acc>_enhanced_test.hic  (layout consumed by evaluate.general_eval_batch)
set -u
LIST=$1; GPU=$2
. "$(dirname "$0")/../../paths.sh"
HERE=$(cd "$(dirname "$0")" && pwd)       # scripts
W=$RA_WORK_ROOT/extended_data/ed_table1             # outputs
PY=$HICFOUNDATION_PY
D=$HIC_DATA_ROOT                          # java (>= 1.8) must be on PATH for .hic writing
while read -r SP A; do
  [ -z "$A" ] && continue
  case $SP in
    human) CHR=chr9,chr10; G=hg38;;
    mouse) CHR=chr1,chr2,chr3,chr4; G=mm10;;
    *) echo "[bad-species] $SP $A"; continue;;
  esac
  mkdir -p $W/$SP $W/tmp_$SP
  FINAL=$W/$SP/${A}_enhanced_test.hic
  if [ -s "$FINAL" ]; then echo "[skip] $A"; continue; fi
  IN=$D/data/hic/downsample/$A.16.seed0.hic
  [ -f "$IN" ] || { echo "[missing-input] $A"; continue; }
  O=$W/tmp_$SP/$A; rm -rf $O; mkdir -p $O
  t0=$(date +%s)
  $PY -u $HERE/infer_subset.py $IN $O 10000 $GPU $CHR > $O/infer.log 2>&1
  [ -s $O/HiCFoundation_enhanced.pkl ] || { echo "[infer-fail] $A"; continue; }
  $PY $HERE/pkl2hic.py $O 10000 $G $FINAL > $O/convert.log 2>&1
  if [ -s "$FINAL" ]; then
    rm -f $O/HiCFoundation_enhanced.pkl $O/input.pkl $O/HiCFoundation_enhanced.hic
    echo "[ok] $SP $A $(( $(date +%s) - t0 ))s"
  else
    echo "[convert-fail] $A"
  fi
done < "$LIST"
echo "[done] $LIST"
