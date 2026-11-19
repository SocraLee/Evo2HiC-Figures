#!/usr/bin/env bash
# HiCFoundation epigenomic prediction (task 4) on GM12878 and H1ESC, chr9/chr10 (K562: same call, see README.md).
. "$(dirname "$0")/../../paths.sh"
HERE=$(cd "$(dirname "$0")" && pwd); W=$RA_WORK_ROOT/extended_data/ed_table1; mkdir -p $W/logs
for pair in "GM12878 4DNFI1UEG1HD" "H1ESC 4DNFIQYQWPF5"; do set -- $pair; mkdir -p $W/epi/$1; $HICFOUNDATION_PY -u $HERE/infer_epi.py $HIC_DATA_ROOT/data/hic/raw_hic/$2.hic $W/epi/$1 3 chr9,chr10 > $W/logs/epi_$1.log 2>&1; done; echo "[done]" >> $W/logs/epi_H1ESC.log
