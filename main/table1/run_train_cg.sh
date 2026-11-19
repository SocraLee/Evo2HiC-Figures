#!/usr/bin/env bash
# Train / recalibrate / validate Evo2HiC-cg (Seq2HiC finetuned in Orca's setting) with the public
# Evo2HiC trainer (CHNFTQ/Evo2HiC main, train/train_CDNA2d_Seq2HiC.py; Evo2HiC-cg support merged in PR #14).
# Base arguments = the released seq2hic run, except
#   -target cg                    Orca-style coarse-grained Micro-C target (Evo2HiC repo: scripts/make_cg_targets.py)
#   --augment-resolution 4000     the cg band lives at 4 kb -> no 2 kb phase-shift augmentation
# Selection uses chr8 (validation) only, never the test chromosomes chr9/10.
# Extra arguments are appended and override the defaults below (argparse: last value wins); the
# final pre-registered recipe is added by rig_ladder2.py (see README.md, "Training").
# Usage: run_train_cg.sh <H1ESC|HFF> <gpu_ids e.g. 0,1,2,3> [port] [extra trainer arguments ...]
set -eu
. "$(dirname "$0")/../../paths.sh"
CELL=$1; GPUS=$2; PORT=${3:-29511}
TARGET=${TARGET:-cg}
case $CELL in
  H1ESC) ACC=4DNFI2TK7L2F;;
  HFF)   ACC=4DNFIPC7P27B;;
  *) echo "cell must be H1ESC or HFF"; exit 1;;
esac
export WANDB_MODE=offline
export PYTHONPATH=$REPO:${PYTHONPATH:-}
NP=$(echo $GPUS | tr ',' '\n' | wc -l)
cd $REPO                     # checkpoints are written to $REPO/checkpoints/<date>_<method-name>_4000 (config.save_dir)
if [ "$NP" -ge 2 ]; then
  LAUNCH="accelerate launch --num_processes $NP --gpu_ids $GPUS --main_process_port $PORT -m"
else
  export CUDA_VISIBLE_DEVICES=$GPUS
  LAUNCH="$PY -m"
fi
$LAUNCH train.train_CDNA2d_Seq2HiC \
  --method-name CDNA2d-Seq2HiC-$TARGET-$CELL \
  -r 4000 --chunk 280 --stride 125 --max-separation 0 \
  -input expected --downsample 16 \
  -target $TARGET --target-specified $ACC \
  --read-count bin-coverage --normalization mixed --max-reads 5000 --denominator 10 \
  -sp human -DNA Yes -evo2 No \
  --augment-resolution 4000 \
  --batch-size 7 --accumulate-step 2 --max-step 20000 --base-learning-rate 0.0001 \
  --flip-prob 0.5 --transpose-prob 0 --DNA-shift 100 --DNA-drop-prob 0 --map-drop-prob 0 \
  --train-pos-per-row 1 --valid-pos-per-row 1 --train-hic-per-pos -1 --valid-hic-per-pos -1 \
  --epoch-step 1000 --seed 0 \
  --initialize-dna-encoder ${DNA_ENCODER_INIT:-$CKPT_ROOT/pretrained_weights/model.pt} \
  "${@:4}"
