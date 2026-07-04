#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the preprocessed ShapeNet data root}"
OUT_DIR="${OUT_DIR:-runs/cfm_sdfusion}"
BASE_CKPT="${BASE_CKPT:-saved_ckpt/sdfusion-snet-all.pth}"
VQ_CKPT="${VQ_CKPT:-saved_ckpt/vqvae-snet-all.pth}"
ITERS="${ITERS:-3000}"
SEED="${SEED:-2026}"

python experiments/cfm_uncond_experiment.py \
  --mode train \
  --name cfm-table-10step-h4-huber \
  --cat table \
  --dataroot "${DATA_ROOT}" \
  --base-ckpt "${BASE_CKPT}" \
  --vq-ckpt "${VQ_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --loss-type ddim_endpoint_multistep_rollout_fm \
  --train-scope all \
  --r-cond-mode delta_add \
  --time-param normalized \
  --time-sampler ddim_endpoint_pair \
  --train-ddim-steps 10 \
  --rollout-hops 4 \
  --loss-kind huber \
  --huber-beta 0.01 \
  --lr 5e-9 \
  --grad-clip 1 \
  --max-dataset-size 256 \
  --batch-size 2 \
  --iters "${ITERS}"

python experiments/cfm_uncond_experiment.py \
  --mode eval \
  --name eval-table-10step-h4-huber \
  --cat table \
  --dataroot "${DATA_ROOT}" \
  --base-ckpt "${BASE_CKPT}" \
  --vq-ckpt "${VQ_CKPT}" \
  --cfm-ckpt "${OUT_DIR}/cfm-table-10step-h4-huber/ckpt/cfm_steps-latest.pth" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --eval-batch 4 \
  --eval-steps 10 50
