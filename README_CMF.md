# SDFusion + CMF Experiments

This experiment fine-tunes the unconditional SDFusion denoiser with pure CMF-style long-hop flow matching. It does not use teacher endpoint distillation; baseline 50-step sampling is used only for evaluation.

Run from the `SDFusion` directory. Replace dataset and checkpoint paths with the paths available on the experiment server.

## Current Findings

The strongest pure-CMF setting so far is:

```bash
--loss-type ddim_endpoint_multistep_rollout_fm
--train-ddim-steps 10
--rollout-hops 4
--r-cond-mode delta_add
--time-param normalized
--time-sampler ddim_endpoint_pair
--loss-kind huber
--huber-beta 0.01
--lr 5e-9
```

On the chair runs, CMF 10-step sampling was about 4.7x faster than baseline 50-step sampling and was much more stable than earlier MSE/random-hop variants. It still did not rigorously match baseline 50-step quality: a few samples kept small fragments or rough geometry.

On the table run, the same setting transferred without widespread collapse and kept a similar speed gain, but visual quality was weaker than chair and still below baseline 50-step.

The current conclusion is negative-to-mixed but usable for a course project: pure CMF can make SDFusion 10-step generation substantially faster and more stable, but without teacher distillation it has not yet fully reached 50-step baseline quality.

## Recommended Scripts

The scripts below wrap the recommended setting. They require `DATA_ROOT` and can optionally override `OUT_DIR`, `BASE_CKPT`, `VQ_CKPT`, `ITERS`, and `SEED`.

```bash
DATA_ROOT=/path/to/shapenet_data \
OUT_DIR=/path/to/cfm_sdfusion \
bash experiments/run_cmf_chair_huber.sh
```

```bash
DATA_ROOT=/path/to/shapenet_data \
OUT_DIR=/path/to/cfm_sdfusion \
bash experiments/run_cmf_table_huber.sh
```

## Chair

Smoke test:

```bash
python experiments/cfm_uncond_experiment.py \
  --mode train \
  --name cfm-chair-smoke \
  --cat chair \
  --dataroot /path/to/shapenet_data \
  --base-ckpt saved_ckpt/sdfusion-snet-all.pth \
  --vq-ckpt saved_ckpt/vqvae-snet-all.pth \
  --out-dir /path/to/cfm_sdfusion \
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
  --iters 200
```

Short run:

```bash
python experiments/cfm_uncond_experiment.py \
  --mode train \
  --name cfm-chair-10step-h4-huber \
  --cat chair \
  --dataroot /path/to/shapenet_data \
  --base-ckpt saved_ckpt/sdfusion-snet-all.pth \
  --vq-ckpt saved_ckpt/vqvae-snet-all.pth \
  --out-dir /path/to/cfm_sdfusion \
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
  --iters 3000
```

Evaluation:

```bash
python experiments/cfm_uncond_experiment.py \
  --mode eval \
  --name eval-chair \
  --cat chair \
  --dataroot /path/to/shapenet_data \
  --base-ckpt saved_ckpt/sdfusion-snet-all.pth \
  --vq-ckpt saved_ckpt/vqvae-snet-all.pth \
  --cfm-ckpt /path/to/cfm_sdfusion/cfm-chair-10step-h4-huber/ckpt/cfm_steps-latest.pth \
  --out-dir /path/to/cfm_sdfusion \
  --seed 2026 \
  --eval-batch 4 \
  --eval-steps 10 50
```

## Table

Use the same commands with:

```bash
--cat table
--name cfm-table-10step-h4-huber
```

The key comparison is CMF 10-step versus baseline 50-step, with baseline 10-step and CMF 50-step kept as diagnostics.
