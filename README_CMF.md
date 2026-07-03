# SDFusion + CMF Experiments

This experiment fine-tunes the unconditional SDFusion denoiser with pure CMF-style long-hop flow matching. It does not use teacher endpoint distillation; baseline 50-step sampling is used only for evaluation.

Run from the `SDFusion` directory. Replace dataset and checkpoint paths with the paths available on the experiment server.

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
