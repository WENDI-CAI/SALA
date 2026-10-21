# Training and fine-tuning

The repository includes the data loader, full model, flow objective, optimizer, EMA tracking, and checkpoint handling. Training uses Hydra configuration files under `conf/` and accepts command-line `key=value` overrides.

## Prepare the environment and data

Install a PyTorch build appropriate for your CUDA environment, then install the package from the repository root:

```bash
python -m pip install -e .
```

Use Python 3.10 or later. `requirements-validated.txt` records direct dependency versions from the release environment; it is not a complete lockfile. Retain the PyTorch build appropriate for your hardware. Optional Weights & Biases logging is available through `python -m pip install -e '.[wandb]'` and is disabled by default.

Prepare your own licensed CIF files with `sala-preprocess`. A complete preprocessing example and loader requirements are in [the data guide](data.md). Confirm that the accepted training and validation splits are both nonempty before starting a run.

No training data are distributed with the code or checkpoint.

## Train from scratch

```bash
sala-train \
  data.paths.pt_root=/absolute/path/to/sala_data \
  hydra.run.dir=outputs/train
```

The defaults select the full 486,282,617-parameter model, one CUDA GPU, float32, a per-device batch cap of one, and zero data-loader worker processes. Validation runs once per epoch. The training budget is 250,000 optimizer steps, subject to a 1,000-epoch limit. Checkpoints, TensorBoard logs, and the resolved Hydra configuration are written under the run directory.

Set the data root once if convenient:

```bash
export SALA_DATA_ROOT=/absolute/path/to/sala_data
sala-train hydra.run.dir=outputs/train
```

The default shard suffix is `_fullcell_q`. If preprocessing used another suffix, pass the same value through `data.paths.shard_suffix`.

### Memory and precision

Memory depends on the expanded full-cell atom count, batch size, optimizer state, precision, and activations. A batch size of one is a starting point, not a fixed memory guarantee. The parameter file alone occupies approximately 1.95 GB in float32; training needs substantially more memory.

Enable activation checkpointing and bfloat16 mixed precision on compatible CUDA hardware:

```bash
sala-train \
  data.paths.pt_root=/absolute/path/to/sala_data \
  model.use_gradient_checkpointing=true \
  trainer.compute.precision=bf16-mixed \
  hydra.run.dir=outputs/train-bf16
```

The loader groups structures by full-cell atom count and shortens batches to respect the configured dense-attention budget. It does not truncate the atoms in a structure. Inspect accepted atom counts before increasing `data.loader.batch_size`.

## Multiple GPUs

For eight GPUs on one machine:

```bash
sala-train \
  data.paths.pt_root=/absolute/path/to/sala_data \
  trainer.compute.devices=8 \
  trainer.compute.strategy=ddp_find_unused_parameters_false \
  trainer.compute.precision=bf16-mixed \
  data.loader.batch_size=32 \
  data.loader.num_workers=12 \
  hydra.run.dir=outputs/train-distributed
```

Tune batch and worker counts for your dataset and hardware. Batch size is a per-device cap; the atom-count budget may produce smaller batches. The data module assigns whole batches to distributed ranks. Keep `trainer.compute.use_distributed_sampler=false`, as in the supplied configuration. Unmatched final whole batches are dropped by default to keep the number of steps equal across ranks.

## Fine-tune the released weights

Download `sala_ema.ckpt` from [Hugging Face](https://huggingface.co/cWendi/SALA), then initialize a new training run:

```bash
sala-train \
  data.paths.pt_root=/absolute/path/to/sala_data \
  finetune_from=/absolute/path/to/sala_ema.ckpt \
  hydra.run.dir=outputs/finetune
```

Fine-tuning loads the model weights and starts a fresh optimizer, scheduler, and training loop. The supplied architecture and flow defaults match the released checkpoint. Keep them compatible when loading these weights; changing the model width or depth does not adapt a checkpoint automatically.

The released EMA checkpoint contains no optimizer or loop state and cannot resume the original training run. Its source step and epoch describe the provenance of its weights.

## Resume your own run

Use a full training checkpoint produced by your run, together with the corresponding training configuration:

```bash
sala-train \
  data.paths.pt_root=/absolute/path/to/sala_data \
  resume_from=/absolute/path/to/your_run/checkpoints/last.ckpt \
  hydra.run.dir=outputs/resumed
```

Use only one of `finetune_from` and `resume_from`. Full training checkpoints preserve the optimizer and loop state required for continuation. Restoring that state does not guarantee bitwise-identical execution across different hardware or software environments.

## Configuration map

| File | Controls |
| :--- | :--- |
| `conf/model/default.yaml` | Full-cell Crystal DiT architecture |
| `conf/flow/default.yaml` | Flow path, objective weights, lattice prior, and sampling defaults |
| `conf/data/default.yaml` | Local paths, filtering, batching, and augmentation |
| `conf/optim/default.yaml` | Optimizer and learning-rate schedule |
| `conf/trainer/default.yaml` | Devices, precision, training budget, logging, and EMA |
| `conf/checkpoint/default.yaml` | Checkpoint saving |

The [model guide](model.md) explains the training objective and released weight specification. Keep run outputs and resolved configurations local when they contain restricted dataset identifiers or paths.
