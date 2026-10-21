# Model and checkpoint

SALA models molecular crystal structures with a velocity field over canonical Cartesian atom coordinates and symmetry-constrained lattice variables. The asymmetric unit supplies the evolving coordinates; explicit crystallographic operations expand it into full-cell tokens for the Crystal DiT. Full-cell atoms are symmetry-related views of that state rather than independently generated coordinates.

## Architecture

| Property | Released model |
| --- | --- |
| Architecture | Full-cell Crystal DiT rectified flow |
| Trainable parameters | 486,282,617 |
| Transformer layers | 24 |
| Hidden width | 1,024 |
| Attention heads | 16 |
| SwiGLU branch width | 2,816 |
| Pair feature width | 128 |
| Covalent message-passing layers | 2 |
| Attention backend | PyTorch scaled dot-product attention |
| Coordinate normalization scale | 8.0 |
| Lattice representation | Constrained `q`, whitened for the model |
| Self-conditioning | Enabled |
| Default sampler | Flow-2M, 24 steps, uniform time grid |

Chemical conditioning includes element types, covalent connectivity, molecular component identities, charge features, and space-group settings. The model uses a shared pair representation for geometric and topological attention biases. The data and model require explicit symmetry operations and general-position orbits.

The CLI obtains these conditions from SMILES with a chosen space group or Hall setting, or from CIF and SALA PT records. Hydrogen atoms are removed by default. SMILES target construction supplies connectivity, composition, and charge features without requiring a reference crystal. It does not enumerate missing stoichiometries or predict a space group. Stereochemical annotations are not a conditioning feature of the released model; SMILES containing explicit stereochemistry are rejected.

## Training objective

The flow follows a linear path from data at time 0 to noise at time 1. Training samples time from a logit-normal distribution with mean 0 and standard deviation 1. The model predicts velocity in normalized Cartesian and whitened lattice coordinates.

The released configuration uses:

- Coordinate velocity loss over the full symmetry orbit, with weight 1.0.
- Lattice velocity loss, with weight 1.0 and normalization by lattice degrees of freedom.
- Contact/clash loss, with weight 0.1, active up to time 0.4, using the input lattice.
- Zero coordinate and lattice endpoint-loss weights.

The contact/clash term combines a target-capped intermolecular distance floor with contact-distance matching. Its distance floor uses a 1 Å offset from the van der Waals radius sum; contact matching uses a 5 Å cutoff and a 0.5 Å Huber threshold. These are training objectives, not guarantees that every generated structure is physically valid.

Self-conditioning uses a detached preliminary clean-state prediction for a randomly selected half of the training graphs. During sampling, the previous clean-state prediction is reused. Flow-2M starts with an Euler step and then uses a second-order Adams–Bashforth update.

## Released weights

The checkpoint is `sala_ema.ckpt`, distributed through [cWendi/SALA on Hugging Face](https://huggingface.co/cWendi/SALA). The `sala-predict` command downloads and caches it automatically unless a local checkpoint is supplied. The source repository includes checkpoint metadata in `checkpoints/model.json`, but does not contain the weight file.

| Property | Value |
| --- | --- |
| Model family | `sala_full_cell_crystal_dit_rectified_flow` |
| Checkpoint schema | 16 |
| Source training step | 249,000 |
| Source epoch | 175 |
| Weight selection | Power-function EMA |
| EMA profile relative standard deviation | 0.05 |
| Floating-point tensor precision | float32 |
| Quantization | None |
| Architecture and flow configuration | Embedded in checkpoint |
| Optimizer, scheduler, and training-loop state | Excluded |
| Dataset paths and data-module state | Excluded |

The selected EMA values are stored directly in `state_dict`. No EMA-profile selection is needed when loading this release. Model buffers needed for inference are retained. The source step and epoch identify the provenance of the weights and do not make the file a resumable training snapshot.

Generation reconstructs the architecture and flow configuration from the checkpoint and loads its state strictly. Changing `conf/model/default.yaml` does not resize the released model during generation. `--num-steps` can override the sampler step count.

Fine-tuning through `finetune_from=/absolute/path/to/sala_ema.ckpt` loads these weights and starts fresh optimizer, scheduler, and loop state. Architecture and flow settings must remain compatible with the checkpoint; the supplied defaults match it. Exact state continuation requires a full training checkpoint produced by your own run and the matching configuration, supplied through `resume_from`. See [training and fine-tuning](training.md) for complete commands.

## Runtime considerations

The distributed training configuration can use bfloat16 mixed precision, while the default single-device configuration uses float32. Checkpoint storage precision and runtime mixed precision are separate settings. The released model is the full architecture in both cases.

Dense attention scales with the square of the expanded full-cell token count. Smaller target batches, smaller sample batches, and gradient checkpointing during training can reduce memory demand. CPU execution is available for loading and generation; CUDA is recommended for training and repeated sampling at this model size.

The generator writes fractional coordinates and cell parameters to CIF files, with crystallographic operations derived from the resolved setting. Optional symmetry-preserving rigid postprocessing and hydrogen placement are separate from the flow sampler and are disabled by default. Hydrogen placement requires sufficient chemical evidence in the input record.

## Scope and limitations

The released model operates on heavy atoms in general-position orbits, conditioned on a supplied chemical composition and crystallographic setting. Special positions are unsupported. Explicit stereochemical SMILES are rejected, and generated candidates must be checked for any required stereochemistry. The model does not predict the space group, discover stoichiometry, or assign an energetic ranking to generated candidates.

The release does not include a benchmark dataset or establish performance on a user's data. A successful parse or generation run does not establish physical validity. Evaluate generated structures for the intended scientific use, including chemical geometry, stereochemistry, contacts, and any required energetic or experimental criteria. Relaxation and stability assessment require a separate workflow.

## License

The code is released under the [MIT License](../LICENSE). Checkpoint terms are documented separately on the [Hugging Face model card](https://huggingface.co/cWendi/SALA); the code license does not specify checkpoint terms. The data used with SALA are subject to their own licenses.
