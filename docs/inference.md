# Inference

SALA samples molecular crystal candidates conditioned on a chemical graph, the asymmetric-unit composition, and a crystallographic setting. Use `sala-predict` for a single SMILES, CIF, or prepared PT target. Use `sala-generate` for folder and dataset-split runs.

## Install and run

From the repository root, install the SMILES dependencies:

```bash
python -m pip install -e '.[inference]'
```

Generate four candidates for aspirin with P1 symmetry:

```bash
sala-predict \
  --smiles 'CC(=O)Oc1ccccc1C(=O)O' \
  --space-group 1 \
  --num-samples 4 \
  --num-steps 24 \
  --seed 42 \
  --device cuda \
  --output outputs/aspirin
```

Replace `cuda` with `cpu` for CPU execution. When `--device` is omitted, SALA selects CUDA if available and otherwise CPU. The full checkpoint is used on either device. CPU inference is useful when a GPU is unavailable, but repeated sampling is better suited to CUDA. Runtime and memory depend on the number of atoms in the expanded unit cell.

The default sampler uses Flow-2M with 24 steps. The default sample count is four and the seed is 42. A fixed seed helps reproduce a run within the same environment; results need not be bitwise identical across devices or software versions.

## What a SMILES input specifies

SMILES supplies the molecular graph: elements, bond orders, connected components, and formal charges. SALA constructs a heavy-atom target and samples its conformation, packing, and lattice. A reference crystal or a precomputed three-dimensional conformer is not required.

The released representation does not include a dedicated stereochemical conditioning feature. SMILES with explicit stereochemistry, such as `@`, `@@`, `/`, and `\`, are rejected. Removing an annotation changes the specified molecular target; generated candidates do not carry a guarantee of the corresponding stereochemistry. Verify stereochemistry and bond geometry in the resulting structures. Hydrogens are excluded from the released model's atom representation.

The parser also rejects isotope labels, radicals, unsupported bond types such as dative bonds, hydrogen-only components, and charge patterns outside the model's representation. Formal atomic charges must be between −2 and +2, and component charges between −16 and +16. Certain concentrated charge assignments on polyatomic components are unsupported even within those ranges. An error identifies the unsupported input rather than silently dropping its chemical features.

### Space groups and settings

`--space-group` accepts an international space-group number from 1 to 230. If omitted for SMILES, it defaults to **1 (P1)**. This is an assumption supplied to generation: SALA does not infer a space group from SMILES or choose the thermodynamically preferred polymorph.

For a precise crystallographic setting, supply its Hall number:

```bash
sala-predict \
  --smiles 'c1ccccc1' \
  --hall-number 81 \
  --output outputs/benzene-hall81 \
  --device cuda
```

Hall number 81 specifies a setting of space group 14, P2₁/c. A Hall number determines the setting and its symmetry operations. If both a space-group number and a Hall number are provided, they must describe the same space group.

Only general-position symmetry orbits are supported. Atoms on special positions, whose symmetry images coincide, are outside the current representation. The full-cell atom count equals the number of asymmetric-unit heavy atoms multiplied by the number of symmetry operations, which can make high-multiplicity groups substantially more expensive to sample. `sala-predict` accepts up to 1,997 heavy atoms after this expansion; memory may impose a lower practical limit.

### Composition and copies

Dot-separated SMILES describes all disconnected components in one asymmetric-unit composition. Write the stoichiometry explicitly, including repeated fragments and formal charges where needed:

```bash
# The example specifies a molecular ion pair, not a known crystal structure.
sala-predict \
  --smiles 'C[NH3+].[Cl-]' \
  --space-group 1 \
  --output outputs/ion-pair \
  --device cuda
```

`--copies` repeats the **entire supplied SMILES composition** in the asymmetric unit. It defaults to one. For example, `--smiles 'CCO.O' --copies 2` supplies two ethanol graphs and two water components before symmetry expansion. It does not repeat only the first fragment or set the total number of molecules in the unit cell. For a single-component molecular input, it controls the number of supplied molecular copies in the asymmetric unit.

A chemically valid input is not evidence that it lies within the released model's training distribution. Assess salts, unusual elements, very small molecules, and large compositions for your use case. [Model limitations](model.md#scope-and-limitations) gives the main representation boundaries.

## Use a CIF or prepared target

Generate from a local CIF:

```bash
sala-predict \
  --input /absolute/path/to/target.cif \
  --num-samples 4 \
  --device cuda \
  --output outputs/target
```

The same interface accepts a SALA PT target:

```bash
sala-predict \
  --input /absolute/path/to/target.pt \
  --device cuda \
  --output outputs/target
```

CIF and PT inputs carry their own graph, composition, and symmetry. The CIF parser needs explicit covalent connectivity and compatible crystallographic metadata; [data preparation](data.md#cif-inputs) describes the accepted input. Single-target inference requires one target record. Load PT files only from sources you trust because they use PyTorch serialization.

Use `--smiles` or `--input` in one invocation. SMILES composition and symmetry flags are intended for constructing a new target; preserve the conditions already encoded in CIF/PT records.

## Checkpoint download and offline use

By default, `sala-predict` retrieves `sala_ema.ckpt` from [cWendi/SALA on Hugging Face](https://huggingface.co/cWendi/SALA). The approximately 1.95 GB float32 checkpoint is cached locally and reused on subsequent runs. Allow additional memory for model loading, activations, and output buffers.

To load an existing file directly:

```bash
sala-predict \
  --smiles 'c1ccccc1' \
  --space-group 1 \
  --checkpoint /absolute/path/to/sala_ema.ckpt \
  --output outputs/local \
  --device cuda
```

To require the checkpoint to be present in the Hugging Face cache, without a network download:

```bash
sala-predict \
  --smiles 'c1ccccc1' \
  --local-files-only \
  --output outputs/offline \
  --device cpu
```

The offline command fails with a missing-cache error if the weights have not been downloaded. A local `--checkpoint` path is also suitable for offline use. `sala-generate` takes an explicit checkpoint path for advanced runs.

## Outputs

Each run writes:

| Path | Contents |
| :--- | :--- |
| `generated/<target_id>/` | Generated CIF files with fractional coordinates, cell parameters, and symmetry operations |
| `generated_results.pt` | Serialized generation results for further local analysis |
| `input.json` | SMILES and target-construction settings, or the source path for CIF/PT input, written by `sala-predict` |
| `manifest.json` | Target identifiers and output paths |
| `summary.json` | Counts and run summary |

Use a separate output directory for each run; `sala-predict` rejects a nonempty directory to avoid overwriting an earlier run. CIFs from the released model describe heavy atoms. Candidate order is sample order, not an energetic ranking. The default inference path does not perform structural relaxation or enforce molecular stereochemistry. Check bond geometry, contacts, stereochemistry, and any required energetic or experimental criteria before interpreting a candidate.

The command operates on local inputs. Generated structures and metadata may carry information derived from licensed data, so retain them under the terms that apply to those inputs.

## Python interface

Use the same prediction workflow from a script or notebook:

```python
from sala.predict import predict

result = predict(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    space_group=1,
    num_samples=4,
    num_steps=24,
    device="cuda",
    output="outputs/aspirin-python",
    seed=42,
)
```

The Python function writes the same output files as the CLI. Use `input_path="/path/to/target.cif"` instead of `smiles` for a file target; omit SMILES-specific symmetry and composition arguments in that case. `checkpoint` and `local_files_only` control weight loading in the same way as their CLI flags.

## Folder and dataset generation

The advanced CLI exposes batching and all generation settings. For a directory of CIF or PT targets:

```bash
sala-generate \
  --checkpoint /absolute/path/to/sala_ema.ckpt \
  --mode folder \
  --input-dir /absolute/path/to/targets \
  --num-samples 20 \
  --num-steps 24 \
  --batch-size 1 \
  --sample-batch-size 1 \
  --output outputs/folder \
  --device cuda \
  --no-reference \
  --seed 42
```

For the validation split of a local preprocessed dataset:

```bash
sala-generate \
  --checkpoint /absolute/path/to/sala_ema.ckpt \
  --mode eval \
  --pt-root /absolute/path/to/sala_data \
  --shard-suffix _fullcell_q \
  --split val \
  --max-targets 10 \
  --num-samples 20 \
  --batch-size 1 \
  --sample-batch-size 1 \
  --output outputs/validation \
  --device cuda \
  --no-reference
```

Input filenames and target identifiers must be unique. `--batch-size` controls the number of targets per batch; `--sample-batch-size` controls simultaneous candidates per target and defaults to one. Start with both set to one, then increase them if memory permits. `--no-reference` suppresses output of the input reference structures.

Training and advanced generation also accept Hydra `key=value` overrides. Keep a single invocation in one configuration style. Run `sala-predict --help` or `sala-generate --help` for the complete command-line reference.

## Common problems

| Symptom | Next step |
| :--- | :--- |
| RDKit is missing | Install `python -m pip install -e '.[inference]'` in the environment running `sala-predict` |
| Checkpoint cannot be downloaded | Check access to the Hugging Face repository, or pass a downloaded file with `--checkpoint` |
| Offline cache lookup fails | Download the checkpoint while online, or use an explicit local checkpoint path |
| CUDA runs out of memory | Use target and sample batch sizes of one; check the expanded full-cell atom count |
| CIF parsing or symmetry validation fails | Check bonds, cell parameters, setting information, and general-position compatibility against [the data guide](data.md) |
| Structure has poor geometry or contacts | Inspect additional independent samples and assess the model's applicability; default outputs are not relaxed or ranked |
