# Data preparation and loading

No dataset or licensed example structure is distributed with this release. Obtain data under an appropriate license and keep the CIF files, shards, indices, manifests, and generated derivatives in your own storage. For inference without a reference structure, `sala-predict --smiles ...` constructs a target directly from a molecular graph and chosen symmetry; see the [inference guide](inference.md).

## CIF inputs

SALA uses CIF records to construct an asymmetric-unit chemical graph and resolve the crystallographic setting. The input needs atom labels and elements, fractional coordinates, unit-cell parameters, covalent connectivity, and sufficient space-group information to determine the Hall setting and symmetry operations. Bond orders, formula moieties, and charge annotations supply additional chemical features when available.

The current representation requires general-position symmetry orbits. Atoms whose symmetry images coincide at special positions are rejected: their stabilizer and molecular permutation information are not represented. The resolved Hall number, setting, symmetry operations, and lattice metric must agree. Disabling the optional quality filter does not bypass these representation checks.

Hydrogen atoms are removed by default before construction of the training graph. Keep that default when using the released checkpoint. `--no-filter-h` changes preprocessing to retain hydrogens and therefore changes the modeling problem.

The generation interface reads CIF or SALA PT inputs containing the chemical graph, molecular components, and crystallographic conditions. These input records also contain reference coordinates and a lattice for format validation and optional reference output. The `sala-predict` interface additionally constructs an inference target from SMILES, without requiring a reference crystal. SMILES target construction does not replace CIF-based training-data preprocessing.

## Preprocess a directory

```bash
sala-preprocess \
  --root /absolute/path/to/licensed_cifs \
  --output /absolute/path/to/sala_data \
  --shard-suffix _fullcell_q \
  --shard-size 1000 \
  --train-ratio 0.9 \
  --val-ratio 0.1 \
  --seed 42 \
  --num-workers 4 \
  --skip-bad \
  --write-jsonl
```

The input directories are scanned for `*.cif` directly inside each directory; traversal is not recursive. Supply several directories after `--root` if needed. Keep CIF identifiers and filenames unique across directories. Write each preprocessing run into its own output directory.

The output layout is:

```text
sala_data/
├── index_fullcell_q.pt
├── index_fullcell_q.jsonl
├── rejected_fullcell_q.jsonl
└── shards/
    ├── shard-00000_fullcell_q.pt
    └── ...
```

The JSONL index is optional. A rejection report is written when there are reportable rejections, unless `--no-reject-jsonl` is specified. `--skip-bad` allows preprocessing to continue after malformed or unsupported inputs; inspect the rejection report and logs to understand which records were excluded. `--max-samples` limits the number of input files considered, not the number accepted.

The split is assigned before quality filtering. The final accepted counts may therefore differ from the requested proportions. For small collections, explicitly select validation files and confirm that both training and validation contain accepted structures.

### Quality policies

`--quality high` is the default. Its standard limits include 2–100 heavy atoms in the asymmetric unit, at least four carbon atoms, and at most 500 heavy atoms in the full unit cell. It requires covalent bonds and excludes metals and symmetry-crossing covalent bonds unless the corresponding options are enabled.

`--quality inclusive_complex_v1` allows a broader set of molecular components, including metals and charged monatomic components, while retaining general-position, connectivity, and charge-consistency requirements. It does not impose the default high-policy full-cell atom cap. `--quality none` omits the optional chemical quality policy but retains parsing, schema, symmetry, and lattice validation.

Use `sala-preprocess --help` to inspect available size limits and policy options. A larger or more permissive dataset does not by itself establish compatibility with the released model's training distribution.

### Explicit validation membership

Prepare a local CSV with a `cif_filename` column listing the files intended for validation. The first column is used when that header is absent. Filenames may include or omit the `.cif` suffix.

```bash
sala-preprocess \
  --root /absolute/path/to/licensed_cifs \
  --output /absolute/path/to/sala_data \
  --val-manifest /absolute/path/to/validation_files.csv \
  --exclude-manifest /absolute/path/to/excluded_files.csv \
  --skip-bad \
  --write-jsonl
```

With `--val-manifest`, listed files are assigned to validation and all remaining files to training. `--exclude-manifest` is optional and may be repeated. Validation and exclusion manifests must not overlap. Keep these manifests private if their identifiers are subject to the source dataset's restrictions.

## Load preprocessed data

The training CLI configures the loader directly:

```bash
sala-train \
  data.paths.pt_root=/absolute/path/to/sala_data \
  data.paths.shard_suffix=_fullcell_q \
  hydra.run.dir=outputs/train
```

`data.paths.index_path=null` resolves to `index_fullcell_q.pt` under the data root. If you choose another suffix at preprocessing time, use the same suffix for training and generation. The environment variable `SALA_DATA_ROOT` supplies the default root when no explicit root override is passed.

For programmatic loading:

```python
from torch.utils.data import DataLoader

from sala.data.collate import asu_collate_fn
from sala.data.datasets.pt_dataset import AsuPtDataset

dataset = AsuPtDataset(
    root="/absolute/path/to/sala_data",
    split="train",
    shard_suffix="_fullcell_q",
    include_unknown=True,
)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=asu_collate_fn,
)
batch = next(iter(loader))
```

The training data module additionally provides atom-count bucketing, shard-local batching, filtering, augmentation, and distributed assignment. The defaults retain partial training batches and include records whose formula class is unknown. Full-cell counts above 1,997 are filtered by the default training configuration. The dense-attention budget is based on `batch_size * (max_full_cell_atoms + 3)^2`; it reduces batch size without truncating structures. Distributed training drops any unmatched final whole batches to keep rank step counts equal.

## Persisted format

The data schema identifier is `full_cell_cartesian_q_v2`; the geometry convention is `row_lattice_v2`. Lattice vectors are rows, so Cartesian coordinates are computed as fractional coordinates multiplied by the lattice matrix.

The PT index is a dictionary containing `entries` and `meta`. Each entry identifies a sample by `shard_id` and `item_idx`, records its split, and includes chemical and crystallographic metadata. Each shard is a list of dictionaries containing an `asu` object and `meta`.

An `AsuInput` records the following groups of fields:

| Content | Fields |
| --- | --- |
| Atomic and molecular identity | `atom_types`, `component_id`, `component_type_id`, `num_atoms` |
| Charge features | `atom_formal_charge`, `atom_charge_mask`, `component_formal_charge` |
| Covalent graph | `edge_index_conv`, `edge_attr_conv`; optional `hybridization` |
| Reference geometry | `frac_coords_gt`, `lattice_gt`; optional `cart_coords_gt` |
| Crystallographic conditions | `spacegroup`, `hall_number`, `setting_id`, `lattice_template_id`, `symmetry_ops` |
| Orbit counts | `n_symops`, `n_heavy_cell`, `n_full_atoms` |

Explicit symmetry operations have shape `[G, 4, 4]`. For the supported general-position representation, `n_full_atoms = num_atoms * G`. The loader checks these counts and the setting against the index; editing the index alone does not convert an incompatible shard.

For generation, a PT file may contain one `AsuInput`, a dictionary with `asu` and optional `meta`, or a dictionary of the required tensor fields. Folder mode also accepts lists of target records and dictionaries containing a `samples` list. Single mode requires exactly one target. Prefer preprocessing CIF inputs with this release instead of constructing serialized records manually.

PT files and checkpoints are loaded using PyTorch serialization. Load files from trusted sources only. Preprocessing metadata includes source file paths and chemical identifiers; generation summaries also record local input and output paths. These files are intended for local use and are not part of the public release.
