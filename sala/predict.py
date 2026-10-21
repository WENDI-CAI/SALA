"""Predict molecular crystal candidates from SMILES or a local structure file."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

MODEL_REPO = "cWendi/SALA"
MODEL_REVISION = "main"
MODEL_FILENAME = "sala_ema.ckpt"
MODEL_SHA256 = "453db9f2dc8af7047d117b156ff3e8568789fbff349f2f3063577620b290804f"
logger = logging.getLogger(__name__)


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> Path:
    """Use a local checkpoint or download and verify the published EMA state."""
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Automatic checkpoint download needs the inference dependencies. "
            "Install with: python -m pip install '.[inference]' "
            "or pass --checkpoint /path/to/sala_ema.ckpt."
        ) from error
    logger.info("Loading %s from %s (download size: 1.95 GB)", MODEL_FILENAME, MODEL_REPO)
    try:
        path = Path(hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        ))
    except Exception as error:
        hint = (
            "The published checkpoint is not in the local cache. "
            "Run once with network access or pass --checkpoint."
            if local_files_only else
            "Could not retrieve the published checkpoint. Check your network "
            "connection or download sala_ema.ckpt from the model page and pass --checkpoint."
        )
        raise RuntimeError(hint) from error
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError("Downloaded checkpoint checksum does not match this release")
    return path


def predict(
    *,
    smiles: str | None = None,
    input_path: str | Path | None = None,
    space_group: int | None = None,
    hall_number: int | None = None,
    copies: int = 1,
    checkpoint: str | Path | None = None,
    output: str | Path = "outputs/prediction",
    device: str = "auto",
    num_samples: int = 4,
    num_steps: int = 24,
    seed: int = 42,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Generate heavy-atom candidates; symmetry and ASU composition are inputs.

    SMILES is parsed locally. Only model weights are downloaded from the Hub.
    An explicit local checkpoint allows fully offline execution.
    """
    if (smiles is None) == (input_path is None):
        raise ValueError("Provide exactly one of smiles or input_path")
    if num_samples < 1 or num_steps < 1:
        raise ValueError("num_samples and num_steps must be positive")
    if input_path is not None and (space_group is not None or hall_number is not None or copies != 1):
        raise ValueError("Space-group, Hall and copies options apply to SMILES input only")
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output already exists and is not empty: {output_dir}. Choose a new directory for this run.")

    import torch
    from sala.generation.cli import (
        EvalTargetFilter,
        GenerationConfig,
        GenerationTarget,
        _seed_generation_rngs,
        generate_for_targets,
        load_checkpoint,
        load_single_target,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu or install a CUDA-enabled PyTorch build")

    # Parse and validate conditioning before any model download or allocation.
    if smiles is not None:
        from sala.inputs import prepare_smiles

        if space_group is None and hall_number is not None:
            import spglib

            if not 1 <= hall_number <= 530:
                raise ValueError("Hall number must lie in [1, 530]")
            setting = spglib.get_spacegroup_type(hall_number)
            if setting is None:
                raise ValueError(f"Unknown Hall number: {hall_number}")
            space_group = int(setting.number)
        elif space_group is None:
            space_group = 1
            logger.warning("No space group supplied; assuming P1 (1). SALA does not infer the space group from SMILES.")
        asu, provenance = prepare_smiles(
            smiles, spacegroup=space_group, hall_number=hall_number,
            copies=copies, seed=seed, target_id="smiles_target",
        )
        target = GenerationTarget(
            target_id="smiles_target", asu=asu,
            source_path=output_dir / "input.json", source_type="smiles", meta=provenance,
        )
    else:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input does not exist: {path}")
        target = load_single_target(path, filter_h=True, input_format="auto")
        provenance = {"source_type": target.source_type, "input_path": str(path), "target_id": target.target_id}
    n_full = int(target.asu.n_full_atoms.item())
    if n_full > 1997:
        raise ValueError(f"The target expands to {n_full} heavy atoms; the supported input limit is 1997")
    path = resolve_checkpoint(checkpoint, local_files_only=local_files_only)
    logger.info("Conditioning: %s heavy atoms in the full unit cell; generating %s candidates", n_full, num_samples)
    model, metadata = load_checkpoint(str(path), device=device)
    config = GenerationConfig(
        checkpoint_path=str(path), mode="single", output_dir=output_dir,
        device=device, num_samples=num_samples, batch_size=1, filter_h=True,
        sample_batch_size=1, save_cif=True, save_pt=True, save_reference=False,
        seed=seed, ema_profile_std=None, num_steps=num_steps,
        input=target.source_path, eval_filter=EvalTargetFilter(),
        eval_num_targets_before_subset=1, eval_num_targets_after_subset=1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    _seed_generation_rngs(seed)
    result = generate_for_targets(model, [target], config, metadata)
    # The existing generator writes tensors, CIFs and a run manifest. Retain
    # the input semantics in its summary so symmetry assumptions are auditable.
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["conditioning"] = provenance
    summary["candidate_status"] = "unranked_unrelaxed_heavy_atom_structures"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s candidates to %s", result["num_generated"], output_dir)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SALA crystal candidates from SMILES, CIF or PT.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles", help="SMILES describing the complete ASU composition, including counterions")
    inputs.add_argument("--input", dest="input_path", type=Path, help="Local CIF or single-target SALA PT file")
    parser.add_argument("--space-group", type=int, help="International space-group number (SMILES default: P1)")
    parser.add_argument("--hall-number", type=int, help="Explicit Hall setting; otherwise the first standard database setting is used")
    parser.add_argument("--copies", type=int, default=1, help="Copies of the entire SMILES composition in the ASU (default: 1)")
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint; otherwise download the published EMA model")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:<index>")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true", help="Use only an existing Hub cache; never download")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        predict(**vars(args))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
