from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

CHEM_GROUPS: tuple[str, ...] = (
    "neutral_single_component",
    "neutral_multi_component",
    "molecular_salt",
    "metal_containing",
)
CRYSTAL_SYSTEMS: tuple[str, ...] = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
CHEM_GROUP_TO_ID = {name: idx for idx, name in enumerate(CHEM_GROUPS)}
CRYSTAL_SYSTEM_TO_ID = {name: idx for idx, name in enumerate(CRYSTAL_SYSTEMS)}
UNKNOWN_GROUP_ID = -1


def crystal_system_from_spacegroup(spacegroup: int | None) -> str:
    sg = int(spacegroup or 0)
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    raise ValueError(f"Invalid spacegroup for crystal system: {spacegroup!r}")


def chem_group_from_entry(entry: Mapping) -> str:
    if bool(entry.get("has_metal", False)):
        return "metal_containing"
    class_label = str(entry.get("class", "unknown")).lower()
    if (not bool(entry.get("is_neutral", True))) or class_label == "salt":
        return "molecular_salt"
    if int(entry.get("n_components") or 1) > 1:
        return "neutral_multi_component"
    return "neutral_single_component"


def add_strata_fields(entry: Mapping) -> dict:
    out = dict(entry)
    chem = str(out.get("chem_group") or chem_group_from_entry(out))
    system = str(out.get("crystal_system") or crystal_system_from_spacegroup(out.get("spacegroup")))
    out["chem_group"] = chem
    out["chem_group_id"] = int(CHEM_GROUP_TO_ID.get(chem, UNKNOWN_GROUP_ID))
    out["crystal_system"] = system
    out["crystal_system_id"] = int(CRYSTAL_SYSTEM_TO_ID.get(system, UNKNOWN_GROUP_ID))
    return out


def stratum_key(entry: Mapping) -> tuple[str, str]:
    e = add_strata_fields(entry)
    return str(e["chem_group"]), str(e["crystal_system"])


def allocate_validation_counts(entries: Sequence[Mapping], val_fraction: float = 0.01) -> dict[tuple[str, str], int]:
    if not entries:
        return {}
    val_fraction = float(val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")
    total_val = int(round(len(entries) * val_fraction))
    counts = Counter(stratum_key(entry) for entry in entries)
    allocations: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    assigned = 0
    for key, count in counts.items():
        exact = count * val_fraction
        base = int(math.floor(exact))
        allocations[key] = base
        assigned += base
        remainders.append((exact - base, key))
    for _frac, key in sorted(remainders, reverse=True)[: max(0, total_val - assigned)]:
        allocations[key] += 1
    return allocations


def stratified_train_val_split(
    entries: Sequence[Mapping],
    *,
    val_fraction: float = 0.01,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    enriched = [add_strata_fields(entry) for entry in entries]
    allocations = allocate_validation_counts(enriched, val_fraction=val_fraction)
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in enriched:
        by_stratum[stratum_key(entry)].append(entry)
    rng = random.Random(int(seed))
    train_entries: list[dict] = []
    val_entries: list[dict] = []
    for key in sorted(by_stratum):
        group = list(by_stratum[key])
        rng.shuffle(group)
        n_val = int(allocations.get(key, 0))
        val_entries.extend(group[:n_val])
        train_entries.extend(group[n_val:])
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    out: list[dict] = []
    for entry in train_entries:
        item = dict(entry)
        item["split"] = "train"
        out.append(item)
    for entry in val_entries:
        item = dict(entry)
        item["split"] = "val"
        out.append(item)
    summary = summarize_strata(out)
    summary["val_fraction"] = float(val_fraction)
    summary["seed"] = int(seed)
    summary["num_train"] = len(train_entries)
    summary["num_val"] = len(val_entries)
    summary["num_total"] = len(out)
    return out, summary


def summarize_strata(entries: Iterable[Mapping]) -> dict:
    entries = [add_strata_fields(entry) for entry in entries]
    by_split: dict[str, dict] = {}
    for split in sorted({str(entry.get("split", "unknown")) for entry in entries}):
        split_entries = [entry for entry in entries if str(entry.get("split", "unknown")) == split]
        by_split[split] = {
            "chem_group": dict(Counter(str(entry["chem_group"]) for entry in split_entries)),
            "crystal_system": dict(Counter(str(entry["crystal_system"]) for entry in split_entries)),
            "cross_stratum": {
                f"{chem}__{system}": count
                for (chem, system), count in Counter(stratum_key(entry) for entry in split_entries).items()
            },
        }
    return {
        "split": dict(Counter(str(entry.get("split", "unknown")) for entry in entries)),
        "chem_group": dict(Counter(str(entry["chem_group"]) for entry in entries)),
        "crystal_system": dict(Counter(str(entry["crystal_system"]) for entry in entries)),
        "cross_stratum": {
            f"{chem}__{system}": count
            for (chem, system), count in Counter(stratum_key(entry) for entry in entries).items()
        },
        "by_split": by_split,
    }


__all__ = [
    "CHEM_GROUPS",
    "CHEM_GROUP_TO_ID",
    "CRYSTAL_SYSTEMS",
    "CRYSTAL_SYSTEM_TO_ID",
    "UNKNOWN_GROUP_ID",
    "add_strata_fields",
    "allocate_validation_counts",
    "chem_group_from_entry",
    "crystal_system_from_spacegroup",
    "stratified_train_val_split",
    "stratum_key",
    "summarize_strata",
]
