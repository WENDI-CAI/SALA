
from __future__ import annotations

from typing import Tuple

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import Tensor

def build_covalent_graph(
    bonds: list[tuple[str, str, float, float]],
    atom_labels: list[str],
) -> Tuple[Tensor, Tensor, Tensor]:
    num_atoms = len(atom_labels)
    if num_atoms == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    label_to_idx = {label: idx for idx, label in enumerate(atom_labels)}
    src_indices: list[int] = []
    dst_indices: list[int] = []
    bond_orders: list[float] = []

    for atom1, atom2, _, bond_order in bonds:
        if atom1 in label_to_idx and atom2 in label_to_idx:
            idx1 = label_to_idx[atom1]
            idx2 = label_to_idx[atom2]
            src_indices.extend([idx1, idx2])
            dst_indices.extend([idx2, idx1])
            bond_orders.extend([bond_order, bond_order])

    if src_indices:
        edge_index_conv = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr_conv = torch.tensor(bond_orders, dtype=torch.float32)
    else:
        edge_index_conv = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_conv = torch.zeros(0, dtype=torch.float32)

    component_id = compute_connected_components(edge_index_conv, num_atoms)
    return edge_index_conv, component_id, edge_attr_conv


def compute_connected_components(edge_index: Tensor, num_nodes: int) -> Tensor:
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.long)
    if edge_index.shape[1] == 0:
        return torch.arange(num_nodes, dtype=torch.long)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    data = [1] * len(src)
    adj_matrix = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    _, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    return torch.tensor(labels, dtype=torch.long)


__all__ = [
    "build_covalent_graph",
    "compute_connected_components",
]
