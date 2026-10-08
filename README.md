# SALA — minimal model architecture

**Symmetry-Aware Lattice and Asymmetric-unit generation.** This release contains the standalone PyTorch velocity-field architecture used by SALA: explicit symmetry expansion, constrained lattice coordinates, full-cell Crystal DiT attention, and orbit pullback.

[Project website](https://wendi-cai.github.io/SALA/) · [Paper](https://arxiv.org/pdf/2609.34690) · [Architecture and input contract](ARCHITECTURE.md)

**Release scope:** model architecture, a small synthetic forward example, and focused tests. Model checkpoints, training data, preprocessing, training losses/optimizers, sampling and relaxation pipelines are not included. Randomly initialized outputs are not generated crystal structures, and this package alone does not reproduce the paper's results.

## Quick start

Python 3.11+ and PyTorch 2.13 are the tested runtime. PyTorch is the only direct runtime dependency; no symmetry database, chemistry toolkit or external attention extension is needed for explicit-operation inputs.

```bash
git clone https://github.com/WENDI-CAI/SALA.git
cd SALA
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python examples/minimal_forward.py
python -m pytest -q
```

The example runs a **small CPU configuration** with hand-written synthetic coordinates and symmetry operations. It does not download data or weights. The velocity heads are zero-initialized by design; zero outputs from an untouched model are expected.

```python
from sala import ModelConfig

model = ModelConfig.tiny().build()  # Small architecture for inspection and tests.
```

The reference configuration is `ModelConfig()`: 24 layers, 1024 channels, 16 heads, a 2816-wide SwiGLU branch, and pair bias plus self-conditioning enabled. It has **486,282,617 trainable parameters**. To inspect its size without allocating the full model's weights:

```python
import torch
from sala import ModelConfig

with torch.device("meta"):
    reference = ModelConfig().build()
print(sum(p.numel() for p in reference.parameters()))
```

Building the reference model on a real device allocates roughly 1.95 GB for FP32 parameters alone; activations and training state need additional memory. The tiny configuration has the same computation structure at a smaller width/depth, and is not a pretrained or quality-equivalent substitute.

## Included modules

| Module | Responsibility |
|---|---|
| `src/sala/models/crystal_dit/` | Covalent embedding, full-cell tokens, periodic pair bias, attention, DiT blocks and output heads |
| `src/sala/core/orbit.py` | Explicit space-group expansion, general-position checks and OrbitPullback |
| `src/sala/core/lattice_manifold.py` | Analytic constrained lattice templates and active-coordinate masks |
| `src/sala/core/types.py` | Packed `AsuBatch` tensor container |
| `src/sala/core/symmetry.py` | Explicit affine operation validation and canonicalization |
| `src/sala/config.py` | Reference and small inspection configurations |
| `examples/` and `tests/` | Synthetic inputs, forward/backward and geometry checks |

The numerical model modules retain the original import paths and parameter names. Supporting modules have been trimmed to the architecture's required operations; no hidden data or serialized geometry tables are bundled. See [ARCHITECTURE.md](ARCHITECTURE.md) for units, required metadata and the release boundary.

## 中文说明

本仓库现开放 SALA 的最小模型架构，保留 ASU 对称展开、受约束晶格、完整晶胞 Transformer、周期配对偏置、OrbitPullback 和自条件输入。`ModelConfig.tiny()` 可在 CPU 上运行合成输入示例；`ModelConfig()` 对应参考架构配置。

当前不包含检查点、真实晶体数据、完整训练/采样或结构弛豫流程。示例使用随机初始化参数，输出是速度场张量，不是已生成的晶体，也不代表论文结果复现。

## License

This architecture release is provided under the [MIT License](LICENSE). PyTorch is installed separately and retains its own license and notices.
