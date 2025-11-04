<img src="./fig4.png" width="400px"></img>

## UltraMem

Implementation of [UltraMem](https://arxiv.org/abs/2411.12364v1), improved Product Key Memory design, from Bytedance AI labs

## Install

```shell
$ pip install ultra-mem
```

## Usage

```python
import torch
from ultra_mem import UltraMem

ultra_mem = UltraMem(
    dim = 512,
    core_heads = 2,
    topk = 32,
)

tokens = torch.randn(1, 1024, 512)

out, mem_indices, aux_loss = ultra_mem(tokens) # (1, 1024, 512), (2, 1, 1024, 32), ()
```

## Char-level LM

```shell
$ uv run train.py
```

## Citations

```bibtex
@misc{huang2025ultrasparsememorynetwork,
    title   = {Ultra-Sparse Memory Network}, 
    author  = {Zihao Huang and Qiyang Min and Hongzhi Huang and Defa Zhu and Yutao Zeng and Ran Guo and Xun Zhou},
    year    = {2025},
    eprint  = {2411.12364},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2411.12364}, 
}
```

```bibtex
@inproceedings{anonymous2025continual,
    title   = {Continual Learning via Sparse Memory Finetuning},
    author  = {Anonymous},
    booktitle = {Submitted to The Fourteenth International Conference on Learning Representations},
    year    = {2025},
    url     = {https://openreview.net/forum?id=LGo7U1m24L},
    note    = {under review}
}
```
