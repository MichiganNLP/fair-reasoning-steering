# Wait, am I Being Fair?

### Characterizing Deductive Stereotyping and Mitigating It with Fair-GCG

[![Paper](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://michigannlp.github.io/fair-reasoning-steering/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code, processed data, and result files for our paper on **reasoning-time fairness steering** in large language models.

> ⚠️ **Content warning:** this repository and the underlying benchmarks contain examples of toxic and offensive stereotypes used for the purpose of studying and mitigating bias.

<p align="center">
  <img src="docs/static/images/teaser.png" width="70%" alt="Fair Reasoning Steering overview">
</p>

## TL;DR

While reasoning generally *improves* fairness in recent LLMs, failures persist. We identify a failure mode we call **deductive stereotyping** — models apply population-level statistical regularities to individual cases, producing logically coherent yet socially biased inferences — and give it a statistical interpretation. To steer models toward fairness-aware reasoning, we introduce a **reasoning-time injection framework** and **Fair-GCG**, a GCG-style discrete search that automatically discovers effective injection phrases. The discovered phrases improve fairness across benchmarks, **generalize from smaller to larger LLMs**, improve reasoning-level fairness, reduce bias in open-ended generation, and transfer to real-world fairness-sensitive tasks.

## Authors

Naihao Deng, Yilun Zhu, Joan Nwatu, Clayton Scott, Rada Mihalcea — University of Michigan ([Language and Information Technologies / MichiganNLP](https://lit.eecs.umich.edu/)).

## Repository layout

```
fair-reasoning-steering/
├── README.md
├── LICENSE                          MIT
├── docs/                            GitHub Pages project site
├── code/
│   ├── src/
│   │   ├── data/                    download + processed-split builders for each benchmark
│   │   ├── inference/               main inference pipeline, Fair-GCG, scoring
│   │   ├── baselines/               Luo / ADBP / IASC / Gallegos SD-{E,R} implementations
│   │   ├── prm/                     fairness PRM scoring
│   │   └── constants.py             project-wide path/flag constants
│   └── scripts/                     bash launchers (data / inference / baselines / prm / slurm)
├── data/
│   └── {bbq, biobias, crowdspairs, genMO, stereoset, winoqueer}/processed/{train,val,test}/
└── results/                         every .jsonl is gzip-compressed (.jsonl.gz)
    ├── multi_choice_main/           Llama 3.1 8B + Qwen 2.5 7B, 7 conditions
    ├── larger_llms/                 Llama 3.1 70B + Qwen 2.5 72B (transfer)
    └── reasoning_llms/              GPT-OSS-20B, DeepSeek-R1-Distill-Llama-70B
```

## Quickstart

```bash
git clone https://github.com/MichiganNLP/fair-reasoning-steering.git
cd fair-reasoning-steering

# minimal environment to re-score the shipped result files
pip install torch transformers datasets accelerate tqdm scikit-learn pandas
# optional, used by the production inference scripts:
pip install "vllm==0.7.2"

export REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT/code
export HF_HOME=/path/to/your/hf-cache
```

Inspect a result file:

```bash
zcat results/multi_choice_main/llama-3.1-8b/bbq/fairgcg.jsonl.gz | head
```

The bash scripts under `code/scripts/` use the placeholders `<REPO_ROOT>` and `<HF_HOME>`; export them as environment variables or `sed`-replace them before running.

## Reproducing the result tables

Result files live under `results/<group>/<model>/<dataset>/<condition>.jsonl.gz`. To recompute accuracy from the raw model outputs:

```bash
cd code
python -m src.inference.calculate_score
```

Edit `src/constants.py` to point the per-dataset paths at the desired result file before running. The scorer applies the BBQ-ambiguous filter used in the paper (DeepSeek-R1 BBQ is scored without it, to match the reported numbers).

### Canonical converged Fair-GCG phrases

| Backbone | Phrase | Notes |
|---|---|---|
| Llama 3.1 8B | length 32, seed `Wait, let me double-check my reasoning step by step.` | phrase ID 114 in our sweep |
| Qwen 2.5 7B | length 8, seed `Wait, am I being fair?` | phrase ID 69 |
| Llama 3.1 70B / Qwen 2.5 72B | the Llama 3.1 8B converged phrase | transfer (Tab. 5) |
| GPT-OSS-20B | length 32 | optimized on GPT-OSS itself |
| DeepSeek-R1-Distill-Llama-70B | `manual` = seed `Wait, am I being fair?`; `fairgcg` = length-10 converged | — |

The Fair-GCG search itself is in `code/src/inference/gcg.py`, launched by `code/scripts/inference/gcg.sh`.

## Datasets

All datasets are publicly released; we ship only the processed splits used in our experiments. To rederive them from raw sources, see `code/src/data/download_*.py` and `code/scripts/data/process_data.sh`.

| Benchmark | Source | Used for |
|---|---|---|
| BBQ | `Elfsong/BBQ` | main tables, reasoning-LLM, appendix |
| CrowS-Pairs | github: `nyu-mll/crows-pairs` | main tables, reasoning-LLM |
| GenMO | github: `divij30bajaj/GenMO` | main tables, reasoning-LLM |
| StereoSet | github: `moinnadeem/StereoSet` | main tables, reasoning-LLM |
| WinoQueer | github: `katyfelkner/winoqueer` | main tables, reasoning-LLM |
| Bias-in-Bios | `LabHC/bias_in_bios` | real-world job-screening transfer |

## Citation

If you find this work useful, please cite:

```bibtex
@article{deng2026fair,
  title   = {Wait, am I Being Fair? Characterizing Deductive Stereotyping and Mitigating It with Fair-GCG},
  author  = {Deng, Naihao and Zhu, Yilun and Nwatu, Joan and Scott, Clayton and Mihalcea, Rada},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). The underlying benchmarks retain their original licenses.
