# QDMouse4M

Official code for **QDMouse4M: A Multi-View 3D Mouse Spontaneous Behavior Dataset with Quantum-Dot Markers** (**NeurIPS 2026 Evaluations and Datasets Track**).

Jingyang Ke, Amartya Pradhan, Xueling Zhang, Weihan Li, Anqi Wu, Jeffrey E. Markowitz

QDMouse4M is a six-view dataset of freely moving mice with over 4 million reflectance frames and physically grounded 2D and 3D keypoint labels derived from subdermal quantum-dot (QD) fluorescence markers. This repository contains the code used in the paper to prepare benchmark data, train and evaluate SLEAP and Lightning Pose 3D baselines, run Keypoint-MoSeq syllable inference, and compute stride-level gait metrics.

## Instructions

Setup, data layout, training, inference, evaluation, Keypoint-MoSeq, and gait-analysis instructions are in [docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md). Further details are in [docs/](docs/).

## Dataset

The dataset will be hosted on the DANDI Archive as [Dandiset 002024](https://dandiarchive.org/dandiset/002024).

## Citation

```bibtex
@inproceedings{ke2026qdmouse4m,
  title     = {{QDMouse4M}: A Multi-View {3D} Mouse Spontaneous Behavior Dataset with Quantum-Dot Markers},
  author    = {Ke, Jingyang and Pradhan, Amartya and Zhang, Xueling and Li, Weihan and Wu, Anqi and Markowitz, Jeffrey E.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). The QDMouse4M dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](DATA_LICENSE.md).
