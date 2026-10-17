# EpiFlow: Bridging Safety and Performance in Offline Reinforcement Learning via Epigraph-Guided Flow Matching
40th Conference on Neural Information Processing Systems (NeurIPS), 2026

Manan Tayal\*†, Mumuksh Tayal\*, Ravi Prakash

\*Equal contribution &nbsp; †Work done while at the Indian Institute of Science, Bengaluru

[**[Paper]**](https://arxiv.org/pdf/2602.08054)  [**[arXiv]**](https://arxiv.org/abs/2602.08054)  [**[Project Page]**](https://tau-intelligence.github.io/EpiFlow/)

EpiFlow formulates safe offline reinforcement learning as a state-constrained optimal control problem, where safety is enforced as a hard state constraint and performance is captured by a reward. It learns an epigraph feasibility value function from a fixed dataset through a bounded Bellman recursion, and extracts a one-step flow-matching policy that maximizes return at the largest safely achievable performance level. Across Boat Navigation, Safety-Gymnasium and Bullet-Safety-Gym tasks, EpiFlow achieves competitive returns with near-zero constraint violations.

# Method
EpiFlow solves the state-constrained problem in two stages:

- Offline learning of an epigraph feasibility value function $\hat V(x, z)$ via a bounded Bellman recursion, with a decomposition regularizer that keeps it sensitive to the return threshold $z$;
- Policy synthesis via flow matching: a flow behavior teacher is distilled into a one-step actor that maximizes feasibility at the state-wise budget $z^\star(x) = \sup\{z \mid \hat V(x, z) \ge 0\}$.

## Installation
``` Bash
conda env create -f environment.yml
pip install -r requirements.txt
```

## Training
``` Bash
bash run.sh
```
which trains on the Safe Boat Navigation environment. See [run.sh](run.sh) for the commands for the Safety-Gymnasium environments.

## Citation
If you find this work useful, please cite:
``` bibtex
@inproceedings{tayal2026epiflow,
  title={Bridging Safety and Performance in Offline Reinforcement Learning via Epigraph-Guided Flow Matching},
  author={Tayal, Manan and Tayal, Mumuksh and Prakash, Ravi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
