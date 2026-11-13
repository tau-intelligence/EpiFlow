"""Evaluate trained EpiFlow policies with the protocol of the paper (Section 4).

    # 500 fixed initial states, five training seeds (mean +- std across seeds)
    python launcher/examples/evaluate_policy.py --run_dir results/BoatRobot/EpiFlow_*

    # action-perturbation study (Appendix C.2): sigma = 5%, 10%, 20% of the action bound
    python launcher/examples/evaluate_policy.py --run_dir results/BoatRobot/EpiFlow_* --action_noise 0.1
"""
import argparse
import json
import os
import sys

sys.path.append(".")

import numpy as np

from launcher.common import load_run
from jaxrl5.evaluation import evaluate_boat_fixed, evaluate_fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", nargs="+", required=True,
                        help="One or more run directories (one per training seed).")
    parser.add_argument("--checkpoint", default="_final",
                        help="Checkpoint suffix: model<checkpoint>.pickle (e.g. _final, 4).")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the evaluation noise.")
    args = parser.parse_args()

    results = []
    for run_dir in args.run_dir:
        config, env, agent = load_run(run_dir, args.checkpoint)
        if config["env_name"] == "BoatRobot":
            info = evaluate_boat_fixed(agent, env, args.episodes, args.action_noise, args.seed)
        elif config["env_name"] == "PointRobot":
            raise ValueError("The Point Robot toy task is not part of the paper's evaluation.")
        else:
            info = evaluate_fixed(agent, env, args.episodes, args.action_noise, args.seed)
        out = os.path.join(run_dir, f"eval{args.checkpoint}_noise{args.action_noise}.json")
        with open(out, "w") as f:
            json.dump(info, f, indent=2)
        print(run_dir, json.dumps(info, indent=2))
        results.append(info)

    if len(results) > 1:
        print("\nAcross runs (mean +- std):")
        for key in ("normalized_return", "normalized_cost", "return", "cost", "safety_rate"):
            vals = np.array([r[key] for r in results])
            print(f"  {key:>18s}: {vals.mean():.3f} +- {vals.std():.3f}")


if __name__ == "__main__":
    main()
