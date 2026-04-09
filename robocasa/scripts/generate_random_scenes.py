#!/usr/bin/env python3
"""
Sample N random RoboCasa kitchen scenes (layout/style/objects/placement) for one task — no policy server.

Uses the same ``create_env`` path as training/eval clients. Each ``reset()`` draws a new scene when
the env uses ``hard_reset`` (default for Kitchen).

Example::

    python -m robocasa.scripts.generate_random_scenes --env-name CloseDrawer --num-scenes 10 --save-dir /tmp/scenes

"""

from __future__ import annotations

import argparse
import os

import imageio
import numpy as np

import robocasa  # noqa: F401 — registers environments
from robocasa.utils.env_utils import create_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-name",
        type=str,
        default="CloseDrawer",
        help="Registered RoboCasa task name (e.g. CloseDrawer, PnPCounterToSink).",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="How many independent resets / scenes to generate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=("pretrain", "target", "all", "none"),
        help="Dataset split: controls layout pool and obj_instance_split (same as eval clients). "
        'Use "none" for split=None.',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Env RNG seed (omit for nondeterministic runs across processes).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, save one RGB PNG per scene from robot0_agentview_center.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="robot0_agentview_center",
        help="MuJoCo camera name for saved frames.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    args = parser.parse_args()

    split = None if args.split == "none" else args.split

    env = create_env(
        env_name=args.env_name,
        split=split,
        seed=args.seed,
        render_onscreen=False,
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for i in range(args.num_scenes):
        env.reset()
        layout = getattr(env, "layout_id", "?")
        style = getattr(env, "style_id", "?")
        print(f"scene {i + 1}/{args.num_scenes}: layout_id={layout} style_id={style}")

        if args.save_dir:
            rgb = env.sim.render(
                height=args.height,
                width=args.width,
                camera_name=args.camera_name,
            )
            rgb = np.asarray(rgb)[::-1]
            path = os.path.join(args.save_dir, f"scene_{i:03d}.png")
            imageio.imwrite(path, rgb)
            print(f"  saved {path}")

    env.close()


if __name__ == "__main__":
    main()
