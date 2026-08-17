#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Carry a policy forward to the next curriculum stage without carrying its prior.

A speed curriculum trains the same motion at 4.0x, then 2.0x, then 1.4x, then
1.0x, each stage warm-starting from the one before. Passing the previous
checkpoint straight to ``run.py --model_file`` does not work, and the way it
fails is silent:

``SMPAgent.save`` writes ``self.state_dict()``, which includes the prior --
100 of the 121 tensors in an M3.1 checkpoint are ``_prior_model.*``. ``load``
then calls ``load_state_dict`` with the default ``strict=True``, so every one
of those tensors is overwritten. The new stage would train against the previous
stage's prior while the environment plays the new stage's clip, and ``sds_loss``
would be scoring the policy against the wrong speed. Nothing raises; the reward
is simply measuring the wrong thing for the whole run.

Merging fixes it. Take the weights that should carry over -- actor, critic, and
the observation/action normalizers -- and drop in the prior that belongs to the
new stage:

    python tools/seed_from_stage.py \\
        --prev_ckpt output/smp_m3_dance_s4/model.pt \\
        --prior     output/smp_prior_vr_m3_1_dance_s2/model.pt \\
        --out       output/seed_dance_s2.pt

    python mimickit/run.py --arg_file args/smp_vr_m3_1_dance_s2_kaggle_args.txt \\
        --model_file output/seed_dance_s2.pt --max_samples 60000000

The prior file stores its tensors unprefixed (``dmodel.*``); inside a checkpoint
the same tensors carry a ``_prior_model.`` prefix. The key sets are otherwise
identical, and this script checks that before writing.

``_sds_normalizer`` is reset by default. It tracks the magnitude of the SDS loss
under the *old* prior, and a new prior at a different clip speed produces losses
on a different scale, so carrying the old value biases the reward until the
running average catches up. Pass ``--keep_sds_norm`` to carry it anyway.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import torch

PRIOR_PREFIX = "_prior_model."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prev_ckpt", required=True,
                    help="Checkpoint from the previous stage; its actor/critic carry over.")
    ap.add_argument("--prior", required=True,
                    help="Prior trained on THIS stage's clip (output/smp_prior_*/model.pt).")
    ap.add_argument("--out", required=True, help="Where to write the seed checkpoint.")
    ap.add_argument("--keep_sds_norm", action="store_true",
                    help="Carry _sds_normalizer over instead of resetting it.")
    args = ap.parse_args()

    for path in (args.prev_ckpt, args.prior):
        if not os.path.isfile(path):
            print("ERROR: no such file: {}".format(path), file=sys.stderr)
            return 1

    ckpt = torch.load(args.prev_ckpt, map_location="cpu", weights_only=False)
    prior = torch.load(args.prior, map_location="cpu", weights_only=False)

    ckpt_prior_keys = {k[len(PRIOR_PREFIX):] for k in ckpt if k.startswith(PRIOR_PREFIX)}
    if not ckpt_prior_keys:
        print("ERROR: {} holds no {}* tensors. Either it is not an SMP checkpoint, "
              "or this MimicKit build keeps the prior outside the agent -- in which "
              "case --model_file is already safe and this script is not needed."
              .format(args.prev_ckpt, PRIOR_PREFIX), file=sys.stderr)
        return 1

    missing = ckpt_prior_keys - set(prior)
    extra = set(prior) - ckpt_prior_keys
    if missing or extra:
        print("ERROR: prior does not match the checkpoint's prior slot.", file=sys.stderr)
        if missing:
            print("  {} key(s) the checkpoint wants and the prior lacks, e.g. {}"
                  .format(len(missing), sorted(missing)[:3]), file=sys.stderr)
        if extra:
            print("  {} key(s) the prior has and the checkpoint does not, e.g. {}"
                  .format(len(extra), sorted(extra)[:3]), file=sys.stderr)
        print("  A different num_layers or num_attention_heads in the tinymdm config "
              "produces exactly this.", file=sys.stderr)
        return 1

    swapped = 0
    shape_clash = []
    out = collections.OrderedDict()
    for key, value in ckpt.items():
        if key.startswith(PRIOR_PREFIX):
            new = prior[key[len(PRIOR_PREFIX):]]
            if hasattr(value, "shape") and tuple(new.shape) != tuple(value.shape):
                shape_clash.append((key, tuple(value.shape), tuple(new.shape)))
            out[key] = new
            swapped += 1
        elif key.startswith("_sds_normalizer.") and not args.keep_sds_norm:
            out[key] = torch.zeros_like(value) if key.endswith("_mean_abs") else value
        else:
            out[key] = value

    if shape_clash:
        print("ERROR: shape mismatch between the two priors:", file=sys.stderr)
        for key, was, now in shape_clash[:5]:
            print("  {}  checkpoint {} vs prior {}".format(key, was, now), file=sys.stderr)
        print("  The disc obs width differs, which means the two stages use different "
              "characters or a different num_disc_obs_steps -- not just a different "
              "clip speed.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)

    carried = len(out) - swapped
    print("wrote {}".format(args.out))
    print("  {} prior tensors replaced from {}".format(swapped, args.prior))
    print("  {} tensors carried over from {}".format(carried, args.prev_ckpt))
    print("  _sds_normalizer: {}".format("carried" if args.keep_sds_norm else "reset"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
