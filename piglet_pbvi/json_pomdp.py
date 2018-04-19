#!/usr/bin/env python2.7
# -*- encoding: utf-8 -*-
"""Load JSON POMDP definitions.

You can use this as a module or run it as a program. When you run it as a
program on a JSON POMDP file, it will execute ten iterations of PBVI on the
POMDP and print the results.
"""

import json
import sys

import numpy as np

import pbvi


def npify(o):
    if isinstance(o, list):
        return np.array(o)
    else:
        return o


def load_pomdp(pomdp_json_path):
    """Load POMDP definition from file and return a PBVI generator."""
    with open(pomdp_json_path, 'r') as f:
        pomdp_data = {k: npify(v) for k, v in json.load(f).items()}

    apbvi = pbvi.PBVI(pomdp_data['transition_matrix'],
                      pomdp_data['observation_matrix'],
                      pomdp_data['reward_matrix'],
                      pomdp_data['discount_factor'])



    epsi, horizon = apbvi.horizon_for_infinite()
    print "Horizon (epsilon = {}): {}".format(epsi, horizon)

    single_b0   = pomdp_data['initial_belief']
    b0          = np.vstack([single_b0, single_b0])
    V0          = np.zeros((1, apbvi.n.s))

    return pbvi.generator(apbvi, V0, b0, horizon)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage:\n    python2.7 json_pomdp.py <path to *.pomdp.json file>"
        sys.exit(1)

    pbvi_gen = load_pomdp(sys.argv[1])

    for _ in xrange(10):
        print next(pbvi_gen)
