# -*- encoding: utf-8 -*-

import json

import numpy as np

import pbvi


def npify(o):
    if isinstance(o, list):
        return np.array(o)
    else:
        return o


def load_pomdp(pomdp_json_path):
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


pbvi_gen = load_pomdp("tiger.95.POMDP.json")


for _ in xrange(10):
    print next(pbvi_gen)
