from point_exp_tools import *
import numpy as np
import argparse
import os
import datetime

import sys
sys.path.append("..")

from relie.lie_distr import SO3Prior

TEST = None
LOG_DIR = "logs/"


def run_test():
    """
    runs a test as specified in TEST
    :return:
    """
    data = create_true_data(n_points=TEST.np, n_views=TEST.nv,
                            lie_group=TEST.lie_group, rot=bool(TEST.learn_rot))
    train_vars = init_train_vars(data[0], n_views=TEST.nv, prob=bool(TEST.prob), lie_group=TEST.lie_group,
                                 z_given=bool(TEST.z_given), rot=bool(TEST.learn_rot),
                                 trans=bool(TEST.learn_trans))

    prior = None
    if bool(TEST.prob) and TEST.lie_group == 'so3':
        prior = SO3Prior()

    model = LieModel(data, train_vars, lie_group=TEST.lie_group, prior=prior, prob=bool(TEST.prob), kl_beta=TEST.beta)
    model.train(n_iter=TEST.it, print_freq=TEST.pf, plot_freq=TEST.plotf)

    os.makedirs(LOG_DIR, exist_ok=True)
    np.save(LOG_DIR + TEST.exp_name,
            np.array([model.losses, model.lie_recons, model.z_recons, vars(TEST)], dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--np', type=int, default=5,
                        help='number of pointa in cloud')
    parser.add_argument('--nv', type=int, default=1,
                        help='number of views to generate')
    parser.add_argument('--lie_group', type=str, default='se3',
                        help='type of lie element, so3 or se3 supported')
    parser.add_argument('--prob', type=int, default=1,
                        help='probabilistic version or not')
    parser.add_argument('--beta', type=float, default=1.,
                        help='scale kl divergence')
    parser.add_argument('--z_given', type=int, default=1,
                        help='estimate z, i.e. depth vector')
    parser.add_argument('--learn_rot', type=int, default=1,
                        help='learn rotation')
    parser.add_argument('--learn_trans', type=int, default=1,
                        help='learn translation')
    parser.add_argument('--it', type=int, default=10000,
                        help='number of iterations to train for')
    parser.add_argument('--pf', type=int, default=1000,
                        help='print frequency during training')
    parser.add_argument('--plotf', type=int, default=10000000,
                        help='plot frequency, default is never')
    parser.add_argument('--exp_name', type=str, default=datetime.datetime.now().isoformat(),
                        help='name of experiment')

    TEST, _ = parser.parse_known_args()

    run_test()
