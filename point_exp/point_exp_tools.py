import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from lie_tools import rodrigues


# noinspection PyCallingNonCallable
def get_pointcloud(k):
    """
    Create a point cloud of k points
    :param k: number of points in cloud
    :return: point cloud as (k, 3) tensor
    """
    x = torch.tensor(np.random.normal(0, 2, (k, 3)), dtype=torch.float32)
    return x


# noinspection PyCallingNonCallable
def get_se3(so3, r):
    """
    Create a SE3 element
    :param so3: so3 group element, (3x3 matrix)
    :param r: translation vectot, (3x1 vector)
    :return: se3 element, (4x4 matrix)
    """
    fill_shape = list(r.shape[:-2])
    filler = torch.tensor([[0., 0., 0., 1.]]).view([1] *
                                                   (len(fill_shape) + 1) + [4]).repeat(fill_shape + [1, 1])
    se3 = torch.cat([so3, r], -1).type(torch.float32)
    se3 = torch.cat([se3, filler], -2)
    return se3


def do_se3_action(se3, x):
    """
    Perform a SE3 group action on pointcloud
    :param se3: se3 element, (tuple1 x4x4 matrix)
    :param x: pointcloude, (tuple2 x 3 matrix)
    :return: rotated pointcloud, (tuple2, tuple1, 3)
    """
    tuple1 = list(se3.shape[:-2])
    tuple2 = list(x.shape[:-1])

    ones1 = [1] * len(tuple1)
    ones2 = [1] * len(tuple2)

    se3 = se3.view(tuple1 + ones2 + [4, 4])
    x_hat = torch.cat([x, torch.ones(tuple2 + [1])], -1)
    x_hat = se3 @ x_hat.view(ones1 + tuple2 + [4, 1])

    return x_hat[..., [0, 1, 2], 0]


def z_diff(z, z_var):
    """
    Finds relative difference of z vectors
    :param z: original z, (k, 3)
    :param z_var: reconstructed z, (k, 3)
    :return: absolute summed difference
    """
    d_pos = ((z - z[0]) - (z_var - z_var[0]).squeeze()).abs().mean()
    d_min = ((z - z[0]) - (-1 * (z_var - z_var[0]).squeeze())).abs().mean()

    return min(d_pos, d_min)


def print_progress(x, x_recon, x_label='original',
                   x_recon_label='recon', title='', s=100):
    """
    Plots pointclouds in 2D
    :param x: pointcloud
    :param x_recon: reconstructed pointcloud
    :param x_label: label of original, string
    :param x_recon_label: label of reconstruction, string
    :param title: title of plot, string
    :param s: size of points, int
    :return: None
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(x_recon.detach().numpy()[..., 0], x_recon.detach().numpy()[..., 1],
                c='r', alpha=0.5, s=s, label=x_recon_label)
    plt.scatter(x.numpy()[..., 0], x.numpy()[..., 1],
                c='g', alpha=0.5, s=s, label=x_label)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()
    plt.title(title)


def xy_z_decomp(xyz):
    """
    Decomposes matrix from (k, xyz) into (k, xy) and (k, z)
    :param xyz: matrix to decompose, (k, 3)
    :return: xy, z decompositions
    """
    return xyz[..., :-1], xyz[..., -1]


# noinspection PyCallingNonCallable
def create_true_data(n_points=5, n_views=1,
                     lie_group='se3', rot=True, show=False):
    """
    Create pointcloud and rotated pointclouds
    :param n_points: number of points in cloud
    :param n_views: number of rotated views
    :param lie_group: group to use, [so3, se3]
    :param rot: no rotation, i.e. only translate
    :param show: plot data
    :return: pointcloud, [rotated pointlcouds], [se3 elements used]
    """
    # generate point cloud
    cloud = get_pointcloud(n_points)

    se3_elements = []
    rotated_clouds = []
    for view in range(n_views):

        v = torch.tensor(np.random.normal(0, 2, 3)) if rot else torch.tensor(np.zeros(3)) + 1e-5

        if lie_group == 'se3':
            r = torch.tensor(np.random.normal(0, 1, (3, 1)))
        elif lie_group == 'so3':
            r = torch.tensor(np.random.normal(0, 0, (3, 1)))
        else:
            raise Exception('use either so3 or se3')

        so3 = rodrigues(v)
        se3 = get_se3(so3, r)
        rotated_cloud = do_se3_action(se3, cloud)

        se3_elements.append(se3)
        rotated_clouds.append(rotated_cloud)

        if show:
            cloud_xy, _ = xy_z_decomp(cloud)
            rotated_cloud_xy, _ = xy_z_decomp(rotated_cloud)
            print_progress(cloud_xy, rotated_cloud_xy,
                           'original', ('rotated %d' % view), title='creation')

    return cloud, rotated_clouds, se3_elements


# noinspection PyCallingNonCallable
def init_train_vars(cloud, n_views=1, z=True, rot=True, trans=True):
    """
    Create trainable params
    :param cloud: pointcloud used as original, (tuple, k, 3)
    :param n_views: number of rotated views generated
    :param z: learn depth, boolean
    :param rot: learn rotation, boolean
    :param trans: learn translation, boolean
    :return: [z inits], [v inits], [r inits], [params to optimize]
    """
    optim_params = []
    v_vars, r_vars = [], []

    # initialize z_axis
    z_var = cloud[..., -1].clone().unsqueeze(-1)
    if z:
        # noinspection PyCallingNonCallable
        z_var = torch.tensor(np.random.normal(0., 1.,
                                              (list(cloud.shape[:-1]) + [1])),
                             dtype=torch.float32,
                             requires_grad=True)
        optim_params.append(z_var)

    for view in range(n_views):
        # se3 rotation
        v_var = torch.tensor(np.zeros(3)) + 1e-5
        if rot:
            v_var = torch.tensor(np.random.normal(0, 1, 3), requires_grad=True)
            optim_params.append(v_var)
        v_vars.append(v_var)

        # se3 translation
        r_var = torch.tensor(np.random.normal(0, 0, (3, 1)))
        if trans:
            r_var = torch.tensor(np.random.normal(0, 1, (3, 1)), requires_grad=True)
            optim_params.append(r_var)
        r_vars.append(r_var)

    return z_var, v_vars, r_vars, optim_params


class DepthEstimatorModel:
    """Model to estimate rotation and depth"""

    def __init__(self, data, train_vars):
        """
        Initialize depth estimator
        :param data: [pointcloud, [rotated pointclouds], [se3 elements]]
        :param train_vars: [[z_vars], [v_vars], [r vars], [optim_params]]
        """

        self.i_trained = 0
        self.cloud, self.rotated_clouds, self.se3_elements = data
        self.z_var, self.v_vars, self.r_vars, self.optim_params = train_vars
        self.optimizer = Adam(self.optim_params)

        self.losses, self.lie_recons, self.z_recons = [], [], []

    @staticmethod
    def forward(v, r, z, xy):
        """
        Conduct forward pass
        :param v: rotation algebra vector
        :param r: translation vector
        :param z: z-axis vector
        :param xy: pointcloud xy axis
        :return: rotated approximation, se3 element
        """
        so3 = rodrigues(v)
        se3 = get_se3(so3, r)

        xyz = torch.cat([xy, z], -1)
        xyz_hat = do_se3_action(se3, xyz)

        xy_hat, z_hat = xy_z_decomp(xyz_hat)
        return xy_hat, z_hat, se3

    @staticmethod
    def loss(xy, xy_hat):
        """
        Loss function rotated view and reconstruction
        :param xy: original rotated view, (tuple2, 2)
        :param xy_hat: reconstructed rotated view, (tuple1, tupl2, 2)
        :return: squared loss
        """
        sqr_loss = (xy - xy_hat) ** 2
        sqr_loss = sqr_loss.sum(-1).mean()

        return sqr_loss

    def train(self, n_iter=10000, print_freq=500, plot_freq=-1):
        """
        Trainer function to learn group element
        :param n_iter: number of training iterations
        :param print_freq: print loss stats frequency
        :param plot_freq: plot progress graphic frequency
        :return: None
        """
        print('train model with %d points in cloud:' % self.cloud.shape[0])
        for i in range(n_iter):
            self.i_trained += 1
            self.optimizer.zero_grad()
            view_losses = 0

            if (i % print_freq) == 0:
                print('  it:%d:' % self.i_trained)

            for j, view in enumerate(self.rotated_clouds):
                cloud_xy, cloud_z = xy_z_decomp(self.cloud)
                rotated_cloud_xy, _ = xy_z_decomp(view)
                xy_hat, z_hat, se3_recon = self.forward(self.v_vars[j],
                                                        self.r_vars[j],
                                                        self.z_var, cloud_xy)
                self.lie_recons.append(se3_recon.detach().numpy())
                view_losses += self.loss(rotated_cloud_xy, xy_hat)

                if (i % print_freq) == 0:
                    print('\t view %d \t loss: %.6f \t z_diff: %.3f' %
                          (j, view_losses, z_diff(cloud_z, self.z_var)))
                if (i % plot_freq) == 0:
                    print_progress(rotated_cloud_xy, xy_hat,
                                   x_label='view %d' % j,
                                   x_recon_label='recon',
                                   title='iter: %d' % self.i_trained)
            self.losses.append(view_losses.detach().numpy())
            self.z_recons.append(self.z_var.detach().numpy())

            view_losses.backward()
            self.optimizer.step()

        self.plot_res()

    def plot_res(self):
        """
        Learned and true group element, as well as depth comparison
        :return: None
        """
        print('-' * 50)
        for i in range(len(self.rotated_clouds)):
            print('view %d' % i)
            so3_rec = rodrigues(self.v_vars[i])
            se3_rec = get_se3(so3_rec, self.r_vars[i])

            print('\nse3 analysis')
            print('true')
            print(self.se3_elements[i])
            print('rec')
            print(se3_rec)
            print('diff')
            print(self.se3_elements[i] - se3_rec)

            print('\nz analysis')
            _, cloud_z = xy_z_decomp(self.cloud)
            cloud_z_rel = (cloud_z - cloud_z[0])
            z_var_rel = (self.z_var - self.z_var[0]).squeeze()
            print('true')
            print(cloud_z_rel)
            print('rec')
            print(z_var_rel)
            print('diff')
            print(cloud_z_rel - z_var_rel)

            print('-' * 50)
