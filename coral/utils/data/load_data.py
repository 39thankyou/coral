import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
# import xarray as xr
from scipy import io
from torch.utils.data import Dataset

# if dataset_name == "shallow-water":
#    index = 0 if data_to_encode == "height" else 1
#    x_train = x_train[..., index].unsqueeze(-1)
#    y_train = y_train[..., index].unsqueeze(-1)

#    x_test = x_test[..., index].unsqueeze(-1)
#    y_test = y_test[..., index].unsqueeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_operator_data(
    data_dir, dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.
    
    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.

    Raises:
        NotImplementedError: _description_

    Returns:
        x_train (torch.Tensor): (ntrain, ..., 1)
        y_train (torch.Tensor): (ntrain, ..., 1)
        x_test (torch.Tensor): (ntest, ..., 1)
        y_test (torch.Tensor): (ntest, ..., 1)
        grid_tr (torch.Tensor): coordinates of x_train, y_train
        grid_te (torch.Tensor): coordinates of x_test, y_test
    """

    data_dir = Path(data_dir)

    if dataset_name == "burgers":
        min_sub = 8  # 4 -> 2048  # 1024
        x_train, y_train, x_test, y_test = get_burgers(
            data_dir / "burgers_data_R10.mat", ntrain, ntest
        )
        x_train, y_train = x_train[:, ::min_sub], y_train[:, ::min_sub]
        x_test, y_test = x_test[:, ::min_sub], y_test[:, ::min_sub]

    elif dataset_name == "darcy":
        min_sub = 1
        x_train, y_train = get_darcy_pwc(
            data_dir / "piececonst_r421_N1024_smooth1.mat", ntrain
        )
        x_test, y_test = get_darcy_pwc(
            data_dir / "piececonst_r421_N1024_smooth2.mat", ntest
        )
        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[:, ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

        # x_max, x_min = x_train.max(), x_train.min()
        # x_train, x_test = (
        #    (x_train - x_min) / (x_max - x_min),
        #    (x_test - x_min) / (x_max - x_min),
        # )
        mu, sigma = x_train.mean(), x_train.std()
        x_train, x_test = (x_train - mu) / sigma, (x_test - mu) / sigma

    elif dataset_name == "kdv":
        min_sub = 8  # 8 -> 1024
        x_train, y_train, x_test, y_test = get_kdv(
            data_dir / "kdv_train_test.mat", ntrain, ntest
        )
        x_train, y_train = x_train[:, ::min_sub], y_train[:, ::min_sub]
        x_test, y_test = x_test[:, ::min_sub], y_test[:, ::min_sub]

    elif dataset_name == "navier-stokes":
        min_sub = 1
        reader = MatReader(data_dir / "ns_V1e-3_N5000_T50.mat")
        u = reader.read_field("u")

        u_train = u[:ntrain, :]
        u_test = u[-ntest:, :]

        x_train, y_train = u_train[..., 9], u_train[..., 19]  # 19 previously
        x_test, y_test = u_test[..., 9], u_test[..., 19]  # 19 previously

    elif dataset_name == "navier-stokes-256":
        min_sub = 2  # 2 usually
        # train_dir = str(data_dir) + "/" + "navier_1e-3_256_2_train.shelve"
        # test_dir = str(data_dir) + "/" + "navier_1e-3_256_2_test.shelve"

        # data_train = dict(shelve.open(train_dir))
        # data_test = dict(shelve.open(test_dir))

        shelve_dir = str(data_dir) + "/" + "ns_1e-3_1200.shelve"
        data = dict(shelve.open(shelve_dir))

        data.pop("a")
        data.pop("t")

        # concatenate dictionary to be of shape (ntrain, 40, 256, 256)

        u = torch.tensor(
            np.concatenate(
                list(
                    map(lambda key: np.expand_dims(
                        np.array(data[key]), 0), data.keys())
                )
            )
        )

        u_train = u[:1000]
        u_test = u[-200:]

        # select the 9 and 19 indices

        x_train, y_train = u_train[..., 9], u_train[..., 19]
        x_test, y_test = u_test[..., 9], u_test[..., 19]

        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[:, ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

    elif dataset_name == "advection":
        min_sub = 1
        x_train, y_train = get_advection(data_dir / "train_IC2.npz", ntrain)
        x_test, y_test = get_advection(data_dir / "test_IC2.npz", ntest)

    elif dataset_name == "pipe":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_pipe(
            data_dir, ntrain=ntrain, ntest=ntest, min_sub=min_sub
        )

    elif dataset_name == "airfoil":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_airfoil(
            data_dir, ntrain=ntrain, ntest=ntest, min_sub=min_sub
        )

    elif dataset_name == "elasticity":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_elasticity(
            data_dir, ntrain=ntrain, ntest=ntest, min_sub=min_sub
        )

    elif dataset_name == "shallow-water":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_shallow_water(
            data_dir, ntrain, ntest, min_sub=1
        )
        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[..., ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

    else:
        raise NotImplementedError

    # expects x_train, y_train, x_test, y_test to be of shape (N, dx, C) or (N, dx1, dx2, C)
    # expects grid_inp_tr, grid_out_tr, x_test, y_test to be of shape (N, dx, C) or (N, dx1, dx2, C)

    if dataset_name in ["elasticity"]:
        average_grid = x_train.mean(0)
        grid_tr = average_grid
        grid_te = average_grid

        grid_tr = einops.repeat(grid_tr, '... -> b ...', b=x_train.shape[0])
        grid_te = einops.repeat(grid_te, '... -> b ...', b=x_test.shape[0])

    elif dataset_name in ["shallow-water"]:
        grid_tr = shape2spherical_coordinates(x_train.shape[1:-1])
        grid_te = shape2spherical_coordinates(x_test.shape[1:-1])
    
    elif dataset_name in ['airfoil', 'pipe']:
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

        # create grid for the high resolution data
        grid_tr = shape2coordinates(x_train.shape[1:-1])
        grid_te = shape2coordinates(x_test.shape[1:-1])

        #if dataset_name == "pipe":
            #grid_tr[..., 0] = grid_tr[..., 0]*2 # the other way around
            #grid_te[..., 0] = grid_te[..., 0]*2
            #grid_tr[..., 1] = grid_tr[..., 1]*0.2
            #grid_te[..., 1] = grid_te[..., 1]*0.2
            #grid_tr[..., 1] = grid_tr[..., 0]*10
            #grid_te[..., 1] = grid_te[..., 0]*10

        grid_tr = einops.repeat(grid_tr, '... -> b ...', b=x_train.shape[0])
        grid_te = einops.repeat(grid_te, '... -> b ...', b=x_test.shape[0])

    else:
        x_train = x_train.unsqueeze(-1)
        x_test = x_test.unsqueeze(-1)

        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)        

        # create grid for the high resolution data
        grid_tr = shape2coordinates(x_train.shape[1:-1])
        grid_te = shape2coordinates(x_test.shape[1:-1])

    # subample data and grid, either uniformly (sub is int) or not (sub is float)

    if isinstance(sub_tr, int):
        grid_tr = operator_subsample(grid_tr, sub_tr)
        x_train = operator_subsample(x_train, sub_tr)
        y_train = operator_subsample(y_train, sub_tr)

    if isinstance(sub_te, int):
        grid_te = operator_subsample(grid_te, sub_te)
        x_test = operator_subsample(x_test, sub_te)
        y_test = operator_subsample(y_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            N = x_train.shape[0]
            C = x_train.shape[-1]
            perm = torch.randperm(x_train.reshape(N, -1, C).shape[1])
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]

            grid_tr = operator_subsample(grid_tr, mask_tr)
            x_train = operator_subsample(x_train, mask_tr)
            y_train = operator_subsample(y_train, mask_tr)
        else:
            x_train, y_train, grid_tr, perm = operator_different_subsample(
                x_train, y_train, grid_tr, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            N = x_test.shape[0]
            C = x_test.shape[-1]
            perm = torch.randperm(x_test.reshape(N, -1, C).shape[1])
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]

            grid_te = operator_subsample(grid_te, mask_te)
            x_test = operator_subsample(x_test, mask_te)
            y_test = operator_subsample(y_test, mask_te)
        else:
            x_test, y_test, grid_te, perm = operator_different_subsample(
                x_test, y_test, grid_te, sub_te
            )

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        grid_tr,
        grid_te,
    )


def get_dynamics_data(
    data_dir,
    dataset_name,
    ntrain,
    ntest,
    seq_inter_len = 20, 
    seq_extra_len = 20,
    sub_from=1,
    sub_tr=1,
    sub_te=1,
    same_grid=True,
):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, all the trajectories avec the same grids.
    
    Raises:
        NotImplementedError: _description_

    Returns:
        u_train (torch.Tensor): (ntrain, ..., T)
        u_test (torch.Tensor): (ntest, ..., T)
        grid_tr (torch.Tensor): coordinates of u_train
        grid_te (torch.Tensor): coordinates of u_test
    """

    data_dir = Path(data_dir)

    u_train_out = None
    u_test_out = None
    u_train_ext = None
    u_test_ext = None
    grid_tr_out = None
    grid_te_out = None
    grid_tr_ext = None
    grid_te_ext = None

    if dataset_name == "navier-stokes-1e-3":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-3_N5000_T50.mat")

    elif dataset_name == "navier-stokes-1e-4":
        u_train, u_test = get_navier_stokes_fno(
            data_dir / "ns_V1e-4_N10000_T30.mat")

    elif dataset_name == "navier-stokes-1e-5":
        filename = data_dir / "NavierStokes_V1e-5_N1200_T20.mat"
        if not filename.exists():
            filename = data_dir / "fno" / "NavierStokes_V1e-5_N1200_T20.mat"
        u_train_full, u_test = get_navier_stokes_fno(
            filename, ntrain=ntrain, ntest=ntest
        )
        total_seq = u_train_full.shape[-1]
        u_train, u_eval_extrapolation, u_test = split_data(
            u_train_full, u_test, seq_inter_len, seq_extra_len, total_seq
        )

    elif dataset_name == "navier-stokes-dino":
        u_train, u_eval_extrapolation, u_test = get_navier_stokes_dino(
            data_dir,
            seq_inter_len,
            seq_extra_len,
            ntrain=ntrain,
            ntest=ntest,
        )

    elif dataset_name == "shallow-water-dino":
        u_train, u_eval_extrapolation, u_test = get_shallow_water_dino(data_dir, seq_inter_len, seq_extra_len, ntrain=ntrain, ntest=ntest)

    else:
        raise NotImplementedError
    
    # u_train should be of shape (N, ..., C, T)
    if dataset_name in ["shallow-water-dino"]:
        grid_tr = shape2spherical_coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2spherical_coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2spherical_coordinates(u_test.shape[1:-2])
    
    #elif dataset_name in ['mp-pde-burgers']:
    #    grid_tr = shape2circular_coordinates(u_train.shape[1:-2])
    #    grid_te = shape2circular_coordinates(u_test.shape[1:-2])

    else:
        grid_tr = shape2coordinates(u_train.shape[1:-2])
        grid_tr_extra = shape2coordinates(u_eval_extrapolation.shape[1:-2])
        grid_te = shape2coordinates(u_test.shape[1:-2])
    if u_train_out is not None:
        grid_tr_out = shape2coordinates(u_train_out.shape[1:-2])
        grid_te_out = shape2coordinates(u_test_out.shape[1:-2])
    if u_train_ext is not None:
        grid_tr_ext = shape2coordinates(u_train_ext.shape[1:-2])
        grid_te_ext = shape2coordinates(u_test_ext.shape[1:-2])

    # grid_tr should be of shape (N, ..., input_dim)
    # we need to artificially create a time dimension for the coordinates

    grid_tr = einops.repeat(
        grid_tr, "... -> b ... t", t=u_train.shape[-1], b=u_train.shape[0]
    )
    grid_tr_extra = einops.repeat(
        grid_tr_extra, "... -> b ... t", t=u_eval_extrapolation.shape[-1], b=u_eval_extrapolation.shape[0]
    )
    grid_te = einops.repeat(
        grid_te, "... -> b ... t", t=u_test.shape[-1], b=u_test.shape[0]
    )
    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_from, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_from)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_from)

    if isinstance(sub_from, int):
        grid_te = dynamics_subsample(grid_te, sub_from)
        u_test = dynamics_subsample(u_test, sub_from)

    if isinstance(sub_from, int):
        grid_tr = dynamics_subsample(grid_tr, sub_from)
        u_train = dynamics_subsample(u_train, sub_from)

    if isinstance(sub_tr, int):
        grid_tr = dynamics_subsample(grid_tr, sub_tr)
        u_train = dynamics_subsample(u_train, sub_tr)

    if isinstance(sub_tr, int):
        grid_tr_extra = dynamics_subsample(grid_tr_extra, sub_tr)
        u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, sub_tr)

    if isinstance(sub_te, int):
        grid_te = dynamics_subsample(grid_te, sub_te)
        u_test = dynamics_subsample(u_test, sub_te)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_train, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr = dynamics_subsample(grid_tr, mask_tr)
            u_train = dynamics_subsample(u_train, mask_tr)

        else:
            print("computing different grids")
            u_train, grid_tr, perm = dynamics_different_subsample(
                u_train, grid_tr, sub_tr
            )

    if isinstance(sub_tr, float) and (sub_tr < 1):
        if same_grid:
            tmp = einops.rearrange(u_eval_extrapolation, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_tr_eval = perm[: int(sub_tr * len(perm))].clone().sort()[0]
            grid_tr_extra = dynamics_subsample(grid_tr_extra, mask_tr_eval)
            u_eval_extrapolation = dynamics_subsample(u_eval_extrapolation, mask_tr_eval)

        else:
            u_eval_extrapolation, grid_tr_extra, perm = dynamics_different_subsample(
                u_eval_extrapolation, grid_tr_extra, sub_tr
            )

    if isinstance(sub_te, float) and (sub_te < 1):
        if same_grid:
            tmp = einops.rearrange(u_test, "b ... c t -> b (...) c t")
            num_points = tmp.shape[1]
            perm = torch.randperm(num_points)
            mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]
            grid_te = dynamics_subsample(grid_te, mask_te)
            u_test = dynamics_subsample(u_test, mask_te)

        else:
            u_test, grid_te, perm = dynamics_different_subsample(
                u_test, grid_te, sub_te
            )

    return u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te


def get_kdv(filename, ntrain, ntest):
    """Get kdv data.

    Args:
        filename (str or Path): path to dataset
        ntrain (int): number of training samples
        ntest (int): number of test samples

    Returns:
        x_train (torch.Tensor): (ntrain, ..., 1)
        y_train (torch.Tensor): (ntrain, ..., 1)
        x_test (torch.Tensor): (ntest, ..., 1)
        y_test (torch.Tensor): (ntest, ..., 1)
    """
    rw_ = io.loadmat(filename)
    x_data = rw_["input"].astype(np.float32)
    y_data = rw_["output"].astype(np.float32)
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def get_advection(filename, ndata):
    """Get advection data.

    Args:
        filename (str or Path): path to dataset

    Returns:
        x (torch.Tensor)
        y (torch.Tensor)
    """
    data = np.load(filename)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # (N, nt, nx)
    N = u.shape[0]

    x = u[:, 0, :].copy()  # (N, nx)
    y = u[:, 1:, :].copy()  # (N, (nt-1), nx)

    x = x[:ndata, :]  # (ndata, nx)
    y = y[:ndata, :, :]  # (ndata, (nt-1), nx)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y


def get_darcy_pwc(filename, ndata):
    """Get darcy data.

    Args:
        filename (str or Path): path to dataset
        ndata (int): number of samples to return

    Returns:
        x (torch.Tensor): (ndata, ..., 1)
        y (torch.Tensor): (ndata, ..., 1)
    """
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29

    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, :, :].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, :, :].astype(np.float32) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    x_branch = torch.from_numpy(x_branch)
    y = torch.from_numpy(y)

    return x_branch, y


def get_burgers(filename, ntrain, ntest):
    """Get burgers data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat(filename)
    x_data = data["a"].astype(np.float32)
    y_data = data["u"].astype(np.float32)
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :].astype(np.float32)  # added
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :].astype(np.float32)  # added

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def get_pipe(filename, ntrain, ntest, min_sub=1):
    """Get pipe data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)

    filename = Path(filename)
    if not (filename / "Pipe_X.npy").exists() and (filename.parent / "pipe" / "Pipe_X.npy").exists():
        filename = filename.parent / "pipe"
    train_size, test_size = 1000, 200
    if not 1 <= ntrain <= train_size or not 1 <= ntest <= test_size:
        raise ValueError("Requested samples exceed the official Pipe split")
    xy = [np.load(filename / f"Pipe_{axis}.npy", mmap_mode="r") for axis in ("X", "Y")]
    stress = np.load(filename / "Pipe_Q.npy", mmap_mode="r")
    # Keep the author's full-training geometry normalization and fixed split.
    def geometry(section):
        channels = []
        for array in xy:
            low, high = float(array[:train_size].min()), float(array[:train_size].max())
            channels.append((torch.from_numpy(np.array(array[section], dtype=np.float32)) - low) / (high-low))
        return torch.stack(channels, dim=-1)[:, ::min_sub, ::min_sub]
    train = slice(0, ntrain)
    test = slice(train_size, train_size+ntest)
    def output(section):
        return torch.from_numpy(np.array(stress[section, 0], dtype=np.float32))[:, ::min_sub, ::min_sub]
    return geometry(train), output(train), geometry(test), output(test)


def get_airfoil(filename, ntrain, ntest, min_sub=1):
    """Get pipe data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)

    filename = Path(filename)
    INPUT_X = filename / "naca" / "NACA_Cylinder_X.npy"
    INPUT_Y = filename / "naca" / "NACA_Cylinder_Y.npy"
    OUTPUT_Sigma = filename / "naca" / "NACA_Cylinder_Q.npy"

    r1 = min_sub
    r2 = min_sub
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    # Keep the authors' fixed 1000/200 split while allowing a smaller prefix
    # for integration tests. mmap avoids materializing the 1+ GB raw arrays.
    train_size = 1000
    test_start = train_size
    inputX_raw = np.load(INPUT_X, mmap_mode="r")
    inputY_raw = np.load(INPUT_Y, mmap_mode="r")
    output_raw = np.load(OUTPUT_Sigma, mmap_mode="r")
    if ntrain > train_size or test_start + ntest > inputX_raw.shape[0]:
        raise ValueError("Requested samples exceed the official NACA split")

    xmax = float(np.max(inputX_raw[:train_size]))
    xmin = float(np.min(inputX_raw[:train_size]))
    ymax = float(np.max(inputY_raw[:train_size]))
    ymin = float(np.min(inputY_raw[:train_size]))
    inputX_train = torch.from_numpy(
        np.asarray(inputX_raw[:ntrain], dtype=np.float32).copy()
    )
    inputY_train = torch.from_numpy(
        np.asarray(inputY_raw[:ntrain], dtype=np.float32).copy()
    )
    inputX_test = torch.from_numpy(
        np.asarray(inputX_raw[test_start:test_start + ntest], dtype=np.float32).copy()
    )
    inputY_test = torch.from_numpy(
        np.asarray(inputY_raw[test_start:test_start + ntest], dtype=np.float32).copy()
    )
    output_train = torch.from_numpy(
        np.asarray(output_raw[:ntrain, 4], dtype=np.float32).copy()
    )
    output_test = torch.from_numpy(
        np.asarray(output_raw[test_start:test_start + ntest, 4], dtype=np.float32).copy()
    )

    inputX_train = (inputX_train - xmin) / (xmax - xmin)
    inputY_train = (inputY_train - ymin) / (ymax - ymin)
    inputX_test = (inputX_test - xmin) / (xmax - xmin)
    inputY_test = (inputY_test - ymin) / (ymax - ymin)
    x_train = torch.stack((inputX_train, inputY_train), dim=-1)
    x_test = torch.stack((inputX_test, inputY_test), dim=-1)
    x_train = x_train[:, ::r1, ::r2, :][:, :s1, :s2]
    x_test = x_test[:, ::r1, ::r2, :][:, :s1, :s2]
    y_train = output_train[:, ::r1, ::r2][:, :s1, :s2]
    y_test = output_test[:, ::r1, ::r2][:, :s1, :s2]

    return x_train, y_train, x_test, y_test


def get_elasticity(filename, ntrain, ntest, min_sub=1):
    filename = Path(filename)
    path_sigma = filename / "Meshes" / "Random_UnitCell_sigma_10.npy"
    path_xy = filename / "Meshes" / "Random_UnitCell_XY_10.npy"
    input_s = np.load(path_sigma, mmap_mode="r")
    input_xy = np.load(path_xy, mmap_mode="r")
    train_size, test_size = 1000, 200
    test_start = input_s.shape[1] - ntest
    if ntrain > train_size or ntest > test_size:
        raise ValueError("Requested samples exceed the official Elasticity split")

    # Statistics use the complete official training partition, even in smoke mode.
    sigma = float(np.asarray(input_s[:, :train_size], dtype=np.float32).std())
    mu = 0  # train_s.mean()

    train_s = torch.from_numpy(
        np.asarray(input_s[:, :ntrain], dtype=np.float32).T.copy()
    ).unsqueeze(-1)
    test_s = torch.from_numpy(
        np.asarray(input_s[:, test_start:], dtype=np.float32).T.copy()
    ).unsqueeze(-1)
    train_xy = torch.from_numpy(
        np.asarray(input_xy[:, :, :ntrain], dtype=np.float32).transpose(2, 0, 1).copy()
    )
    test_xy = torch.from_numpy(
        np.asarray(
            input_xy[:, :, test_start:], dtype=np.float32
        ).transpose(2, 0, 1).copy()
    )

    train_s = (train_s - mu) / sigma
    test_s = (test_s - mu) / sigma

    return train_xy, train_s, test_xy, test_s


def get_shallow_water(filename, ntrain, ntest, min_sub=1):
    path_to_file = os.path.join(
        filename, "../shallow_water/data_t0180_freq6_N1200.h5")
    rf = h5py.File(path_to_file, "r")

    # shape (N, T, long, lat)
    # shape (1200, 6, 256, 128)

    initial_time = 0
    target_time = 5

    height_scale = 3 * 1e3
    vorticity_scale = 2

    height = torch.Tensor(rf["height"][()])
    vorticity = torch.Tensor(rf["vorticity"][()])

    # permute long and lat
    # create an extra dimension
    height = (height_scale * height).permute(0, 1, 3, 2).unsqueeze(-1)
    vorticity_scale = (vorticity_scale * vorticity).permute(0,
                                                            1, 3, 2).unsqueeze(-1)

    x_train = torch.cat(
        [height[:ntrain, initial_time], vorticity_scale[:ntrain, initial_time]], axis=-1
    )
    y_train = torch.cat(
        [height[:ntrain, target_time], vorticity_scale[:ntrain, target_time]], axis=-1
    )
    x_test = torch.cat(
        [height[-ntest:, initial_time], vorticity_scale[-ntest:, initial_time]], axis=-1
    )
    y_test = torch.cat(
        [height[-ntest:, target_time], vorticity_scale[-ntest:, target_time]], axis=-1
    )

    return x_train, y_train, x_test, y_test


def get_navier_stokes_fno(filename, ntrain=1000, ntest=200, sequence_length=None, index_start=None):
    # reader = MatReader(data_dir / "NavierStokes_V1e-5_N1200_T20.mat")
    reader = MatReader(filename)
    u = reader.read_field("u")

    # u of shape (N, Dx, Dy, T)
    u_train = u[:ntrain, :]
    u_test = u[-ntest:, :]

    print('before index start', u_train.shape)
    
    if index_start is not None:
        u_train = u_train[..., index_start:]
        u_test = u_test[..., index_start:]

    print('seq length', u_train.shape)

    if sequence_length is not None:
        # this creates sub_trajectories according of size sequence_length. Increasing the number of samples.
        u_train = einops.rearrange(
            u_train, "b w h (d t)-> (b d) w h t", t=sequence_length
        )
        u_test = einops.rearrange(
            u_test, "b w h (d t)-> (b d) w h t", t=sequence_length
        )
    print('unsqueeze', u_train.shape)
    # return u_train, u_test with shape (N, Dx, Dy, 1, Seq)
    return u_train.unsqueeze(-2), u_test.unsqueeze(-2)


def get_navier_stokes_dino(
    filename,
    seq_inter_len=20,
    seq_extra_len=20,
    ntrain=256,
    ntest=16,
):
    train_path = str(filename) + "/dino/navier_1e-3_256_2_train.shelve"
    test_path = str(filename) + "/dino/navier_1e-3_256_2_test.shelve"

    def read_split(path, requested, split):
        with shelve.open(str(path), flag="r") as database:
            # GNU dbm does not guarantee insertion-order iteration. DINo uses
            # integer trajectory ids, so numeric sorting fixes sample order.
            keys = sorted(database.keys(), key=int)
            if requested > len(keys):
                raise ValueError(
                    f"Requested {requested} {split} trajectories, "
                    f"but {path} contains {len(keys)}"
                )
            arrays = [np.asarray(database[key]["data"]) for key in keys[:requested]]
        return torch.from_numpy(np.concatenate(arrays, axis=0)).float()

    # Each shelve value has shape (1, 40, 64, 64).
    u_train = read_split(train_path, ntrain, "train")
    u_test = read_split(test_path, ntest, "test")

    # if sequence_length is not None:
    #     u_train = einops.rearrange(
    #         u_train, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )
    #     u_test = einops.rearrange(
    #         u_test, "b (d t) w h -> (b d) t w h", t=sequence_length
    #     )

    u_train = einops.rearrange(u_train, "b t w h -> b w h t").unsqueeze(-2)
    u_test = einops.rearrange(u_test, "b t w h -> b w h t").unsqueeze(-2)

    # output of shape (N, Dx, Dy, 1, T)

    return split_data(u_train, u_test, seq_inter_len, seq_extra_len, 40)


def split_data(u_train, u_test, seq_inter_len, seq_extra_len, total_seq):
    
    if (seq_inter_len is not None) & (seq_extra_len is not None):
        if total_seq % (seq_inter_len+seq_extra_len) == 0:
            u_train = einops.rearrange(u_train, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
            u_test = einops.rearrange(u_test, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
        elif total_seq % (seq_inter_len+seq_extra_len) != 0:
            u_train = u_train[..., total_seq - total_seq // (seq_inter_len+seq_extra_len) * (seq_inter_len+seq_extra_len):]
            u_test = u_test[..., total_seq - total_seq // (seq_inter_len+seq_extra_len) * (seq_inter_len+seq_extra_len):]
            u_train = einops.rearrange(u_train, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
            u_test = einops.rearrange(u_test, 'b ... (d t) -> (b d) ... t', t=seq_inter_len+seq_extra_len)
    u_eval_extrapolation = u_train
    u_train = u_train[..., :seq_inter_len]
    return u_train, u_eval_extrapolation, u_test


def get_shallow_water_dino(filename, seq_inter_len=20, seq_extra_len=20, ntrain=None, ntest=None):
    """Read only requested temporal windows; counts are windows, not HDF5 trajectories."""
    window = seq_inter_len + seq_extra_len
    def read(split, count):
        name = "16_160_128_256_train" if split == "train" else "2_160_128_256_test"
        path = Path(filename) / "dino" / f"shallow_water_{name}.h5"
        with h5py.File(path, "r") as f:
            total_seq = f["height"].shape[1]
            windows = total_seq // window
            available = f["height"].shape[0] * windows
            count = available if count is None else int(count)
            if windows < 1 or not 1 <= count <= available:
                raise ValueError(f"SW {split} supports up to {available} windows")
            trajectories = (count + windows - 1) // windows
            height = torch.from_numpy(f["height"][:trajectories]).float() * 3000
            vorticity = torch.from_numpy(f["vorticity"][:trajectories]).float() * 2
        fields = torch.stack((height, vorticity), dim=-1)
        fields = einops.rearrange(fields, 'b t long lat c -> b lat long c t')
        fields = fields[..., total_seq % window:]
        fields = einops.rearrange(fields, 'b ... (d t) -> (b d) ... t', t=window)
        return fields[:count]
    train, test = read("train", ntrain), read("test", ntest)
    return train[..., :seq_inter_len], train, test


class MatReader(object):
    """Loader for navier-stokes data"""

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except BaseException:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x


def set_seed(seed=33):
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def operator_subsample(x, sub=1, missing_batch=False):
    """
    WARNING: This functions does not work for graph data.
    Subsample data and coordinates in the same way.

    Args:
        x (torch.Tensor): data to be subsampled, of shape (N, Dx, Dy, C)
        sub (int or Tensor, optional): When set to int, subsamples x as x[::sub]. When set to Tensor of indices, slices x in the 1st dim. Defaults to 1.
        missing_batch (bool, optional): Coordinates are missing batch dimension at this stage and should be aligned with data wehn set to True. Defaults to True.

    Returns:
        x (torch.Tensor): subsampled array.
    """

    if missing_batch:
        x = x.unsqueeze(0)

    if isinstance(sub, int):
        # regular slicing
        if x.ndim == 3:
            x = x[:, ::sub]
        if x.ndim == 4:
            x = x[:, ::sub, ::sub]

    if isinstance(sub, torch.Tensor):
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x[:, sub]


    if missing_batch:
        x = x.squeeze(0)
    return x


def operator_different_subsample(x, y, grid, draw_ratio):
    """
    WARNING: This functions does not work for graph data.
    Performs different subsampling per sample for operator data.
    Args:
        data (torch.Tensor): univariate time series (batch_size, num_points, num_channels)
        grid (torch.Tensor): timesteps coordinates (batch_size, num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    N = x.shape[0]
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    y = y.reshape(y.shape[0], -1, y.shape[-1])
    grid = grid.reshape(grid.shape[0], -1, grid.shape[-1])

    in_channels = x.shape[-1]
    out_channels = y.shape[-1]
    input_dim = grid.shape[-1]
    partial_grid_size = int(draw_ratio * grid.shape[1])

    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[1])[:partial_grid_size].unsqueeze(0)
        for ii in range(N)
    ]
    permutations = torch.cat(permutations, axis=0)

    small_x = torch.gather(
        x, 1, permutations.unsqueeze(-1).repeat(1, 1, in_channels))
    small_y = torch.gather(
        y, 1, permutations.unsqueeze(-1).repeat(1, 1, out_channels))
    small_grid = torch.gather(
        grid, 1, permutations.unsqueeze(-1).repeat(1, 1, input_dim)
    )

    return small_x, small_y, small_grid, permutations


def dynamics_subsample(x, sub=1, missing_batch=False):
    """
    WARNING: This functions does not work for graph data.

    Subsample data and coordinates in the same way.

    Args:
        x (torch.Tensor): data to be subsampled, of shape (N, Dx, Dy, C, T)
        sub (int or Tensor, optional): When set to int, subsamples x as x[::sub]. When set to Tensor of indices, slices x in the 1st dim. Defaults to 1.
        missing_batch (bool, optional): Coordinates are missing batch dimension at this stage and should be aligned with data wehn set to True. Defaults to True.

    Returns:
        x (torch.Tensor): subsampled array.
    """

    if missing_batch:
        x = x.unsqueeze(0)
    if isinstance(sub, int):
        # regular slicing
        if x.ndim == 4:  # 1D data (N, Dx, C, T)
            x = x[:, ::sub]
        if x.ndim == 5:  # 2D data (N, Dx, Dy, C, T)
            x = x[:, ::sub, ::sub]

    if isinstance(sub, torch.Tensor):
        x = einops.rearrange(
            x, "b ... c t -> b (...) c t"
        )  # x.reshape(x.shape[0], -1, x.shape[-1])
        x = x[:, sub]

    if missing_batch:
        x = x.squeeze(0)
    return x

def dynamics_different_subsample(u, grid, draw_ratio):
    """
    Performs subsampling for univariate time series
    Args:
        u (torch.Tensor): univariate time series (batch_size, num_points, num_channels, T)
        grid (torch.Tensor): timesteps coordinates (batch_size, num_points, input_dim)
        draw_ratio (float): draw ratio
    Returns:
        small_data: subsampled data
        small_grid: subsampled grid
        permutations: draw indexs
    """
    u = einops.rearrange(u, "b ... c t -> b (...) c t")
    grid = einops.rearrange(grid, "b ... c t -> b (...) c t")

    N = u.shape[0]
    C = u.shape[-2]
    dims = grid.shape[-2]
    T = u.shape[-1]
    input_dim = grid.shape[-2]
    partial_grid_size = int(draw_ratio * grid.shape[1])
    
    # Create draw indexes
    permutations = [
        torch.randperm(grid.shape[1])[:partial_grid_size].unsqueeze(0)
        for ii in range(N)
    ]
    permutations = torch.cat(permutations, axis=0)
    small_u = torch.gather(u, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, C, T))
    small_grid = torch.gather(grid, 1, permutations.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, dims, T))

    return small_u, small_grid, permutations


def shape2coordinates(spatial_shape, max_value=1.0):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, max_value, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def shape2circular_coordinates(spatial_shape):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, 2*np.pi, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    coords = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)

    new_coords = torch.zeros(*coords.shape[:-1], 2)
    new_coords[..., 0] = torch.sin(coords[..., 0])
    new_coords[..., 1] = torch.cos(coords[..., 0])
    return new_coords

def shape2spherical_coordinates(spatial_shape):
    """Returns spherical coordinates on a uniform latitude and longitude grid.
    Args:
        spatial_shape (tuple of int): Tuple (num_lats, num_lons) containing
            number of latitudes and longitudes in grid.
    """
    num_lats, num_lons = spatial_shape
    # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
    latitude = torch.linspace(90.0, -90.0, num_lats)
    longitude = torch.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)
    # Create a grid of latitude and longitude values (num_lats, num_lons)
    longitude_grid, latitude_grid = torch.meshgrid(
        longitude, latitude, indexing="xy")
    # Create coordinate tensor
    # Spherical coordinates have 3 dimensions
    coordinates = torch.zeros(latitude_grid.shape + (3,))
    long_rad = deg_to_rad(longitude_grid)
    lat_rad = deg_to_rad(latitude_grid)
    coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
    coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
    coordinates[..., 2] = torch.sin(lat_rad)
    return coordinates


def deg_to_rad(degrees):
    return torch.pi * degrees / 180.0


def rad_to_deg(radians):
    return 180.0 * radians / torch.pi


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.
    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates
