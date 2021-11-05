
import numpy as np


def world_to_img(coordinates_3d, intrinsics, R=None, T=None):
    if R is None and T is None:
        projected_coord = coordinates_3d @ intrinsics.T
    else:
        projected_coord = (((coordinates_3d - T.T) @ R.T) @ intrinsics.T)
    projected_coord[:, :2] = projected_coord[:, :2] / projected_coord[:, 2].reshape(-1, 1)
    return projected_coord


def img_to_world(coordinates_2dz, intrinsics, R=None, T=None):
    coordinates_2dz[:, :2] = coordinates_2dz[:, :2] * coordinates_2dz[:, 2].reshape(-1, 1)
    if R is None and T is None:
        coordinates_2dz = coordinates_2dz @ np.linalg.inv(intrinsics.T)
    else:
        coordinates_2dz = coordinates_2dz @ np.linalg.inv(intrinsics.T) @ np.linalg.inv(R.T) + T.T
    return coordinates_2dz
