#!/usr/bin/env python3

import torch

from constants import *


def rotation_matrix_from_axis_combined(rotations):
    return rotation_matrix_from_axis(rotations[:, 0], rotations[:, 1], rotations[:, 2])


def rotation_matrix_from_axis(azimuths, elevations, rolls):
    sin_azimuths = torch.sin(azimuths).view(-1, 1, 1)
    cos_azimuths = torch.cos(azimuths).view(-1, 1, 1)
    sin_elevations = torch.sin(elevations).view(-1, 1, 1)
    cos_elevations = torch.cos(elevations).view(-1, 1, 1)
    sin_rolls = torch.sin(rolls).view(-1, 1, 1)
    cos_rolls = torch.cos(rolls).view(-1, 1, 1)

    batch_size = azimuths.size(0)

    def zero_column():
        return torch.zeros(batch_size, 1, 1, device=device)

    def one_column():
        return torch.ones(batch_size, 1, 1, device=device)

    rot_azimuths = torch.cat((torch.cat((cos_azimuths,
                                         -sin_azimuths,
                                         zero_column()), dim=2),
                              torch.cat((sin_azimuths,
                                         cos_azimuths,
                                         zero_column()), dim=2),
                              torch.cat((zero_column(),
                                         zero_column(),
                                         one_column()), dim=2)
                              ), dim=1)
    rot_elevation = torch.cat((torch.cat((cos_elevations,
                                          zero_column(),
                                          sin_elevations), dim=2),
                               torch.cat((zero_column(),
                                          one_column(),
                                          zero_column()), dim=2),
                               torch.cat((-sin_elevations,
                                          zero_column(),
                                          cos_elevations), dim=2)
                               ), dim=1)
    rot_roll = torch.cat((torch.cat((one_column(),
                                     zero_column(),
                                     zero_column()), dim=2),
                          torch.cat((zero_column(),
                                     cos_rolls,
                                     -sin_rolls), dim=2),
                          torch.cat((zero_column(),
                                     sin_rolls,
                                     cos_rolls), dim=2)
                          ), dim=1)
    return torch.bmm(torch.bmm(rot_azimuths, rot_elevation), rot_roll)
