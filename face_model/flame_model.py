import time

import cv2
import torch
import torch.nn as nn
import pickle
import numpy as np
from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.sphere import Sphere

from constants import *
from pt_renderer import PTRenderer


class FlameModel(nn.Module):
    def __init__(self, model_path, albedo_path, mask_path, landmark_path, masks=None):
        super().__init__()

        # Load geometry 3DMM.
        with open(model_path, "rb") as f:
            model_file = pickle.load(f, encoding="latin1")
        template_v = model_file["v_template"]
        shape_dirs = model_file["shapedirs"].x
        faces = model_file["f"]

        # template_v shape: (1, 5023, 3).
        template_v = template_v - template_v.mean(0) + np.array([0.0377, 0.0166, -0.0266])
        self.register_buffer("template_v", torch.from_numpy(template_v)
                             .to(device=device, dtype=torch.float32).unsqueeze(0))
        # shape_dirs shape: (5023, 3, 400).
        self.register_buffer("shape_dirs", torch.from_numpy(shape_dirs).to(device=device, dtype=torch.float32))
        # faces shape: (9976, 3).
        self.register_buffer("faces", torch.from_numpy(faces.astype(np.long)).to(device))

        # Load albedo 3DMM.
        albedo_file = np.load(albedo_path)
        albedo_mean = albedo_file["MU"]
        albedo_pc = albedo_file["PC"]
        albedo_ft = albedo_file["ft"]
        albedo_vt = albedo_file["vt"]
        self.register_buffer("albedo_mean", torch.from_numpy(albedo_mean).to(device, dtype=torch.float32))
        self.register_buffer("albedo_pc", torch.from_numpy(albedo_pc).to(device, dtype=torch.float32))
        self.register_buffer("albedo_ft", torch.from_numpy(albedo_ft.astype(np.long)).to(device, torch.long))
        self.register_buffer("albedo_vt", torch.from_numpy(albedo_vt).to(device, torch.float32))

        # Load landmarks.
        # with open(landmark_path, "rb") as f:
        #     landmark_file = pickle.load(f, encoding="latin1")
        # f_landmarks = torch.from_numpy(landmark_file["lmk_face_idx"].astype(np.long)).to(device, torch.long)
        # self.landmarks0 = self.faces[f_landmarks[14:19], 0]
        # self.landmarks1 = self.faces[f_landmarks[14:19], 1]
        # self.landmarks2 = self.faces[f_landmarks[14:19], 2]
        landmarks = torch.tensor([3764, 2566, 335, 3153, 3712, 673, 3863, 16, 2138, 3892,  # Eyebrows.
                                  3553, 3561, 3501, 3563,  # Nose ridge.
                                  2746, 2795, 3552, 1678, 1618,  # Nose bottom.
                                  2437, 2453, 2494, 3632, 2293, 2333,  # Left eye.
                                  3833, 1343, 1218, 1175, 955, 881],  # Right eye.
                                 dtype=torch.long, device=device)
        self.register_buffer("landmarks", landmarks)

        # Load masks.
        with open(mask_path, "rb") as f:
            masks_file = pickle.load(f, encoding="latin1")
        if masks is not None:
            total_mask = []
            for m in masks:
                self.register_buffer(m + "_mask", torch.from_numpy(masks_file[m]).to(device, torch.long))
                total_mask.append(masks_file[m])
            total_mask = np.sort(np.unique(np.concatenate(total_mask)))
            self.register_buffer("mask", torch.from_numpy(total_mask).to(device, torch.long))
            self.mask_faces(self.mask)

            masked_landmarks = []
            for lm in landmarks:
                masked_landmarks.append(torch.nonzero(self.mask == lm.item()).flatten())
            self.register_buffer("masked_landmarks", torch.cat(masked_landmarks))
        else:
            for mk in masks_file:
                self.register_buffer(mk + "_mask", torch.from_numpy(masks_file[mk]).to(device, torch.long))
            self.mask = None

    def forward(self, shape_params, albedo_params):
        vert = self.template_v + torch.einsum('bl,mkl->bmk', [shape_params, self.shape_dirs])
        tex = self.albedo_mean + torch.einsum('bl,mnkl->bmnk', [albedo_params, self.albedo_pc])

        tex = torch.clip(tex, 0., 1.)
        return vert, tex

    def mask_faces(self, mask):
        """
        Remove the masked out vertices and the related triangles and texture coordinates.

        :param mask:
        :return:
        """
        masked_faces = []
        masked_face_indices = []
        for i in range(self.faces.size(0)):
            if self.faces[i][0] in mask and self.faces[i][1] in mask and self.faces[i][2] in mask:
                masked_faces.append([torch.nonzero(mask == self.faces[i][0]).item(),
                                     torch.nonzero(mask == self.faces[i][1]).item(),
                                     torch.nonzero(mask == self.faces[i][2]).item()])
                masked_face_indices.append(i)
        masked_faces = torch.tensor(masked_faces, dtype=torch.long, device=device)
        self.register_buffer("faces", masked_faces)

        vt_mask = torch.sort(torch.unique(self.albedo_ft[masked_face_indices]))[0]
        masked_ft = []
        for i in self.albedo_ft[masked_face_indices]:
            masked_ft.append([torch.nonzero(vt_mask == i[0]).item(),
                              torch.nonzero(vt_mask == i[1]).item(),
                              torch.nonzero(vt_mask == i[2]).item()])
        masked_ft = torch.tensor(masked_ft, dtype=torch.long, device=device)
        self.register_buffer("albedo_vt", self.albedo_vt[vt_mask])
        self.register_buffer("albedo_ft", masked_ft)


def visualise_model():
    fm = FlameModel(FLAME_PATH + "FLAME2020/generic_model.pkl", FLAME_PATH + "albedoModel2020_FLAME_albedoPart.npz",
                    FLAME_PATH + "FLAME_masks/FLAME_masks.pkl", FLAME_PATH + "flame_static_embedding.pkl",
                    masks=["right_eyeball", "left_eyeball", "nose", "eye_region"])

    renderer = PTRenderer(fm.faces, fm.albedo_ft.unsqueeze(0), fm.albedo_vt.unsqueeze(0))
    mesh_viewer = MeshViewers(shape=(1, 1))
    while True:
        vert, tex = fm(torch.randn((1, 400), dtype=torch.float32, device=device),
                       torch.randn((1, 145), dtype=torch.float32, device=device))

        vert = vert - vert[:, fm.right_eyeball_mask].mean(1) + torch.tensor([[0.0061, 0.0383, -0.0026]], device=device)
        # vert = vert - vert[:, fm.right_eyeball_mask].mean(1) +
        # torch.tensor([[-0.1205, -0.0031,  0.4866]], device=device)  # Ball.

        lms = []
        # Visualise triangle landmarks for selection.
        # for i in range(fm.landmarks0.shape[0]):
        #     lms.append(Sphere(vert[0, fm.landmarks0[i]].cpu().numpy().flatten(), 0.001).to_mesh((255, 0, 0)))
        #     lms.append(Sphere(vert[0, fm.landmarks1[i]].cpu().numpy().flatten(), 0.001).to_mesh((0, 255, 0)))
        #     lms.append(Sphere(vert[0, fm.landmarks2[i]].cpu().numpy().flatten(), 0.001).to_mesh((0, 0, 255)))
        #     print(fm.landmarks0[i].item(), fm.landmarks1[i].item(), fm.landmarks2[i].item())
        # Visualise selected landmarks.
        for i in range(fm.landmarks.shape[0]):
            lms.append(Sphere(vert[0, fm.landmarks[i]].cpu().numpy().flatten(), 0.001).to_mesh())

        vert = vert[:, fm.mask]

        # Visualise using MeshViewer.
        mesh = Mesh(v=vert[0].cpu().numpy(), f=fm.faces.cpu().numpy())
        cv2.imwrite("juju.png", tex[0].cpu().numpy() * 255.)
        mesh.vt = fm.albedo_vt.cpu().numpy()
        mesh.ft = fm.albedo_ft.cpu().numpy()
        mesh.set_texture_image("juju.png")
        mesh_viewer[0][0].set_dynamic_meshes([mesh] + lms)
        mesh.write_obj("juju.obj")

        # Visualise using PT Renderer.
        # img = renderer(vert, tex)
        img = renderer(vert, tex, [torch.tensor([[[1., 0., 0.],
                                                  [0., -1., 0.],
                                                  [0., 0., -1.]]], device=device),
                                   torch.tensor([[0., 0., 1.]], device=device),
                                   torch.tensor([[[522.335571, 0.000000, 323.681061, 0.],
                                                  [0.000000, 522.346497, 269.110931, 0.],
                                                  [0.000000, 0.000000, 0.000000, 1.],
                                                  [0, 0, 1., 0.]]], device=device)])
        img = img[0].cpu().numpy()
        bg = cv2.imread("test_frame.png")
        img_coords = np.where(np.any(img != 0, axis=2))
        img_coords_shifted = (img_coords[0], img_coords[1] + 80)
        bg[img_coords_shifted] = img[img_coords] * 255.
        cv2.imshow("", bg)
        cv2.waitKey(1)
        time.sleep(0.25)


if __name__ == '__main__':
    visualise_model()
