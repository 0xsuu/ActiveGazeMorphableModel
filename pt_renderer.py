
from pytorch3d.renderer import rasterize_meshes, BlendParams, MeshRenderer,\
    SoftPhongShader, softmax_rgb_blend, AmbientLights
from pytorch3d.renderer.mesh.rasterizer import Fragments, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV
import torch
import torch.nn as nn

from constants import *


class ProjectedMeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes which have already been transformed.
    """

    def __init__(self, raster_settings=None):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.raster_settings = raster_settings

    def forward(self, meshes_screen, **kwargs) -> Fragments:
        """
        Args:
            meshes_screen: a Meshes object representing a batch of meshes with
                          coordinates already projected.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        return Fragments(
            pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists
        )


class NoLightShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        if blend_params is None:
            blend_params = BlendParams()

        self.blend_params = blend_params

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        pixel_colors = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images


class PTRenderer(nn.Module):
    def __init__(self, faces, faces_uvs, verts_uvs, image_size=(480, 480)):
        super().__init__()

        self.image_size = image_size

        blend_params = BlendParams(background_color=(0., 1., 0.))
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                perspective_correct=True)

        lights = AmbientLights(device=device)

        self.renderer = MeshRenderer(
            rasterizer=ProjectedMeshRasterizer(raster_settings=raster_settings).to(device),
            shader=SoftPhongShader(
                device=device,
                lights=lights,
                blend_params=blend_params)
        ).to(device)

        self.register_buffer("faces", faces)
        self.register_buffer("faces_uvs", faces_uvs)
        self.register_buffer("verts_uvs", verts_uvs)

    @staticmethod
    def project_world_to_img(vertices, camera_parameters):
        cam_R, cam_T, cam_K = camera_parameters

        cameras = PerspectiveCameras(R=cam_R, T=cam_T, K=cam_K, device=device)

        vertices_img = cameras.transform_points(vertices)

        return vertices_img, cameras

    def forward(self, vertices, textures, camera_parameters=None):
        if camera_parameters is not None:
            # # Camera configured as recommended. Not projecting to the right position though.
            # cameras2 = PerspectiveCameras(R=cam_R, T=cam_T,
            #                               principal_point=cam_K[:, :2, 2],
            #                               focal_length=torch.diagonal(cam_K, dim1=1, dim2=2)[:, :2],
            #                               device=device, image_size=(self.image_size,))

            # # Screen to NDC and NDC to screen. Not working, don't know why.
            # vertices_ndc[:, :, 0] = 1 - (vertices_ndc[:, :, 0] * 2) / (image_width - 1.0)
            # vertices_ndc[:, :, 1] = 1 - (vertices_ndc[:, :, 1] * 2) / (image_height - 1.0)
            # ndc_z = vertices_ndc[..., 2]
            # screen_x = (image_width - 1.0) / 2.0 * (1.0 - vertices_ndc[..., 0])
            # screen_y = (image_height - 1.0) / 2.0 * (1.0 - vertices_ndc[..., 1])
            # vertices = torch.stack((screen_x, screen_y, ndc_z), dim=2)

            vertices, cameras = self.project_world_to_img(vertices, camera_parameters)
            projected_vertices = vertices[:, :, :2].clone()

            # # Inplace version starts.
            # vertices[:, :, 0] -= 80
            # vertices[:, :, :2] = vertices[:, :, :2] / 480 * 2 - 1
            # vertices[:, :, :2] *= -1
            #
            # vertices[:, :, 2] *= -1
            # vertices[:, :, 2] -= vertices[:, :, 2].min() - 0.01
            # vertices[:, :, 2] /= vertices[:, :, 2].max() + 0.01
            # # Inplace version ends.

            # Non-inplace version.
            vertices_01 = torch.stack([vertices[:, :, 0] - 80, vertices[:, :, 1]], dim=2)
            vertices_01 = (vertices_01 / 240 - 1) * -1

            vertices_2 = vertices[:, :, 2] * -1
            vertices_2 = vertices_2 - (vertices_2.min() - 0.01)
            vertices_2 = vertices_2 / (vertices_2.max() + 0.01)

            vertices = torch.cat([vertices_01, vertices_2.unsqueeze(-1)], dim=2)

            # img = torch.zeros((1, 480, 480, 3), dtype=torch.uint8, device=device)
            # img[:, vertices[:, :, 0].to(torch.long), vertices[:, :, 1].to(torch.long), :] = torch.tensor([1., 1., 0.], dtype=torch.uint8, device=device)
            # img = img.transpose(1, 2)
            # return img
        else:
            vertices[:, :, 0] -= vertices[:, :, 0].min()
            vertices[:, :, 0] /= vertices[:, :, 0].max()
            vertices[:, :, 0] = vertices[:, :, 0] * 2 - 1
            vertices[:, :, 1] -= vertices[:, :, 1].min()
            vertices[:, :, 1] /= vertices[:, :, 1].max()
            vertices[:, :, 1] = vertices[:, :, 1] * 2 - 1

            # Reverse Z-axis.
            vertices[:, :, 2] *= -1
            vertices[:, :, 2] -= vertices[:, :, 2].min() - 0.01
            vertices[:, :, 2] /= vertices[:, :, 2].max() + 0.01

            cameras = None
            projected_vertices = None

        mesh = Meshes(verts=vertices, faces=self.faces.repeat(vertices.shape[0], 1, 1),
                      textures=TexturesUV(maps=textures,
                                          faces_uvs=self.faces_uvs.repeat(vertices.shape[0], 1, 1),
                                          verts_uvs=self.verts_uvs.repeat(vertices.shape[0], 1, 1)))

        rendered_image = self.renderer(meshes_world=mesh, cameras=cameras)
        # rendered_image = rendered_image.transpose(1, 2)
        return rendered_image[:, :, :, :3], projected_vertices
