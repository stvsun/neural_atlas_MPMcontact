"""Analytical coordinate chart decoders for simple geometries.

These replace neural ChartDecoders for geometries with known analytical
mappings. They provide exact Jacobians, avoiding training entirely.

Each decoder maps reference coordinates xi in [-1,1]^3 to physical
coordinates x in R^3 via a known closed-form transformation.
"""

import math

import torch


class TubeSectorDecoder(torch.nn.Module):
    """Maps a cube [-1,1]^3 to a tube sector in cylindrical coordinates.

    Mapping:
        xi_0 -> theta (circumferential): theta_c + xi_0 * theta_span/2
        xi_1 -> z (axial): z_c + xi_1 * L/2
        xi_2 -> r (radial): r_mid + xi_2 * t_half

    Physical coords:
        x = r * cos(theta)
        y = r * sin(theta)
        z = z_axial

    Parameters
    ----------
    theta_center : float
        Center angle of the sector (radians).
    theta_span : float
        Angular width of the sector (radians).
    r_mid : float
        Mid-wall radius.
    t_half : float
        Half-wall thickness.
    z_center : float
        Axial center.
    L_half : float
        Half axial length.
    """

    def __init__(
        self,
        theta_center: float = 0.0,
        theta_span: float = math.pi / 1.5,  # 120 degrees
        r_mid: float = 2.925,
        t_half: float = 0.075,
        z_center: float = 2.5,
        L_half: float = 2.5,
    ):
        super().__init__()
        self.theta_center = theta_center
        self.theta_span = theta_span
        self.r_mid = r_mid
        self.t_half = t_half
        self.z_center = z_center
        self.L_half = L_half

    def forward(self, xi, **kwargs):
        """Map reference coords to physical tube sector.

        Parameters
        ----------
        xi : torch.Tensor (N, 3)
            Reference coordinates in [-1, 1]^3.

        Returns
        -------
        x : torch.Tensor (N, 3)
            Physical coordinates [x, y, z].
        """
        theta = self.theta_center + xi[:, 0] * self.theta_span / 2
        r = self.r_mid + xi[:, 2] * self.t_half
        z = self.z_center + xi[:, 1] * self.L_half

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        return torch.stack([x, y, z], dim=1)


class BoxDecoder(torch.nn.Module):
    """Maps [-1,1]^3 to an axis-aligned box [x0,x1] x [y0,y1] x [z0,z1].

    Useful for rectangular specimens (DCB, uniaxial tension rods).

    Parameters
    ----------
    center : tuple of 3 floats
        Box center.
    half_extents : tuple of 3 floats
        Half-widths in each direction.
    """

    def __init__(self, center=(0, 0, 0), half_extents=(1, 1, 1)):
        super().__init__()
        self.center = torch.tensor(center, dtype=torch.float64)
        self.half_extents = torch.tensor(half_extents, dtype=torch.float64)

    def forward(self, xi, **kwargs):
        return self.center.unsqueeze(0) + xi * self.half_extents.unsqueeze(0)


class CylinderDecoder(torch.nn.Module):
    """Maps [-1,1]^3 to a solid cylinder sector.

    xi_0 -> theta, xi_1 -> z, xi_2 -> r (from 0 to R).

    Parameters
    ----------
    theta_center, theta_span : float
        Angular center and width.
    R : float
        Outer radius.
    z_center, L_half : float
        Axial center and half-length.
    """

    def __init__(self, theta_center=0.0, theta_span=math.pi/1.5,
                 R=1.0, z_center=0.0, L_half=1.0):
        super().__init__()
        self.theta_center = theta_center
        self.theta_span = theta_span
        self.R = R
        self.z_center = z_center
        self.L_half = L_half

    def forward(self, xi, **kwargs):
        theta = self.theta_center + xi[:, 0] * self.theta_span / 2
        r = (xi[:, 2] + 1) / 2 * self.R  # [0, R]
        z = self.z_center + xi[:, 1] * self.L_half

        return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)
