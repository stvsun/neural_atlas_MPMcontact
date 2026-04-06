"""Analytical coordinate chart decoders for simple geometries.

These replace neural ChartDecoders for geometries with known analytical
mappings. They provide exact Jacobians, avoiding training entirely.

Each decoder maps reference coordinates xi in [-1,1]^3 to physical
coordinates x in R^3 via a known closed-form transformation.

Includes CrackTipDecoder which absorbs the 1/sqrt(r) stress singularity
into the coordinate mapping via radial squaring (r_phys ~ xi^2).
"""

import math
from typing import Optional

import numpy as np
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

    def inverse(self, x_phys):
        """Closed-form inverse: physical (x,y,z) -> reference (xi0,xi1,xi2).

        Parameters
        ----------
        x_phys : torch.Tensor (N, 3)

        Returns
        -------
        xi : torch.Tensor (N, 3)
        """
        x, y, z = x_phys[:, 0], x_phys[:, 1], x_phys[:, 2]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        xi0 = (theta - self.theta_center) / (self.theta_span / 2)
        xi1 = (z - self.z_center) / self.L_half
        xi2 = (r - self.r_mid) / self.t_half

        return torch.stack([xi0, xi1, xi2], dim=1)


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

    def inverse(self, x, **kwargs):
        """Map physical x -> reference xi: xi = (x - center) / half_extents."""
        c = self.center.to(x.device, x.dtype).unsqueeze(0)
        he = self.half_extents.to(x.device, x.dtype).unsqueeze(0)
        return (x - c) / he

    def jacobian(self, xi, **kwargs):
        """Constant Jacobian: diag(half_extents)."""
        n = xi.shape[0]
        J = torch.zeros(n, 3, 3, device=xi.device, dtype=xi.dtype)
        he = self.half_extents.to(xi.device, xi.dtype)
        J[:, 0, 0] = he[0]
        J[:, 1, 1] = he[1]
        J[:, 2, 2] = he[2]
        return J


class GradedBoxDecoder(torch.nn.Module):
    """Maps [-1,1]^3 to a box with element concentration near the center.

    Like BoxDecoder, but applies a power-law grading along one axis so that
    elements are packed near center[grade_axis] (e.g., the crack tip) and
    stretched in the far-field. This gives h-refinement exactly where it's
    needed without multi-chart overhead.

    Forward mapping (graded axis k):
        x_k = center_k + sign(xi_k) * |xi_k|^grade_power * half_extents_k

    Other axes are linear (same as BoxDecoder).

    Element size ratio (center vs boundary):
        h_center / h_boundary = 1 / grade_power   (at xi=0 vs xi=±1)
        For grade_power=2: 2× finer at center
        For grade_power=3: 3× finer at center

    Parameters
    ----------
    center : tuple of 3 floats
        Box center. The graded axis concentrates elements at this point.
    half_extents : tuple of 3 floats
        Half-widths in each direction.
    grade_axis : int (0, 1, or 2)
        Which axis to grade (0=x, 1=y, 2=z).
    grade_power : float
        Grading exponent. 1.0 = uniform (BoxDecoder), 2.0 = quadratic, etc.
    """

    def __init__(self, center=(0, 0, 0), half_extents=(1, 1, 1),
                 grade_axis=1, grade_power=2.0):
        super().__init__()
        self.center = torch.tensor(center, dtype=torch.float64)
        self.half_extents = torch.tensor(half_extents, dtype=torch.float64)
        self.grade_axis = grade_axis
        self.grade_power = grade_power

    def forward(self, xi, **kwargs):
        c = self.center.to(xi.device, xi.dtype).unsqueeze(0)
        he = self.half_extents.to(xi.device, xi.dtype).unsqueeze(0)
        x = c + xi * he  # start with linear mapping

        # Apply power-law grading on the specified axis
        k = self.grade_axis
        p = self.grade_power
        xi_k = xi[:, k]
        # sign-preserving power: f(xi) = sign(xi) * |xi|^p
        x_k = c[0, k] + torch.sign(xi_k) * torch.abs(xi_k).pow(p) * he[0, k]
        x = x.clone()
        x[:, k] = x_k
        return x

    def inverse(self, x, **kwargs):
        c = self.center.to(x.device, x.dtype).unsqueeze(0)
        he = self.half_extents.to(x.device, x.dtype).unsqueeze(0)
        xi = (x - c) / he  # start with linear inverse

        # Invert the power-law on graded axis: xi = sign(dx) * |dx/he|^(1/p)
        k = self.grade_axis
        p = self.grade_power
        dx_k = x[:, k] - c[0, k]
        normalized = dx_k / he[0, k]
        xi_k = torch.sign(normalized) * torch.abs(normalized).pow(1.0 / p)
        xi = xi.clone()
        xi[:, k] = xi_k
        return xi

    def jacobian(self, xi, **kwargs):
        n = xi.shape[0]
        he = self.half_extents.to(xi.device, xi.dtype)
        J = torch.zeros(n, 3, 3, device=xi.device, dtype=xi.dtype)

        # Linear axes
        for d in range(3):
            J[:, d, d] = he[d]

        # Graded axis: dx/dxi = he * p * |xi|^(p-1)
        k = self.grade_axis
        p = self.grade_power
        xi_k = xi[:, k]
        # Clamp |xi| away from 0 to avoid zero Jacobian at the center
        abs_xi = torch.abs(xi_k).clamp(min=1e-6)
        J[:, k, k] = he[k] * p * abs_xi.pow(p - 1)
        return J


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

    def inverse(self, x_phys):
        """Closed-form inverse: physical (x,y,z) -> reference (xi0,xi1,xi2)."""
        x, y, z = x_phys[:, 0], x_phys[:, 1], x_phys[:, 2]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        xi0 = (theta - self.theta_center) / (self.theta_span / 2)
        xi1 = (z - self.z_center) / self.L_half
        xi2 = 2 * r / self.R - 1  # r = (xi2+1)/2 * R => xi2 = 2r/R - 1

        return torch.stack([xi0, xi1, xi2], dim=1)


class CrackTipDecoder(torch.nn.Module):
    """Maps [-1,1]^3 to a crack-tip region with radial coordinate squaring.

    Absorbs the 1/sqrt(r) stress singularity into the coordinate mapping
    so that standard P1 elements capture it naturally without enrichment.

    Coordinate layout:
        xi_0: along crack front tangent (tangent1 direction)
        xi_1: along crack face in-plane (tangent2 direction)
        xi_2: radial from crack tip (mapped via power law)

    Radial mapping:
        r_phys = radius * ((xi_2 + 1) / 2) ^ power

    With power=2 (default):
        - Physical spacing concentrates near the tip (dr/dxi -> 0 at xi=-1)
        - sqrt(r_phys) = sqrt(radius) * (xi_2+1)/2, which is LINEAR in xi
        - Williams displacement u ~ sqrt(r) becomes smooth in xi-space
        - P1 elements capture the singularity without h-refinement

    Works with all benchmark geometries — constructed from crack tip
    position + crack frame (normal, tangent1, tangent2) + support radius.

    Parameters
    ----------
    center : array-like (3,)
        Crack tip position in physical space.
    normal : array-like (3,)
        Crack opening direction (unit normal to crack plane).
    tangent1 : array-like (3,)
        In-plane direction along crack front.
    tangent2 : array-like (3,)
        In-plane direction perpendicular to crack front.
    radius : float
        Support radius of the chart.
    power : float
        Radial squaring exponent (default 2.0 for sqrt(r) singularity).
        Use power=3 for r^(1/3) singularities, etc.
    in_plane_scale : float
        Scale factor for in-plane (tangent) directions. Default = radius.
    """

    def __init__(
        self,
        center=(0, 0, 0),
        normal=(1, 0, 0),
        tangent1=(0, 1, 0),
        tangent2=(0, 0, 1),
        radius: float = 0.5,
        power: float = 2.0,
        in_plane_scale: Optional[float] = None,
    ):
        super().__init__()
        self.center_np = np.asarray(center, dtype=np.float64)
        self.normal_np = np.asarray(normal, dtype=np.float64)
        self.normal_np = self.normal_np / np.linalg.norm(self.normal_np)
        self.tangent1_np = np.asarray(tangent1, dtype=np.float64)
        self.tangent1_np = self.tangent1_np / np.linalg.norm(self.tangent1_np)
        self.tangent2_np = np.asarray(tangent2, dtype=np.float64)
        self.tangent2_np = self.tangent2_np / np.linalg.norm(self.tangent2_np)
        self.radius = radius
        self.power = power
        self.in_plane_scale = in_plane_scale if in_plane_scale is not None else radius

        # Store as buffers for torch compatibility
        self.register_buffer('_center', torch.tensor(self.center_np, dtype=torch.float64))
        self.register_buffer('_normal', torch.tensor(self.normal_np, dtype=torch.float64))
        self.register_buffer('_tangent1', torch.tensor(self.tangent1_np, dtype=torch.float64))
        self.register_buffer('_tangent2', torch.tensor(self.tangent2_np, dtype=torch.float64))

    def forward(self, xi, **kwargs):
        """Map reference coordinates to physical crack-tip region.

        Parameters
        ----------
        xi : torch.Tensor (N, 3)
            Reference coordinates in [-1, 1]^3.

        Returns
        -------
        x : torch.Tensor (N, 3)
            Physical coordinates.
        """
        # In-plane: linear mapping
        d_t1 = xi[:, 0] * self.in_plane_scale  # along tangent1
        d_t2 = xi[:, 1] * self.in_plane_scale  # along tangent2

        # Radial: power-law mapping (concentrates mesh near tip)
        # xi_2 in [-1, 1] -> t in [0, 1] -> r in [0, radius]
        # Floor at t=0.01 prevents singular Jacobian at the crack tip
        t = (xi[:, 2] + 1.0) / 2.0  # [0, 1]
        t = torch.clamp(t, min=0.01)
        r_phys = self.radius * t ** self.power  # [~0, radius]

        # Physical position: center + in-plane + radial*normal
        x = (self._center.unsqueeze(0)
             + d_t1.unsqueeze(1) * self._tangent1.unsqueeze(0)
             + d_t2.unsqueeze(1) * self._tangent2.unsqueeze(0)
             + r_phys.unsqueeze(1) * self._normal.unsqueeze(0))

        return x

    def inverse(self, x_phys):
        """Closed-form inverse: physical -> reference coordinates.

        Parameters
        ----------
        x_phys : torch.Tensor (N, 3)

        Returns
        -------
        xi : torch.Tensor (N, 3)
        """
        dx = x_phys - self._center.unsqueeze(0)

        # Project onto frame basis
        d_t1 = torch.sum(dx * self._tangent1.unsqueeze(0), dim=1)
        d_t2 = torch.sum(dx * self._tangent2.unsqueeze(0), dim=1)
        d_n = torch.sum(dx * self._normal.unsqueeze(0), dim=1)

        xi0 = d_t1 / self.in_plane_scale
        xi1 = d_t2 / self.in_plane_scale

        # Invert radial: r = radius * t^power => t = (r/radius)^(1/power)
        r_phys = torch.clamp(d_n, min=0.0)
        t = torch.clamp((r_phys / self.radius) ** (1.0 / self.power), max=1.0)
        xi2 = 2.0 * t - 1.0

        return torch.stack([xi0, xi1, xi2], dim=1)

    def jacobian(self, xi, **kwargs):
        """Explicit analytical Jacobian dx/dxi.

        Returns
        -------
        J : torch.Tensor (N, 3, 3)
            J[n, i, j] = dx_i / dxi_j at point n.
        """
        N = xi.shape[0]
        J = torch.zeros(N, 3, 3, device=xi.device, dtype=xi.dtype)

        t1 = self._tangent1  # (3,)
        t2 = self._tangent2  # (3,)
        n = self._normal     # (3,)

        # dx/dxi_0 = in_plane_scale * tangent1
        J[:, :, 0] = self.in_plane_scale * t1.unsqueeze(0)

        # dx/dxi_1 = in_plane_scale * tangent2
        J[:, :, 1] = self.in_plane_scale * t2.unsqueeze(0)

        # dx/dxi_2 = dr/dxi_2 * normal
        # r = radius * ((xi_2+1)/2)^power
        # dr/dxi_2 = radius * power * ((xi_2+1)/2)^(power-1) * (1/2)
        t_val = (xi[:, 2] + 1.0) / 2.0  # [0, 1]
        t_val = torch.clamp(t_val, min=0.01)  # floor prevents singular Jacobian at tip
        dr_dxi2 = self.radius * self.power * t_val ** (self.power - 1) / 2.0

        J[:, :, 2] = dr_dxi2.unsqueeze(1) * n.unsqueeze(0)

        return J

    @classmethod
    def from_spawned_pair(cls, pair, side="plus", power=2.0):
        """Construct from a SpawnedChartPair produced by ChartSpawner.

        Parameters
        ----------
        pair : SpawnedChartPair
            From atlas.topo.chart_spawn.
        side : str
            "plus" or "minus" — which side of the crack.
        power : float
            Radial squaring exponent.

        Returns
        -------
        decoder : CrackTipDecoder
        """
        seed = getattr(pair, f"seed_{side}")
        frame = getattr(pair, f"frame_{side}")

        return cls(
            center=seed,
            normal=frame[2],      # row 2 = normal
            tangent1=frame[0],    # row 0 = tangent1
            tangent2=frame[1],    # row 1 = tangent2
            radius=pair.radius,
            power=power,
        )

    @classmethod
    def from_crack_tip(cls, tip_position, crack_direction, opening_direction,
                       radius=0.5, power=2.0):
        """Construct from crack tip geometry (convenience for all benchmarks).

        Parameters
        ----------
        tip_position : array-like (3,)
            Physical location of the crack tip.
        crack_direction : array-like (3,)
            Direction the crack propagates (tangent to crack line).
        opening_direction : array-like (3,)
            Normal to crack plane (opening direction).
        radius : float
            Support radius.
        power : float
            Radial squaring exponent.
        """
        cd = np.asarray(crack_direction, dtype=np.float64)
        od = np.asarray(opening_direction, dtype=np.float64)
        cd = cd / np.linalg.norm(cd)
        od = od / np.linalg.norm(od)
        # Third axis via cross product
        t2 = np.cross(od, cd)
        t2 = t2 / np.linalg.norm(t2)

        return cls(
            center=tip_position,
            normal=od,
            tangent1=cd,
            tangent2=t2,
            radius=radius,
            power=power,
        )
