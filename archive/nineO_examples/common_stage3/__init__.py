"""Stage 3: XFEM-critic tests exposing fracture mechanics limitations.

These tests are designed by an XFEM expert to reveal the structural weaknesses
of the neural atlas approach compared to mature XFEM implementations. They test:

  X1: Williams enrichment completeness (angular modes, not just radial)
  X2: Crack-face traction-free enforcement
  X3: Mixed-mode capability (K_II extraction, M-integral)
  X4: Crack path accuracy (curved, kinked, branching)
  X5: Displacement discontinuity handling across crack faces
  X6: Stiffness matrix conditioning
  X7: Element integration accuracy near singularities
  X8: Partition of unity at chart overlaps
  X9: Nucleation regularization (mesh independence)

Reference: Moes et al. (1999), Belytschko & Black (1999), Areias & Belytschko (2005)
"""
