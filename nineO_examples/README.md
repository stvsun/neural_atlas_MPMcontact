# Nine Circles of Elastic Brittle Fracture

Implementation plans for validating the neural atlas chart-based FEM against the nine challenge problems from Kamarei, Zeng, Dolbow & Lopez-Pamies (2026), CMAME 448, 118449.

**Reference:** "Nine circles of elastic brittle fracture: A series of challenge problems to assess fracture models"

**Data:** [Illinois Data Bank](https://databank.illinois.edu/datasets/IDB-6684845) | [Duke Repository](https://research.repository.duke.edu/record/401)

---

## Progress Overview

| # | Challenge | Type | Status | Error |
|---|-----------|------|--------|-------|
| 1 | Uniaxial tension | Strength nucleation (Mode I) | PLANNED | — |
| 2 | **Biaxial tension** | Strength nucleation | **VALIDATED** | **3.9%** |
| 3 | Torsion | Strength nucleation (shear) | PARTIAL | mesh resolution |
| 4 | Pure-shear fracture | Griffith nucleation | PLANNED | — |
| 5 | Single edge notch | Strength-Griffith mediated | PLANNED | — |
| 6 | Indentation | Strength-Griffith mediated | PLANNED | — |
| 7 | Poker-chip | Strength-Griffith mediated | PLANNED | — |
| 8 | DCB | Griffith propagation (Mode I) | PARTIAL (analytical) | — |
| 9 | Trousers | Griffith propagation (Mode III) | PLANNED | — |

## Priority Order for Development

### Tier 1 — Ready to implement (all infrastructure exists)
1. **Challenge 1: Uniaxial tension** — simplest geometry, uniform stress, should pass easily
2. **Challenge 4: Pure-shear fracture** — Griffith nucleation, rectangular strip with crack
3. **Challenge 8: DCB** — Mode I propagation, beam geometry, analytical reference exists

### Tier 2 — Needs moderate new work
4. **Challenge 5: Single edge notch** — strength-toughness transition, multiple crack lengths
5. **Challenge 3: Torsion** — needs anisotropic mesh or cylindrical-native formulation

### Tier 3 — Needs significant new infrastructure
6. **Challenge 6: Indentation** — axisymmetric FEM, contact BCs, cone crack
7. **Challenge 7: Poker-chip** — near-incompressibility, Neo-Hookean, mixed formulation
8. **Challenge 9: Trousers** — large deformation, Mode III, Neo-Hookean, near-incompressibility

## Materials

| Material | E (MPa) | nu | sigma_ts (MPa) | sigma_hs (MPa) | G_c (N/m) |
|----------|---------|-----|----------------|----------------|-----------|
| Soda-lime glass | 70,000 | 0.22 | 40 | 27.8 | 10 |
| PU elastomer | mu=0.52, Lam=85.77 | ~0.4997 | 0.3 | 1.0 | 41 |

## Folder Structure

```
nineO_examples/
├── README.md              (this file)
├── 1_uniaxial_tension/    PLAN.md
├── 2_biaxial_tension/     PLAN.md, README.md (validated)
├── 3_torsion/             README.md (assessment + progress)
├── 4_pure_shear/          PLAN.md
├── 5_single_edge_notch/   PLAN.md
├── 6_indentation/         PLAN.md
├── 7_poker_chip/          PLAN.md
├── 8_dcb/                 PLAN.md
└── 9_trousers/            PLAN.md
```
