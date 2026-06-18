"""atlas/topo — topology-aware atlas construction via persistent homology."""
from atlas.topo.filtration import sample_sdf_on_grid, clip_to_interior
from atlas.topo.persistence import compute_persistence_diagrams, betti_numbers_at
from atlas.topo.ls_category import compute_m_min, certify_atlas
from atlas.topo.monitor import TopologyMonitor, TopologyEvent
from atlas.topo.chart_spawn import ChartSpawner, SpawnedChartPair

__all__ = [
    "sample_sdf_on_grid", "clip_to_interior",
    "compute_persistence_diagrams", "betti_numbers_at",
    "compute_m_min", "certify_atlas",
    "TopologyMonitor", "TopologyEvent",
    "ChartSpawner", "SpawnedChartPair",
]
