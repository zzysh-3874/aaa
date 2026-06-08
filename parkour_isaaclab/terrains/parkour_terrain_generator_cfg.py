

from __future__ import annotations
from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg

@configclass
class ParkourSubTerrainBaseCfg(HfTerrainBaseCfg):
    border_width: float = 0.0
    horizontal_scale: float = 0.05
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""
    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""
    platform_len: float = 2.5
    platform_height: float = 0.
    slope_threshold: float | None = 1.5
    edge_width_thresh = 0.05
    use_simplified: bool = False
    
@configclass
class ParkourTerrainGeneratorCfg(TerrainGeneratorCfg):
    num_goals: int = 8 
    terrain_names: list[str] = [] 
    random_difficulty: bool = False 
    # START-style terrain progressive curriculum (TerProg). When set, it maps
    # each sub-terrain NAME to a difficulty-level band (start_frac, end_frac)
    # in [0, 1] over the row axis (row 0 = easiest, row num_rows-1 = hardest).
    # A sub-terrain only appears in rows whose normalised difficulty
    # ``row / (num_rows - 1)`` falls within its band, so early (low) levels
    # contain only the easy terrains (e.g. flat + low-randomness stepping
    # stones) and harder terrains (gaps / hurdles / narrow beams) are
    # introduced only at higher levels. Within the eligible types for a row,
    # columns are allocated proportionally to each type's ``proportion``.
    # When None (default), the original column-based type assignment is used
    # (all types present at every difficulty level).
    terprog_bands: dict[str, tuple[float, float]] | None = None