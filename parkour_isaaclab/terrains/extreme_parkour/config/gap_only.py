from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import (
    GapOnlyTerrainCfg,
)

GAP_ONLY_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(8.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.04,
    vertical_scale=0.005,
    slope_threshold=0.3,
    difficulty_range=(0.0, 0.15),
    use_cache=False,
    curriculum=True,
    num_goals=1,
    sub_terrains={
        "gap_corridor": GapOnlyTerrainCfg(
            proportion=1.0,
            apply_roughness=False,
        ),
    },
)
