from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import (
    StairsOnlyTerrainCfg,
)

STAIRS_ONLY_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=0.3,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs_pyramid": StairsOnlyTerrainCfg(
            proportion=1.0,
            apply_roughness=False,
        ),
    },
)
