from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour import *  # noqa: F401,F403
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import (
    StairsBeamTerrainCfg,
)

STAIRS_BEAM_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # Curriculum warmup: stairs only and beam only, so the policy can
        # learn each skill in isolation before facing the combined track.
        "stairs_only": StairsBeamTerrainCfg(
            proportion=0.3,
            mode="stairs_only",
            apply_roughness=False,
        ),
        "beam_only": StairsBeamTerrainCfg(
            proportion=0.3,
            mode="beam_only",
            apply_roughness=False,
        ),
        "stairs_beam": StairsBeamTerrainCfg(
            proportion=0.4,
            mode="full",
            apply_roughness=False,
        ),
    },
)
