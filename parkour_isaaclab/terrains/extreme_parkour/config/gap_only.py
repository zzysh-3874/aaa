from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import (
    GapOnlyTerrainCfg,
    ExtremeParkourHurdleTerrainCfg,
)

GAP_ONLY_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(12.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.04,
    vertical_scale=0.005,
    slope_threshold=0.3,
    difficulty_range=(0.0, 0.15),
    use_cache=False,
    curriculum=True,
    num_goals=3,
    sub_terrains={
        "gap_corridor": GapOnlyTerrainCfg(
            proportion=0.7,
            apply_roughness=False,
        ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
            proportion=0.3,
            apply_roughness=False,
            apply_flat=True,
            x_range=(1.2, 2.2),
            # Force goals onto the centerline. The default y_range=(-0.4, 0.4)
            # makes mid-tile goals offset sideways, which causes ParkourCommand
            # target_yaw to point off-axis and the policy gets dragged
            # diagonally instead of walking straight.
            y_range=(0.0, 0.1),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.15*difficulty",
        ),
    },
)
