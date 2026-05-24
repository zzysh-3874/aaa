"""Teacher-style multi-terrain mix.

Mirrors EXTREME_PARKOUR_TERRAINS_CFG (5 sub-terrains used by the original
Teacher policy: gap, hurdle, flat, step, generic parkour). All sub-terrains
share the goal-with-y-offset placement strategy that Teacher relies on, so
the policy must learn to actually face each goal direction instead of
running in a straight line.
"""

from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour import (
    ExtremeParkourGapTerrainCfg,
    ExtremeParkourHurdleTerrainCfg,
    ExtremeParkourStepTerrainCfg,
    ExtremeParkourTerrainCfg,
    ExtremeParkourDemoTerrainCfg,
)


FULL_PARKOUR_WITH_GAP_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    # num_goals=8 (default in ParkourTerrainGeneratorCfg) — matches Teacher.
    sub_terrains={
        # Teacher 5-terrain mix (20% each).
        "parkour_gap": ExtremeParkourGapTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(0.8, 1.5),
            half_valid_width=(0.6, 1.2),
            gap_size="0.1 + 0.7*difficulty",
        ),
        "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.25*difficulty",
        ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            apply_flat=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.15*difficulty",
        ),
        "parkour_step": ExtremeParkourStepTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range=(0.3, 1.5),
            half_valid_width=(0.5, 1),
            step_height="0.1 + 0.35*difficulty",
        ),
        "parkour": ExtremeParkourTerrainCfg(
            proportion=0.2,
            apply_roughness=True,
            x_range="-0.1, 0.1+0.3*difficulty",
            y_range="0.2, 0.3+0.1*difficulty",
            stone_len="0.9 - 0.3*difficulty, 1 - 0.2*difficulty",
            incline_height="0.25*difficulty",
            last_incline_height="incline_height + 0.1 - 0.1*difficulty",
        ),
        "parkour_demo": ExtremeParkourDemoTerrainCfg(
            proportion=0.0,
            apply_roughness=True,
        ),
    },
)
