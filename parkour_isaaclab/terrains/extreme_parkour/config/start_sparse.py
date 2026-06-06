"""START-paper sparse-foothold terrain mix.

Reproduces the four terrain archetypes from START (arXiv 2512.13153):
stepping stones (梅花桩), balance beams (独木桥), and gaps. (The paper's
"stepping beams" is a progressive blend of stones+beams; here we approximate
the curriculum by mixing stones and beams directly.)

All sub-terrains use centre-line goals so the standard ParkourEvent waypoint
logic works unchanged. A small ``parkour_flat`` slot is kept as an auxiliary
flat-ground warmup terrain, matching START's "auxiliary flat terrains to learn
basic locomotion skills" at the start of the curriculum.
"""

from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import (
    SteppingStonesTerrainCfg,
    BalanceBeamTerrainCfg,
    GapOnlyTerrainCfg,
    ExtremeParkourHurdleTerrainCfg,
)


START_SPARSE_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(12.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.04,
    vertical_scale=0.005,
    slope_threshold=0.3,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    num_goals=8,
    sub_terrains={
        # Auxiliary flat ground for basic-locomotion warmup (START keeps a
        # flat slot in the early curriculum).
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
            proportion=0.1,
            apply_roughness=False,
            apply_flat=True,
            x_range=(1.2, 2.2),
            y_range=(0.0, 0.1),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.15*difficulty",
        ),
        # 梅花桩 — stepping stones.
        "stepping_stones": SteppingStonesTerrainCfg(
            proportion=0.35,
            apply_roughness=False,
        ),
        # 独木桥 — balance beam.
        "balance_beam": BalanceBeamTerrainCfg(
            proportion=0.30,
            apply_roughness=False,
        ),
        # Gaps.
        "gap_corridor": GapOnlyTerrainCfg(
            proportion=0.25,
            apply_roughness=False,
        ),
    },
)
