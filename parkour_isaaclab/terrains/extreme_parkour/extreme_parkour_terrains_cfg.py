from isaaclab.utils import configclass
from ..parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg
from . import extreme_parkour_terrians

@configclass
class ExtremeParkourRoughTerrainCfg(ParkourSubTerrainBaseCfg):
    apply_roughness: bool = True 
    apply_flat: bool = False 
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float,float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1 
    pad_height: float = 0.0

@configclass
class ExtremeParkourGapTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_gap_terrain
    gap_size: str = '0.1 + 0.7*difficulty'
    gap_depth: tuple[float, float] = (0.2, 1) 

@configclass
class ExtremeParkourHurdleTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_hurdle_terrain
    stone_len: str = '0.1 + 0.3 * difficulty'
    hurdle_height_range: str = '0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty'

@configclass
class ExtremeParkourStepTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_step_terrain
    step_height: str = '0.1 + 0.35*difficulty'

@configclass
class ExtremeParkourTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_terrain
    pit_depth: tuple[float, float] = (0.2, 1)
    stone_width: float = 1.0
    last_stone_len: float =1.6
    x_range: str = '-0.1, 0.1+0.3*difficulty'
    y_range: str = '0.2, 0.3+0.1*difficulty'
    stone_len: str = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty'
    incline_height: str = '0.25*difficulty'
    last_incline_height: str = 'incline_height + 0.1 - 0.1*difficulty'

@configclass
class ExtremeParkourDemoTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = extreme_parkour_terrians.parkour_demo_terrain


@configclass
class StairsBeamTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Stairs followed by a narrow balance beam.

    Layout along +x: start_platform -> N stair steps -> top_platform ->
    balance_beam -> landing_platform. Waypoints are placed on the centre line
    of this path so the standard parkour ParkourEvent goals still work.
    """

    function = extreme_parkour_terrians.stairs_beam_terrain
    # What to include. Use "stairs_only" / "beam_only" for curriculum warmup,
    # "full" for the combined task.
    mode: str = "full"
    # Stairs parameters.
    num_stair_steps: int = 3
    stair_step_depth: float = 0.35  # length of each step along +x (m)
    stair_step_height: str = "0.05 + 0.13 * difficulty"  # m per step
    # Top platform between stairs and the beam.
    top_platform_len: float = 1.0  # m
    # Balance beam parameters.
    beam_length: float = 3.0  # m
    beam_width: str = "0.8 - 0.5 * difficulty"  # m (wider is easier)
    beam_pit_depth: str = "0.2 + 0.6 * difficulty"  # m (deeper is scarier)
    # Landing platform after the beam.
    landing_platform_len: float = 1.5  # m


@configclass
class StairsOnlyTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Pure staircase terrain: up N steps then down N steps (pyramid).

    Layout along +x: start_platform -> up_stairs -> top_platform ->
    down_stairs -> end_platform.
    """

    function = extreme_parkour_terrians.stairs_only_terrain
    num_steps_up: int = 15
    num_steps_down: int = 15
    step_depth: float = 0.28  # m per step along +x
    step_height: str = "0.05 + 0.10 * difficulty"  # m per step (max)
    step_height_min: float = 0.05  # m per step (min); each ring samples
    # uniformly from [step_height_min, step_height(difficulty)].
    top_platform_len: float = 2.5  # m — central plateau width (wide enough
    # that the 2x2 m origin window in parkour_field_to_mesh stays inside the
    # plateau, so the robot spawns at the bowl bottom rather than a ring).
    slope_threshold: float = 0.3  # horizontal_scale / vertical_scale ratio;
    # lower = more step risers become fully vertical in the mesh. At
    # hs=0.08, vs=0.005, 0.3 gives a threshold of 4.8 vertical quanta, so
    # any riser ≥ 0.025 m becomes vertical.


@configclass
class GapOnlyTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Straight corridor with random-width gaps between platforms.

    Layout along +x:
        [start_platform] → [gap] → [platform] → [gap] → ... → [end_platform]

    Each platform length is random in [plat_len_min, plat_len_max].
    Each gap width is random in [gap_width_min, gap_width_max(difficulty)].
    Gap depth is fixed (deep enough to terminate on contact).
    The corridor is full-width in y (no side pits).

    This terrain forces vision dependence: the policy must see the gap
    ahead and decide to jump or adjust stride. Proprio alone cannot
    detect a gap before the foot is already over it.
    """

    function = extreme_parkour_terrians.gap_only_terrain
    # Platform parameters.
    start_platform_len: float = 2.0  # m — corridor before first gap
    plat_len_min: float = 1.5  # m — min platform length between gaps
    plat_len_max: float = 2.5  # m — max platform length between gaps (randomized)
    # Gap parameters.
    gap_width_min: float = 0.05  # m — narrowest gap (5cm, trivially crossable)
    gap_width_max: str = "0.05 + 0.70 * difficulty"  # m — widest gap
    gap_depth: float = 2.0  # m — deep enough to terminate on contact
    # Number of gaps per tile.
    num_gaps: int = 3
