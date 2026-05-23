
from __future__ import annotations

import numpy as np
import random
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from ..utils import parkour_field_to_mesh
if TYPE_CHECKING:
    from . import extreme_parkour_terrains_cfg

"""
Reference from https://arxiv.org/pdf/2309.14341
"""

def padding_height_field_raw(
    height_field_raw:np.ndarray, 
    cfg:extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg
    )->np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw

def random_uniform_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg,
    height_field_raw: np.ndarray,
    ):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled 
    return height_field_raw 

@parkour_field_to_mesh
def parkour_gap_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourGapTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        mid_y = length_pixels // 2  # length is actually y width
        gap_size = eval(cfg.gap_size,{"difficulty":difficulty})
        gap_size = round(gap_size / cfg.horizontal_scale)

        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale) + gap_size
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) + gap_size

        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        gap_depth = -round(np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1]) / cfg.vertical_scale)
        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len - 1, mid_y]
        dis_x = platform_len
        last_dis_x = dis_x
        for i in range(num_goals - 2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            dis_x += rand_x
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if not cfg.apply_flat:
                height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale

@parkour_field_to_mesh
def parkour_hurdle_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourHurdleTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = round(stone_len / cfg.horizontal_scale)

        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        hurdle_height_range = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
        hurdle_height_max = round(hurdle_height_range[1] / cfg.vertical_scale)
        hurdle_height_min = round(hurdle_height_range[0] / cfg.vertical_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        dis_x = platform_len
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height

        goals[0] = [platform_len - 1, mid_y]

        for i in range(num_goals-2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            dis_x += rand_x
            if not cfg.apply_flat:
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_step_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourStepTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_height = eval(cfg.step_height,{'difficulty':difficulty} )
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        step_height = round(step_height / cfg.vertical_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        dis_x = platform_len
        last_dis_x = dis_x
        stair_height = 0
        goals = np.zeros((num_goals, 2))
        goals[0] = [platform_len - round(1 / cfg.horizontal_scale), mid_y]
        goal_heights = np.ones((num_goals)) * platform_height

        num_stones = num_goals - 2
        for i in range(num_stones):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if i < num_stones // 2:
                stair_height += step_height
            elif i > num_stones // 2:
                stair_height -= step_height
            height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
            dis_x += rand_x
            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
            goal_heights[i+1] = stair_height
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        # import ipdb; ipdb.set_trace()
        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights 

@parkour_field_to_mesh
def parkour_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        height_field_raw[:] = -round(np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1]) / cfg.vertical_scale)
        mid_y = length_pixels // 2  # length is actually y width
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = np.random.uniform(*stone_len)
        stone_len = 2 * round(stone_len / 2.0, 1)
        stone_len = round(stone_len / cfg.horizontal_scale)
        x_range = eval(cfg.x_range, {"difficulty": difficulty})
        y_range = eval(cfg.y_range, {"difficulty": difficulty})
        dis_x_min = stone_len + round(x_range[0] / cfg.horizontal_scale)
        dis_x_max = stone_len + round(x_range[1] / cfg.horizontal_scale)
        dis_y_min = round(y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        
        stone_width = round(cfg.stone_width / cfg.horizontal_scale)
        last_stone_len = round(cfg.last_stone_len / cfg.horizontal_scale)

        incline_height = eval(cfg.incline_height, {"difficulty": difficulty})
        last_incline_height = eval(cfg.last_incline_height, {"difficulty": difficulty, "incline_height":incline_height})
        last_incline_height = round(last_incline_height / cfg.vertical_scale)
        incline_height = round(incline_height / cfg.vertical_scale)

        dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len -  stone_len // 2, mid_y]
        left_right_flag = np.random.randint(0, 2)
        dis_z = 0
        num_stones = num_goals - 2
        for i in range(num_stones):
            dis_x += np.random.randint(dis_x_min, dis_x_max)
            pos_neg = round(2*(left_right_flag - 0.5))
            dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
            if i == num_stones - 1:
                dis_x += last_stone_len // 4
                heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
                height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            else:
                heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            
            goals[i+1] = [dis_x, dis_y]
            goal_heights[i+1] = np.mean(heights.astype(int))

            left_right_flag = 1 - left_right_flag
        final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
        final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // cfg.horizontal_scale)
        height_field_raw[final_platform_start:, :] = platform_height
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale




@parkour_field_to_mesh
def parkour_demo_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourDemoTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals = np.zeros((num_goals, 2))
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    mid_y = length_pixels // 2  # length is actually y width

    height_field_raw = np.zeros((width_pixels, length_pixels))
    goal_heights = np.ones((num_goals)) * round(cfg.platform_height / cfg.vertical_scale)
    platform_length = round(2 / cfg.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / cfg.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / cfg.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height

    platform_length += round(np.random.uniform(1.5, 2.5) / cfg.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / cfg.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    goal_heights[1] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    goal_heights[2] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / cfg.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    goal_heights[3] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    goal_heights[4] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / cfg.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / cfg.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / cfg.horizontal_scale)
    slope_width = round(1.0 / cfg.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / cfg.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[5] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[6] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size + round(0.4 / cfg.horizontal_scale)
    goals[-1] = [platform_length, left_y]

    height_field_raw = padding_height_field_raw(height_field_raw,cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale




@parkour_field_to_mesh
def stairs_beam_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.StairsBeamTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stairs followed by a narrow balance beam.

    Layout along +x:
        [0, start_platform_len]       start platform (at z=0 or slightly above)
        [..., + stair_depth_total]    num_stair_steps stairs going up
        [..., + top_platform_len]     top platform (highest stair height)
        [..., + beam_length]          balance beam (narrow strip at top height,
                                      everything to the sides is at -pit_depth)
        [..., + landing_platform_len] landing platform at top height

    Goals are spread along the centre line of the path (x increasing).
    """
    mode = cfg.mode
    step_height_m = eval(cfg.stair_step_height, {"difficulty": difficulty})
    beam_w_m = eval(cfg.beam_width, {"difficulty": difficulty})
    pit_depth_m = eval(cfg.beam_pit_depth, {"difficulty": difficulty})

    # Pixel conversions.
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    width_pixels = int(cfg.size[0] / hs)
    length_pixels = int(cfg.size[1] / hs)
    mid_y = length_pixels // 2

    start_len_p = int(cfg.platform_len / hs)
    step_depth_p = int(cfg.stair_step_depth / hs)
    top_len_p = int(cfg.top_platform_len / hs)
    beam_len_p = int(cfg.beam_length / hs)
    landing_len_p = int(cfg.landing_platform_len / hs)
    beam_half_w_p = max(1, int((beam_w_m / 2.0) / hs))

    step_h_q = int(round(step_height_m / vs))
    pit_depth_q = int(round(pit_depth_m / vs))

    num_steps = cfg.num_stair_steps if mode != "beam_only" else 0
    include_stairs = mode in ("stairs_only", "full")
    include_beam = mode in ("beam_only", "full")

    # Base height field filled with the top platform height (so stair section
    # is carved from zero up, beam section is at top height and pit is below).
    top_h_q = step_h_q * num_steps  # final stair height in quanta
    height_field_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # Start platform at z=0.
    x = 0
    height_field_raw[x : x + start_len_p, :] = 0
    x += start_len_p

    # Stairs going up.
    if include_stairs:
        for i in range(num_steps):
            x_end = min(x + step_depth_p, width_pixels)
            stair_h = step_h_q * (i + 1)
            height_field_raw[x:x_end, :] = stair_h
            x = x_end

    # Top platform.
    if include_stairs:
        x_end = min(x + top_len_p, width_pixels)
        height_field_raw[x:x_end, :] = top_h_q
        x = x_end
        beam_base_h = top_h_q
    else:
        # beam-only mode: raise the whole surface so the pit fits in height.
        beam_base_h = max(pit_depth_q, step_h_q * 3)
        height_field_raw[:x, :] = beam_base_h

    # Balance beam: narrow strip at beam_base_h, pit on either side.
    if include_beam:
        x_beam_start = x
        x_beam_end = min(x + beam_len_p, width_pixels)
        # Fill beam section with pit (below base), then carve beam back to base.
        height_field_raw[x_beam_start:x_beam_end, :] = beam_base_h - pit_depth_q
        y_lo = max(0, mid_y - beam_half_w_p)
        y_hi = min(length_pixels, mid_y + beam_half_w_p)
        height_field_raw[x_beam_start:x_beam_end, y_lo:y_hi] = beam_base_h
        x = x_beam_end

    # Landing platform at beam_base_h.
    x_end = min(x + landing_len_p, width_pixels)
    height_field_raw[x:x_end, :] = beam_base_h
    x = x_end

    # If there is any tail space left, keep it at beam_base_h so the robot
    # doesn't fall off by accident.
    if x < width_pixels:
        height_field_raw[x:, :] = beam_base_h

    # Waypoints (num_goals=8 total). Distribute them by stage so the whole
    # track is covered:
    #   goal[0]: start platform
    #   goal[1..num_steps]: one per stair step (or beam start if stairs_only)
    #   goal[num_steps+1]: top platform / beam entry
    #   goal[...]: beam waypoints
    #   goal[-1]: landing platform end
    goals = np.zeros((num_goals, 2), dtype=np.float32)
    goal_heights = np.zeros((num_goals,), dtype=np.float32)

    def _clip_goal_x(gx: int | float) -> int:
        return max(0, min(width_pixels - 1, int(gx)))

    # Collect key x positions along the path for goal placement.
    start_goal_x = _clip_goal_x(start_len_p - int(round(1.0 / hs)))
    stair_end_x = start_len_p + num_steps * step_depth_p
    top_center_x = stair_end_x + top_len_p // 2
    beam_start_x = stair_end_x + (top_len_p if include_stairs else 0)
    landing_start_x = beam_start_x + (beam_len_p if include_beam else 0)

    if mode == "full" and num_goals == 8 and include_stairs and include_beam and num_steps > 0:
        # Aggressive spacing: keep only one early stair waypoint and one top
        # platform waypoint, then spend four waypoints on the narrow beam.
        # This avoids dense stair goals flipping target direction while giving
        # the beam enough center-line guidance.
        first_stair_mid_x = start_len_p + step_depth_p // 2
        key_xs = [
            start_goal_x,
            _clip_goal_x(first_stair_mid_x),
            _clip_goal_x(top_center_x),
        ]
        for frac in (0.12, 0.38, 0.64, 0.90):
            key_xs.append(_clip_goal_x(beam_start_x + frac * beam_len_p))
        key_xs.append(_clip_goal_x(landing_start_x + landing_len_p // 2))
    else:
        key_xs = [start_goal_x]
        # Stair step goals (middle of each step).
        stair_x = start_len_p
        for i in range(num_steps):
            step_mid = stair_x + step_depth_p // 2
            key_xs.append(_clip_goal_x(step_mid))
            stair_x += step_depth_p
        # Top platform goal.
        if include_stairs:
            key_xs.append(_clip_goal_x(stair_x + top_len_p // 2))
        # Beam goals (split beam into 2-3 segments).
        if include_beam:
            beam_segments = 3
            for seg in range(beam_segments):
                frac = (seg + 0.5) / beam_segments
                bx = beam_start_x + frac * beam_len_p
                key_xs.append(_clip_goal_x(bx))
        # Landing goal.
        key_xs.append(_clip_goal_x(landing_start_x + landing_len_p // 2))

    # Trim or pad to exactly num_goals.
    if len(key_xs) > num_goals:
        # Evenly subsample.
        indices = np.linspace(0, len(key_xs) - 1, num_goals).astype(int)
        key_xs = [key_xs[i] for i in indices]
    elif len(key_xs) < num_goals:
        # Pad by repeating last.
        key_xs += [key_xs[-1]] * (num_goals - len(key_xs))

    for gi, gx in enumerate(key_xs):
        gx = max(0, min(width_pixels - 1, gx))
        goals[gi] = [gx, mid_y]
        goal_heights[gi] = height_field_raw[gx, mid_y]

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        # Only roughen the start and landing platforms; leave stairs and beam
        # clean so the curriculum signal stays sharp.
        rough = random_uniform_terrain(difficulty, cfg, np.zeros_like(height_field_raw))
        mask = np.zeros_like(height_field_raw, dtype=bool)
        mask[:start_len_p, :] = True
        mask[int(goals[-1, 0] / hs) :, :] = True
        height_field_raw = np.where(mask, height_field_raw + rough, height_field_raw)

    return height_field_raw, goals * hs, goal_heights


@parkour_field_to_mesh
def stairs_only_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.StairsOnlyTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IsaacLab-style inverted pyramid stairs (2D bowl) with random step heights.

    Each ring has a *random* step height sampled from
    ``[step_h_min, step_h_max]`` where ``step_h_max`` is controlled by
    ``difficulty``. This forces the policy to use depth vision to predict
    the upcoming step height rather than relying on a fixed rhythm.

    Geometry:
      * Outer rim at z=0.
      * Each ring descends by a random amount (cumulative).
      * Innermost region is the deepest platform.
    """
    step_h_max_m = eval(cfg.step_height, {"difficulty": difficulty})
    step_h_min_m = getattr(cfg, "step_height_min", 0.02)

    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    width_pixels = int(cfg.size[0] / hs)
    length_pixels = int(cfg.size[1] / hs)
    mid_y = length_pixels // 2

    step_width_p = max(int(cfg.step_depth / hs), 1)
    center_plat_p = max(int(cfg.top_platform_len / hs), 1)

    max_rings_y = max((length_pixels - center_plat_p) // (2 * step_width_p), 0)
    max_rings_x = max((width_pixels - center_plat_p) // (2 * step_width_p), 0)
    num_rings = min(cfg.num_steps_down, cfg.num_steps_up, max_rings_x, max_rings_y)

    # Sample a random step height (in quanta) for each ring.
    step_heights_m = np.random.uniform(step_h_min_m, step_h_max_m, size=num_rings)
    step_heights_q = np.round(step_heights_m / vs).astype(int)
    # Cumulative depth for each ring.
    cumulative_q = np.cumsum(step_heights_q)

    # Outer rim at z=0, inner rings progressively lower.
    height_field_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    for k in range(num_rings):
        start_x += step_width_p
        stop_x -= step_width_p
        start_y += step_width_p
        stop_y -= step_width_p
        height_field_raw[start_x:stop_x, start_y:stop_y] = -cumulative_q[k]

    # Goals: evenly distributed along +x at y=mid.
    goals = np.zeros((num_goals, 2), dtype=np.float32)
    goal_heights = np.zeros((num_goals,), dtype=np.float32)
    for gi in range(num_goals):
        frac = gi / max(num_goals - 1, 1)
        gx = int(frac * (width_pixels - 1))
        gx = max(0, min(width_pixels - 1, gx))
        goals[gi] = [gx, mid_y]
        goal_heights[gi] = height_field_raw[gx, mid_y]

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * hs, goal_heights


@parkour_field_to_mesh
def gap_only_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.GapOnlyTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight corridor with random-width gaps between platforms.

    The terrain is a flat corridor along +x with periodic gaps (pits).
    Platform lengths and gap widths are randomized. The robot must use
    depth vision to detect upcoming gaps and jump or adjust stride.

    Layout along +x:
        [start_platform] → [gap₁] → [plat₁] → [gap₂] → ... → [end fill]
    """
    gap_width_max_m = eval(cfg.gap_width_max, {"difficulty": difficulty})
    gap_width_min_m = cfg.gap_width_min

    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    width_pixels = int(cfg.size[0] / hs)
    length_pixels = int(cfg.size[1] / hs)
    mid_y = length_pixels // 2

    gap_depth_q = -int(round(cfg.gap_depth / vs))
    start_plat_p = int(cfg.start_platform_len / hs)

    # Initialize height field at z=0 (platform level).
    height_field_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # Build the corridor: start platform, then alternating gaps and platforms.
    x = start_plat_p
    gap_starts = []  # track gap positions for goal placement

    for _ in range(cfg.num_gaps):
        if x >= width_pixels:
            break
        # Random gap width.
        gap_w_m = np.random.uniform(gap_width_min_m, max(gap_width_min_m, gap_width_max_m))
        gap_w_p = max(int(round(gap_w_m / hs)), 1)
        # Carve the gap (full y-width).
        x_end = min(x + gap_w_p, width_pixels)
        height_field_raw[x:x_end, :] = gap_depth_q
        gap_starts.append((x, x_end))
        x = x_end

        if x >= width_pixels:
            break
        # Random platform after the gap.
        plat_len_m = np.random.uniform(cfg.plat_len_min, cfg.plat_len_max)
        plat_len_p = max(int(round(plat_len_m / hs)), 1)
        x_end = min(x + plat_len_p, width_pixels)
        # Platform stays at z=0 (already initialized).
        x = x_end

    # Goals: one per platform AFTER each gap (not on start platform).
    # Robot spawns on the start platform, so placing a goal there is
    # meaningless (instantly reached). Goals should only be on platforms
    # that require crossing a gap to reach.
    plat_mids = []
    for i, (gs, ge) in enumerate(gap_starts):
        # Platform after this gap: from ge to next gap start (or tile end)
        if i + 1 < len(gap_starts):
            next_gs = gap_starts[i + 1][0]
        else:
            next_gs = width_pixels
        plat_mid = (ge + next_gs) // 2
        plat_mid = max(0, min(width_pixels - 1, plat_mid))
        plat_mids.append(plat_mid)

    # Fill goals array: use as many platform midpoints as we have goals.
    goals = np.zeros((num_goals, 2), dtype=np.float32)
    goal_heights = np.zeros((num_goals,), dtype=np.float32)
    for gi in range(num_goals):
        if gi < len(plat_mids):
            gx = plat_mids[gi]
        else:
            # Repeat last platform if more goals than platforms.
            gx = plat_mids[-1]
        gx = max(0, min(width_pixels - 1, gx))
        goals[gi] = [gx, mid_y]
        goal_heights[gi] = height_field_raw[gx, mid_y]

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    # Gap intervals in metres along +x of the sub-terrain (before the mesh
    # decorator centres the mesh). Each row is [x_start_m, x_end_m]. The
    # decorator forwards these unchanged so downstream code can build gap
    # zones for reward masking; the values never enter the observation.
    if len(gap_starts) > 0:
        gap_intervals_m = np.array(
            [(s * hs, e * hs) for s, e in gap_starts], dtype=np.float32
        )
    else:
        gap_intervals_m = np.zeros((0, 2), dtype=np.float32)

    return height_field_raw, goals * hs, goal_heights, gap_intervals_m
