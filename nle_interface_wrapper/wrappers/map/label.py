import itertools
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from nle_interface_wrapper.wrappers.properties.glyph import SS, G
from nle_interface_wrapper.wrappers.properties.utils import isin


def print_boolean_array_ascii(arr):
    # Normalize the array to (0, 1)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        normalized = np.full(arr.shape, 0.5)
    else:
        normalized = (arr - min_val) / (max_val - min_val)
    # Create ASCII visualization
    chars = " ._-=+*#%@"

    for row in normalized:
        line = ""
        for val in row:
            index = int(val * (len(chars) - 1))
            line += chars[index]
        print(line)


def save_boolean_array_pillow(arr):
    # Ensure array is numpy array
    arr = np.asarray(arr)

    # Get unique labels and remap them to consecutive integers
    unique_values = np.unique(arr)
    label_map = {val: idx for idx, val in enumerate(unique_values)}
    remapped_arr = np.zeros_like(arr)
    for val in unique_values:
        remapped_arr[arr == val] = label_map[val]

    # Create colormap
    num_colors = len(unique_values)
    colormap = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # Set background to black

    # Create RGB image
    height, width = arr.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, label in enumerate(unique_values):
        mask = arr == label
        rgb_image[mask] = colormap[idx]

    # Save image
    try:
        img = Image.fromarray(rgb_image)
        img.save("labeled_rooms.png")
    except Exception as e:
        print(f"Error saving image: {e}")


def label_dungeon_features(glyphs, level):
    """
    Labels the dungeon features (rooms, corridors, and doors) in the current level of the bot.
    Args:
        bot (Bot): The bot instance containing the current level and entity position.
    Returns:
        tuple: A tuple containing:
            - labeled_features (np.ndarray): An array with labeled rooms and corridors.
            - num_rooms (int): The number of labeled rooms.
            - num_corridors (int): The number of labeled corridors.
    """
    structure = ndimage.generate_binary_structure(2, 1)

    # rooms
    room_floor = frozenset({SS.S_room, SS.S_darkroom})
    rooms = isin(level.objects, room_floor)
    labeled_rooms, num_rooms = ndimage.label(rooms, structure=structure)

    # corridors
    corridor_floor = frozenset({SS.S_corr, SS.S_litcorr})
    corridors = isin(level.objects, corridor_floor)
    labeled_corridors, num_corridors = ndimage.label(corridors, structure=structure)

    # doors
    doors = isin(level.objects, frozenset({SS.S_ndoor}), G.DOOR_OPENED)

    # combine rooms and corridors
    labeled_features = np.zeros_like(level.objects)
    labeled_features[rooms] = labeled_rooms[rooms]
    labeled_features[corridors] = labeled_corridors[corridors] + num_rooms

    def label_walkable_features(position):
        # if all neighbors which are dungeon features have the same label
        neighbors = []
        height, width = glyphs.shape
        for x, y in itertools.product([-1, 0, 1], repeat=2):
            if x == 0 and y == 0:
                continue

            # make sure we don't have out of bounds
            if not (0 <= position[0] + y < height) or not (0 <= position[1] + x < width):
                continue

            neighbor = labeled_features[position[0] + y, position[1] + x]
            if neighbor != 0:
                neighbors.append(neighbor)
        neighbors = np.array(neighbors)

        if len(neighbors) > 0:
            # if all neighbors are rooms or corridors
            if np.all(neighbors <= num_rooms):
                rooms[position] = True
            else:
                corridors[position] = True

    # missing features (items, corpses, monsters, chests, etc.)
    # this includes our position
    for p in np.argwhere(np.logical_and(level.walkable, labeled_features == 0)):
        label_walkable_features(tuple(p))

    # we include doors only at the end to be able to detect if we are standing on the door, overall we treat doors as part of the corridor
    corridors[doors] = True
    rooms[doors] = False  # we have to exclude doors from rooms as well, because we could stand on the door

    labeled_rooms, num_rooms = ndimage.label(rooms, structure=structure)
    labeled_corridors, num_corridors = ndimage.label(corridors, structure=structure)
    labeled_features = np.zeros_like(level.objects)
    labeled_features[rooms] = labeled_rooms[rooms]
    labeled_features[corridors] = labeled_corridors[corridors] + num_rooms

    return labeled_features, num_rooms, num_corridors


def room_detection(glyphs, level) -> Tuple[np.ndarray, int]:
    labeled_features, num_rooms, num_corridors = label_dungeon_features(glyphs, level)
    labeled_features[labeled_features > num_rooms] = 0

    return labeled_features, num_rooms


def corridor_detection(glyphs, level) -> Tuple[np.ndarray, int]:
    labeled_features, num_rooms, num_corridors = label_dungeon_features(glyphs, level)
    labeled_features[labeled_features <= num_rooms] = 0
    labeled_features -= num_rooms
    labeled_features[labeled_features < 0] = 0

    return labeled_features, num_corridors


def features_detection(glyphs, level) -> Tuple[np.ndarray, int, int]:
    labeled_features, num_rooms, num_corridors = label_dungeon_features(glyphs, level)

    return labeled_features, num_rooms + num_corridors
