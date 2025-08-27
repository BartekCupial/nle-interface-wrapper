import re
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy import ndimage

from nle_interface_wrapper.interface.blstats import BLStats
from nle_interface_wrapper.interface.entity import Entity
from nle_interface_wrapper.interface.glyph import SHOP, G
from nle_interface_wrapper.interface.map.label import corridor_detection, room_detection
from nle_interface_wrapper.interface.map.level import Level
from nle_interface_wrapper.interface.map.utils import get_revelable_positions
from nle_interface_wrapper.interface.utils import isin


class Map:
    def __init__(self):
        self.overview = {}
        self.terrain_features = defaultdict(dict)
        self.shops = defaultdict(list)
        self.levels = {}
        self.map_description = ""

    def update(self, blstats: BLStats, message: str, glyphs, entity: Entity, entities: List[Entity]):
        self.current_level = self.get_current_level(blstats)

        self.current_level.update(glyphs, blstats)
        self.update_terrain_features(glyphs, blstats)
        self.update_shops(blstats, message, entity, entities)

        self.map_description = self.describe_map(glyphs, blstats, entity)

    def update_terrain_features(self, glyphs, blstats: BLStats, force: bool = False):
        current_features = self.get_terrain_features(glyphs)

        if not force:
            past_features = self.terrain_features[(blstats.dungeon_number, blstats.level_number)].get("features", {})

            # Handle stairs persistence
            for key in ("stairs up", "stairs down"):
                past_positions = past_features.get(key)
                curr_positions = current_features.get(key)

                if past_positions is not None and curr_positions is not None:
                    # Merge (union) rows from both arrays
                    merged = np.vstack((past_positions, curr_positions))
                    merged_unique = np.unique(merged, axis=0)
                    current_features[key] = merged_unique

                elif past_positions is not None and curr_positions is None:
                    # Current scan missed it -> carry past memory
                    current_features[key] = past_positions

        self.terrain_features[(blstats.dungeon_number, blstats.level_number)] = {
            "features": current_features,
            "time": blstats.time,
        }

    def get_terrain_features(self, glyphs) -> Dict[str, Any]:
        """
        Returns the terrain features of the current level.
        """
        name_glyph = {
            "stairs down": G.STAIR_DOWN,
            "stairs up": G.STAIR_UP,
            "altar": G.ALTAR,
            "fountain": G.FOUNTAIN,
            "throne": G.THRONE,
            "sink": G.SINK,
            "trap": G.TRAPS,
            "grave": G.GRAVE,
        }

        terrain_features = {}
        for name, glyph in name_glyph.items():
            mask = isin(glyphs, glyph)
            positions = np.argwhere(mask)
            if len(positions) > 0:
                terrain_features[name] = positions

        return terrain_features

    def update_shops(self, blstats: BLStats, message: str, entity: Entity, entities: List[Entity]):
        shop_type = None
        matches = re.search(f"Welcome( again)? to [a-zA-Z' ]*({'|'.join(SHOP.name2id.keys())})!", message)

        if matches is None:
            return

        shop_name = matches.groups()[1]
        assert shop_name in SHOP.name2id, shop_name
        shop_type = SHOP.name2id[shop_name]
        shop_string = SHOP.id2string[shop_type]

        shop_keepers = [ent.position for ent in entities if ent.name == "shopkeeper"]
        assert len(shop_keepers) > 0, "No shopkeepers found"

        # TODO: chebyshev distance
        closest_shop_keeper = min(shop_keepers, key=lambda sk: np.linalg.norm(np.array(sk) - np.array(entity.position)))

        self.shops[(blstats.dungeon_number, blstats.level_number)].append(
            {
                "name": shop_string,
                "type": shop_type,
                "position": closest_shop_keeper,
            }
        )

    def describe_room(
        self,
        blstats: BLStats,
        entity: Entity,
        room_mask: np.ndarray,
        dilated_corridors: np.ndarray,
        dilated_doors: np.ndarray,
        dilated_bars: np.ndarray,
        revelable_positions: np.ndarray,
    ):
        def direction_to(from_xy, to_xy):
            """
            Returns a string describing the direction.
            """
            dy, dx = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
            dirs = []
            if dy < 0:
                dirs.append("north")
            elif dy > 0:
                dirs.append("south")
            if dx < 0:
                dirs.append("west")
            elif dx > 0:
                dirs.append("east")
            if not dirs:
                return "here"
            return " ".join(dirs)

        def get_distance_name(distance):
            """
            Returns a string describing the distance.
            """
            distance_order = {
                "very far to the": 32,
                "far to the": 16,
                "to the": 8,
                "a short distance to the": 4,
                "immediately": 1,
                "": 0,
            }

            for name, dist in distance_order.items():
                if distance >= dist:
                    return name
            return "unknown"

        room_coords = np.argwhere(room_mask)
        py, px = entity.position

        # Describe the exploration status, first check for revelable positions
        if len(revelable_positions) > 0 and np.any(
            np.all(room_coords[:, None] == revelable_positions[None, :], axis=-1)
        ):
            # Visited the room
            if np.any(np.logical_and(self.current_level.was_on, room_mask)):
                explored = "Partially explored"
            else:
                explored = "Unexplored"
        else:
            explored = "Explored"

        # Compute the number of exits
        corridor_exits = np.argwhere(np.logical_and(dilated_corridors, room_mask))
        door_exits = np.argwhere(np.logical_and(dilated_doors, room_mask))
        bar_exits = np.argwhere(np.logical_and(dilated_bars, room_mask))
        num_exits = len(corridor_exits) + len(door_exits) + len(bar_exits)
        num_closed_doors = len(door_exits)
        num_bars = len(bar_exits)

        # Info about features: stairs, fountains, sinks, altars, etc.
        # TODO: add shops
        map_features = self.terrain_features[(blstats.dungeon_number, blstats.level_number)].get("features", {})
        room_features = defaultdict(int)
        for feature_name, positions in map_features.items():
            for pos in positions:
                if room_mask[tuple(pos)]:
                    room_features[feature_name] += 1

        name_plural = {
            "stairs down": ("stairs down", "stairs down"),
            "stairs up": ("stairs up", "stairs up"),
            "altar": ("an altar", "altars"),
            "fountain": ("a fountain", "fountains"),
            "throne": ("a throne", "thrones"),
            "sink": ("a sink", "sinks"),
            "trap": ("a trap", "traps"),
            "grave": ("a grave", "graves"),
        }

        features = []
        for feature, count in room_features.items():
            if count == 1:
                features.append(name_plural[feature][0])
            elif count > 1:
                features.append(f"{count} {name_plural[feature][1]}")

        shop_name = None
        for shop_info in self.shops[blstats.dungeon_number, blstats.level_number]:
            if room_mask[shop_info["position"]]:
                shop_name = shop_info["name"]
                break

        # Compute distance from the player to the room
        room_distances = np.sum(np.abs(np.array(entity.position) - room_coords), axis=1)
        idx = np.argmin(room_distances)

        # Describe the distance
        distance = get_distance_name(room_distances[idx])

        # Describe the direction
        in_this_room = room_distances[idx] == 0
        if in_this_room:
            direction = "here"
        else:
            direction = direction_to((py, px), room_coords[idx])

        return {
            "explored": explored,
            "distance": distance,
            "direction": direction,
            "num_exits": num_exits,
            "num_closed_doors": num_closed_doors,
            "num_bars": num_bars,
            "features": features,
            "shop_name": shop_name,
        }

    def describe_map(self, glyphs, blstats, entity):
        labeled_rooms, num_rooms = room_detection(glyphs, self.current_level)
        labeled_corridors, num_corridors = corridor_detection(glyphs, self.current_level)
        revelable_positions = get_revelable_positions(self.current_level, labeled_rooms)

        dilated_corridors = ndimage.binary_dilation(labeled_corridors)
        dilated_doors = ndimage.binary_dilation(isin(glyphs, G.DOOR_CLOSED))
        dilated_bars = ndimage.binary_dilation(isin(glyphs, G.BARS))

        rooms_info = []
        for room_id in range(1, num_rooms + 1):
            room = labeled_rooms == room_id

            room_info = self.describe_room(
                blstats, entity, room, dilated_corridors, dilated_doors, dilated_bars, revelable_positions
            )
            room_info["room_id"] = room_id

            rooms_info.append(room_info)

        desc = []
        # overview = self.get_cached_overview()
        # if overview:
        #     desc.append("Dungeon overview:")
        #     desc.extend(overview.split("\n"))

        # desc.append("Local map:")

        for room_info in rooms_info:
            room_id = room_info["room_id"]
            explored = room_info["explored"]
            distance = room_info["distance"]
            direction = room_info["direction"]
            num_exits = room_info["num_exits"]
            num_closed_doors = room_info["num_closed_doors"]
            num_bars = room_info["num_bars"]
            shop_name = room_info["shop_name"]
            features = "    Objects: " + ", ".join(room_info["features"]) + "." if room_info["features"] else ""

            if direction == "here":
                here = "<- You are here."
                direction = ""
                punctuation = ":" if features else ""
            else:
                here = ""
                direction = direction
                punctuation = ":" if features else "."

            if shop_name is not None:
                detail_text = f"{explored} {shop_name}"
            else:
                detail_text = f"{explored} room"

            if num_exits:
                exits_text = f"with {num_exits} {'exit' if num_exits == 1 else 'exits'}"

                blocked_exits = []
                if num_closed_doors > 0:
                    blocked_exits.append(f"{num_closed_doors} closed doors")
                if num_bars > 0:
                    blocked_exits.append(f"{num_bars} iron bars")
                exits_text += f" ({'and '.join(blocked_exits)})" if blocked_exits else ""

                detail_text += " " + exits_text

            text = " ".join([e for e in [detail_text, distance, direction, punctuation, here] if e])
            text = text.replace(" :", ":")
            text = text.replace(" .", ".")

            desc.append(text)

            if features:
                desc.append(features)

        return "\n".join(desc)

    def get_current_level(self, blstats: BLStats) -> Level:
        """
        :return: Level object of the current level
        """
        key = (blstats.dungeon_number, blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        return self.levels[key]

    def __str__(self):
        return self.map_description

    def __repr__(self):
        return self.map_description
