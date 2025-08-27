import numpy as np
from scipy import ndimage


def get_revelable_positions(level, labeled_features):
    """
    Finds walkable tiles that unvail new areas, based on the edges
    """
    # get unexplored positions of the room
    structure = ndimage.generate_binary_structure(2, 2)
    unexplored_edges = np.logical_and(ndimage.binary_dilation(~level.seen, structure), level.seen)
    walkable_edges = np.logical_and(unexplored_edges, level.walkable)  # we use level.walkable to exclude walls etc.
    discovery_potential = np.logical_and(walkable_edges, ~level.was_on)
    feature_unexplored = np.logical_and(labeled_features, discovery_potential)
    unexplored_positions = np.argwhere(feature_unexplored)

    return unexplored_positions
