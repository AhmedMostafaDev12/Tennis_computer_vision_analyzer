def get_center_of_bbox(bbox):
    """
    Given a bounding box in the format [x1, y1, x2, y2],
    return the center point (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2 )
    cy = int((y1 + y2) / 2 )
    return (cx, cy)

def measure_distance(p1, p2):
    """
    Measure Euclidean distance between the centers of two bounding boxes.
    Each bbox is in the format [x1, y1, x2, y2].
    """
    import math
    distance = math.sqrt((p1[0]- p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return distance

def get_foot_position(bbox):
    """
    Given a bounding box in the format [x1, y1, x2, y2],
    return the foot position (bottom center) of the bbox.
    """
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2 )
    foot_y = int(y2)
    return (foot_x, foot_y)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    """
    Find the closest keypoint to a given point based on vertical distance.

    This function is used to determine which court keypoint (e.g., baseline, service line, net)
    is closest to a player's position, comparing only vertical (y-coordinate) distances.

    Args:
        point: A tuple (x, y) representing a position, typically a player's foot position
        keypoints: A flat list of keypoint coordinates [x1, y1, x2, y2, x3, y3, ...]
                   representing court line points
        keypoint_indices: A list of indices indicating which keypoints to check

        note :
        keypoints = ALL the points (the complete dataset)
        keypoint_indices = The CONCERNED/RELEVANT points you want to check (a subset)

    Returns:
        int: The index of the closest keypoint from keypoint_indices

    Note:
        Distance is calculated using only the y-coordinate (vertical distance),
        not full Euclidean distance, as vertical position on the court is typically
        more important for tennis court analysis.
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_indix in keypoint_indices:
        keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
        distance = abs(point[1]-keypoint[1])

        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix

    return key_point_ind

def get_height_of_bbox(bbox):
    """
    Given a bounding box in the format [x1, y1, x2, y2],
    return the height of the bbox.
    """
    x1, y1, x2, y2 = bbox
    height = int(y2 - y1)
    return height

def measure_xy_distance(p1, p2):
    """
    Measure the horizontal (x) and vertical (y) distance between the centers of two bounding boxes.
    Each bbox is in the format [x1, y1, x2, y2].
    Returns a tuple (dx, dy).
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    return (dx, dy)

def get_center_of_bbox(bbox):
    """
    Given a bounding box in the format [x1, y1, x2, y2],
    return the center point (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2 )
    cy = int((y1 + y2) / 2 )
    return (cx, cy)