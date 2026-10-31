import numpy as np

from harnessstudio.tools.geometry import pixel_to_world, world_to_pixel


def test_pixel_world_round_trip_with_horizontal_flip():
    K = np.array([[200.0, 0, 127.5], [0, 200.0, 127.5], [0, 0, 1]])
    world_to_cam = np.eye(4)
    world_to_cam[:3, 3] = [0.1, -0.05, 1.5]          # camera looks along +z at points ~1.5 m away
    w2p = np.eye(4)
    w2p[:3, :3] = K
    w2p = w2p @ world_to_cam
    cam = {"world_to_pixel": w2p, "height": 256, "width": 256, "pixel_flip": "horizontal"}
    point = np.array([0.2, 0.1, 0.0])
    u, v = world_to_pixel(point, cam)
    depth = np.full((256, 256), float((world_to_cam @ np.append(point, 1))[2]), dtype=np.float32)
    back = pixel_to_world(u, v, depth, cam)
    assert np.linalg.norm(back - point) < 0.01   # pixel rounding only
