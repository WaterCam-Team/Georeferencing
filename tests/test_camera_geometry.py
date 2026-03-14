import numpy as np

from camera_geometry import build_rotation_matrix


def test_build_rotation_matrix_straight_down_north():
    """
    heading=0, pitch=-90, roll=0 → camera looks straight down over North.

    In ENU, the camera forward (Z_cam) expressed in world coords should be ~[0, 0, -1],
    right ~[1, 0, 0], down ~[0, -1, 0].
    """
    R = build_rotation_matrix(heading_deg=0.0, pitch_deg=-90.0, roll_deg=0.0)

    # Columns of R_cam_to_world are [right, down, forward]
    # R is world→camera, so its transpose is camera→world.
    R_cam_to_world = R.T
    right_w = R_cam_to_world[:, 0]
    down_w = R_cam_to_world[:, 1]
    forward_w = R_cam_to_world[:, 2]

    assert np.allclose(forward_w, np.array([0.0, 0.0, -1.0]), atol=1e-6)
    assert np.allclose(right_w, np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(down_w, np.array([0.0, -1.0, 0.0]), atol=1e-6)


def test_build_rotation_matrix_level_north():
    """
    heading=0, pitch=0, roll=0 → camera looks North at horizon.

    Forward ~[0, 1, 0] (North), right ~[1, 0, 0] (East), down ~[0, 0, -1] (toward ground).
    """
    R = build_rotation_matrix(heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
    R_cam_to_world = R.T
    right_w = R_cam_to_world[:, 0]
    down_w = R_cam_to_world[:, 1]
    forward_w = R_cam_to_world[:, 2]

    assert np.allclose(forward_w, np.array([0.0, 1.0, 0.0]), atol=1e-6)
    assert np.allclose(right_w, np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(down_w, np.array([0.0, 0.0, -1.0]), atol=1e-6)

