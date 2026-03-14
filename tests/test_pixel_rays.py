import numpy as np


def pixel_ray_from_K(K: np.ndarray, u: float, v: float) -> np.ndarray:
    """Helper: compute normalized camera-frame ray for pixel (u, v)."""
    K_inv = np.linalg.inv(K)
    ray = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def test_center_pixel_points_along_camera_z():
    """
    For a symmetric pinhole camera, the image center should map to the
    camera forward direction (0, 0, 1) in camera coordinates.
    """
    w, h = 1920, 1080
    fx = fy = 1200.0
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    ray = pixel_ray_from_K(K, cx, cy)
    # Expect approximately [0, 0, 1]
    assert np.allclose(ray, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_off_center_pixel_has_expected_direction():
    """
    A pixel offset from the center should produce a ray with corresponding
    normalized x/y components in camera space.
    """
    w, h = 1920, 1080
    fx = fy = 1200.0
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # Pixel 100 px to the right and 50 px up from center
    u = cx + 100.0
    v = cy - 50.0
    ray = pixel_ray_from_K(K, u, v)

    # In ideal pinhole, before normalization: [dx/fx, dy/fy, 1]
    dx = 100.0 / fx
    dy = -50.0 / fy
    ideal = np.array([dx, dy, 1.0])
    ideal /= np.linalg.norm(ideal)

    assert np.allclose(ray, ideal, atol=1e-6)

