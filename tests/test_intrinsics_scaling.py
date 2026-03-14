import numpy as np

from georeference_tool import scale_intrinsics_for_resolution


def test_scale_intrinsics_for_resolution_center_invariant():
    """
    Scaling K to a new resolution should keep the principal point at the same
    relative location in the image (i.e. cx/w and cy/h stay constant).
    """
    calib_w, calib_h = 2000, 1000
    fx = 1500.0
    fy = 1500.0
    cx = calib_w / 3.0
    cy = calib_h / 4.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    new_w, new_h = 3000, 2000
    K_scaled = scale_intrinsics_for_resolution(K, calib_w, calib_h, new_w, new_h)

    fx_s, fy_s = K_scaled[0, 0], K_scaled[1, 1]
    cx_s, cy_s = K_scaled[0, 2], K_scaled[1, 2]

    # Focal lengths scale linearly with resolution
    assert np.isclose(fx_s / fx, new_w / calib_w)
    assert np.isclose(fy_s / fy, new_h / calib_h)

    # Principal point stays at same relative position
    assert np.isclose(cx / calib_w, cx_s / new_w)
    assert np.isclose(cy / calib_h, cy_s / new_h)

