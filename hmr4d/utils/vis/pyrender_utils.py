import numpy as np


def get_campose_pyrender(p_from, p_to, gravity_vec):
    """
    Get camera pose in pyrender format. (OpenGL coordinate)
    Args:
        p_from: (3)
        p_to: (3)
        gravity_vec: (3)
    Returns:
        T_w2c: (4, 4)
    """
    vec = p_to - p_from
    vec = vec / np.linalg.norm(vec)
    z_axis = -vec  # z pointing to backward
    x_axis = np.cross(z_axis, gravity_vec)

    # this coordinate doesn't follow right-hand rule
    y_axis = -np.cross(x_axis, z_axis)

    # Compose
    R_w2c = np.vstack([x_axis, y_axis, z_axis]).T
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = p_from

    return T_w2c
