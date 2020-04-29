import numpy as np
import matplotlib.image

N_CONSTANTS = {
    "air": {"value": 1, "lower_bound": -1, "higher_bound": 0},  # Air is the default
    "concrete": {"value": 2.16 - 0.021j, "lower_bound": 0, "higher_bound": 0.90},
}


def parse_image(filename):
    """
    Imports a greyscale png image, and determines where the walls are from the greyscale value.
    Assuming concrete walls.
    """
    read_img = matplotlib.image.imread(filename)

    if len(np.shape(read_img)) > 2:  # In case image is not grayscale.
        read_img = read_img[:, :, 0]

    read_img = read_img.astype(np.complex64)
    # Create a mask size of read_img that is false
    all_mask = read_img != read_img
    for mat in N_CONSTANTS.values():
        material_mask = (read_img >= mat["lower_bound"]) & (
            read_img < mat["higher_bound"]
        )
        # Only convert each element once
        # In the case where a conversion would cause the next conversion to be a false positive
        material_mask &= ~all_mask
        read_img[material_mask] = mat["value"]
        all_mask |= material_mask

    # convert rest to air
    read_img[~all_mask] = N_CONSTANTS["air"]["value"]

    return read_img
