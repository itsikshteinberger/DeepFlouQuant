import numpy as np
import typing as t
from Algorithm.PreProcess import norm
import matplotlib.colors as mcolors
import cv2

colors = ['cyan', 'pink', 'yellow', 'gold', 'lightcoral', 'lightgreen', 'lightskyblue', 'mediumorchid', 'lightsalmon']


def create_uniform_image(r: int, g: int, b: int, size: tuple) -> np.ndarray:
    """
    :param r: red
    :param g: green
    :param b: blue
    :param size: size of the output image
    :return: Uniform image with the rgb input values
    """

    size = (size[0], size[1], 4)
    img = np.zeros(size, dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    img[:, :, 3] = 255

    return img


def color_name_to_rgb(color_name: str) -> list:
    """
    :param color_name: The color name as string
    :return: the rgb value of this color
    """
    try:
        rgb = mcolors.to_rgb(color_name)
        return rgb
    except ValueError:
        raise Exception(f"Error: '{color_name}' is not a valid color name.")


def drawNis(org_imgs: t.List[np.ndarray]) -> t.List[np.ndarray]:
    """
    :param org_imgs: the original channels images
    :return: show the top 99.99% percentile pixels as in the nis image
    """

    stack_images = []
    for i, org in enumerate(org_imgs):
        t = np.percentile(org, 99.8)
        org[org < t] = 0
        org = norm(org, costume_max=np.percentile(org, 99.99))
        alpha_values = org / np.max(org)

        rgb_color = color_name_to_rgb(colors[i])

        # Create RGBA array
        rgba_image = np.zeros((*org.shape, 4), dtype=np.float32)
        rgba_image[:, :, :3] = np.array(rgb_color)
        rgba_image[:, :, 3] = alpha_values

        stack_images.append(rgba_image)

    background_img = np.zeros((*org_imgs[0].shape, 4), dtype=np.uint8)

    for i, img in enumerate(stack_images):
        if i > 0:
            stack_images[i] = (img * 255).astype(np.uint8)
            background_img += stack_images[i]

    background_img[background_img != 0] = 255

    stack_images = [background_img] + stack_images

    return stack_images


def draw_border(dapi: dict) -> np.ndarray:
    """
    :param dapi: the dapi data
    :return: the dapi image with border around each nuclei
    """

    result_img = cv2.cvtColor(norm(dapi['img'].copy()), cv2.COLOR_GRAY2RGB)

    borders = dapi['details']['coord']

    for border in borders:
        border = border.T.reshape((-1, 1, 2)).astype(int)

        x = border[:, :, 0].copy()
        y = border[:, :, 1].copy()

        border[:, :, 1] = x
        border[:, :, 0] = y
        cv2.polylines(result_img, [border], isClosed=True, color=(0, 0, 255),
                      thickness=2)

    return result_img
