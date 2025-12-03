import cv2
import tifffile
import typing as t
import numpy as np
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def get_channels_name(path: str) -> t.Union[None, t.List[str]]:
    """
    :param path: the path of the tiff file
    :return: the channels names
    """

    assert path.split('.')[-1] in ['tif', 'tiff']

    with tifffile.TiffFile(path) as tiff:
        metadata = tiff.pages[0].tags
    if 'ImageDescription' in metadata:
        image_description = metadata['ImageDescription'].value
        channel_names = [line.strip() for line in image_description.split('\n') if line.strip().startswith('Name')]
        channel_names = [c.replace('Name: ', '') for c in channel_names]
        return channel_names
    return None


def readImg(path: str, c: int = -1) -> t.Tuple[t.Union[t.List[str], str], t.Union[t.List[np.ndarray], np.ndarray]]:
    """
    :param path: the path the tiff image
    :param c: The channel idx to return (optional)
    :return: the tiff image as stack of numpy 2d matrices + their names (according to c)
    """

    assert path.split('.')[-1] in ['tif', 'tiff']

    ret, imgs = cv2.imreadmulti(path, [], cv2.IMREAD_UNCHANGED)
    imgs = list(imgs)
    names = get_channels_name(path)
    if c != -1:
        return names[1 + c], imgs[1 + c]
    else:
        return names, imgs


def get_dapi(names: t.List[str], images: t.List[np.ndarray]) -> t.Tuple[np.ndarray, t.List[np.ndarray], t.List[str]]:
    """
    :param names: list of the channels names
    :param images: list of the images (2d matrix array each)
    :return: The images and the channels names but separated from the dapi image
    """

    try:
        dapi_index = [element.lower() for element in names].index('dapi')
    except:
        raise Exception('Make sure your image have a Dapi Channel')

    Dapi = images.pop(dapi_index)
    names.pop(dapi_index)
    names = names[::-1]
    return Dapi, images, names


def norm(image: np.ndarray, costume_min: float = None, costume_max: float = None) -> np.ndarray:
    """
    :param image: The image you wanna norm
    :param costume_min: the min value for the min-max normalization (default is the real min)
    :param costume_max: the max value for the min-max normalization (default is the real max)
    :return: The image after min max norm in 8-bit floating point format
    """

    minValue = np.min(image) if not costume_min else costume_min
    maxValue = np.max(image) if not costume_max else costume_max
    image = np.clip(image, minValue, maxValue)
    image = 255 * ((image - minValue) / (maxValue - minValue))
    return image.astype(np.uint8)


def multi_norm(images: t.List[np.ndarray], thresholds: t.List[float]) -> t.List[np.ndarray]:
    """
    :param images: List of images you wanna norm
    :param thresholds: The max value for each image you wanna norm with in the min-max normalization
    :return: The images after min max norm in 8-bit floating point format
    """

    assert len(images) == len(thresholds)

    images = [norm(img.copy(), costume_max=threshold) for img, threshold in zip(images, thresholds)]
    return images


def process_dapi(img: np.ndarray) -> dict:
    """
    :param img: the dapi image
    :return: dictionary contains all the nuclei segmentation result
    """

    img = normalize(img)
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, details = model.predict_instances(img)
    # result_img = draw_border({'img': img, 'details': details})
    return {'labels': labels, 'details': details, 'img': img}


if __name__ == "__main__":
    # Some test
    p = '../Data/DAPI-Moxd1-Fezf1-Cartpt-s22_1_x20_amy-lowerCHswitch_crop5600x3200.tif'
    thresholds = [4000, 20000, 10000]
    names, images = readImg(path=p)
    dapi, images, names = get_dapi(names=names, images=images)
    images = multi_norm(images=images, thresholds=thresholds)

    for img, name in zip([dapi] + images, ['dapi'] + names):
        plt.title(name)
        plt.imshow(img, cmap='gray')
        plt.show()
