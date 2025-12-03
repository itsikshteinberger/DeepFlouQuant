import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tensorflow.keras as keras
from skimage.transform import resize
import typing as t
from .PreProcess import readImg, get_dapi, multi_norm
import os


MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Model')


def numpy2tensor(array: np.ndarray) -> t.Union[tf.Tensor, tuple]:
    """
    :param array: numpy matrix
    :return: 256x256 tensor version of this numpy matrix + it original shape
    """

    image, image_shape = array.copy(), array.shape  # Ensure the original array remains unchanged
    image = resize(image, (256, 256)) if image.shape != (256, 256) else image
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image, image_shape


def custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    :param y_true: the real target
    :param y_pred: the predicted target (output of the ann)
    :return: loss function (mse + cross entropy)
    """

    loss_fn = keras.losses.BinaryCrossentropy()
    bce = loss_fn(y_true, y_pred)
    max_pred = tf.reduce_max(y_true)
    max_true = tf.reduce_max(y_pred)
    mse_max = (max_true - max_pred) ** 2
    loss_value = bce + mse_max
    return loss_value


def pred(patch: np.ndarray, model: t.Optional[tf.keras.Model] = None) -> np.ndarray:
    """
    :param patch: input image
    :param model: neural network
    :return: output images - model(input_image)
    """

    model = load_model() if not model else model

    image, image_shape = numpy2tensor(patch)
    img1, img2 = image.numpy().reshape(256, 256), model(tf.expand_dims(image, axis=0)).numpy().reshape(256, 256)

    return resize(img2, image_shape)


def modelEval(img: np.ndarray, model: t.Optional[tf.keras.Model] = None, s: int = 300) -> np.ndarray:
    """
    :param img: input image
    :param model: neural network
    :param s: size of patch
    :return: the patches, each processed by the model, stitched together into a single image
    """

    result_img = np.zeros_like(img, dtype='float')

    model = load_model() if not model else model

    for v1, i in enumerate(tqdm(range(0, img.shape[0], s))):
        for v2, j in enumerate(range(0, img.shape[1], s)):
            i_start = i if (i + s) < img.shape[0] else (img.shape[0] - s)
            j_start = j if (j + s) < img.shape[1] else (img.shape[1] - s)
            i_end = i + s if (i + s) < img.shape[0] else img.shape[0]
            j_end = j + s if (j + s) < img.shape[1] else img.shape[1]
            patch = img[i_start:i_end, j_start:j_end]

            prediction = pred(patch=patch, model=model)

            prediction = prediction[i - i_start:, j - j_start:]
            result_img[i:i_end, j:j_end] = prediction

    return result_img


def load_model() -> tf.keras.Model:
    """
    :return: the DeepSpot model
    """

    model = tf.keras.models.load_model(MODEL_PATH,
                                       custom_objects={'custom_loss': custom_loss})
    return model


if __name__ == "__main__":
    # Some test
    p = '../Data/DAPI-Moxd1-Fezf1-Cartpt-s22_1_x20_amy-lowerCHswitch_crop5600x3200.tif'
    thresholds = [4000, 20000, 10000]
    names, images = readImg(path=p)
    dapi, images, names = get_dapi(names=names, images=images)
    images = multi_norm(images=images, thresholds=thresholds)
    model = load_model()

    result_images = [modelEval(img=img.copy(), model=model) for img in images]

    for img, res_img, name in zip(images, result_images, names):
        fig, axs = plt.subplots(1, 2, sharey='all', sharex='all')

        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(res_img, cmap='gray')

        plt.suptitle(f'{name} channel')
        axs[0].set_title('Input')
        axs[1].set_title('Output')

        plt.show()
