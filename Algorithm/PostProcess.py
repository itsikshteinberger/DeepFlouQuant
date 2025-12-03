from scipy.spatial import distance_matrix
from skimage.feature import blob_log
import numpy as np
import pandas as pd
import typing as t


def get_df(image: np.ndarray) -> pd.DataFrame:
    """
    :param image: input image (2d gray scale matrix) in range 0-1
    :return: df of image blobs (x,y and radius)
    """

    data = {'x': [], 'y': [], 'r': []}

    image[image < 0.2] = 0
    blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)

    data['x'] = list(blobs_log[:, 1])
    data['y'] = list(blobs_log[:, 0])
    data['r'] = list(blobs_log[:, 2] * np.sqrt(2))

    return pd.DataFrame.from_dict(data)


def clean_noise(dataframes: t.List[pd.DataFrame], img: np.ndarray, s: int = 300, min_t: int = 1) -> t.List[pd.DataFrame]:
    """
    Over-lapping / almost over-lapping blobs are considered as noise and to be removed
    :param dataframes: blobs dataframes of each channel
    :param img: some image of one of the channels
    :param s: the window size (s x s) we looked at to remove noise
    :param min_t: the max distance between two blobs to be considered as noise
    :return: the dataframes without noises
    """

    indices_to_remove = [[] for _ in dataframes]

    for v1, i in enumerate(range(0, img.shape[0], s)):
        for v2, j in enumerate(range(0, img.shape[1], s)):

            sub_dataframes = [df[(df['x'] >= j) & (df['x'] < j + s) & (df['y'] >= i) & (df['y'] < i + s)] for df in
                              dataframes]

            for i, sub_dataframe in enumerate(sub_dataframes):
                dfs = pd.concat([df for idx, df in enumerate(sub_dataframes) if idx != i], ignore_index=True)
                d_matrix = distance_matrix(sub_dataframe.values, dfs.values)
                if d_matrix.shape[0] * d_matrix.shape[1] > 0:
                    indices_to_remove[i] = indices_to_remove[i] + sub_dataframe.iloc[
                        np.where(np.min(d_matrix, axis=1) < min_t)[0]].index.tolist()

    for i, df in enumerate(dataframes):
        dataframes[i] = df[~df.index.isin(indices_to_remove[i])]

    return dataframes


def count(dapi: dict, names: t.List[str], dataframes: t.List[pd.DataFrame]) -> pd.DataFrame:
    """
    :param dapi: the dapi dict contain all the segmentation details
    :param names: the channels names
    :param dataframes: the blobs dataframes per channel
    :return: result df contain how much blobs for each channel there are on each nuclei from the dapi image
    """

    result = pd.DataFrame()
    result['x_center'] = dapi['details']['points'][:, 1]
    result['y_center'] = dapi['details']['points'][:, 0]
    result['cell_index'] = result.index + 1

    for c, df in zip(names, dataframes):
        blobs_img = np.zeros((dapi['img'].shape[0], dapi['img'].shape[1]))
        x = df['x'].round().astype(int)
        y = df['y'].round().astype(int)
        blobs_img[y, x] = 1

        labels = dapi['labels'][blobs_img == 1]
        labels = labels[labels != 0]

        label, count = np.unique(labels, return_counts=True)
        count_dict = dict(zip(label, count))
        result[c] = result['cell_index'].map(count_dict).fillna(0)

    label, count = np.unique(dapi['labels'], return_counts=True)
    count_dict = dict(zip(label, count))
    result['size'] = result['cell_index'].map(count_dict).fillna(0)

    return result
