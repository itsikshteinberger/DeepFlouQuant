import os
import shutil
import typing as t
import cv2
import numpy as np
import pandas as pd
from Algorithm import PreProcess, NeuralNetwrok, PostProcess
import matplotlib.pyplot as plt


def run(path: str, thresholds: t.List[float], is_save: bool = True) -> pd.DataFrame:
    """
    :param path: the path for the tiff image
    :param thresholds: all the max values for the min-max normalization
    :param is_save: save the results according to the process
    :return: The result df contain info about each nuclei
    """

    # ---- PreProcess ----

    print('Read and normalize image')

    names, images = PreProcess.readImg(path=path)
    dapi, images, names = PreProcess.get_dapi(names=names, images=images)
    images = PreProcess.multi_norm(images=images, thresholds=thresholds)

    for i in range(3):
        images[i] = images[i]#[200: 800, 200: 800]
    #dapi = dapi[200: 800, 200: 800]

    print('Segment the dapi image')

    dapi_data = PreProcess.process_dapi(img=dapi)

    # ---- Neural Network ----

    print('Load model')
    model = NeuralNetwrok.load_model()

    print('Process images')
    ann_images = [NeuralNetwrok.modelEval(img=img, model=model) for img in images]

    # ---- PostProcess ----

    print('Get blobs and clean noises')
    dataFrames = [PostProcess.get_df(img) for img in images]
    dataFrames = PostProcess.clean_noise(dataframes=dataFrames, img=images[0])

    print('Get the result df')
    result_df = PostProcess.count(dapi=dapi_data, names=names, dataframes=dataFrames)

    if is_save:
        if os.path.exists('result'):
            shutil.rmtree('result')
        os.makedirs('result')

        for i in range(len(images)):
            save(data=images[i], name=f'{names[i]}_pre')
            save(data=ann_images[i], name=f'{names[i]}_ann')
            save(data=dataFrames[i], name=f'{names[i]}_df')
        save(data=result_df, name='result_df')

    return result_df


def save(data: t.Union[pd.DataFrame, np.ndarray], name: str):
    """
    :param data: the data you wanna save
    :param name: the file name for this data
    :return: Save the file in the result folder
    """

    assert not isinstance(data, list)

    if isinstance(data, pd.DataFrame):
        data.to_csv(f'result/{name}.csv')
        print(f'Saved {name}.csv successfully!')

    elif isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            data = PreProcess.norm(data)
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(f'result/{name}.png', data)
        else:
            cv2.imwrite(f'result/{name}.png', data)

        print(f'Saved {name}.png successfully!')
    else:
        raise Exception('Just numpy array / data frame can be valid inputs to this function')


if __name__ == "__main__":
    p = 'Data/DAPI-Moxd1-Fezf1-Cartpt-s22_1_x20_amy-lowerCHswitch_crop5600x3200.tif' # Example, change as you want
    thresholds = [4000, 8000, 4000]
    run(p, thresholds, True)
