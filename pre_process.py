import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, exposure, io, transform
from skimage.feature import hog
from sklearn.model_selection import train_test_split


class PreProcess:

    def __init__(self):
        # self.data = file_list
        pass

    def file_labelling(self, cat_path, dog_path):
        cat_dir = os.listdir(cat_path)
        cat_file = [os.path.join(cat_path, i) for i in cat_dir]
        cat_df = pd.DataFrame(data=cat_file, columns=['file'])
        cat_df['target'] = 0

        dog_dir = os.listdir(dog_path)
        dog_file = [os.path.join(dog_path, i) for i in dog_dir]
        dog_df = pd.DataFrame(data=dog_file, columns=['file'])
        dog_df['target'] = 1

        return cat_df, dog_df

    def test_file_labelling(self, path):
        test_dir = os.listdir(path)
        test_file = [os.path.join(path, i) for i in test_dir]
        test_df = pd.DataFrame(data=test_file, columns=['Location'])
        return test_df

    def extract_hog_feature(self, image_path):
        image = io.imread(image_path)
        # Resize the image to a consistent size
        resized_image = transform.resize(image, (128, 128))
        # Convert the resized image to grayscale
        gray_image = color.rgb2gray(resized_image)
        fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        return fd.flatten(), image_path

    def split_dataset(self, df):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=100)
        return X_train, X_test, y_train, y_test
