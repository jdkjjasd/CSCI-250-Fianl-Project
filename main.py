#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Assignment
# @File      :main.py
# @Author    :Miao Sun
# @Time      :12/14/2021

# 0 for masked
# 1 for no_mask

import trainer
import image_formatter as fm
import image_rename as rn
import separate_image as sp

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import load_model
from keras.preprocessing import image

# from tensorflow.keras.applications import preprocess_input
import numpy as np
import os
import shutil


def main():
    creat_folder()

    choice_list = [1, 2, 3, 0]

    while True:
        print_title()
        choice = input()
        if choice.isdigit():
            choice = int(choice)
        if choice not in choice_list:
            continue

        if choice == 1:
            regenerate_dataset()
        elif choice == 2:
            retrain()
        elif choice == 3:
            test_model()
        else:
            os.exit()


def creat_folder():
    if not os.path.exists("./test_file/"):
        os.makedirs("./test_file/")

    if not os.path.exists("./raw_image/have_mask/"):
        os.makedirs("./raw_image/have_mask/")

    if not os.path.exists("./rwa_image/no_mask/"):
        os.makedirs("./rwa_image/no_mask/")


def print_title():
    print("----------------------------------------------------------")
    print(
        'how to use: add image into "raw_image" folder to change the dataset and train model  '
    )
    print('\t add image into "test_file folder" to test model')
    print("----------------------------------------------------------")
    print("1. regenerate_dataset")
    if (
        os.path.exists("./dataset_image/train/have_mask")
        and len(os.listdir("./dataset_image/train/have_mask")) > 0
    ):
        print("2. retrain the model use current dataset")

    if os.path.exists("./model/model.h5") and len(os.listdir("./test_file")) > 0:
        print("3. use current model to test image in test_file folder")

    print("0. exit progrom")
    print("----------------------------------------------------------")


def test_model():
    ld_model()
    root = "./test_file"

    files = os.listdir("./test_file")
    total = len(files)

    n = math.ceil(total ** 0.5)

    fig = plt.gcf()
    fig.set_size_inches(n * 2, n * 2)

    for i, file in enumerate(files):
        filename = os.path.join(root, file)
        img = image.load_img(filename, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)

        sp = plt.subplot(n, n, i + 1)
        sp.axis("off")

        img = mpimg.imread(filename)
        plt.imshow(img)

        title = "have_mask" if preds[0][0] == 0 else "no_mask"
        plt.title(title)

    plt.show()

    """
    img = image.load_img("./test_file/42.jpg", target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    preds = model.predict(x)

    print(preds[0][0]==0)
    """


def ld_model():

    global model

    model = load_model("./model/model.h5")


def regenerate_dataset():
    rn.del_old_file()
    fm.del_old_file()
    sp.del_old_file()

    rn.copy_tree()
    rn.rename(1, "./rename_image/no_mask/")
    rn.rename(0, "./rename_image/have_mask/")

    fm.start()

    sp.separater()


def retrain():
    shutil.rmtree("./model", True)
    trainer.trainer()


if __name__ == "__main__":
    main()
