#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Final proj
# @File      :image_formatter.py
# @Author    :Miao Sun
# @Time      :12/14/2021

from os import path
from PIL import Image

# from os import path
import os
import glob
import shutil


def converter(input: path, output: path, w=128, h=128):
    """convert jpg to 128*128 and save to output floder without change filename

    Args:
        input (path): input file path
        output (path): output file path
        w (int, optional): [description]. Defaults to 128.
        h (int, optional): [description]. Defaults to 128.
    """
    img = Image.open(input)
    try:
        con_image = img.resize((w, h), Image.BILINEAR)
        con_image.save(os.path.join(output, os.path.basename(input)))
    except Exception as e:
        print(e)


def del_old_file(file_path="./convert_image/"):
    """
    delete old convert file(.jpg)
    """
    shutil.rmtree(file_path, True)
    have_mask_path = os.path.join(file_path, "have_mask/")
    no_mask_path = os.path.join(file_path, "no_mask/")
    os.makedirs(have_mask_path)
    os.makedirs(no_mask_path)


def start():
    for file in glob.glob("./rename_image/have_mask/*.jpg"):
        converter(
            file, "./convert_image/have_mask/",
        )

    for file in glob.glob("./rename_image/no_mask/*.jpg"):
        converter(
            file, "./convert_image/no_mask/",
        )


# del_old_file()
# start()
