#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Assignment
# @File      :separate_image.py
# @Author    :Miao Sun
# @Time      :12/14/2021

import os
import shutil


def del_old_file(root_path="./dataset_image/"):
    shutil.rmtree(root_path, True)
    os.makedirs("./dataset_image/train/have_mask")
    os.makedirs("./dataset_image/train/no_mask")
    os.makedirs("./dataset_image/validation/have_mask")
    os.makedirs("./dataset_image/validation/no_mask")


def sep(file_path, train_path, test_path, testcase: int):
    files = os.listdir(file_path)
    case = 0
    for file in files:
        full_file_path = os.path.join(file_path, file)

        if case % testcase == 0:
            shutil.copy(full_file_path, test_path)
        else:
            shutil.copy(full_file_path, train_path)
        case += 1


def separater(
    have_mask_dir="./convert_image/have_mask", no_mask_dir="./convert_image/no_mask"
):

    # total_have_mask_case = len(os.listdir(have_mask_dir))
    # total_no_mask_case = len(os.listdir(no_mask_dir))

    sep(
        have_mask_dir,
        "./dataset_image/train/have_mask",
        "./dataset_image/validation/have_mask",
        10,
    )

    sep(
        no_mask_dir,
        "./dataset_image/train/no_mask",
        "./dataset_image/validation/no_mask",
        10,
    )


# del_old_file()

# separater()
