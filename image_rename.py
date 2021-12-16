#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Final
# @File      :image_rename.py
# @Author    :Miao Sun
# @Time      :12/14/2021

import os
import shutil


def copy_tree(copy_from="./raw_image", copy_to="./rename_image"):
    shutil.copytree(copy_from, copy_to)


def del_old_file(file_path="./rename_image"):
    shutil.rmtree(file_path, True)


def rename(pre: int, file_path):
    """
    rename image name as pre+_+oldname

    Args:
        pre (int): prefix
        file_path (path): the path need to rename
    """
    files = os.listdir(file_path)

    for file in files:
        oldname = os.path.join(file_path, file)
        filename = str(pre) + "_" + file
        newname = os.path.join(file_path, filename)
        os.rename(oldname, newname)


# del_old_file()
# copy_tree()
# rename(1, "./rename_image/no_mask/")
# rename(0, "./rename_image/have_mask/")
