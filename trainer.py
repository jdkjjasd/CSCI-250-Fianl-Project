#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Assignment
# @File      :trainer.py
# @Author    :Miao Sun
# @Time      :12/14/2021


import os
import matplotlib.pyplot as plt

# import matplotlib.image as mpimg

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

# from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def trainer():
    base_dir = "./dataset_image/"
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")

    train_have_mask_dir = os.path.join(train_dir, "have_mask")
    train_no_mask_dir = os.path.join(train_dir, "no_mask")

    validation_have_mask_dir = os.path.join(validation_dir, "have_mask")
    validation_no_mask_dir = os.path.join(validation_dir, "no_mask")

    train_have_mask_fname = os.listdir(train_have_mask_dir)
    train_no_mask_fname = os.listdir(train_no_mask_dir)

    count_have_mask = len(train_have_mask_fname)
    count_no_mask = len(train_no_mask_fname)
    count_val = len(os.listdir(validation_no_mask_dir)) + len(
        os.listdir(validation_have_mask_dir)
    )
    """
    def pic_test():
        nrows = 4
        ncols = 4
        pic_index = 0

        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)

        pic_index += 8
        next_have_mask_pic = [
            os.path.join(train_have_mask_dir, fname)
            for fname in train_have_mask_fname[pic_index - 8: pic_index]
        ]
        next_no_mask_pic = [
            os.path.join(train_no_mask_dir, fname)
            for fname in train_no_mask_fname[pic_index - 8: pic_index]
        ]

        for i, img_path in enumerate(next_have_mask_pic + next_no_mask_pic):
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis("off")
            img = mpimg.imread(img_path)
            plt.imshow(img)
        plt.show()
    """
    # build CNN
    img_input = layers.Input(shape=(128, 128, 3))

    x = layers.Conv2D(16, 3, activation="relu")(img_input)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = Model(img_input, output)

    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # pic_test()
    model.compile(
        loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["acc"]
    )

    # generate image data

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(128, 128),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode="binary",
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_dir, target_size=(128, 128), batch_size=20, class_mode="binary"
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(
            (count_have_mask + count_no_mask) / 20
        ),  # 2000 images = batch_size * steps
        epochs=15,
        validation_data=validation_generator,
        validation_steps=int(count_val / 20),  # 1000 images = batch_size * steps
        verbose=2,
    )

    model.save("./model/model.h5")
    model_json = model.to_json()
    with open("./model/model.json", "w") as json_file:
        json_file.write(model_json)

    def evaluate_acc():
        acc = history.history["acc"]
        val_acc = history.history["val_acc"]

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs = range(len(acc))

        plt.plot(epochs, acc, label="acc")
        plt.plot(epochs, val_acc, label="val_acc")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.savefig("acc.png")

        plt.figure()

        plt.plot(epochs, loss, label="loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.savefig("loss.png")

    evaluate_acc()


trainer()
