import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PreProcess_Data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img = os.path.join(dpath, i, train_class[j])
                img = cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count += 1
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train images : {}\n'.format(len(train)))
        print('Number of train images labels: {}\n'.format(len(label)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        return train, label, retina_df

    def generate_train_test_images(self, train, label):
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        train_data, test_data = train_test_split(retina_df, test_size=0.2)
        print(test_data)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            validation_split=0.15)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode='categorical',
            subset='validation'

        )
        test_generator = train_datagen.flow_from_dataframe(
            test_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            batch_size=32,
            class_mode='categorical',

        )
        print(f"Train image shape:{train_data.shape}")
        print(f"Test image shape:{test_data.shape}")
        return train_generator, test_generator, validation_generator

    def plot_history(self, history):
        # plot loss
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('training and validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show(block=True)

        # plot accuracy
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('training and validation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show(block=True)
