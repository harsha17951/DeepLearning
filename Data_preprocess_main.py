import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import classification_model as cm
import Data_preprocess as dp

if __name__ == "__main__":
    images_folder_path_hard = 'train'

    imdata_hard = dp.PreProcess_Data()
    #mdata_normal = dp.PreProcess_Data()

    imdata_hard.visualization_images(images_folder_path_hard, 2)
   # imdata_normal.visualization_images(images_folder_path_normal, 2)

    train_hard, label_hard, _ = imdata_hard.preprocess(images_folder_path_hard)
    #train_normal, label_normal, _ = imdata_normal.preprocess(images_folder_path_normal)

    train = train_hard
    label = label_hard
    train_generator, test_generator, validate_generator = imdata_hard.generate_train_test_images(train, label)

    AnnModel = cm.DeepANN()
    Model1 = AnnModel.simple_model()
    print("train generator", train_generator)
    ANN_history = Model1.fit(train_generator, epochs=150, validation_data=validate_generator)

    Ann_test_loss, Ann_test_acc = Model1.evaluate(test_generator)
    print(f'Test accuracy:{Ann_test_acc}')
    Model1.save("my_model1.keras")
    print("ann architecture")
    print(Model1.summary())
    print("plot graph")
    imdata_hard.plot_history(ANN_history)

    image_shape = (28, 28, 3)
    model_adam = cm.DeepANN().simple_model(image_shape, optimizer='adam')
    model_sgd = cm.DeepANN().simple_model(image_shape, optimizer='sgd')
    model_rmsprop = cm.DeepANN().simple_model(image_shape, optimizer='rmsprop')

    cm.compare_models([model_adam, model_sgd, model_rmsprop], train_generator, validate_generator, epochs=3)
    my_model_test_loss, my_model_test_acc = model_adam.evaluate(test_generator)
    print(f'Test accuracy: {my_model_test_acc}')
