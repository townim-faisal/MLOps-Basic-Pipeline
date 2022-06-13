import os

import numpy as np

from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.losses import SparseCategoricalCrossentropy 

def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    loss_object = SparseCategoricalCrossentropy(from_logits=True)

    return loss_object(y_true=y, y_pred=y_)

def generate_train_dataset(config):
    #train
    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1)

    train_generator = train_datagen.flow_from_directory(config[0],
                                                        target_size=(config[2], config[3]),
                                                        color_mode="rgb",
                                                        batch_size=config[4],
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")

    #valid
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(config[1],
                                                        target_size=(config[2], config[3]),
                                                        color_mode="rgb",
                                                        batch_size=config[5],
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    
    return train_generator, valid_generator

def generate_test_dataset(config):
    #test
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(config[0],
                                                        target_size=(config[1], config[2]),
                                                        color_mode="rgb",
                                                        batch_size=config[3],
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    return test_generator


def evaluate(config):
    # get the test_dataset
    # config :
    #     0 -> test_dir
    #     1 -> image_height
    #     2 -> image_width
    #     3 -> batch_size 
    test_data_config = (config[0], config[1], config[2], config[3])
    test_dataset = generate_test_dataset(test_data_config)
        

    result_save_path = os.path.join(result_dir, model)
    model_name = "{}_{}_dogcat".format(model, version)
    model_save_path = os.path.join(result_save_path, model_name)
   
    loaded_model = keras.models.load_model(model_save_path)
    
    #confusion matrix
    predict_x = loaded_model.predict(test_dataset) 
    
    pred = np.round(predict_x)
    pred = np.argmax(pred,axis=1)

    pred = pred.tolist()
    
    label = test_dataset.classes.tolist()
    
    cf= confusion_matrix(pred, label)
    
    accuracy = loaded_model.evaluate(test_dataset)

    return pred, test_dataset, cf, accuracy