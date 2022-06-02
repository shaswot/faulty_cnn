import os
import sys
import git
import pathlib

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from libs import utils
############################################################################################
# create mnist32-CNN model
def create_model(image_x_size, image_y_size,seed):
    kernel_initializer = tf.keras.initializers.HeUniform(seed)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (4, 4), 
                                  activation='relu', 
                                  kernel_initializer = kernel_initializer,
                                  input_shape=(image_x_size, image_y_size, 1),
                                  padding='same',
                                  name="c0"))
    model.add(keras.layers.MaxPooling2D((2, 2), name="p0"))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(32, (4, 4), 
                                  activation='relu', 
                                  kernel_initializer = kernel_initializer,
                                  padding='same',
                                  name="c1"))
    model.add(keras.layers.MaxPooling2D((2, 2), name="p1"))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Flatten(name="flatten"))
    
    model.add(keras.layers.Dense(1024, 
                                 activation="relu",
                                 kernel_initializer = kernel_initializer,
                                 name="h0"))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(10, 
                                 activation=tf.nn.softmax,
                                 kernel_initializer = kernel_initializer,
                                 name="op"))
    return model
############################################################################################
def train(model_instance, show_summary=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    # load dataset  
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    image_x_size = 28
    image_y_size = 28
    
    # get model parameters
    dataset, model_arch, model_config, layer_widths, seed = utils.instancename2metadata(model_instance)
    model_meta_type, model_type, model_instance = utils.metadata2instancenames(dataset, 
                                                                               model_arch, 
                                                                               layer_widths, 
                                                                               seed)
    # set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    # create model
    model = create_model(image_x_size, image_y_size,seed)
    
    # compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    
    if show_summary:
        model.summary()
    
    # fit model
    model.fit(train_images, train_labels,
              batch_size=128, 
              epochs=25, 
              verbose=show_summary,
              validation_data=(test_images, test_labels))
    
    # Folder to save models (create if necessary)
    model_folder = pathlib.Path(PROJ_ROOT_PATH / "models" / model_type)
    pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)

    # save model file
    model_filename = model_instance + ".h5"
    model_file = pathlib.Path(model_folder/ model_filename)
    print("Saved model: ", model_file)
    model.save(model_file)
    return model_file
    
#############################################################################################
# The meaning of testing a model is quite ambiguous.
# What does the test accuracy represent?
# Is it 
# a. the ability to generalize classification over ONLY unseen images (i.e., test set) ?
# b. the ability to recognize and classify over both seen and unseen images (i.e., test set + train set )?

# def test_mnist32(model_file, show_summary=False):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             print(e)
            
#     # Prepare dataset
#     # Combine test and train images together into one dataset
#     DATASET_PATH = str(pathlib.Path(PROJ_ROOT_PATH / "datasets" / "mnist.npz" ))
#     (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=DATASET_PATH)
#     train_images = train_images.astype(np.float32) / 255.0
#     test_images = test_images.astype(np.float32) / 255.0  

#     all_images =np.concatenate([train_images, test_images], axis=0)
#     all_labels =np.concatenate([train_labels, test_labels], axis=0)
#     all_images = np.expand_dims(all_images, axis=-1)
    
#     # resize the input shape , i.e. old shape: 28, new shape: 32
#     image_x_size = 32
#     image_y_size = 32
#     all_images = tf.image.resize(all_images, [image_x_size, image_y_size]) 

#     # Load model
#     model = tf.keras.models.load_model(model_file)
#     if show_summary:
#         model.summary()

#     # Evaluate model
#     y_pred = model.predict(all_images) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#     Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels

#     accuracy = accuracy_score(y_true=all_labels, y_pred=Y_pred)
#     conf_matrix = confusion_matrix(y_true=all_labels, y_pred=Y_pred) 
          
#     return [accuracy, conf_matrix]

# #############################################################################################