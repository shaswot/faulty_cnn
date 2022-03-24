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
def create_model(layer_widths, seed):
    image_x_size = 32
    image_y_size = 32
    kernel_initializer = keras.initializers.HeUniform(seed)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (4, 4), 
                                  activation='relu', 
                                  kernel_initializer = kernel_initializer,
                                  input_shape=(image_x_size, image_y_size, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    for idx in range(len(layer_widths)):
        model.add(keras.layers.Dense(layer_widths[idx], 
                                     activation="relu", 
                                     kernel_initializer = kernel_initializer,
                                     name="fc_"+str(idx)))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax, name="op_layer"))
    return model
############################################################################################
def train_mnist32(model_instance, show_summary=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    # load dataset
    DATASET_PATH = str(pathlib.Path(PROJ_ROOT_PATH / "datasets" / "mnist.npz" ))
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=DATASET_PATH)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    ## resize the input shape , i.e. old shape: 28, new shape: 32
    image_x_size = 32
    image_y_size = 32
    train_images = tf.image.resize(train_images, [image_x_size, image_y_size]) 
    test_images = tf.image.resize(test_images, [image_x_size, image_y_size])

    ## one hot encode target values
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    
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
    model = create_model(layer_widths, seed)
    
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, 
                                  momentum=0.9)
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    if show_summary:
        model.summary()
    
    # fit model
    model.fit(train_images, train_labels,
              batch_size=1024, 
              epochs=25, 
              verbose=show_summary,
              validation_data=(test_images, test_labels))
    
    # Folder to save models (create if necessary)
    model_folder = pathlib.Path(PROJ_ROOT_PATH / "models" / model_meta_type / model_type)
    pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)

    # save model file
    model_filename = model_instance + ".h5"
    model_file = pathlib.Path(model_folder/ model_filename)
    print("Saved model: ", model_file)
    model.save(model_file)
    return model_file
    
#############################################################################################
def test_mnist32(model_file, show_summary=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    # Prepare dataset
    # Combine test and train images together into one dataset
    DATASET_PATH = str(pathlib.Path(PROJ_ROOT_PATH / "datasets" / "mnist.npz" ))
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=DATASET_PATH)
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0  

    all_images =np.concatenate([train_images, test_images], axis=0)
    all_labels =np.concatenate([train_labels, test_labels], axis=0)
    all_images = np.expand_dims(all_images, axis=-1)
    
    # resize the input shape , i.e. old shape: 28, new shape: 32
    image_x_size = 32
    image_y_size = 32
    all_images = tf.image.resize(all_images, [image_x_size, image_y_size]) 

    # Load model
    model = tf.keras.models.load_model(model_file)
    if show_summary:
        model.summary()

    # Evaluate model
    y_pred = model.predict(all_images) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels

    accuracy = accuracy_score(y_true=all_labels, y_pred=Y_pred)
    conf_matrix = confusion_matrix(y_true=all_labels, y_pred=Y_pred) 
          
    return [accuracy, conf_matrix]

#############################################################################################