import os
import sys
import git
import pathlib

import numpy as np

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)


#############################################################################################
# function to create model instance name with given parameters
def metadata2instancenames(dataset, model_arch, layer_widths, seed):
    model_config = '_'.join([str(layer_width) for layer_width in layer_widths])
    model_tag = model_arch + '_' + model_config 

    model_meta_type = dataset + '-' + model_arch
    model_type = dataset + '-' + model_tag
    model_instance = model_type + '-' + str(seed)
    
    return [model_meta_type, model_type, model_instance]

#############################################################################################
# function to infer model parameters from given model_instance_name
def instancename2metadata(model_instance):
    # get all metdata about model from the model_instance name
    # model_instance = dataset-modelarch_modelconfig-seed
    # e.g. 'mnist-dense_32_32-170100'
    dataset = model_instance.split("-")[0]
    model_tag = model_instance.split("-")[1]

    model_arch  = model_tag.split("_")[0]
    model_config = model_tag.replace(model_arch, "")[1:]
    seed = int(model_instance.split("-")[2])
    
    layer_widths = [int(x) for x in model_config.split("_")]   

    # Recontruct the tags from inferred metadata
    model_tag = model_arch + '_' + model_config
    model_meta_type = dataset + '-' + model_arch
    model_type = dataset + '-' + model_tag
    model_instance = model_type + '-' + str(seed)
    
    return [dataset, model_arch, model_config, layer_widths, seed]
############################################################################################# 
# entropy function
# https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
#############################################################################################
def segregate_dataset_4Q_mean(all_images, all_labels, half_size):
    
    # Differentiate dataset into 4 different types depending on which quadrant has highest mean
    """
    q1 | q2
    -------
    q3 | q4
    
    """
    q1_images = []
    q2_images = []
    q3_images = []
    q4_images = []
    
    q1_labels = []
    q2_labels = []
    q3_labels = []
    q4_labels = []
    
    for image, label in zip(all_images, all_labels):
        q1 = np.mean(image[:half_size,:half_size].flatten())
        q2 = np.mean(image[:half_size,half_size:].flatten())
        q3 = np.mean(image[half_size:,:half_size].flatten())
        q4 = np.mean(image[half_size:,half_size:].flatten())
    
        max_ent_q = np.argmax([q1,q2,q3,q4])
    
        if max_ent_q == 0:
            q1_images.append(image)
            q1_labels.append(label)
        elif max_ent_q == 1:
            q2_images.append(image)
            q2_labels.append(label)
        elif max_ent_q == 2:
            q3_images.append(image)
            q3_labels.append(label)
        elif max_ent_q == 3:
            q4_images.append(image)
            q4_labels.append(label)
        else:
            print("max_ent_q has invalid value")
    return (np.array(q1_images), np.array(q1_labels),
            np.array(q2_images), np.array(q2_labels),
            np.array(q3_images), np.array(q3_labels),
            np.array(q4_images), np.array(q4_labels))
#############################################################################################
def segregate_dataset_4Q_ent(all_images, all_labels, half_size):
    
    # Differentiate dataset into 4 different types depending on which quadrant has highest entropy
    """
    q1 | q2
    -------
    q3 | q4
    
    """
    q1_images = []
    q2_images = []
    q3_images = []
    q4_images = []
    
    q1_labels = []
    q2_labels = []
    q3_labels = []
    q4_labels = []
    
    for image, label in zip(all_images, all_labels):
        q1_ent = entropy(image[:half_size,:half_size].flatten())
        q2_ent = entropy(image[:half_size,half_size:].flatten())
        q3_ent = entropy(image[half_size:,:half_size].flatten())
        q4_ent = entropy(image[half_size:,half_size:].flatten())
    
        max_ent_q = np.argmax([q1_ent,q2_ent,q3_ent,q4_ent])
    
        if max_ent_q == 0:
            q1_images.append(image)
            q1_labels.append(label)
        elif max_ent_q == 1:
            q2_images.append(image)
            q2_labels.append(label)
        elif max_ent_q == 2:
            q3_images.append(image)
            q3_labels.append(label)
        elif max_ent_q == 3:
            q4_images.append(image)
            q4_labels.append(label)
        else:
            print("max_ent_q has invalid value")
    return (np.array(q1_images), np.array(q1_labels),
            np.array(q2_images), np.array(q2_labels),
            np.array(q3_images), np.array(q3_labels),
            np.array(q4_images), np.array(q4_labels))
#############################################################################################