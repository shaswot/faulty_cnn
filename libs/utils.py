import os
import sys
import git
import pathlib

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

