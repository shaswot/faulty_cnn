import tensorflow as tf
import pathlib
import git


print(tf.__version__)
print(pathlib.Path.cwd())

PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
print(PROJ_ROOT_PATH)
