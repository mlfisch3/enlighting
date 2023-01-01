import streamlit as st
import os
import sys
import json

BASE_DIR_PATH = os.path.split(sys.path[0])[0] # sys.path[0] returns absolute path to FILE main.py instead of current working directory.  unexpected behavior
DEBUG_FILE_PATH = os.path.join(BASE_DIR_PATH, 'debug.txt')  # debug mode is on/off depending on existance/absence of a file with this exact filepath name 
CONFIG_DIR_PATH = os.path.join(BASE_DIR_PATH, 'config')
EXAMPLES_FILE_PATH = os.path.join(CONFIG_DIR_PATH, 'examples.json')
NPY_DIR_PATH = os.path.join(BASE_DIR_PATH, 'NPY')
IMAGE_DIR_PATH = os.path.join(BASE_DIR_PATH, 'IMAGES')
DATA_DIR_PATH = os.path.join(BASE_DIR_PATH, 'DATA')
EXAMPLES_DIR_PATH = os.path.join(BASE_DIR_PATH, 'examples')

with open(EXAMPLES_FILE_PATH, 'r') as f:
    examples = json.load(f)

images = examples['images']

image_labels = [image['label'] for image in images]
image_filenames = [os.path.join(EXAMPLES_DIR_PATH, image['filename']) for image in images]

EXAMPLES = image_labels
EXAMPLE_PATHS = dict(zip(image_labels, image_filenames))