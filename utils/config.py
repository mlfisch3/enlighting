import streamlit as st
import os
import sys

BASE_DIR_PATH = os.path.split(sys.path[0])[0] # sys.path[0] returns absolute path to FILE main.py instead of current working directory.  unexpected behavior
DEBUG_FILE_PATH = os.path.join(BASE_DIR_PATH, 'debug.txt')  # debug mode is on/off depending on existance/absence of a file with this exact filepath name 

NPY_DIR_PATH = os.path.join(BASE_DIR_PATH, 'NPY')
IMAGE_DIR_PATH = os.path.join(BASE_DIR_PATH, 'IMAGES')
DATA_DIR_PATH = os.path.join(BASE_DIR_PATH, 'DATA')
EXAMPLES_DIR_PATH = os.path.join(BASE_DIR_PATH, 'examples')

SCRAPYARD_FILE_NAME = 'scrapyard.jpg'
SELFIE_FILE_NAME = 'selfie.jpg'
CYLINDER_FILE_NAME = 'cylinder.jpg'
PARK_FILE_NAME = 'park.jpg'
SCHOOL_FILE_NAME = 'school.jpg'
SPIRAL_FILE_NAME = 'spiral.jpg'


SCRAPYARD_FILE_PATH = os.path.join(EXAMPLES_DIR_PATH, SCRAPYARD_FILE_NAME)
SELFIE_FILE_PATH = os.path.join(EXAMPLES_DIR_PATH, SELFIE_FILE_NAME)
CYLINDER_FILE_PATH = os.path.join(EXAMPLES_DIR_PATH, CYLINDER_FILE_NAME)
PARK_FILE_PATH = os.path.join(EXAMPLES_DIR_PATH, PARK_FILE_NAME)
SCHOOL_FILE_PATH = os.path.join(EXAMPLES_DIR_PATH, SCHOOL_FILE_NAME)
SPIRAL_FILE_PATH = os.path.join(EXAMPLES_DIR_PATH, SPIRAL_FILE_NAME)

EXAMPLES = ('scrapyard', 'selfie', 'cylinder', 'park', 'school', 'spiral')
EXAMPLE_PATHS = {
                    'scrapyard': SCRAPYARD_FILE_PATH, 
                    'selfie': SELFIE_FILE_PATH, 
                    'cylinder': CYLINDER_FILE_PATH, 
                    'park':PARK_FILE_PATH, 
                    'school':SCHOOL_FILE_PATH, 
                    'spiral':SPIRAL_FILE_PATH
                }