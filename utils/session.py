import streamlit as st
import pandas as pd
import os
import sys
from random import randint
from utils.logging import timestamp
from utils.config import BASE_DIR_PATH, DEBUG_FILE_PATH, EXAMPLES_DIR_PATH, NPY_DIR_PATH, IMAGE_DIR_PATH, DATA_DIR_PATH

pd.options.display.width = sys.maxsize
pd.options.display.max_colwidth = 9999
pd.options.display.colheader_justify ='center'
pd.options.display.max_rows = 500

default_granularity=0.1
default_power=0.5
default_smoothness=0.3
default_texture_style='I'
default_kernel_parallel=5
default_kernel_orthogonal=1
default_sharpness=0.001
CG_TOL=0.1
LU_TOL=0.015
MAX_ITER=50
FILL=50
default_dim_size=(50)
default_dim_threshold=0.5
default_a=-0.3293
default_b=1.1258
default_lo=1
default_hi=7
default_exposure_ratio_in=-1
default_color_gamma=0.3981

def set_debug():

    st.session_state.debug = False
    if os.path.isfile(st.session_state.debug_file_path):
        st.session_state.debug = True


def report_runs(tag=''):

    update = {
                "timestamp" : [timestamp()],
                "run_counts (main(), run_app()):" : f'({st.session_state.completed_main_runs}/{st.session_state.total_main_runs}) ({st.session_state.completed_app_runs}/{st.session_state.total_app_runs})',
                "tag" : [tag],
                "input_key" : [st.session_state.input_key],
                "input_source" : [st.session_state.input_source],
                "source_last_updated" : [st.session_state.source_last_updated],
                "fImage_is_not_None" : [st.session_state.fImage_is_not_None],
                "input_file_path" : [st.session_state.input_file_path],
                "upload_key" : [st.session_state.upload_key],
                "viewer_selection" : [st.session_state.viewer_selection],
                "viewer_selection_key" : [st.session_state.viewer_selection_key]
            }

    update_df = pd.DataFrame(data=update)
    st.session_state.state_history = pd.concat([st.session_state.state_history, update_df]).reset_index(drop=True)


def initialize_session():

    if 'granularity_options' not in st.session_state:
        st.session_state.granularity_options = ('standard', 'boost', 'max')
    
    if 'granularity_selection_key' not in st.session_state:
        st.session_state.granularity_selection_key = str(randint(1000, 10000000))

    if 'granularity_selection_index' not in st.session_state:
        st.session_state.granularity_selection_index = 0

    if 'granularity_dict' not in st.session_state:
        st.session_state.granularity_dict = {'standard': 0.1, 'boost': 0.3, 'max': 0.5}
    
    if 'granularity_selection' not in st.session_state:
        st.session_state.granularity_selection = st.session_state.granularity_options[st.session_state.granularity_selection_index]

    if 'viewer_options' not in st.session_state:
        st.session_state.viewer_options = ("Enhanced Image", "Original vs Enhanced", "Comparisons (interactive)", "Show All Processing Steps")
    
    if 'viewer_selection_key' not in st.session_state:
        st.session_state.viewer_selection_key = str(randint(1000, 10000000))

    if 'viewer_selection_index' not in st.session_state:
        st.session_state.viewer_selection_index = 1     

    if 'viewer_selection' not in st.session_state:
        st.session_state.viewer_selection = st.session_state.viewer_options[st.session_state.viewer_selection_index]

    if 'layout_options' not in st.session_state:
        st.session_state.layout_options = ('horizontal', 'vertical', 'auto')

    if 'layout_selection_index' not in st.session_state:
        st.session_state.layout_selection_index = 0
            
    if 'layout_selection_key' not in st.session_state:
        st.session_state.layout_selection_key = str(randint(1000, 10000000))
    
    if 'layout_selection' not in st.session_state:
        st.session_state.layout_selection = st.session_state.layout_options[st.session_state.layout_selection_index]

    if 'comparison_options' not in st.session_state:
        st.session_state.comparison_options = ("Original Image", "Enhanced Image", "Illumination Map", "Total Variation", "Fusion Weights", "Max Entropy Exposure", "Texture Weights", "Fine Texture Map", "Enhancement Map")

    if 'left_image_selection_key' not in st.session_state:
        st.session_state.left_image_selection_key = str(randint(1000, 10000000))

    if 'left_image_selection_index' not in st.session_state:
        st.session_state.left_image_selection_index = 0     

    if 'left_image_selection' not in st.session_state:
        st.session_state.left_image_selection = st.session_state.comparison_options[st.session_state.left_image_selection_index]

    if 'right_image_selection_key' not in st.session_state:
        st.session_state.right_image_selection_key = str(randint(1000, 10000000))

    if 'right_image_selection_index' not in st.session_state:
        st.session_state.right_image_selection_index = 1     

    if 'right_image_selection' not in st.session_state:
        st.session_state.right_image_selection = st.session_state.comparison_options[st.session_state.right_image_selection_index]

    if 'texture_weight_calculator_options' not in st.session_state:
        st.session_state.texture_weight_calculator_options = ('I', 'II', 'III', 'IV', 'V')
    
    if 'texture_weight_calculator_selection_key' not in st.session_state:
        st.session_state.texture_weight_calculator_selection_key = str(randint(1000, 10000000))

    if 'texture_weight_calculator_selection_index' not in st.session_state:
        st.session_state.texture_weight_calculator_selection_index = 0

    if 'texture_weight_calculator_dict' not in st.session_state:
        st.session_state.texture_weight_calculator_dict = {
                                'I':  ('I', CG_TOL, LU_TOL, MAX_ITER, FILL),
                                'II': ('II', CG_TOL, LU_TOL, MAX_ITER, FILL),
                                'III':('III', 0.1*CG_TOL, LU_TOL, 10*MAX_ITER, FILL),
                                'IV': ('IV', 0.5*CG_TOL, LU_TOL, MAX_ITER, FILL/2),
                                'V':  ('V', CG_TOL, LU_TOL, MAX_ITER, FILL)
                                }
    
    if 'texture_weight_calculator_selection' not in st.session_state:
        st.session_state.texture_weight_calculator_selection = st.session_state.texture_weight_calculator_options[st.session_state.texture_weight_calculator_selection_index]

    if 'base_dir' not in st.session_state:
        st.session_state.base_dir_path = BASE_DIR_PATH

    if 'examples_dir' not in st.session_state:
        st.session_state.examples_dir = EXAMPLES_DIR_PATH
        
    if 'npy_dir' not in st.session_state:
        st.session_state.npy_dir = NPY_DIR_PATH

    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = DATA_DIR_PATH

    if 'image_dir' not in st.session_state:
        st.session_state.image_dir = IMAGE_DIR_PATH

    if 'debug_file_path' not in st.session_state:
        st.session_state.debug_file_path = DEBUG_FILE_PATH

    if 'debug' not in st.session_state:
        st.session_state.debug = False

    if 'low_resources' not in st.session_state:
        st.session_state.low_resources = False

    if 'cache_checked' not in st.session_state:
        st.session_state.cache_checked = False

    if 'data_checked' not in st.session_state:
        st.session_state.data_checked = False

    # if 'left_image_selection' not in st.session_state:
    #     st.session_state.left_image_selection = "Original Image"
        
    # if 'right_image_selection' not in st.session_state:
    #     st.session_state.right_image_selection = "Enhanced Image"     

    # if 'left_image_selection_index' not in st.session_state:
    #     st.session_state.left_image_selection_index = 0

    # if 'right_image_selection_index' not in st.session_state:
    #     st.session_state.right_image_selection_index = 1

    if 'query_params' not in st.session_state:
        st.session_state.query_params = {}
        st.session_state.query_params['console'] = False
        st.session_state.query_params['resources'] = False
        st.session_state.query_params['cache'] = False

    if 'total_main_runs' not in st.session_state:
        st.session_state.total_main_runs = 0

    if 'fImage' not in st.session_state:
        st.session_state.fImage = None

    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = str(randint(1000, 1000000))

    if 'fImage_is_not_None' not in st.session_state:
        st.session_state.fImage_is_not_None = False

    if 'completed_main_runs' not in st.session_state:
        st.session_state.completed_main_runs = 0

    if 'incomplete_main_runs' not in st.session_state:
        st.session_state.incomplete_main_runs = 0

    if 'total_app_runs' not in st.session_state:
        st.session_state.total_app_runs = 0

    if 'last_run_exited_early' not in st.session_state:
        st.session_state.last_run_exited_early = False

    if 'source_last_updated' not in st.session_state:
        st.session_state.source_last_updated = 'local'
        
    if 'completed_app_runs' not in st.session_state:
        st.session_state.completed_app_runs = 0

    if 'auto_reloads' not in st.session_state:
        st.session_state.auto_reloads = 0

    if 'purge_count' not in st.session_state:
        st.session_state.purge_count = 0

    if 'memmapped' not in st.session_state:
        st.session_state.memmapped = {}

    if 'input_selection' not in st.session_state:
        st.session_state.input_selection = ''

    if 'input_example_path' not in st.session_state:
        st.session_state.input_example_path = ''

    if 'input_data_path' not in st.session_state:
        st.session_state.input_data_path = ''

    if 'input_key' not in st.session_state:
        st.session_state.input_key = ''

    if 'input_file_name' not in st.session_state:
        st.session_state.input_file_name = ''

    if 'input_file_path' not in st.session_state:
        st.session_state.input_file_path = ''

    if 'input_source' not in st.session_state:
        st.session_state.input_source = 'E'

    if 'input_file_ext' not in st.session_state:
        st.session_state.input_file_ext = ''

    if 'input_shape' not in st.session_state:
        st.session_state.input_shape = (-1,-1,-1)

    if 'cache_clearance' not in st.session_state:
        st.session_state.cache_clearance = False
    else:
        st.session_state.cache_clearance = False

    if 'show_resource_usage' not in st.session_state:
        st.session_state.show_resource_usage = False
    else:
        st.session_state.show_resource_usage = False

    if 'show_console' not in st.session_state:
        st.session_state.show_console = False

    if 'console_out' not in st.session_state:
        st.session_state.console_out = ''

    if 'console_in' not in st.session_state:
        st.session_state.console_in = ''

    if 'keys_to_npy' not in st.session_state:
        st.session_state.keys_to_npy = {}

    if 'keys_to_images' not in st.session_state:
        st.session_state.keys_to_images = {}

    if 'keys_to_shape' not in st.session_state:
        st.session_state.keys_to_shape = {}

    if 'named_keys' not in st.session_state:
        st.session_state.named_keys = {}
    
    if 'exposure_ratio' not in st.session_state:
        st.session_state.exposure_ratio = -1

    if 'mmap_wrefs' not in st.session_state:
        st.session_state.mmap_wrefs = {}       # store weak references to variables bound via memory map to file

    if 'mmap_file_wrefs_lookup' not in st.session_state:  # key: name of weak ref variable, value: (name of stong ref, name of memmapped file)
        st.session_state.mmap_file_wref_lookup = {} 

    if 'saved_images' not in st.session_state:
        st.session_state.saved_images = {}

    if 'paths' not in st.session_state:
        st.session_state.paths = {}

    if 'keys_' not in st.session_state:
        st.session_state.keys_ = Keys("initializing", 
                                     default_granularity, 
                                     default_kernel_parallel, 
                                     default_kernel_orthogonal,
                                     default_sharpness, 
                                     default_texture_style, 
                                     default_smoothness, 
                                     default_power, 
                                     default_a,
                                     default_b,
                                     default_exposure_ratio_in, 
                                     default_color_gamma,
                                     default_lo,
                                     default_hi)

    if 'exposure_ratios' not in st.session_state:
        st.session_state.exposure_ratios = {}

    set_debug()

    st.session_state.state_history = pd.DataFrame(data={
                                                            "timestamp" : [timestamp()],
                                                            "tag" : ['session.py|initialize_session|236'],
                                                            "input_key" : st.session_state.input_key,
                                                            "input_source" : st.session_state.input_source,
                                                            "source_last_updated" : st.session_state.source_last_updated,
                                                            "fImage_is_not_None" : st.session_state.fImage_is_not_None,
                                                            "input_file_path" : st.session_state.input_file_path,
                                                            "upload_key" : st.session_state.upload_key,
                                                        }
                                                    )

class Keys:

    def __init__(self, image_input_key, 
                    scale, 
                    kernel_parallel, 
                    kernel_orthogonal,
                    sharpness, 
                    texture_style, 
                    lamda, 
                    power, 
                    exposure_ratio_in, 
                    color_gamma,
                    a,
                    b,
                    min_gain,
                    max_gain):
        
        self.image_input_key = image_input_key
        self.scale = scale
        self.kernel_parallel = kernel_parallel
        self.kernel_orthogonal = kernel_orthogonal
        self.sharpness = sharpness
        self.texture_style = texture_style
        self.lamda = lamda
        self.power = power
        self.exposure_ratio_in = exposure_ratio_in
        self.color_gamma = color_gamma
        self.a = a
        self.b = b
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.image_reduced_key = f'{self.image_input_key}{int(100*scale):02d}'
        self.gradients_key = f'{self.image_reduced_key}G'
        self.convolutions_key = f'{self.gradients_key}C{self.kernel_parallel}{self.kernel_orthogonal}'
        self.texture_weights_key = f'{self.convolutions_key}{int(1000*self.sharpness):003d}{self.texture_style}'   
        self.texture_weights_map_key = f'{self.texture_weights_key}WM'    
        self.total_variation_map_key = f'{self.texture_weights_key}VM'    
        self.smoother_output_fullsize_key = f'{self.texture_weights_key}{int(1000*self.lamda):003d}'
        self.fine_texture_map_key = f'{self.smoother_output_fullsize_key}FM'
        self.fusion_weights_key = f'{self.smoother_output_fullsize_key}{int(1000*self.power):003d}'
        self.exposure_ratio_in_key = f'{self.smoother_output_fullsize_key}{self.exposure_ratio_in}{self.min_gain}{self.max_gain}{int(1000*self.a):0004d}{int(10000*self.b):00005d}'
        self.exposure_ratio_out_key = f'{self.exposure_ratio_in_key}R'
        self.adjusted_exposure_key = f'{self.exposure_ratio_in_key}{int(1000*self.color_gamma):003d}'  # include camera parameter values a & b for completeness
        self.enhanced_image_key = f'{self.adjusted_exposure_key}{int(1000*self.power):003d}EI'  # depends on all parameters
        self.enhancement_map_key = f'{self.enhanced_image_key}EM'


    def __repr__(self):

        output_str = f'image_input_key: {self.image_input_key}                                    \n'
        output_str += f'scale: {self.scale}                                                       \n'
        output_str += f'kernel_parallel: {self.kernel_parallel}                                                    \n'
        output_str += f'kernel_orthogonal: {self.kernel_orthogonal}                                                    \n'
        output_str += f'sharpness: {self.sharpness}                                                    \n'
        output_str += f'texture_style: {self.texture_style}                                                    \n'
        output_str += f'lamda: {self.lamda}                                                    \n'
        output_str += f'power: {self.power}                                                    \n'
        output_str += f'exposure_ratio_in: {self.exposure_ratio_in}                                                    \n'
        output_str += f'color_gamma: {self.color_gamma}                                                    \n'
        output_str += f'a: {self.a}                                                    \n'
        output_str += f'b: {self.b}                                                    \n'
        output_str += f'min_gain: {self.min_gain}                                                    \n'
        output_str += f'max_gain: {self.max_gain}                                                    \n'                           
        output_str += f'gradients_key: {self.gradients_key}                                       \n'
        output_str += f'convolutions_key: {self.convolutions_key}                                 \n'
        output_str += f'texture_weights_key: {self.texture_weights_key}                           \n'
        output_str += f'smoother_output_fullsize_key: {self.smoother_output_fullsize_key}         \n'
        output_str += f'exposure_ratio_in_key: {self.exposure_ratio_in_key}                       \n'
        output_str += f'exposure_ratio_out_key: {self.exposure_ratio_out_key}                     \n'
        output_str += f'fusion_weights_key: {self.fusion_weights_key}                             \n'
        output_str += f'adjusted_exposure_key: {self.adjusted_exposure_key}                       \n'
        output_str += f'enhancement_map_key: {self.enhancement_map_key}                           \n'
        output_str += f'enhanced_image_key: {self.enhanced_image_key}                             \n'
        
        return output_str

    def __str__(self):
 
        return self.__repr__()


    def __print__(self):
        print(f'image_input_key: {self.image_input_key}')
        print(f'gradients_key: {self.gradients_key}')
        print(f'convolutions_key: {self.convolutions_key}')
        print(f'texture_weights_key: {self.texture_weights_key}')
        print(f'smoother_output_fullsize_key: {self.smoother_output_fullsize_key}')
        print(f'exposure_ratio_in_key: {self.exposure_ratio_in_key}')
        print(f'exposure_ratio_out_key: {self.exposure_ratio_out_key}')
        print(f'fusion_weights_key: {self.fusion_weights_key}')
        print(f'adjusted_exposure_key: {self.adjusted_exposure_key}')
        print(f'enhancement_map_key: {self.enhancement_map_key}')
        print(f'enhanced_image_key: {self.enhanced_image_key}')