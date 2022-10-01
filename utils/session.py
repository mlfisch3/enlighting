import streamlit as st
from utils.config import NPY_DIR, IMAGE_DIR, DATA_DIR

def initialize_session():

    if 'low_resources' not in st.session_state:
        st.session_state.low_resources = False

    if 'cache_checked' not in st.session_state:
        st.session_state.cache_checked = False

    if 'data_checked' not in st.session_state:
        st.session_state.data_checked = False

    if 'query_params' not in st.session_state:
        st.session_state.query_params = {}
        st.session_state.query_params['console'] = False

    if 'total_main_runs' not in st.session_state:
        st.session_state.total_main_runs = 0

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

    if 'input_key' not in st.session_state:
        st.session_state.input_key = ''

    if 'input_file_name' not in st.session_state:
        st.session_state.input_file_name = ''

    if 'input_file_path' not in st.session_state:
        st.session_state.input_file_path = ''

    if 'input_source' not in st.session_state:
        st.session_state.input_source = ''

    if 'input_file_ext' not in st.session_state:
        st.session_state.input_file_ext = ''

    if 'show_console' not in st.session_state:
        st.session_state.show_console = False

    if 'console_out' not in st.session_state:
        st.session_state.console_out = ''

    if 'command' not in st.session_state:
        st.session_state.command = ''

    if 'npy_dir' not in st.session_state:
        st.session_state.npy_dir = NPY_DIR

    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = DATA_DIR

    if 'image_dir' not in st.session_state:
        st.session_state.image_dir = IMAGE_DIR

    if 'keys_to_npy' not in st.session_state:
        st.session_state.keys_to_npy = {}

    if 'keys_to_images' not in st.session_state:
        st.session_state.keys_to_images = {}

    if 'named_keys' not in st.session_state:
        st.session_state.named_keys = {}
    
    # if 'input_shape' not in st.session_state:
    #     st.session_state.input_shape = (-1,-1)

    if 'exposure_ratio' not in st.session_state:
        st.session_state.exposure_ratio = -1

    if 'mmap_wrefs' not in st.session_state:
        st.session_state.mmap_wrefs = {}       # store weak references to variables bound via memory map to file

    if 'mmap_file_wrefs_lookup' not in st.session_state:  # key: name of weak ref variable, value: (name of stong ref, name of memmapped file)
        st.session_state.mmap_file_wref_lookup = {} 

    if 'saved_images' not in st.session_state:
        st.session_state.saved_images = {}

    # if 'last_image_key' not in st.session_state:
    #     st.session_state.last_image_key = ''

    if 'keys_' not in st.session_state:
        st.session_state.keys_ = {}

    if 'exposure_ratios' not in st.session_state:
        st.session_state.exposure_ratios = {}

    # if 'active_keys' not in st.session_state:
    #     st.session_state.active_keys = {}
    #     st.session_state.active_keys['image_input'] = ('-1','-1','-1')       #  input_file_name: load_image()  -->  image_np, image_01, image_01_maxRGB
    #     st.session_state.active_keys['image_reduced'] = '-1'            #  image_01_maxRGB, scaling: imresize() --> image_01_maxRGB_reduced
    #     st.session_state.active_keys['gradients'] = ('-1','-1')        #  image_01_maxRGB_reduced: delta() --> gradient_v, gradient_h
    #     st.session_state.active_keys['convolutions'] = ('-1','-1')     #   gradient_v, gradient_h, kernel_shape: convolve()   -->  convolved_v, convolved_h
    #     st.session_state.active_keys['texture_weights'] = ('-1','-1')  #   gradient_v, gradient_h, convolved_v, convolved_h, sharpness, texture_style:  calculate_texture_weights() --> texture_weights_v, texture_weights_h
    #     st.session_state.active_keys['A'] = ('-1')                     #  texture_weights_v, texture_weights_h, lamda:   construct_map_cyclic() --> A
    #     st.session_state.active_keys['image_01_maxRGB_reduced_smooth'] = ('-1')  #  A, image_01_maxRGB_reduced: solve_sparse_system() --> image_01_maxRGB_reduced_smooth
    #     st.session_state.active_keys['illumination_map'] = ('-1')                #  image_01_maxRGB_reduced_smooth: imresize()  --> image_01)maxRGB_smooth --> illumination_map
    #     st.session_state.active_keys['gradients_fullsize'] = ('-1','-1')         #  gradient_v, gradient_h: imresize()  -->   gradient_v_fullsize, gradient_h_fullsize
    #     st.session_state.active_keys['convolutions_fullsize'] = ('-1','-1')     # 
    #     st.session_state.active_keys['texture_weights_fullsize'] = ('-1','-1')  #   texture_weights_v, texture_weights_h: imresize()  -->    texture_weights_v_fullsize, texture_weights_h_fullsize
    #     st.session_state.active_keys['smoother_output_fullsize'] = ('-1')       #   illumination_map, texture_weights_v_fullsize, texture_weights_h_fullsize, gradient_v_fullsize, gradient_h_fullsize
    #     st.session_state.active_keys['fusion_weights'] = ('-1')                 #   illumination_map, power:  calculate_fusion_weights()  --> fusion_weights
    #     st.session_state.active_keys['adjusted_exposure_params'] = ('-1')        #     exposure_ratio, color_gamma, lo, hi
    #     st.session_state.active_keys['image_adjusted_exposure'] = ('-1')        #   image_np, exposure_ratio, color_gamma, lo, hi  --> image_exposure_adjusted, exposure_ratio_auto
    #     st.session_state.active_keys['enhancement_output'] = ('-1')             #   fusion_weights, image_np, image_exposure_adjusted  fuse_image() --->  enhancement_map, enhanced_image


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
        self.exposure_ratio_in_key = f'{self.smoother_output_fullsize_key}{self.exposure_ratio_in}{self.min_gain}{self.max_gain}{int(1000*self.a):0004d}{int(10000*self.b):00005d}'
        self.exposure_ratio_out_key = f'{self.exposure_ratio_in_key}R'
        self.fusion_weights_key = f'{self.smoother_output_fullsize_key}{int(1000*self.power):003d}'
        self.adjusted_exposure_key = f'{self.exposure_ratio_in_key}{int(1000*self.color_gamma):003d}'  # include camera parameter values a & b for completeness
        self.enhancement_map_key = f'{self.fusion_weights_key}EM'
        self.enhanced_image_key = f'{self.fusion_weights_key}EI'
        

    def __repr__(self):
        output_str = f'image_input_key: {self.image_input_key}                                    \n'
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