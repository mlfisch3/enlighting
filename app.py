import streamlit as st
import subprocess
import cv2
import numpy as np
from matplotlib import image as img
import datetime
from psutil import virtual_memory, swap_memory, Process
import os
from os import getpid
import sys
from random import randint
import gc
from utils.io_tools import change_extension, load_binary, load_image, mkpath
from utils.config import NPY_DIR_PATH, IMAGE_DIR_PATH, EXAMPLE_PATHS, EXAMPLES, DATA_DIR_PATH, EXAMPLES_DIR_PATH, DEBUG_FILE_PATH
from utils.sodef import bimef
from utils.array_tools import float32_to_uint8, uint8_to_float32, normalize_array, array_info#, mono_float32_to_rgb_uint8
from utils.logging import timestamp, log_memory
from utils.mm import get_mmaps, get_weakrefs, references_dead_object, clear_cache, clear_data, clear
from utils.session import Keys, report_runs
import weakref
from pathlib import Path
from streamlit_image_comparison import image_comparison

def set_source(source='local'):
    print('\n')
    print('↓↓↓↓↓↓↓↓↓↓↓↓')

    # if all([source == 'local', st.session_state.source_last_updated == 'upload']):
    #     st.session_state.upload_key = str(randint(1000, 10000000))

    st.session_state.source_last_updated = source

    if st.session_state.source_last_updated == 'local':
        st.session_state.input_source = 'E'
        st.session_state.upload_key = str(randint(1000, 10000000))
    else:
        st.session_state.input_source = 'U'

    print('\n')
#    print(globals().keys())
    print('\n')
#    print(f'[{timestamp()}|app.py|set_source_31]')
    report_runs('app.py|set_source|59')   
    print('↑↑↑↑↑↑↑↑↑↑↑↑')
    print('\n')

def run_command():#command, ):
    #print(f'[{timestamp()}] command: {command}')
    print(f'[{timestamp()}] st.session_state.console_in: {st.session_state.console_in}')
    try:
        st.session_state.console_out = str(subprocess.check_output(st.session_state.console_in, shell=True, text=True))
        st.session_state.console_out_timestamp = f'{timestamp()}'
    except subprocess.CalledProcessError as e:
        #print(vars(e))
        st.session_state.console_out = f'exited with error\nreturncode: {e.returncode}\ncmd: {e.cmd}\noutput: {e.output}\nstderr: {e.stderr}'
        st.session_state.console_out_timestamp = f'{timestamp()}'

    print(f'[{timestamp()}] st.session_state.console_out: {st.session_state.console_out}')

def run_app(default_power=0.5, 
            default_smoothness=0.3, 
            default_texture_style='I',
            default_kernel_parallel=5, 
            default_kernel_orthogonal=1,
            default_sharpness=0.001,
            CG_TOL=0.1, 
            LU_TOL=0.015, 
            MAX_ITER=50, 
            FILL=50,
            default_dim_size=(50), 
            default_dim_threshold=0.5, 
            default_a=-0.3293, 
            default_b=1.1258, 
            default_lo=1, 
            default_hi=7,
            default_exposure_ratio=-1, 
            default_color_gamma=0.3981):

    print('══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗')

#    print('╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩╩')
    if st.session_state.debug:
        with st.expander("session_state 0:"):
            st.write(st.session_state)

    st.session_state.total_app_runs += 1

    #print(f'[{timestamp()}|app.py|77]')
    report_runs('app.py|run_app|105')
    container = st.sidebar.container()
    with st.sidebar:
        with st.expander("About", expanded=True):
            st.markdown("<h1 style='text-align: left; color: white;'>Welcome to Enlighting</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: left; color: yellow'>Check out the examples to see what's possible</h3>", unsafe_allow_html=True)

            st.markdown("<h3 style='text-align: left; color: yellow'>Upload your own image to enhance</h3>", unsafe_allow_html=True)
        pid = getpid()
        placeholder = st.empty()
        if st.session_state.show_console:
            with placeholder.container():
                with st.expander("console"):
                    with st.form('console'):
                        command = st.text_input(f'[{pid}] {timestamp()}', str(st.session_state.console_in), key="console_in")
                        submitted = st.form_submit_button('run', help="coming soon", on_click=run_command)#, args=(command,))

                        st.write(f'IN: {command}')
                        #st.write(f'IN: {st.session_state.console_in}')
                        
                        st.text(f'OUT:\n{st.session_state.console_out}')
                        #with st.expander('OUT'):
                            #st.text(f'{st.session_state.console_out}')
                            # st.text(f'OUT: {console_out}')
        else:
             placeholder.empty()
            
        if st.session_state.low_resources:
            clear_cache()
            st.session_state.low_resources = False

        with st.expander(f'{Process(pid).memory_info()[0]/float(2**20):.2f}'):

            with st.form("Clear"):
                st.session_state.cache_checked = st.checkbox("Clear Cache", help="coming soon", value=False)
                st.session_state.data_checked = st.checkbox("Clear Data", help="coming soon", value=False)
                st.form_submit_button("Clear", on_click=clear, args=([st.session_state.cache_checked, st.session_state.data_checked]), help="coming soon")
        with st.expander("Input", expanded=True):
            source_tab0, source_tab1 = st.tabs(["• Example Selector","• Image Uploader"])
            with source_tab0:
                report_runs('app.py|input_selection|138') 
                st.session_state.input_selection = st.radio("Select Example:", EXAMPLES, horizontal=True, on_change=set_source, kwargs=dict(source='local'), help="coming soon", key='local_example')
                #print(f'[{timestamp()}|app.py|input_selection|113]')
                report_runs('app.py|input_selection|141')
                st.session_state.input_example_path = EXAMPLE_PATHS[st.session_state.input_selection]
                #print(f'[{timestamp()}|app.py|input_example_path|116]')
            with source_tab1:
                report_runs('app.py|input_example_path|143')
                fImage = st.file_uploader("Or Upload Your Own Image:", on_change=set_source, kwargs=dict(source='upload'), help="coming soon", key=st.session_state.upload_key) #("Process new image:")
                #print(f'[{timestamp()}|app.py|st.file_uploader|119]')
                report_runs('app.py|st.file_uploader|147')

        if fImage is not None:            
            input_file_name = str(fImage.__dict__['name'])
            input_data_path = os.path.join(st.session_state.data_dir, input_file_name)
            if not os.path.isfile(input_data_path):
                np_array = np.frombuffer(fImage.getvalue(), np.uint8)
                image_u8 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)                
                st.session_state.saved_images[input_data_path] = cv2.imwrite(input_data_path, image_u8)
                del np_array, image_u8

            st.session_state.input_file_name = input_file_name
            st.session_state.input_data_path = input_data_path
            st.session_state.upload_key = str(randint(1000, 10000000))
            fImage = None
            st.experimental_rerun()

        if st.session_state.source_last_updated == 'upload':
            st.session_state.input_file_path = st.session_state.input_data_path
        else:
            st.session_state.input_file_path = st.session_state.input_example_path

        st.session_state.input_file_name = os.path.basename(st.session_state.input_file_path)
        container.write(f'Image Name:   {st.session_state.input_file_name}')
    
    
    
    if st.session_state.debug:
        with st.expander("session_state 0.2:"):
            st.write(st.session_state)

    with st.sidebar:
        
        with st.expander("Settings"):
            with st.form('Settings'):                
                submitted = st.form_submit_button('Apply Changes', help="coming soon")

                param_tab0, param_tab1, param_tab2, param_tab3 = st.tabs(["• Viewer", "• Basic", "• Exposure", "• Weights"])
                
                with param_tab0:
                    viewer_selection = st.radio("Viewer", ("comparison", "enhanced", "side-by-side", "all"), help="Coming soon", key="viewer_selection")

                    # show_enhanced_only_checkbox = st.checkbox('Show Enhanced Image Only', value=True, help="coming soon")
                    # show_all_checkbox = st.checkbox('Show All Process Images', value=True, help="coming soon")
        
                with param_tab1:
                    granularity_selection = st.radio("Illumination detail", ('standard', 'boost', 'max'), horizontal=True, help="coming soon")
                    granularity_dict = {'standard': 0.1, 'boost': 0.3, 'max': 0.5}
                    granularity = granularity_dict[granularity_selection]
                    power = float(st.text_input(f'Power     (default = {default_power})', str(default_power), help="coming soon"))

                with param_tab2:

                    a = float(st.text_input(f'Camera A   (default = {default_a})', str(default_a), help="coming soon"))
                    b = float(st.text_input(f'Camera B   (default = {default_b})', str(default_b), help="coming soon"))
                    lo = int(st.text_input(f'Min Gain   (default = {default_lo})', str(default_lo), help="Sets lower bound of search range for optimal Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                    hi = int(st.text_input(f'Max Gain   (default = {default_hi})', str(default_hi), help="Sets upper bound of search range for optimal Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                    exposure_ratio_in = float(st.text_input(f'Exposure Ratio   (default = -1 (auto))', str(default_exposure_ratio), help="coming soon"))
                    color_gamma = float(st.text_input(f'Color Gamma   (default = {default_color_gamma})', str(default_color_gamma), help="coming soon"))
                with param_tab3:
                    kernel_parallel = int(st.text_input(f'Kernel Parallel   (default = {default_kernel_parallel})', str(default_kernel_parallel), help="coming soon"))
                    kernel_orthogonal = int(st.text_input(f'Kernel Orthogonal   (default = {default_kernel_orthogonal})', str(default_kernel_orthogonal), help="coming soon")) 
                    smoothness = float(st.text_input(f'Smoothness   (default = {default_smoothness})', str(default_smoothness), help="coming soon"))
                    sharpness = float(st.text_input(f'Sharpness   (default = {default_sharpness})', str(default_sharpness), help="coming soon"))
                    # texture_weight_calculator = st.radio("Select texture weight calculator", ('I', 'II', 'III', 'IV', 'V'), horizontal=True, help="coming soon") 
                    # texture_weight_calculator_dict = {
                    #             'I':  ('I', CG_TOL, LU_TOL, MAX_ITER, FILL),
                    #             'II': ('II', CG_TOL, LU_TOL, MAX_ITER, FILL),
                    #             'III':('III', 0.1*CG_TOL, LU_TOL, 10*MAX_ITER, FILL),
                    #             'IV': ('IV', 0.5*CG_TOL, LU_TOL, MAX_ITER, FILL/2),
                    #             'V':  ('V', CG_TOL, LU_TOL, MAX_ITER, FILL)
                    #             }
                    texture_weight_calculator = st.radio("Select texture weight calculator", ('I', 'II', 'III'), horizontal=True, help="coming soon") 
                    texture_weight_calculator_dict = {
                                'I':  ('I', CG_TOL, LU_TOL, MAX_ITER, FILL),
                                'II': ('II', CG_TOL, LU_TOL, MAX_ITER, FILL),
                                'III':('III', 0.1*CG_TOL, LU_TOL, 10*MAX_ITER, FILL)
                                }
                    texture_style, cg_tol, lu_tol, max_iter, fill = texture_weight_calculator_dict[texture_weight_calculator]


    start = datetime.datetime.now()
    #print(f'[{timestamp()}|app.py|156]')
    report_runs('app.py|run_app|198')
    image_input_key = load_image(st.session_state.input_file_path, st.session_state.input_source)#, reload_previous=st.session_state.last_run_exited_early)  ###############################<<<<<<<<<<<<<<<<<<<
    report_runs('app.py|run_app|201')
    

    if st.session_state.debug:
        with st.expander("session_state 0.3:"):
            st.write(st.session_state)

    st.session_state.keys_ = Keys(image_input_key, 
                                 granularity, 
                                 kernel_parallel, 
                                 kernel_orthogonal,
                                 sharpness, 
                                 texture_style, 
                                 smoothness, 
                                 power, 
                                 a,
                                 b,
                                 exposure_ratio_in, 
                                 color_gamma,
                                 lo,
                                 hi)

    report_runs('app.py|run_app|222')

    if st.session_state.debug:
        with st.expander("session_state 0.4:"):
            st.write(st.session_state.keys_)

    input_image = st.session_state.memmapped[image_input_key]
    image_np, image_01, image_01_maxRGB = input_image
    shape = image_01_maxRGB.shape
    st.session_state.keys_to_shape[image_input_key] = shape
    st.session_state.input_shape = st.session_state.keys_to_shape[image_input_key]
    #container.write(f'Image Size:   {str(st.session_state.input_shape[0])}   ×   {str(st.session_state.input_shape[1])}')
    container.write(f'Image Size:   {str(shape[0])}   ×   {str(shape[1])}')
    report_runs('app.py|run_app|231')
   
    if st.session_state.keys_.enhanced_image_key not in st.session_state.memmapped:

        report_runs('app.py|run_app|235')

        enhanced_image, exposure_ratio_out = bimef(image_01, 
                                                       image_01_maxRGB,
                                                       exposure_ratio_in=exposure_ratio_in, 
                                                       scale=granularity, 
                                                       power=power, 
                                                       lamda=smoothness, 
                                                       a=a, 
                                                       b=b, 
                                                       lo=lo, 
                                                       hi=hi,
                                                       texture_style=texture_style, 
                                                       kernel_shape=(kernel_parallel, kernel_orthogonal),
                                                       sharpness=sharpness, 
                                                       color_gamma=color_gamma,
                                                       CG_TOL=cg_tol, 
                                                       LU_TOL=lu_tol, 
                                                       MAX_ITER=max_iter, 
                                                       FILL=fill,
                                                       return_texture_weights=True)

    report_runs('app.py|run_app|257')

    end = datetime.datetime.now()
    process_time = (end - start).total_seconds()
    print(f'[{datetime.datetime.now().isoformat()}]  Processing time: {process_time:.5f} s')
    sys.stdout.flush()

    
    if st.session_state.debug:
        with st.expander("session_state 0.5:"):
            st.write(st.session_state)

# Welcome to Light-Fix, an image enhancement application
# Welcome to Light-Fix, an image enhancement application that shines new light on backlit photos
    # with st.expander(" ", expanded=True):
    #     st.markdown("<h1 style='text-align: left; color: white;'>Welcome to Light-Fix</h1>", unsafe_allow_html=True)
    #     st.markdown("<h3 style='text-align: left; color: yellow'>Check out the examples to see what's possible</h3>", unsafe_allow_html=True)

    #     st.markdown("<h3 style='text-align: left; color: yellow'>Upload your own image to enhance</h3>", unsafe_allow_html=True)

    shape = st.session_state.memmapped[image_input_key][2].shape
    
    granularity_param_str = f'_{granularity*100:.0f}'
    convolution_param_str = granularity_param_str + f'_{kernel_parallel:d}_{kernel_orthogonal:d}'
    texture_param_str = convolution_param_str + f'_{sharpness*1000:.0f}_{texture_weight_calculator:s}'
    smooth_param_str = texture_param_str + f'_{smoothness*100:.0f}'
    fusion_param_str = smooth_param_str + f'_{color_gamma*100:.0f}_{power*100:.0f}_{-a*1000:.0f}_{b*1000:.0f}_{exposure_ratio_in*100:.0f}'

    input_file_name = st.session_state.input_file_name
    input_file_ext = '.' + str(input_file_name.split('.')[-1])
    input_file_basename = input_file_name.replace(input_file_ext, '')

    if viewer_selection == "comparison":

        with st.expander(" "):
            with st.form("Comparison"):
                left_image_selection = st.radio("Select Left", ("Original Image", "Enhanced Image", "Illumination Map", "Total Variation", "Fusion Weights", "Max Entropy Exposure", "Texture Weights", "Fine Texture Map", "Enhancement Map"), help="Coming Soon", key="compare_left")
                right_image_selection = st.radio("Select Right", ("Enhanced Image", "Original Image", "Illumination Map", "Total Variation", "Fusion Weights", "Max Entropy Exposure", "Texture Weights", "Fine Texture Map", "Enhancement Map"), help="Coming Soon", key="compare_right")
                submitted = st.form_submit_button("Update Comparison")

        paths = {
                    "Original Image" : st.session_state.input_file_path, 
                    "Enhanced Image" : st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key], 
                    "Illumination Map" : st.session_state.keys_to_images[st.session_state.keys_.smoother_output_fullsize_key], 
                    "Total Variation" : st.session_state.keys_to_images[st.session_state.keys_.total_variation_map_key], 
                    "Fusion Weights" : st.session_state.keys_to_images[st.session_state.keys_.fusion_weights_key], 
                    "Max Entropy Exposure" : st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key], 
                    "Texture Weights" : st.session_state.keys_to_images[st.session_state.keys_.texture_weights_map_key], 
                    "Fine Texture Map" : st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key],
                    "Enhancement Map" : st.session_state.keys_to_images[st.session_state.keys_.enhancement_map_key]
                }
                

        left_image = cv2.cvtColor(cv2.imread(paths[left_image_selection]), cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(paths[right_image_selection]), cv2.COLOR_BGR2RGB)
    
        image_comparison(
            img1=left_image,
            img2=right_image,
            label1=left_image_selection,
            label2=right_image_selection,
            width=1250,
            show_labels=True
            )


    elif viewer_selection == 'enhanced':

        st.markdown("<h3 style='text-align: center; color: white;'>Enhanced Image</h3>", unsafe_allow_html=True)
        st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]), channels="BGR")
        output_fused_file_name = input_file_basename + '_FUSION' + fusion_param_str + input_file_ext    
        output_fused_file_name = st.text_input('Download Enhanced Image As', output_fused_file_name)
        ext = '.' + output_fused_file_name.split('.')[-1]
        button = st.download_button(label = "Download Enhanced Image",  
                                        data=Path(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]).read_bytes(),
                                        file_name = output_fused_file_name, 
                                        mime = f"image/{input_file_ext}", 
                                        key='ei'
                                   )   

    elif viewer_selection == 'side-by-side':
        col1, col2 = st.columns(2)

        with col1:        
            
            st.markdown("<h3 style='text-align: center; color: white;'>Original Image</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.input_file_path), channels="BGR")

        with col2:

            st.markdown("<h3 style='text-align: center; color: white;'>Enhanced Image</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]), channels="BGR")

            output_fused_file_name = input_file_basename + '_FUSION' + fusion_param_str + input_file_ext    
            output_fused_file_name = st.text_input('Download Enhanced Image As', output_fused_file_name)
            ext = '.' + output_fused_file_name.split('.')[-1]
            button = st.download_button(label = "Download Enhanced Image",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]).read_bytes(),
                                            file_name = output_fused_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='ei'
                                       )   

    else:

        col10, col20, col30 = st.columns(3)


        with col10:
            
            st.markdown("<h3 style='text-align: center; color: white;'>Original Image</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.input_file_path), channels="BGR")#, caption=['↓'])
            #st.markdown("<h5 style='text-align: center; color: white;'>↓</h5>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: white;'>Texture Weights</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.texture_weights_map_key], cv2.IMREAD_UNCHANGED), clamp=True)
            st.markdown("<h3 style='text-align: center; color: white;'>Total Variation</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.total_variation_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

        with col20:

            st.markdown("<h3 style='text-align: center; color: white;'>Enhancement Layer</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.enhancement_map_key]), channels="BGR")
            st.markdown("<h3 style='text-align: center; color: white;'>Max Entropy Exposure</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key]), channels="BGR")            
            st.markdown("<h3 style='text-align: center; color: white;'>Illumination Map</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.smoother_output_fullsize_key], cv2.IMREAD_UNCHANGED), clamp=True)

        with col30:
        
            st.markdown("<h3 style='text-align: center; color: white;'>Enhanced Image</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]), channels="BGR")
            st.markdown("<h3 style='text-align: center; color: white;'>Fusion Weights</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fusion_weights_key], cv2.IMREAD_UNCHANGED), clamp=True)          
            st.markdown("<h3 style='text-align: center; color: white;'>Texture Map</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key], cv2.IMREAD_UNCHANGED), clamp=True)
        
            
        with st.expander("Download Selected Results"):           

            output_fused_file_name = input_file_basename + '_FUSION' + fusion_param_str + input_file_ext    
            output_fused_file_name = st.text_input('Download Enhanced Image As', output_fused_file_name)
            ext = '.' + output_fused_file_name.split('.')[-1]
            button = st.download_button(label = "Download Enhanced Image",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]).read_bytes(),
                                            file_name = output_fused_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='ei'
                                       )   


            output_enhancement_layer_file_name = input_file_basename + '_SIM' + fusion_param_str + input_file_ext
            output_enhancement_layer_file_name = st.text_input('Download Enhancement Layer As', output_enhancement_layer_file_name)
            ext = '.' + output_fused_file_name.split('.')[-1]
            button = st.download_button(label = "Download Enhancement Layer",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.enhancement_map_key]).read_bytes(),
                                            file_name = output_enhancement_layer_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='em'
                                       )   

            output_fine_texture_map_file_name = input_file_basename + '_FTM' + smooth_param_str + input_file_ext
            output_fine_texture_map_file_name = st.text_input('Download Fine Texture Map As', output_fine_texture_map_file_name)
            ext = '.' + output_fine_texture_map_file_name.split('.')[-1]
            button = st.download_button(
                                            label = "Download Fine Texture Map", 
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key]).read_bytes(),
                                            file_name = output_fine_texture_map_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='ftm'
                                       )


            output_wls_map_file_name = input_file_basename + '_WLS' + texture_param_str + input_file_ext
            output_wls_map_file_name = st.text_input('Download Texture Weights As', output_wls_map_file_name)
            ext = '.' + output_wls_map_file_name.split('.')[-1]
            button = st.download_button(label = "Download Texture Weights",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.texture_weights_map_key]).read_bytes(),
                                            file_name = output_wls_map_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='tw'
                                       )
            
            output_L1_map_file_name = input_file_basename + '_L1' + texture_param_str + input_file_ext
            output_L1_map_file_name = st.text_input('Download Total Variation', output_L1_map_file_name)
            ext = '.' + output_L1_map_file_name.split('.')[-1]
            button = st.download_button(label = "Download Total Variation",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.total_variation_map_key]).read_bytes(),
                                            file_name = output_L1_map_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='tv'
                                       )


            output_illumination_map_file_name = input_file_basename + '_ILL' + smooth_param_str + input_file_ext
            output_illumination_map_file_name = st.text_input('Download Illumination Map As', output_illumination_map_file_name)
            ext = '.' + output_illumination_map_file_name.split('.')[-1]
            button = st.download_button(label = "Download Illumination Map",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.smoother_output_fullsize_key]).read_bytes(),
                                            file_name = output_illumination_map_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='ill'
                                       )

            output_fusion_weights_file_name = input_file_basename + '_FW' + fusion_param_str + input_file_ext
            output_fusion_weights_file_name = st.text_input('Download Fusion Weights As', output_fusion_weights_file_name)
            ext = '.' + output_fusion_weights_file_name.split('.')[-1]
            button = st.download_button(label = "Download Fusion Weights",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.fusion_weights_key]).read_bytes(),
                                            file_name = output_fusion_weights_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='fw'
                                       )


            output_exposure_maxent_file_name = input_file_basename + '_EME' + smooth_param_str + input_file_ext
            output_exposure_maxent_file_name = st.text_input('Download Maxent Exposure As', output_exposure_maxent_file_name)
            ext = '.' + output_exposure_maxent_file_name.split('.')[-1]
            button = st.download_button(label = "Download Maxent Exposure",  
                                            data=Path(st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key]).read_bytes(),
                                            file_name = output_exposure_maxent_file_name, 
                                            mime = f"image/{input_file_ext}", 
                                            key='maxent'
                                       )   

    with st.sidebar:
        
        if st.checkbox("View Image Info", help="coming soon"):
            col_left, col_mid, col_right = st.columns(3)

            image_np_info, image_np_info_str = array_info(image_np, print_info=False, return_info=True, return_info_str=True, name='Original Image')
            
            image_np_fused_info, image_np_fused_info_str = array_info(st.session_state.memmapped[st.session_state.keys_.enhanced_image_key], print_info=False, return_info=True, return_info_str=True, name='Fused Image')
            exposure_ratio = st.session_state.exposure_ratios[st.session_state.keys_.exposure_ratio_out_key]
            st.text(f'exposure ratio: {exposure_ratio:.4f}')
            
            entropy_change_abs = image_np_fused_info['entropy'] - image_np_info['entropy']
            entropy_change_rel = (image_np_fused_info['entropy'] / image_np_info['entropy']) - 1.0
            st.text(f'entropy change: {entropy_change_abs:.4f} ({entropy_change_rel * 100.0:.4f} %)\n')   

            st.text(image_np_info_str)
            
            st.text("\n\n\n\n\n")
           
            st.text(image_np_fused_info_str)
 
        with st.expander("Resource Usage"):
                                              
            pid = getpid()
            mem = Process(pid).memory_info()[0]/float(2**20)
            virt = virtual_memory()[3]/float(2**20)
            swap = swap_memory()[1]/float(2**20)

            print(f'[{timestamp()}] mem: {mem}') 
            if mem > 950:
                clear_cache()
            elif mem > 800:
                st.session_state.low_resources = True
            mem = Process(pid).memory_info()[0]/float(2**20)
            print(f'[{timestamp()}] mem: {mem}') 

            st.text(f'[{timestamp()}]\nPID: {pid}')
            st.text(f'rss: {mem:.2f} MB\nvirt: {virt:.2f} MB\nswap: {swap:.2f} MB')

    # with st.form("Download Batch"):
    #     st.text('Download All Output Files to Local Folder:')     

    #     colI, colII = st.columns(2)
    #     with colI:
    #         default_dir_path = DEFAULT_DIR_PATH
    #         dir_path = st.text_input('Folder Name:', default_dir_path)

    #         last_download_time = '-'

    #     colA, colB, colC, colD, colE = st.columns(5)
    #     with colA:
    #         ext_batch = st.text_input('File extension:', 'jpg')

    #         illumination_map_fullpath = os.path.join(dir_path,output_illumination_map_file_name)
    #         wls_map_fullpath = os.path.join(dir_path, output_wls_map_file_name)
    #         L1_map_fullpath = os.path.join(dir_path, output_L1_map_file_name)
    #         fine_texture_map_fullpath = os.path.join(dir_path, output_fine_texture_map_file_name)
    #         simulation_fullpath = os.path.join(dir_path,output_simulation_file_name)
    #         exposure_maxent_fullpath = os.path.join(dir_path,output_exposure_maxent_file_name)
    #         fusion_weights_fullpath = os.path.join(dir_path,output_fusion_weights_file_name)
    #         fused_fullpath = os.path.join(dir_path,output_fused_file_name)

    #     with colB:
    #         st.text('\n')
    #         st.text('\n')
    #         if st.form_submit_button('DOWNLOAD ALL', on_click=mkpath, args=[dir_path]):
    #             mkpath(dir_path)
    #             img.imsave(change_extension(wls_map_fullpath, ext_batch), texture_weights_map)
    #             img.imsave(change_extension(L1_map_fullpath, ext_batch), image_np_TV_map)
    #             img.imsave(change_extension(illumination_map_fullpath, ext_batch), illumination_map)
    #             img.imsave(change_extension(fine_texture_map_fullpath, ext_batch), image_np_fine_texture_map)
    #             img.imsave(change_extension(simulation_fullpath, ext_batch), image_np_simulation)
    #             img.imsave(change_extension(fusion_weights_fullpath, ext_batch), fusion_weights)
    #             img.imsave(change_extension(exposure_maxent_fullpath, ext_batch), image_exposure_maxent)
    #             img.imsave(change_extension(fused_fullpath, ext_batch), image_np_fused)
    #             last_download_time = datetime.datetime.now()

    #     st.text(f'last batch download completed at {last_download_time}')
    
    st.session_state.completed_app_runs += 1
    #print(f'[{timestamp()}|app.py|509]')
    report_runs('app.py|run_app|562')

    if st.session_state.debug:
        with st.expander("session_state 1.0:"):
            st.write(st.session_state)

    
    #print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')
    #print('○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○')
#    print('≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡')
    print('══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝')

    return True

if __name__ == '__main__':
    total_start = datetime.datetime.now()
    log_memory('main|run_app|B')

    run_app()

    log_memory('main|run_app|E')
    total_end = datetime.datetime.now()
    total_process_time = (total_end - total_start).total_seconds()
    print(f'[{timestamp()}]  Total processing time: {total_process_time:.5f} s')
    sys.stdout.flush()


