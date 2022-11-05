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
from utils.array_tools import float32_to_uint8, uint8_to_float32, normalize_array, array_info
from utils.logging import timestamp, log_memory
from utils.mm import get_mmaps, get_weakrefs, references_dead_object, clear_cache, clear_data, clear
from utils.session import Keys, report_runs
import weakref
from pathlib import Path
from streamlit_image_comparison import image_comparison

def set_source(source='local'):
    print('\n')
    print('↓↓↓↓↓↓↓↓↓↓↓↓')

    st.session_state.source_last_updated = source

    if st.session_state.source_last_updated == 'local':
        st.session_state.input_source = 'E'
        st.session_state.upload_key = str(randint(1000, 10000000))
    else:
        st.session_state.input_source = 'U'

    st.session_state.granularity_selection_key = str(randint(1000, 10000000))
    st.session_state.granularity_selection_index = st.session_state.granularity_options.index(st.session_state.granularity_selection)

    st.session_state.texture_weight_calculator_selection_key = str(randint(1000, 10000000))
    st.session_state.texture_weight_calculator_selection_index = st.session_state.texture_weight_calculator_options.index(st.session_state.texture_weight_calculator_selection)

    st.session_state.viewer_selection_key = str(randint(1000, 10000000))
    st.session_state.viewer_selection_index = st.session_state.viewer_options.index(st.session_state.viewer_selection)

    st.session_state.left_image_selection_key = str(randint(1000, 10000000))
    st.session_state.left_image_selection_index = st.session_state.comparison_options.index(st.session_state.left_image_selection)

    st.session_state.right_image_selection_key = str(randint(1000, 10000000))
    st.session_state.right_image_selection_index = st.session_state.comparison_options.index(st.session_state.right_image_selection)

    print('\n')
    print('\n')
    report_runs('app.py|set_source|41')   
    print('↑↑↑↑↑↑↑↑↑↑↑↑')
    print('\n')

def run_command():
    print(f'[{timestamp()}] st.session_state.console_in: {st.session_state.console_in}')
    try:
        st.session_state.console_out = str(subprocess.check_output(st.session_state.console_in, shell=True, text=True))
        st.session_state.console_out_timestamp = f'{timestamp()}'
    except subprocess.CalledProcessError as e:
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

    if st.session_state.debug:
        with st.expander("session_state 0.1:"):
            st.write(st.session_state)

    st.session_state.total_app_runs += 1

    report_runs('app.py|run_app|83')
    container = st.sidebar.container()
    with container:
        with st.expander("About ", expanded=True):
            about_tab, details_tab = st.tabs(["• Intro", "• Details"])
            with about_tab:
            
                st.markdown("<h1 style='text-align: left; color: white;'>Welcome to InLight</h1>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: left; color: yellow'>Check out the examples to see what's possible</h3>", unsafe_allow_html=True)

                st.markdown("<h3 style='text-align: left; color: yellow'>Upload your own images to enhance</h3>", unsafe_allow_html=True)
            with details_tab:
                detailed_info = f'InLight restores lighting detail to underexposed image regions.\n\r\n\rThe app is fully functional, but help descriptions are still being added.\n\rThis detailed information section will be expanded.\n\rA detailed explanation of the underlying algorithm is also in preparation'
                details = f""" 
                <style>
                p.a {{
                    font: bold 14px Arial;
                }}
                </style>
                <p class="a">{detailed_info}</p>
                """
                st.markdown(detailed_info, unsafe_allow_html=True)

        pid = getpid()
        placeholder = st.empty()
        if st.session_state.show_console:
            with placeholder.container():
                with st.expander("console", expanded=True):
                    with st.form('console'):
                        command = st.text_input(f'[{pid}] {timestamp()}', str(st.session_state.console_in), key="console_in")
                        submitted = st.form_submit_button('run', help="coming soon", on_click=run_command)

                        st.write(f'IN: {command}')
                        st.text(f'OUT:\n{st.session_state.console_out}')
                    file_name = st.text_input("File Name", "")
                    if os.path.isfile(file_name):
                        button = st.download_button(label="Download File", data=Path(file_name).read_bytes(), file_name=file_name, key="console_download")
        else:
             placeholder.empty()
            
        if st.session_state.low_resources:
            clear_cache()
            st.session_state.low_resources = False

        if st.session_state.show_resource_usage:
            with st.expander(f'{Process(pid).memory_info()[0]/float(2**20):.2f}', expanded=True):
                with st.form("Clear"):
                    st.session_state.cache_checked = st.checkbox("Clear Cache", help="coming soon", value=False)
                    st.session_state.data_checked = st.checkbox("Clear Data", help="coming soon", value=False)
                    st.form_submit_button("Clear", on_click=clear, args=([st.session_state.cache_checked, st.session_state.data_checked]), help="coming soon")

        with st.expander("Image Source", expanded=True):
            source_tab0, source_tab1 = st.tabs([f"• Image Uploader", f'• Example Selector'])

            with source_tab0:
                report_runs('app.py|input_example_path|135')
                fImage = st.file_uploader("Upload Your Own Image:", on_change=set_source, kwargs=dict(source='upload'), help="coming soon", key=st.session_state.upload_key)
                report_runs('app.py|st.file_uploader|137')

            with source_tab1:
                report_runs('app.py|input_selection|140') 
                st.session_state.input_selection = st.radio("Select Example:", EXAMPLES, horizontal=True, on_change=set_source, kwargs=dict(source='local'), help="coming soon", key='local_example')
                report_runs('app.py|input_selection|142')
                st.session_state.input_example_path = EXAMPLE_PATHS[st.session_state.input_selection]

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
        image_name_html = f""" 
        <style>
        p.a {{
            font: bold 14px Arial;
        }}
        </style>
        <p class="a">{st.session_state.input_file_name}</p>
        """
        container.markdown(image_name_html, unsafe_allow_html=True)

    if st.session_state.debug:
        with st.expander("session_state 0.2:"):
            st.write(st.session_state)
    
    with st.sidebar:
        
        with st.expander("Parameters", expanded=True):
            with st.form('Parameters'):                
                submitted = st.form_submit_button('Apply Changes', help="coming soon")

                param_tab1, param_tab2, param_tab3 = st.tabs(["• Illumination", "• Exposures", "• Power"])

                with param_tab1:
                    
                    st.session_state.granularity_selection = st.radio("Resolution", st.session_state.granularity_options, key=st.session_state.granularity_selection_key, index=st.session_state.granularity_selection_index, horizontal=True, help="coming soon")
                    granularity = st.session_state.granularity_dict[st.session_state.granularity_selection]

                    kernel_parallel = int(st.text_input(f'Kernel Parallel   (default = {default_kernel_parallel})', str(st.session_state.keys_.kernel_parallel), help="coming soon"))
                    kernel_orthogonal = int(st.text_input(f'Kernel Orthogonal   (default = {default_kernel_orthogonal})', str(st.session_state.keys_.kernel_orthogonal), help="coming soon")) 

                    smoothness = float(st.text_input(f'Smoothness   (default = {default_smoothness})', str(st.session_state.keys_.lamda), help="coming soon"))

                    sharpness = float(st.text_input(f'Sharpness   (default = {default_sharpness})', str(st.session_state.keys_.sharpness), help="coming soon"))

                    st.session_state.texture_weight_calculator_selection = st.radio("Select texture weight calculator", st.session_state.texture_weight_calculator_options, key=st.session_state.texture_weight_calculator_selection_key, index=st.session_state.texture_weight_calculator_selection_index, horizontal=True, help="coming soon") 
                    texture_style, cg_tol, lu_tol, max_iter, fill = st.session_state.texture_weight_calculator_dict[st.session_state.texture_weight_calculator_selection]

                with param_tab2:

                    a = float(st.text_input(f'Camera A   (default = {default_a})', str(default_a), help="coming soon"))
                    b = float(st.text_input(f'Camera B   (default = {default_b})', str(default_b), help="coming soon"))
                    lo = int(st.text_input(f'Min Gain   (default = {default_lo})', str(default_lo), help="Sets lower bound of search range for optimal Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                    hi = int(st.text_input(f'Max Gain   (default = {default_hi})', str(default_hi), help="Sets upper bound of search range for optimal Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                    exposure_ratio_in = float(st.text_input(f'Exposure Ratio   (default = -1 (auto))', str(default_exposure_ratio), help="coming soon"))
                    
                with param_tab3:
                   
                    power = float(st.text_input(f'Power     (default = {default_power})', str(default_power), help="coming soon"))
                    color_gamma = float(st.text_input(f'Color Spread Attenuation   (default = {default_color_gamma})', str(default_color_gamma), help="Color Spread Attenuation (CSA)).  Increasing CSR suppresses false colorization.  However, true colors may become washed out as Γ → 1.  The default value 0.3981 was found to work well on many test images."))






    start = datetime.datetime.now()

    report_runs('app.py|run_app|229')
    
    # LOAD IMAGE
    #▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲
    
    image_input_key = load_image(st.session_state.input_file_path, st.session_state.input_source)
    
    #▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲

    report_runs('app.py|run_app|238')
    if st.session_state.debug:
        with st.expander("session_state 0.3:"):
            st.write(st.session_state)

    # Create all key names unique to the current image and parameter set
    st.session_state.keys_ = Keys(image_input_key, 
                                 granularity, 
                                 kernel_parallel, 
                                 kernel_orthogonal,
                                 sharpness, 
                                 texture_style, 
                                 smoothness, 
                                 power, 
                                 exposure_ratio_in, 
                                 color_gamma,
                                 a,
                                 b,
                                 lo,
                                 hi)

    report_runs('app.py|run_app|259')
    if st.session_state.debug:
        with st.expander("session_state 0.4:"):
            st.write(st.session_state.keys_)

    input_image = st.session_state.memmapped[image_input_key]
    image_np, image_01, image_01_maxRGB = input_image
    shape = image_01_maxRGB.shape
    st.session_state.keys_to_shape[image_input_key] = shape
    st.session_state.input_shape = st.session_state.keys_to_shape[image_input_key]
    container.write(f'{str(shape[0])}   ×   {str(shape[1])}')
    report_runs('app.py|run_app|270')
   
    if st.session_state.keys_.enhanced_image_key not in st.session_state.memmapped:

        report_runs('app.py|run_app|274')

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


    end = datetime.datetime.now()
    process_time = (end - start).total_seconds()
    print(f'[{datetime.datetime.now().isoformat()}]  Processing time: {process_time:.5f} s')
    sys.stdout.flush()
    
    report_runs('app.py|run_app|302')
    if st.session_state.debug:
        with st.expander("session_state 0.5:"):
            st.write(st.session_state)

    # prepare to encode parameter settings in filenames of optional downloads
    granularity_param_str = f'_{granularity*100:.0f}'
    convolution_param_str = granularity_param_str + f'_{kernel_parallel:d}_{kernel_orthogonal:d}'
    texture_param_str = convolution_param_str + f'_{sharpness*1000:.0f}_{texture_style:s}'
    smooth_param_str = texture_param_str + f'_{smoothness*100:.0f}'
    fusion_param_str = smooth_param_str + f'_{color_gamma*100:.0f}_{power*100:.0f}_{-a*1000:.0f}_{b*1000:.0f}_{exposure_ratio_in*100:.0f}'

    input_file_name = st.session_state.input_file_name
    input_file_ext = '.' + str(input_file_name.split('.')[-1])
    input_file_basename = input_file_name.replace(input_file_ext, '')

    with st.expander(f'Select Viewer', expanded=True):

        st.session_state.viewer_selection = st.radio(" ", st.session_state.viewer_options, help="Coming soon", key=st.session_state.viewer_selection_key, horizontal=True, index=st.session_state.viewer_selection_index)#, on_change=prepare_next_key)

    if  st.session_state.viewer_selection == "Comparisons (interactive)":
    
        with st.expander("Comparison Options "):
            with st.form("Comparison"):
                submitted = st.form_submit_button("Update Comparison")
                col_left, col_right, _ = st.columns(3)
                with col_left:
                    st.session_state.left_image_selection = st.radio("Select Left", st.session_state.comparison_options, help="Coming Soon", key=st.session_state.left_image_selection_key, index=st.session_state.left_image_selection_index)
                   # st.session_state.left_image_selection_index = st.session_state.comparison_options.index(st.session_state.left_image_selection)
                with col_right:
                    st.session_state.right_image_selection = st.radio("Select Right", st.session_state.comparison_options, help="Coming Soon", key=st.session_state.right_image_selection_key, index=st.session_state.right_image_selection_index)
                    #st.session_state.right_image_selection_index = st.session_state.comparison_options.index(st.session_state.right_image_selection)
                
        st.session_state.paths = {
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
                
        left_image = cv2.cvtColor(cv2.imread(st.session_state.paths[st.session_state.left_image_selection]), cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(st.session_state.paths[st.session_state.right_image_selection]), cv2.COLOR_BGR2RGB)
        image_comparison(
                img1=left_image,
                img2=right_image,
                label1=st.session_state.left_image_selection,
                label2=st.session_state.right_image_selection,
                width=1080,
                starting_position=50,
                show_labels=True
            )
       

    elif st.session_state.viewer_selection == "Enhanced Image":

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

    elif  st.session_state.viewer_selection == "Original vs Enhanced":
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
            st.markdown("<h3 style='text-align: center; color: white;'>Fine Texture Map</h3>", unsafe_allow_html=True)
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

    if  st.session_state.show_resource_usage:
        with st.expander("Resource Usage", expanded=True):
            st.text(f'[{timestamp()}]\nPID: {pid}')
            st.text(f'rss: {mem:.2f} MB\nvirt: {virt:.2f} MB\nswap: {swap:.2f} MB')
    
    st.session_state.completed_app_runs += 1
    report_runs('app.py|run_app|559')

    if st.session_state.debug:
        with st.expander("session_state 1.0:"):
            st.write(st.session_state)
    
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


