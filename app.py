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
import gc
from utils.io_tools import change_extension, load_binary, load_image, mkpath
from utils.config import NPY_DIR_PATH, IMAGE_DIR_PATH, EXAMPLE_PATHS, EXAMPLES, DATA_DIR_PATH, EXAMPLES_DIR_PATH
from utils.sodef import bimef
from utils.array_tools import float32_to_uint8, uint8_to_float32, normalize_array, array_info, mono_float32_to_rgb_uint8
from utils.logging import timestamp, log_memory
from utils.mm import get_mmaps, get_weakrefs, references_dead_object, clear_cache, clear_data, clear
from utils.session import Keys
import weakref
from pathlib import Path

def reset():
    # pass
    # if 'image_np' in st.session_state:
    #     del st.session_state.image_np
    if 'input_file_name' in st.session_state:
        del st.session_state.input_file_name

def set_source(source='local'):
    st.session_state.source_last_updated = source

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

    with st.expander("session_state 0:"):
        st.write(st.session_state)

    st.session_state.total_app_runs += 1

    st.session_state.npy_dir = NPY_DIR_PATH
    st.session_state.image_dir = IMAGE_DIR_PATH
    st.session_state.data_dir = DATA_DIR_PATH
    st.session_state.examples_dir = EXAMPLES_DIR_PATH

    with st.sidebar:

        pid = getpid()
        #st.text(f'PID: {pid}')
        placeholder = st.empty()
        if st.session_state.show_console:
            with placeholder.container():
                with st.form('console'):

                    submitted = st.form_submit_button('run', help="coming soon")#, on_click=run_command, args=[command])
                    command = st.text_input("in")
                    try:
                        console_out = str(subprocess.check_output(command, shell=True, text=True))
                    except subprocess.CalledProcessError as e:
                        #print(vars(e))
                        console_out = f'exited with error\nreturncode: {e.returncode}\ncmd: {e.cmd}\noutput: {e.output}\nstderr: {e.stderr}'

            # st.write(f'IN: {st.session_state.command}')
            # st.text(f'OUT: {st.session_state.console_out}')
                st.write(f'IN: {command}')
                st.text(f'OUT: {console_out}')
        else:
             placeholder.empty()
            
        if st.session_state.low_resources:
            clear_cache()
            st.session_state.low_resources = False

        with st.expander(f'{Process(pid).memory_info()[0]/float(2**20):.2f}'):

            with st.form("Clear"):
                st.session_state.cache_checked = st.checkbox("Clear Cache", help="coming soon")
                st.session_state.data_checked = st.checkbox("Clear Data", help="coming soon")
                st.form_submit_button("Clear", on_click=clear, args=([st.session_state.cache_checked, st.session_state.data_checked]), help="coming soon")

        input_selection = st.radio("Select Example:", EXAMPLES, horizontal=True, on_change=set_source, kwargs=dict(source='local'), help="coming soon")
        image_example_path = EXAMPLE_PATHS[input_selection]
        fImage = st.file_uploader("Or Upload Your Own Image:", on_change=set_source, kwargs=dict(source='upload'), help="coming soon") #("Process new image:")
        # input_selection = st.radio("Select Example:", EXAMPLES, horizontal=True, help="coming soon")
        # image_example_path = EXAMPLE_PATHS[input_selection]
        # fImage = st.file_uploader("Or Upload Your Own Image:", help="coming soon") #("Process new image:")
        if all([fImage is None, st.session_state.source_last_updated == 'upload', st.session_state.last_run_exited_early]):
            st.write(st.session_state.input_file_name)

    with st.expander("session_state 0.2:"):
        st.write(st.session_state)

    with st.sidebar:
        with st.expander("Parameter Settings"):
            with st.form('Parameter Settings'):
                submitted = st.form_submit_button('Apply', help="coming soon")
                granularity_selection = st.radio("Illumination detail", ('standard', 'boost', 'max'), horizontal=True, help="coming soon")
                granularity_dict = {'standard': 0.1, 'boost': 0.3, 'max': 0.5}
                granularity = granularity_dict[granularity_selection]
                power = float(st.text_input(f'Power     (default = {default_power})', str(default_power), help="coming soon"))
                smoothness = float(st.text_input(f'Smoothness   (default = {default_smoothness})', str(default_smoothness), help="coming soon"))
                sharpness = float(st.text_input(f'Sharpness   (default = {default_sharpness})', str(default_sharpness), help="coming soon"))
                kernel_parallel = int(st.text_input(f'Kernel Parallel   (default = {default_kernel_parallel})', str(default_kernel_parallel), help="coming soon"))
                kernel_orthogonal = int(st.text_input(f'Kernel Orthogonal   (default = {default_kernel_orthogonal})', str(default_kernel_orthogonal), help="coming soon")) 
                a = float(st.text_input(f'Camera A   (default = {default_a})', str(default_a), help="coming soon"))
                b = float(st.text_input(f'Camera B   (default = {default_b})', str(default_b), help="coming soon"))
                lo = int(st.text_input(f'Min Gain   (default = {default_lo})', str(default_lo), help="Sets lower bound of search range for optimal Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                hi = int(st.text_input(f'Max Gain   (default = {default_hi})', str(default_hi), help="Sets upper bound of search range for optimal Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                exposure_ratio_in = float(st.text_input(f'Exposure Ratio   (default = -1 (auto))', str(default_exposure_ratio), help="coming soon"))
                color_gamma = float(st.text_input(f'Color Gamma   (default = {default_color_gamma})', str(default_color_gamma), help="coming soon"))
                texture_weight_calculator = st.radio("Select texture weight calculator", ('I', 'II', 'III', 'IV', 'V'), horizontal=True, help="coming soon") 
                texture_weight_calculator_dict = {
                            'I':  ('I', CG_TOL, LU_TOL, MAX_ITER, FILL),
                            'II': ('II', CG_TOL, LU_TOL, MAX_ITER, FILL),
                            'III':('III', 0.1*CG_TOL, LU_TOL, 10*MAX_ITER, FILL),
                            'IV': ('IV', 0.5*CG_TOL, LU_TOL, MAX_ITER, FILL/2),
                            'V':  ('V', CG_TOL, LU_TOL, MAX_ITER, FILL)
                            }

                texture_style, cg_tol, lu_tol, max_iter, fill = texture_weight_calculator_dict[texture_weight_calculator]

        checkbox = st.checkbox('Show Process Images', value=True, help="coming soon")

    start = datetime.datetime.now()

    image_input_key = load_image(fImage, example_path=image_example_path, reload_previous=st.session_state.last_run_exited_early)  ###############################<<<<<<<<<<<<<<<<<<<
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

    with st.expander("session_state 0.4:"):
        st.write(st.session_state.keys_)

    input_image = st.session_state.memmapped[image_input_key]
    image_np, image_01, image_01_maxRGB = input_image

    
    if st.session_state.keys_.enhanced_image_key not in st.session_state.memmapped:


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

    granularity_param_str = f'_{granularity*100:.0f}'
    convolution_param_str = granularity_param_str + f'_{kernel_parallel:d}_{kernel_orthogonal:d}'
    texture_param_str = convolution_param_str + f'_{sharpness*1000:.0f}_{texture_weight_calculator:s}'
    smooth_param_str = texture_param_str + f'_{smoothness*100:.0f}'
    fusion_param_str = smooth_param_str + f'_{color_gamma*100:.0f}_{power*100:.0f}_{-a*1000:.0f}_{b*1000:.0f}_{exposure_ratio_in*100:.0f}'

    input_file_name = st.session_state.input_file_name
    input_file_ext = '.' + str(input_file_name.split('.')[-1])
    input_file_basename = input_file_name.replace(input_file_ext, '')
    output_wls_map_file_name = input_file_basename + '_WLS' + texture_param_str + input_file_ext
    output_L1_map_file_name = input_file_basename + '_L1' + texture_param_str + input_file_ext
    output_fine_texture_map_file_name = input_file_basename + '_FTM' + smooth_param_str + input_file_ext
    output_illumination_map_file_name = input_file_basename + '_ILL' + smooth_param_str + input_file_ext
    output_simulation_file_name = input_file_basename + '_SIM' + fusion_param_str + input_file_ext
    output_exposure_maxent_file_name = input_file_basename + '_EME' + smooth_param_str + input_file_ext
    output_fusion_weights_file_name = input_file_basename + '_FW' + fusion_param_str + input_file_ext
    output_fused_file_name = input_file_basename + '_FUSION' + fusion_param_str + input_file_ext    
    
    with st.expander("session_state 0.5:"):
        st.write(st.session_state)

    col1, col2, col3 = st.columns(3)

    with col1:        
        
        st.markdown("<h3 style='text-align: center; color: white;'>Original</h3>", unsafe_allow_html=True)
        #st.image(st.session_state.memmapped[st.session_state.keys_.image_input_key][0], clamp=True, channels="BGR")#[:,:,[2,1,0]])
        #st.image(image_np, channels="BGR")#[:,:,[2,1,0]])
        #impath = st.session_state.input_file_path
        print(f'st.session_state.input_file_path: {st.session_state.input_file_path}')
        st.image(cv2.imread(st.session_state.input_file_path), channels="BGR")#, cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], clamp=True)
        # input_file_name = st.text_input('Download Original Image As', input_file_name)
        # ext = '.' + input_file_name.split('.')[-1]
        # image_np_binary = cv2.imencode(ext, image_np)[1].tobytes()
        # button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")

  #      if checkbox:
  #          st.markdown("<h3 style='text-align: center; color: white;'>Texture Weights</h3>", unsafe_allow_html=True)
  #          st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.texture_weights_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_wls_map_file_name = st.text_input('Download Texture Weights As', output_wls_map_file_name)
            # ext = '.' + output_wls_map_file_name.split('.')[-1]
            # texture_weights_map_binary = load_binary(ext, texture_weights_map, color_channel='bgr')

            # button = st.download_button(label = "Download Texture Weights", data = texture_weights_map_binary, file_name = output_wls_map_file_name, mime = "image/png")
            
 #           st.markdown("<h3 style='text-align: center; color: white;'>Total Variation</h3>", unsafe_allow_html=True)
 #           st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.total_variation_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_L1_map_file_name = st.text_input('Download Total Variation', output_L1_map_file_name)
            # ext = '.' + output_L1_map_file_name.split('.')[-1]
            # image_np_TV_map_binary = cv2.imencode(ext, image_np_TV_map[:,:,[2,1,0]])[1].tobytes()

            # button = st.download_button(label = "Download Total Variation", data = image_np_TV_map_binary, file_name = output_L1_map_file_name, mime = "image/png")

    with col2:
        st.markdown("<h3 style='text-align: center; color: white;'>Enhancement Layer</h3>", unsafe_allow_html=True)
        #st.image(st.session_state.keys_to_images[st.session_state.keys_.enhancement_map_key], clamp=True)
        #st.image(cv2.imread(os.path.join(st.session_state.image_dir,'scrapyardjpgexample10GC51001I300500E.png'), cv2.IMREAD_UNCHANGED), clamp=True)
        impath = st.session_state.keys_to_images[st.session_state.keys_.enhancement_map_key]
        st.image(cv2.imread(impath), channels="BGR")#, cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], clamp=True)

        # output_simulation_file_name = st.text_input('Download Enhancement Map As', output_simulation_file_name)
        # ext = '.' + output_simulation_file_name.split('.')[-1]
        # image_np_simulation_binary = cv2.imencode(ext, image_np_simulation[:,:,[2,1,0]])[1].tobytes()

        # button = st.download_button(label = "Download Enhancement Map", data = image_np_simulation_binary, file_name = output_simulation_file_name, mime = "image/png")        

#        if checkbox:

#            st.markdown("<h3 style='text-align: center; color: white;'>Illumination Map</h3>", unsafe_allow_html=True)
#            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.smoother_output_fullsize_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_illumination_map_file_name = st.text_input('Download Illumination Map As', output_illumination_map_file_name)
            # ext = '.' + output_illumination_map_file_name.split('.')[-1]
            # illumination_map_binary = cv2.imencode(ext, illumination_map[:,:,[2,1,0]])[1].tobytes()

            # button = st.download_button(label = "Download Illumination Map", data = illumination_map_binary, file_name = output_illumination_map_file_name, mime = "image/png")

            #st.markdown("<h3 style='text-align: center; color: white;'>Fusion Weights</h3>", unsafe_allow_html=True)
            #st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fusion_weights_key], cv2.IMREAD_UNCHANGED), clamp=True)
#            st.markdown("<h3 style='text-align: center; color: white;'>Max RGB</h3>", unsafe_allow_html=True)
#            st.image(st.session_state.memmapped[st.session_state.keys_.image_input_key][2], clamp=True)

            # output_fusion_weights_file_name = st.text_input('Download Fusion Weights As', output_fusion_weights_file_name)
            # ext = '.' + output_fusion_weights_file_name.split('.')[-1]
            # fusion_weights_binary = cv2.imencode(ext, fusion_weights)[1].tobytes()

            # button = st.download_button(label = "Download Fusion Weights", data = fusion_weights_binary, file_name = output_fusion_weights_file_name, mime = "image/png")

    with col3:

        st.markdown("<h3 style='text-align: center; color: white;'>Enhanced Image</h3>", unsafe_allow_html=True)
        st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key]), channels="BGR")#, cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], clamp=True)

        # output_fused_file_name = st.text_input('Download Fused Image As', output_fused_file_name)
        # ext = '.' + output_fused_file_name.split('.')[-1]
        # image_np_fused_binary = cv2.imencode(ext, image_np_fused[:,:,[2,1,0]])[1].tobytes()

        # button = st.download_button(label = "Download Fused Image", data = image_np_fused_binary, file_name = output_fused_file_name, mime = "image/png")
    
#        if checkbox:
            # st.image([cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key], cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], 
            #           cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key], cv2.IMREAD_UNCHANGED)
            #          ],
            #          caption=[
            #                      "Simulated Exposure",
            #                      "Preserved Dark"
            #                  ]
            #          )
#           st.markdown("<h3 style='text-align: center; color: white;'>Simulated Exposure</h3>", unsafe_allow_html=True)
#           st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key]), channels="BGR")#, cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], clamp=True)

            # output_exposure_maxent_file_name = st.text_input('Download Maxent Exposure As', output_exposure_maxent_file_name)
            # ext = '.' + output_exposure_maxent_file_name.split('.')[-1]
            # image_exposure_maxent_binary = load_binary(ext, image_exposure_maxent[:,:,[2,1,0]], color_channel='bgr')

            # button = st.download_button(label = "Download Maxent Exposure", data = image_exposure_maxent_binary, file_name = output_exposure_maxent_file_name, mime = "image/png")            
            
 #          st.markdown("<h3 style='text-align: center; color: white;'>Preserved Dark</h3>", unsafe_allow_html=True)
 #          st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_fine_texture_map_file_name = st.text_input('Download Fine Texture Map As', output_fine_texture_map_file_name)
            # ext = '.' + output_fine_texture_map_file_name.split('.')[-1]
            # image_np_fine_texture_map_binary = cv2.imencode(ext, image_np_fine_texture_map[:,:,[2,1,0]])[1].tobytes()

            # button = st.download_button(label = "Download Fine Texture Map", data = image_np_fine_texture_map_binary, file_name = output_fine_texture_map_file_name, mime = "image/png")


    col10, col20, col30 = st.columns(3)




    if checkbox:

        with col10:
            st.markdown("<h3 style='text-align: center; color: white;'>Texture Weights</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.texture_weights_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_wls_map_file_name = st.text_input('Download Texture Weights As', output_wls_map_file_name)
            # ext = '.' + output_wls_map_file_name.split('.')[-1]
            # texture_weights_map_binary = load_binary(ext, texture_weights_map, color_channel='bgr')

            # button = st.download_button(label = "Download Texture Weights", data = texture_weights_map_binary, file_name = output_wls_map_file_name, mime = "image/png")
            
            st.markdown("<h3 style='text-align: center; color: white;'>Total Variation</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.total_variation_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_L1_map_file_name = st.text_input('Download Total Variation', output_L1_map_file_name)
            # ext = '.' + output_L1_map_file_name.split('.')[-1]
            # image_np_TV_map_binary = cv2.imencode(ext, image_np_TV_map[:,:,[2,1,0]])[1].tobytes()

            # button = st.download_button(label = "Download Total Variation", data = image_np_TV_map_binary, file_name = output_L1_map_file_name, mime = "image/png")



        with col20:
            
            st.markdown("<h3 style='text-align: center; color: white;'>Illumination Map</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.smoother_output_fullsize_key], cv2.IMREAD_UNCHANGED), clamp=True)

            # output_illumination_map_file_name = st.text_input('Download Illumination Map As', output_illumination_map_file_name)
            # ext = '.' + output_illumination_map_file_name.split('.')[-1]
            # illumination_map_binary = cv2.imencode(ext, illumination_map[:,:,[2,1,0]])[1].tobytes()

            # button = st.download_button(label = "Download Illumination Map", data = illumination_map_binary, file_name = output_illumination_map_file_name, mime = "image/png")

            st.markdown("<h3 style='text-align: center; color: white;'>Fusion Weights</h3>", unsafe_allow_html=True)
            st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fusion_weights_key], cv2.IMREAD_UNCHANGED), clamp=True)
            #st.markdown("<h3 style='text-align: center; color: white;'>Max RGB</h3>", unsafe_allow_html=True)
            #st.image(st.session_state.memmapped[st.session_state.keys_.image_input_key][2], clamp=True)

            # output_fusion_weights_file_name = st.text_input('Download Fusion Weights As', output_fusion_weights_file_name)
            # ext = '.' + output_fusion_weights_file_name.split('.')[-1]
            # fusion_weights_binary = cv2.imencode(ext, fusion_weights)[1].tobytes()

            # button = st.download_button(label = "Download Fusion Weights", data = fusion_weights_binary, file_name = output_fusion_weights_file_name, mime = "image/png")


        with col30:
            # st.image([cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key], cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], 
            #           cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key], cv2.IMREAD_UNCHANGED)
            #          ],
            #          caption=[
            #                      "Simulated Exposure",
            #                      "Preserved Dark"
            #                  ]
            #          )
           st.markdown("<h3 style='text-align: center; color: white;'>Max Entropy Exposure</h3>", unsafe_allow_html=True)
           st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key]), channels="BGR")#, cv2.IMREAD_UNCHANGED)[:,:,[2,1,0]], clamp=True)

            # output_exposure_maxent_file_name = st.text_input('Download Maxent Exposure As', output_exposure_maxent_file_name)
            # ext = '.' + output_exposure_maxent_file_name.split('.')[-1]
            # image_exposure_maxent_binary = load_binary(ext, image_exposure_maxent[:,:,[2,1,0]], color_channel='bgr')

            # button = st.download_button(label = "Download Maxent Exposure", data = image_exposure_maxent_binary, file_name = output_exposure_maxent_file_name, mime = "image/png")            
            
           st.markdown("<h3 style='text-align: center; color: white;'>Texture Map</h3>", unsafe_allow_html=True)
           st.image(cv2.imread(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key], cv2.IMREAD_UNCHANGED), clamp=True)

           output_fine_texture_map_file_name = st.text_input('Download Fine Texture Map As', output_fine_texture_map_file_name)
           ext = '.' + output_fine_texture_map_file_name.split('.')[-1]
            # image_np_fine_texture_map_binary = cv2.imencode(ext, image_np_fine_texture_map[:,:,[2,1,0]])[1].tobytes()
           #imbytes = Path(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key]).read_bytes()
           button = st.download_button(
                                          label = "Download Fine Texture Map", 
                                          data=Path(st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key]).read_bytes(),
                                          file_name = output_fine_texture_map_file_name, 
                                          mime = f"image/{input_file_ext}"
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

    with st.expander("session_state 1:"):
     st.write(st.session_state)

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


