import numpy as np
import datetime
import os
import streamlit as st
import weakref
import cv2

from utils.array_tools import float32_to_uint8, imresize, normalize_array
from utils.logging import timestamp
from utils.edge_preserving_smoother import smooth
from utils.exposure import adjust_exposure
from utils.fusion import calculate_fusion_weights, fuse_image
from utils.io_tools import save_to_npy

def bimef(image_01, 
          image_01_maxRGB,
          exposure_ratio_in=-1, 
          power=0.5, 
          a=-0.3293, 
          b=1.1258, 
          lamda=0.5, 
          texture_style='I',
          kernel_shape=(5,1), 
          scale=0.3, 
          sharpness=0.001, 
          dim_threshold=0.5, 
          dim_size=(50,50), 
          solver='cg', 
          CG_prec='ILU', 
          CG_TOL=0.1, 
          LU_TOL=0.015, 
          MAX_ITER=50, 
          FILL=50, 
          lo=1, 
          hi=7, 
          npoints=20, 
          color_gamma=0.3981,
          verbose=False, 
          print_info=True, 
          return_texture_weights=True):


    tic = datetime.datetime.now()
    
    if st.session_state.keys_.image_reduced_key not in st.session_state.memmapped:

        fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.image_reduced_key}.npy')
        st.session_state.keys_to_npy[st.session_state.keys_.image_reduced_key] = fpath

        if not os.path.isfile(fpath):    
            if (scale <= 0) | (scale >= 1) : 
                save_to_npy(fpath, image_01_maxRGB)
            else: 
                save_to_npy(fpath, imresize(image_01_maxRGB, scale))

        st.session_state.memmapped[st.session_state.keys_.image_reduced_key] = np.lib.format.open_memmap(fpath, mode='r')
        st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(st.session_state.memmapped[st.session_state.keys_.image_reduced_key]), 'image_01_maxRGB_reduced')
    
    image_01_maxRGB_reduced = st.session_state.memmapped[st.session_state.keys_.image_reduced_key]

    if st.session_state.keys_.smoother_output_fullsize_key not in st.session_state.memmapped: 

        smoother_output_fullsize_key = smooth(image_01_maxRGB_reduced, 
                                            image_01_maxRGB.shape, 
                                            texture_style=texture_style, 
                                            kernel_shape=kernel_shape, 
                                            sharpness=sharpness, 
                                            lamda=lamda, 
                                            solver=solver, 
                                            CG_prec=CG_prec, 
                                            CG_TOL=CG_TOL, 
                                            LU_TOL=LU_TOL, 
                                            MAX_ITER=MAX_ITER, 
                                            FILL=FILL, 
                                            return_texture_weights=return_texture_weights)

        illumination_map = st.session_state.memmapped[smoother_output_fullsize_key]

        impath_IM = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.smoother_output_fullsize_key}.png')
        st.session_state.keys_to_images[st.session_state.keys_.smoother_output_fullsize_key] = impath_IM
        
        # save fine_texture_map = (image_01_maxRGB - illumination_map) as image file
        impath_FM = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.fine_texture_map_key}.png')
        st.session_state.keys_to_images[st.session_state.keys_.fine_texture_map_key] = impath_FM
        if not all([os.path.isfile(impath_FM), os.path.isfile(impath_IM)]):   # save illumination_map as image file
            st.session_state.saved_images[impath_IM] = cv2.imwrite(impath_IM, float32_to_uint8(illumination_map))
            image_01_maxRGB = st.session_state.memmapped[st.session_state.keys_.image_input_key][2]
            st.session_state.saved_images[impath_FM] = cv2.imwrite(impath_FM, float32_to_uint8(normalize_array(image_01_maxRGB - illumination_map, is_read_only=True)))
        
    illumination_map = st.session_state.memmapped[st.session_state.keys_.smoother_output_fullsize_key]

    if st.session_state.keys_.fusion_weights_key not in st.session_state.memmapped:

        fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.fusion_weights_key}.npy')
        impath = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.fusion_weights_key}.png')
        st.session_state.keys_to_images[st.session_state.keys_.fusion_weights_key] = impath
        
        if not all([os.path.isfile(fpath), os.path.isfile(impath)]):
            fusion_weights = calculate_fusion_weights(illumination_map, power, image_01.ndim)

            # save weights as image (if it doesn't already exist)
            st.session_state.saved_images[impath] = cv2.imwrite(impath, float32_to_uint8(fusion_weights.squeeze())) # squeeze removes the empty extra dim used to broadcast to rgb
            
            save_to_npy(fpath, fusion_weights)
            del fusion_weights

        st.session_state.memmapped[st.session_state.keys_.fusion_weights_key] = np.lib.format.open_memmap(fpath, mode='r')
        st.session_state.mmap_file_wref_lookup[fpath] = (weakref.ref(st.session_state.memmapped[st.session_state.keys_.fusion_weights_key]), 'fusion_weights')
    
    fusion_weights = st.session_state.memmapped[st.session_state.keys_.fusion_weights_key]


    if st.session_state.keys_.adjusted_exposure_key not in st.session_state.memmapped:

        fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.adjusted_exposure_key}.npy')
        impath = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.adjusted_exposure_key}.png')
        st.session_state.keys_to_images[st.session_state.keys_.adjusted_exposure_key] = impath

        if not all([os.path.isfile(fpath), os.path.isfile(impath), st.session_state.keys_.exposure_ratio_out_key in st.session_state.exposure_ratios]): 
            
            adjusted_exposure, exposure_ratio_out = adjust_exposure(image_01, 
                                                          illumination_map, 
                                                          a, 
                                                          b, 
                                                          exposure_ratio=exposure_ratio_in, 
                                                          dim_threshold=dim_threshold, 
                                                          dim_size=dim_size, 
                                                          lo=lo, 
                                                          hi=hi, 
                                                          color_gamma=color_gamma, 
                                                          npoints=npoints)

            st.session_state.exposure_ratios[st.session_state.keys_.exposure_ratio_out_key] = exposure_ratio_out
            
            if not os.path.isfile(impath):
                st.session_state.saved_images[impath] = cv2.imwrite(impath, float32_to_uint8(adjusted_exposure))

            save_to_npy(fpath, adjusted_exposure)
            del adjusted_exposure

        st.session_state.memmapped[st.session_state.keys_.adjusted_exposure_key] = np.lib.format.open_memmap(fpath, mode='r')       
        st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(st.session_state.memmapped[st.session_state.keys_.adjusted_exposure_key]), 'image_exposure_adjusted')
    
    exposure_ratio_out = st.session_state.exposure_ratios[st.session_state.keys_.exposure_ratio_out_key]
    image_exposure_adjusted = st.session_state.memmapped[st.session_state.keys_.adjusted_exposure_key]

    if st.session_state.keys_.enhanced_image_key not in st.session_state.memmapped:

        fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.enhanced_image_key}.npy')
        st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key] = fpath
        if fpath in st.session_state.mmap_file_wref_lookup:
            if not st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.closed:
                st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.close()
                _ = st.session_state.mmap_file_wref_lookup.pop(fpath, None)

                print(f'[{timestamp()}] Lingering open file was closed via weakref:')
                print(f'[{timestamp()}]      FILENAME: {fpath}\nMMAP_NAME: {st.session_state.mmap_file_wref_lookup[fpath][1]}')    

        impath_EM = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.enhancement_map_key}.png')
        st.session_state.keys_to_images[st.session_state.keys_.enhancement_map_key] = impath_EM
        impath_EI = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.enhanced_image_key}.png')
        st.session_state.keys_to_images[st.session_state.keys_.enhanced_image_key] = impath_EI

        if not all([os.path.isfile(fpath), os.path.isfile(impath_EM), os.path.isfile(impath_EI)]):
            enhancement_map, enhanced_image = fuse_image(image_01, image_exposure_adjusted, fusion_weights) 

        # save enhancement images 
            st.session_state.saved_images[impath_EM] = cv2.imwrite(impath_EM, float32_to_uint8(enhancement_map))
            st.session_state.saved_images[impath_EI] = cv2.imwrite(impath_EI, float32_to_uint8(enhanced_image))

            save_to_npy(fpath, enhanced_image)
            del enhancement_map, enhanced_image

        enhanced_image = np.lib.format.open_memmap(fpath, mode='r')
        st.session_state.memmapped[st.session_state.keys_.enhanced_image_key] = enhanced_image
        st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(enhanced_image), 'enhanced')

    enhanced_image = st.session_state.memmapped[st.session_state.keys_.enhanced_image_key]

    return enhanced_image, exposure_ratio_out

    toc = datetime.datetime.now()

    if print_info:
        print(f'[{timestamp()}] exposure_ratio: {exposure_ratio:.4f}, power: {power:.4f}, lamda: {lamda:.4f}, scale: {scale:.4f}, runtime: {(toc-tic).total_seconds():.4f}s')
    
    
