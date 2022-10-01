import streamlit as st
import numpy as np
from utils.array_tools import float32_to_uint8

def calculate_fusion_weights(illumination_map, enhance, ndim): 
    fusion_weights = np.power(illumination_map, enhance)  
    if ndim==3:                                           
        fusion_weights = np.expand_dims(fusion_weights, axis=2)  
    fusion_weights  = np.where(fusion_weights>1,1,fusion_weights)
    return fusion_weights

    
def fuse_image(image_01, image_exposure_adjusted, fusion_weights):

    enhancement_map = float32_to_uint8(image_exposure_adjusted * (1 - fusion_weights))
    image_enhanced = enhancement_map + float32_to_uint8(image_01 * fusion_weights)

    return [enhancement_map, image_enhanced]

