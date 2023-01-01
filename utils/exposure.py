import numpy as np
import streamlit as st

from utils.array_entropy import entropy
from utils.array_tools import geometric_mean, imresize, normalize_array

def applyK(G, k, a=-0.3293, b=1.1258, verbose=False):
    if k==1.0:
        return G.astype(np.float32)

    if k<=0:
        return np.ones_like(G, dtype=np.float32)

    gamma = k**a
    beta = np.exp((1-gamma)*b)

    if verbose:
        print(f'a: {a:.4f}, b: {b:.4f}, k: {k:.4f}, gamma: {gamma:.4f}, beta: {beta}.  ----->  output = {beta:.4} * image^{gamma:.4f}')
    return (np.power(G,gamma)*beta).astype(np.float32)


def get_dim_pixels(image,dim_pixels,dim_size=(50,50)):
    dim_pixels = imresize(dim_pixels,size=dim_size)

    image = imresize(image,size=dim_size)
    image = np.where(image>0,image,0)
    Y = geometric_mean(image)
    return Y[dim_pixels]

def optimize_exposure_ratio(array, a, b, lo=1, hi=7, npoints=20):
    if sum(array.shape)==0:
        return 1.0

    sample_ratios = np.r_[lo:hi:complex(0,npoints)].tolist()
    entropies = np.array(list(map(lambda k: entropy(applyK(array, k, a, b)), sample_ratios)))
    optimal_index = np.argmax(entropies)
    return sample_ratios[optimal_index]

    return fusion_weights

def adjust_exposure(image_01, illumination_map, a, b, exposure_ratio=-1, dim_threshold=0.5, dim_size=(50,50), lo=1, hi=7, npoints=20, color_gamma=0.3981, verbose=False):
    #image_01 = normalize_array(image)
    dim_pixels = np.zeros_like(illumination_map)
    
    if exposure_ratio==-1:
        dim_pixels = illumination_map<dim_threshold
        Y = get_dim_pixels(image_01, dim_pixels, dim_size=dim_size) 
        exposure_ratio = optimize_exposure_ratio(Y, a, b, lo=lo, hi=hi, npoints=npoints)


    image_01_K = applyK(image_01, exposure_ratio, a, b) # adjustment applied equally to all color channels
    if (image_01.ndim == 2) or ((exposure_ratio < 1) & (exposure_ratio > 0)):
        image_exposure_adjusted = image_01_K  # color artifacts not an issue if image is grayscale, or if further reducing exposure of dim pixels
    else:
        image_01_ave = image_01.mean(axis=2)[:,:,None]
        image_01_dRGB = image_01 - image_01_ave 
        image_01_ave_K = applyK(image_01_ave, exposure_ratio, a, b)
        image_exposure_adjusted = color_gamma * (image_01_ave_K + image_01_dRGB) + (1 - color_gamma)* image_01_K

        # color_gamma = 0     →     adjustment applied to array (gives result from paper)
        # color_gamma = 1     →     apply exposure scaling to the average value of each pixel, then add back original rgb deviations  δr = r - (r+g+b)/3

    image_exposure_adjusted = np.where(image_exposure_adjusted>1,1,image_exposure_adjusted)    

    return image_exposure_adjusted, exposure_ratio
