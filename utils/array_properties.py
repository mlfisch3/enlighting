import streamlit as st
import numpy as np
import sys
from .array_entropy import entropy

def array_info(array, print_info=True, return_info=False, return_info_str=False, name=None):
    '''
    Calculate array properties and store in dictionary
    
    Returns:
        Only print formatted string (Default)
        Return info as dictionary (return_info=True)
        Return info as formatted string (return_info_str=True)
        
    Example:
    >>> x = np.random.randint(0,255,size=(1080, 1920, 3), dtype=np.uint8)
    >>> array_info(x, name='x')

        **********************************
        **********************************
            x
        **********************************

        bytes: 6220944
        uint8  (1080, 1920, 3)
        nonzero: 6196350 / 6220800  (99.6)
        min:  0.00   max: 254.00
        mean: 127.01   std: 73.63     layer_variation: 54.83
        entropy: 7.99
        **********************************

    '''
    info = {}
    info['name'] = str(name)
    info['bytes'] = sys.getsizeof(array)
    info['dtype'] = array.dtype
    info['ndim'] = array.ndim
    info['shape'] = array.shape
    info['max'] = array.max()
    info['min'] = array.min()
    info['mean'] = array.mean()
    info['std'] = array.std()
    info['size'] = array.size
    info['nonzero'] = np.count_nonzero(array)
    info['layer_variation'] = 0
    info['entropy'] = entropy(array)

    if array.ndim > 2:
        info['layer_variation'] = array.std(axis=array.ndim-1).mean()

    info['pct'] = 100 * info['nonzero'] / info['size']

    if print_info:
        if info["name"] is not None:
            print(f'\n**********************************\n**********************************\n    {info["name"]}\n**********************************\n')
        print(f'bytes: {info["bytes"]}')
        print(f'{info["dtype"]}  {info["shape"]}')
        print(f'nonzero: {info["nonzero"]} / {info["size"]}  ({info["pct"]:.1f} %)')
        print(f'min:  {info["min"]:.2f}   max: {info["max"]:.2f}')
        print(f'mean: {info["mean"]:.2f}   std: {info["std"]:.2f}', end="")
        if info["ndim"] > 2:
            print(f'     layer_variation: {info["layer_variation"]:.2f}')
        else:
            print('\n')

        print(f'entropy: {info["entropy"]:.2f}')#, end="")
        print(f'**********************************\n')    
    out = []
    if return_info:
        out.append(info)
    if return_info_str:
        if name is None:
          info_str = ''
        else:
          info_str = f'name: {info["name"]}\n'
        info_str += f'bytes: {info["bytes"]}\n'
        info_str += f'shape: {info["shape"]}\n'
        info_str += f'size: {info["size"]}\nnonzero: {info["nonzero"]}  ({info["pct"]:.4f} %)\n'
        info_str += f'min: {info["min"]}    max: {info["max"]}\n'
        info_str += f'mean: {info["mean"]:.4f}    std: {info["std"]:.4f}\n'
        if array.ndim > 2:
            info_str += f'layer_variation: {info["layer_variation"]:.4f}\n'
        else:
            print('\n')
            
        info_str += f'entropy: {info["entropy"]:.4f}\n'

        out.append(info_str)

    if return_info or return_info_str:
        if len(out)==1:
            return out[0]
        else:
            return out


def print_array_info(info):
    '''
    Print items in info

    info: dictionary [Note:  Must be created by or equivalent to the dictionary returned by array_info(array, return_info=True)]

    Example:

    >>> x = np.random.randint(0,255,size=(1080, 1920, 3), dtype=np.uint8)
    >>> array_info(x, name='x')

        **********************************
        **********************************
            x
        **********************************

        bytes: 6220944
        uint8  (1080, 1920, 3)
        nonzero: 6196350 / 6220800  (99.6)
        min:  0.00   max: 254.00
        mean: 127.01   std: 73.63     layer_variation: 54.83
        entropy: 7.99
        **********************************

    >>> y = array_info(x, name='x', return_info=True, print_info=False)
    >>> print_array_info(y)

        **********************************
        **********************************
            x
        **********************************

        bytes: 6220944
        uint8  (1080, 1920, 3)
        nonzero: 6196350 / 6220800  (99.6)
        min:  0.00   max: 254.00
        mean: 127.01   std: 73.63     layer_variation: 54.83
        entropy: 7.99
        **********************************

    '''

    if info["name"] is not None:
        print(f'\n**********************************\n**********************************\n    {info["name"]}\n**********************************\n')
    print(f'bytes: {info["bytes"]}')
    print(f'{info["dtype"]}  {info["shape"]}')
    print(f'nonzero: {info["nonzero"]} / {info["size"]}  ({info["pct"]:.1f})')
    print(f'min:  {info["min"]:.2f}   max: {info["max"]:.2f}')
    print(f'mean: {info["mean"]:.2f}   std: {info["std"]:.2f}', end="")
    if info["ndim"] > 2:
        print(f'     layer_variation: {info["layer_variation"]:.2f}')
    else:
        print('\n')

    print(f'entropy: {info["entropy"]:.2f}')#, end="")
    print(f'**********************************\n')    
