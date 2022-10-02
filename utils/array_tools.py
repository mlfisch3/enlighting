import sys
import numpy as np
from PIL import Image
from pympler import asizeof
from utils.array_entropy import entropy
import copy

def float32_to_uint8(array):
    array_max = np.max(array)
    if array_max > 1:
        array = array / array_max
    array = (255 * array).astype(np.uint8)
    return array

def mono_float32_to_rgb_uint8(array):
    return np.tile(float32_to_uint8(normalize_array(array, is_read_only=True)).T, (3,1,1)).T
#    return np.tile(float32_to_uint8(array).T, (3,1,1)).T
#    return np.tile(float32_to_uint8(np.clip(array, 0, 1)).T, (3,1,1)).T

def uint8_to_float32(array):
    array = array.astype(np.float32) / np.float32(255)
    return array

def diff(x, axis=0):
    if axis==0:
        return x[1:,:]-x[:-1,:]
    else:
        return x[:,1:]-x[:,:-1]

def cyclic_diff(x,axis=0):
    if axis==0:
        return x[0,:]-x[-1,:]
    else:
        return (x[:,0]-x[:,-1])[None,:].T


def geometric_mean(image):
    try:
        assert image.ndim == 3, 'Warning: Expected a 3d-array.  Returning input as-is.'
        return np.power(np.prod(image, axis=2), 1/3)
    except AssertionError as msg:
        print(msg)
        return image

def normalize_array(array, is_read_only=False):
    if is_read_only:
        array_ = np.array(array)
    else:
        array_ = array

    lo = np.min(array_)
    array_ -= lo    
    hi = np.max(array_)
    try:
        assert hi > 0, f'normalize_array cannot map null array to interval [0,1]'
    except AssertionError as msg:
        print(msg)
        return array_

    array_ = array_ / hi
    return array_.astype(np.float32)

def normalize_arrays(A, B):

    lo = np.min(np.hstack([A,B]))
    
    A = A - lo
    B = B - lo

    hi = np.max(np.hstack([A,B]))
    A = A / hi
    B = B / hi
    return A, B


def imresize(image, scale=-1, size=(-1,-1)):
    ''' image: numpy array with shape (n, m) or (n, m, 3)
       scale: mulitplier of array height & width (if scale > 0)
       size: (num_rows, num_cols) 2-tuple of ints > 0 (only used if scale <= 0)'''
    if (image.shape == size) | (scale == 1):
        return image

    dtype = image.dtype
    if dtype == 'float64':
        dtype = 'float32'

    if image.ndim==2:
        im = Image.fromarray(image)
        if scale > 0:
            width, height = im.size
            newsize = (int(width*scale), int(height*scale))
        else:
            newsize = (size[1],size[0])    #         numpy index convention is reverse of PIL
        return np.array(im.resize(newsize), dtype=dtype)

    if scale > 0:
        height = max(1,image.shape[0])  # changed from np.max()  [mdf: 20220706]
        width = max(1,image.shape[1])  # changed from np.max()  [mdf: 20220706]
        newsize = (int(width*scale), int(height*scale))
    else:
        newsize = (size[1],size[0])    

    tmp = np.zeros((newsize[1],newsize[0],3), dtype=dtype)
    for i in range(3):
        im = Image.fromarray(image[:,:,i])
        tmp[:,:,i] = np.array(im.resize(newsize), dtype=dtype)
    return tmp


def L1(array2d):
    return np.max(np.abs(array2d).sum(axis=0))

def recsum(x):
    if x.ndim>1:
        return recsum(x.sum())
    else:
        return x.sum()

def pnorm(a,p=2):
    A = np.power(np.abs(a),p)
    return np.power(recsum(A),1/p)

def Lp(a, p):
    return np.power(np.power(a.ravel(), p).sum(),1/p)

def array_info(array, print_info=True, return_info=False, return_info_str=False, name=None):
    '''

    Calculate array properties
    Print formatted string [Optional]  (Default)

    Returns:            
        Return info as dictionary (return_info=True) [Optional]
        Return info as formatted string (return_info_str=True) [Optional]
        
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
    #info['bytes'] = asizeof.asizeof(array)
    info['bytes'] = array.nbytes
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
        print(f'nonzero: {info["nonzero"]} / {info["size"]}  ({info["pct"]:.1f})')
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
