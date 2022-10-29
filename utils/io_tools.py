import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image, ImageOps
from utils.array_tools import normalize_array
from utils.mm import get_mmaps, references_dead_object, clear_cache
from utils.logging import timestamp
import weakref

def change_extension(filename, ext):
    return '.'.join(filename.split('.')[:-1] + [ext])

def path2tuple(path):
    '''    
    recursively call os.path.split 
    return path components as tuple, preserving hierarchical order

    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> path2tuple(newdir)
    ('C:\\', 'temp', 'subdir0', 'subdir1', 'suubdir2')
          

    '''
    (a,b) = os.path.split(path)
    if b == '':
        return a,
    else:
        return *path2tuple(a), b

def mkpath(path):
    '''
    Similar to os.mkdir except mkpath also creates implied directory structure as needed.

    For example, suppose the directory "C:\\temp" is empty. Build the hierarchy "C:\\temp\\subdir0\\subdir1\\subdir2" with single call:
    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> mkpath(newdir)        

    '''
    u = list(path2tuple(path))    
    pth=u[0]

    for i,j in enumerate(u, 1):
        if i < len(u):
            pth = os.path.join(pth,u[i])
            if not any([os.path.isdir(pth), os.path.isfile(pth)]):
                os.mkdir(pth)

def save_to_npy(fpath, array):

    try:
        np.save(fpath, array)
    except OSError:
        print(f'[{timestamp()}] encountered OSError while trying to save npy file {fpath}.  will retry after attempting to find and fix issues')
        wref = st.session_state.mmap_file_wref_lookup.get(fpath)
        if wref is not None:
            print(f'[{timestamp()}] retrieved weak reference to memory mapped file {fpath}')
            if not references_dead_object(wref[0]):
                print(f'[{timestamp()}] object named {wref[1]} has been garbage collected (weak reference is referencing a dead object)')
            if not wref[0]()._mmap.closed:
                wref[0]()._mmap.close()
                if wref[0]()._mmap.closed:
                    print(f"[{timestamp()}] Successfully closed memory mapped file {vars(wref[0]())['filename']} bound to key {wref[1]}")
                    del wref[0]
                    del st.session_state.mmap_file_wref_lookup[wref[0]]

        else:
            print(f'[{timestamp()}] could not find weak reference to {fpath}')


            clear_cache()

        try:
            np.save(fpath, array)
        except OSError:
                print(f'[{timestamp()}] unable to save file {fpath}')



def load_binary(ext, image_array, color_channel='rgb'):
    if color_channel=='rgb':
        return cv2.imencode(ext, image_array)[1].tobytes()
    else: # convert bgr to rgb
        return cv2.imencode(ext, image_array[:,:,[2,1,0]])[1].tobytes()


def get_npy_properties_(fpath_npy):
    '''
    Get properties of the array that was saved as a .npy file.
    Main purpose is to provide the array shape and dtype for use with np.lib.format.open_memmap(npy_file, mode, shape, dtype)
    
    This version might be loading the entire .npy file into memory (testing this is a TODO item), which would be a waste since just want to read the first line
    Update: NOT loading the entire file. (increment < 1kb according to memory_profiler)
    >>> z = np.ones((10,10,3), dtype=np.uint8)
    >>> np.save('z.npy',z)
    >>> get_npy_properties_('z.npy')
    {'descr': '|u1', 'fortran_order': False, 'shape': (10, 10, 3)}

    '''    
    properties = ''
    with open(fpath_npy, 'rb') as f:
        for line in f:
            properties = str(line.strip())
            break

    d = {}
    props = properties.replace('\': \'',':').replace('\', \'','~').replace(', \'','~').replace('\': ',':').replace('\'','').replace(', }\"','~').split('{')[1].split('~')
    for prop in props:
        kv = prop.split(':')
        if len(kv)==2:
            if kv[0]=='shape':
                d['shape'] = tuple([int(x) for x in kv[1].replace('(','').replace(')','').split(',')])
            elif kv[0]=='fortran_order':
                if kv[1]=='True':
                    d['fortran_order'] = True
                else:
                    d['fortran_order'] = False
            else:
                d[kv[0]] = kv[1]

    return d


def get_npy_properties(fpath_npy):
    '''
    Get properties of the array that was saved as a .npy file.
    Main purpose is to provide the array shape and dtype for use with np.lib.format.open_memmap(npy_file, mode, shape, dtype)
    
    This version should load only the first 128 bytes into memory

    >>> z = np.ones((10,10,3), dtype=np.uint8)
    >>> np.save('z.npy',z)
    >>> get_npy_properties('z.npy')
    {'descr': '|u1', 'fortran_order': False, 'shape': (10, 10, 3)}

    '''
    with open(fpath_npy, 'rb') as f:
        while properties := f.read(128):
            break

    properties = properties[1:].decode('utf-8').strip()

    d = {}
    properties = properties[1:].decode('utf-8').strip()
    props = properties.split('{')[1].replace('\': \'',':').replace('\', \'','~').replace(', \'','~').replace('\': ',':').replace('\'','').replace(', }','~').split('~')
    for prop in props:
        kv = prop.split(':')
        if len(kv)==2:
            if kv[0]=='shape':
                d['shape'] = tuple([int(x) for x in kv[1].replace('(','').replace(')','').split(',')])
            elif kv[0]=='fortran_order':
                if kv[1]=='True':
                    d['fortran_order'] = True
                else:
                    d['fortran_order'] = False
            else:
                d[kv[0]] = kv[1]

    return d


def memmap_npy(fpath_npy, mode='r'):
    
    try:
        assert os.path.isfile(fpath_npy), f'[{timestamp()}] file not found: {fpath_npy}'
    except AssertionError as msg:
        print(msg)

    shape = get_npy_properties_(fpath_npy)['shape']

    return np.lib.format.open_memmap(fpath_npy, shape=shape, mode=mode)




def get_keyed_variable(prefix_key, function, args, keyable_argvals, keyname):

    keyable_argval_str = ''.join([str(keyable_argval) for keyable_argval in keyable_argvals])
    this_key = f'{prefix_key}{function.__name__}{keyable_argval_str}'
    
    st.session_state.named_keys[keyname] = this_key
    
    if this_key not in st.session_state.memmapped:
        if not os.path.isfile(fpath):
            output_array_ = function(args)
            fpath = os.path.join(st.session_state.npy_dir, f'{this_key}.npy')
            np.save(fpath, output_array_)
            del output_array_

        st.session_state.memmapped[this_key] = np.lib.format.open_memmap(fpath, mode='r')

    return st.session_state.memmapped[this_key]


def load_image(input_file_path, input_source):

    print('─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐')

    OS_NAME = os.name
    if OS_NAME == 'posix':
        input_file_name = input_file_path.split('/')[-1]
    elif OS_NAME == 'nt':
        input_file_name = input_file_path.split('\\')[-1]

    st.session_state.input_file_name = input_file_name 
    st.session_state.input_file_ext = input_file_name.split('.')[-1]

    input_key = (input_file_name + input_source).replace('.','')
    st.session_state.input_key = input_key
    st.session_state.keys_to_images[input_key] = input_file_path

    # save input arrays to disk
    if input_key not in st.session_state.memmapped:

        fpath_u8 = os.path.join(st.session_state.npy_dir, input_key + '_u8.npy')
        fpath_f32 = os.path.join(st.session_state.npy_dir, input_key + '_f32.npy')
        fpath_f32_maxRGB = os.path.join(st.session_state.npy_dir, input_key + '_f32maxRGB.npy')

        files = [fpath_u8, fpath_f32, fpath_f32_maxRGB]    
        if input_key not in st.session_state.keys_to_npy:  # should only be true if the input key hasn't been processed before
            st.session_state.keys_to_npy[input_key] = tuple(files)  # make tuple to force copy and make immutable

            print(f'[{timestamp()}] checking if {fpath_u8} exists')

            if not all([os.path.isfile(fpath_u8), os.path.isfile(fpath_f32), os.path.isfile(fpath_f32_maxRGB)]):
                print(f'[{timestamp()}] Did not find existing npy files for image_input. Creating them now ...')
                image_u8 = cv2.imread(st.session_state.input_file_path)
                #st.session_state.keys_to_shape[input_key] = image_u8.shape

                image_f32 = normalize_array(image_u8)
                image_f32_maxRGB = image_f32.max(axis=2)
                
                print(f'[{timestamp()}] saving image_u8 array to new npy file: {fpath_u8}')           
                save_to_npy(fpath_u8, image_u8)
                del image_u8

                save_to_npy(fpath_f32, image_f32)
                del image_f32
                
                save_to_npy(fpath_f32_maxRGB, image_f32_maxRGB)
                del image_f32_maxRGB

        # open saved arrays as memory maps. keep track of them with weakrefs
        image_u8_ = np.lib.format.open_memmap(fpath_u8, mode='r', dtype=np.uint8)
        st.session_state.mmap_file_wref_lookup[fpath_u8] = (weakref.ref(image_u8_), 'image_u8_')

        image_f32_ = np.lib.format.open_memmap(fpath_f32, mode='r', dtype=np.float32)
        st.session_state.mmap_file_wref_lookup[fpath_f32] = (weakref.ref(image_f32_), 'image_f32_')

        image_f32_maxRGB_ = np.lib.format.open_memmap(fpath_f32_maxRGB, mode='r', dtype=np.float32)
        st.session_state.mmap_file_wref_lookup[fpath_f32_maxRGB] = (weakref.ref(image_f32_maxRGB_), 'image_f32_maxRGB_')

        st.session_state.memmapped[input_key] = (image_u8_, image_f32_, image_f32_maxRGB_)
    
    print('─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘')
    #return st.session_state.memmapped[input_key]
    return input_key