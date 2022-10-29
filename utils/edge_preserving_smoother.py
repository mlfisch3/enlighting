import streamlit as st
import os
import numpy as np
import weakref
import cv2

from scipy import signal
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator, use_solver

from utils.array_tools import diff, cyclic_diff, imresize, float32_to_uint8, normalize_array
from utils.io_tools import save_to_npy

def delta(x):   

    gradient_v = np.vstack([diff(x, axis=0),cyclic_diff(x,axis=0)])
    gradient_h = np.hstack([diff(x,axis=1),cyclic_diff(x,axis=1)])

    return gradient_v, gradient_h

def convolve(gradient_v, gradient_h, kernel_shape=(5,1), return_level=1):
    ''' 
        gradient_v:         forward-difference in vertical direction
        gradient_h:         forward-difference in horizontal direction 
        kernel_shape:       2-tuple of odd positive integers. (Auto-incremented if even)
        return_level 1:     return only standard convolution
        return_level 2:     return standard convolution and convolution of absolute values
        return_level -2:    return only convolution of absolute values
    '''

    sigma_v, sigma_h = kernel_shape
    sigma_v += 1-sigma_v%2
    sigma_h += 1-sigma_h%2

    n_pad = [int(sigma_v/2),int(sigma_h/2)]
    kernel = np.ones(kernel_shape)

    out = []
    if (return_level > 0) or (return_level==-1):
        convolution_v = signal.convolve(gradient_v, kernel, method='fft')[n_pad[0]:n_pad[0]+gradient_v.shape[0],n_pad[1]:n_pad[1]+gradient_v.shape[1]]
        convolution_h = signal.convolve(gradient_h, kernel.T, method='fft')[n_pad[1]:n_pad[1]+gradient_h.shape[0],n_pad[0]:n_pad[0]+gradient_h.shape[1]]
        out.append(convolution_v)
        out.append(convolution_h)

    if (return_level > 1) or (return_level==-2):
        convolution_v_abs = signal.convolve(np.abs(gradient_v), kernel, method='fft')[n_pad[0]:n_pad[0]+gradient_v.shape[0],n_pad[1]:n_pad[1]+gradient_v.shape[1]]
        convolution_h_abs = signal.convolve(np.abs(gradient_h), kernel.T, method='fft')[n_pad[1]:n_pad[1]+gradient_h.shape[0],n_pad[0]:n_pad[0]+gradient_h.shape[1]]
        out.append(convolution_v_abs)
        out.append(convolution_h_abs)

    return tuple(out)

def calculate_texture_weights(image_01_maxRGB_reduced, restore_shape, kernel_shape=(5,1), sharpness=0.001, texture_style='I'):
    '''
    returns memory-maps to the following 2d arrays stored on disk as .npy files (all arrays are "reduced"-size (i.e., scaled down)):
    gradient_v, gradient_h
    texture_weights_v, texture_weights_h

    '''
    if st.session_state.keys_.texture_weights_key not in st.session_state.memmapped:        

        if st.session_state.keys_.gradients_key not in st.session_state.memmapped:        
            fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.gradients_key}.npy')
            if fpath in st.session_state.mmap_file_wref_lookup:
                if not st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.closed:
                    st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.close()
                    _ = st.session_state.mmap_file_wref_lookup.pop(fpath, None)

                    print(f'[{timestamp()}] Lingering open file was closed via weakref:')
                    print(f'[{timestamp()}]      FILENAME: {fpath}\nMMAP_NAME: {st.session_state.mmap_file_wref_lookup[fpath][1]}')    

            if not os.path.isfile(fpath):            
                save_to_npy(fpath, np.array(delta(image_01_maxRGB_reduced)))

            st.session_state.keys_to_npy[st.session_state.keys_.gradients_key] = fpath
            gradients = np.lib.format.open_memmap(fpath, mode='r')
            st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(gradients), 'gradients')
            st.session_state.memmapped[st.session_state.keys_.gradients_key] = (gradients[0], gradients[1])

        gradient_v, gradient_h = st.session_state.memmapped[st.session_state.keys_.gradients_key]
        
        
        if st.session_state.keys_.convolutions_key not in st.session_state.memmapped:        
            fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.convolutions_key}.npy')
            if not os.path.isfile(fpath):
                save_to_npy(fpath, np.array(convolve(gradient_v, gradient_h, kernel_shape)))
            st.session_state.keys_to_npy[st.session_state.keys_.convolutions_key] = fpath
            convolutions = np.lib.format.open_memmap(fpath, mode='r')
            st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(convolutions), 'convolutions')
            st.session_state.memmapped[st.session_state.keys_.convolutions_key] = (convolutions[0], convolutions[1])

        convolution_v, convolution_h = st.session_state.memmapped[st.session_state.keys_.convolutions_key]
   

        fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.texture_weights_key}.npy')
        if fpath in st.session_state.mmap_file_wref_lookup:
            if not st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.closed:
                st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.close()
                _ = st.session_state.mmap_file_wref_lookup.pop(fpath, None)

                print(f'[{timestamp()}] Lingering open file was closed via weakref:')
                print(f'[{timestamp()}]      FILENAME: {fpath}\nMMAP_NAME: {st.session_state.mmap_file_wref_lookup[fpath][1]}')  
        
        impath = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.total_variation_map_key}.{st.session_state.input_file_ext}')            
        st.session_state.keys_to_images[st.session_state.keys_.total_variation_map_key] = impath
        impath_M = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.texture_weights_map_key}.{st.session_state.input_file_ext}')
        st.session_state.keys_to_images[st.session_state.keys_.texture_weights_map_key] = impath_M

        if not all([os.path.isfile(fpath), os.path.isfile(impath_M), os.path.isfile(impath)]): 
            if texture_style == 'I':
                texture_weights_v = 1/(np.abs(convolution_v) * np.abs(gradient_v) + sharpness)
                texture_weights_h = 1/(np.abs(convolution_h) * np.abs(gradient_h) + sharpness)

            elif texture_style == 'II':
                texture_weights_v = 1/(np.abs(gradient_v) + sharpness)
                texture_weights_h = 1/(np.abs(gradient_h) + sharpness)

            elif texture_style == 'III':
                texture_weights_v = 1/(np.abs(convolution_v) + sharpness)
                texture_weights_h = 1/(np.abs(convolution_h) + sharpness)
            else:
                convolution_v_abs, convolution_h_abs = convolve(gradient_v, gradient_h, kernel_shape, return_level=-2)

                if texture_style == 'IV':
                    texture_weights_v = convolution_v_abs/(np.abs(convolution_v) + sharpness)
                    texture_weights_h = convolution_h_abs/(np.abs(convolution_h)+ sharpness)

                elif texture_style == 'V':
                    texture_weights_v = convolution_v_abs/(np.abs(convolution_v) + sharpness)
                    texture_weights_h = convolution_h_abs/(np.abs(convolution_h)+ sharpness)

            # create total_variation_map, upsampled to full-size, save as image to disk, then free the resource 
        
            total_variation_map = np.abs(gradient_v * texture_weights_v + gradient_h * texture_weights_h)
            total_variation_map = total_variation_map.clip(min=0, max=total_variation_map.ravel().mean()+total_variation_map.ravel().std())
            st.session_state.saved_images[impath] = cv2.imwrite(impath, float32_to_uint8(normalize_array(imresize(total_variation_map, size=restore_shape))))
            del total_variation_map

            #texture_weights_map = np.dstack([np.zeros_like(texture_weights_v),texture_weights_v, texture_weights_h])
            #texture_weights_map = np.sqrt(np.power(texture_weights_v,2) + np.power(texture_weights_h,2))
            texture_weights_map = (np.abs(texture_weights_v) + np.abs(texture_weights_h))/2
            texture_weights_mean = texture_weights_map.ravel().mean()
            texture_weights_std = texture_weights_map.ravel().std()
            print(f'texture_weights_mean: {texture_weights_mean}')
            print(f'texture_weights_std: {texture_weights_std}')
            #texture_weights_map = texture_weights_map.clip(min=texture_weights_mean-texture_weights_std, max=texture_weights_mean+texture_weights_std)
            texture_weights_map = texture_weights_map.clip(min=0, max=texture_weights_mean+texture_weights_std)  # prevent value range from extending to 1/sharpness ~ 10^4
            st.session_state.saved_images[impath] = cv2.imwrite(impath_M, float32_to_uint8(normalize_array(imresize(texture_weights_map, size=restore_shape))))
            del texture_weights_map

            save_to_npy(fpath, np.array([texture_weights_v, texture_weights_h]))
            del texture_weights_v, texture_weights_h

        st.session_state.keys_to_npy[st.session_state.keys_.texture_weights_key] = fpath
        texture_weights = np.lib.format.open_memmap(fpath, mode='r')
        st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(texture_weights), 'texture_weights')
        st.session_state.memmapped[st.session_state.keys_.texture_weights_key] = (texture_weights[0], texture_weights[1])

    return st.session_state.memmapped[st.session_state.keys_.texture_weights_key]


def construct_map_cyclic(texture_weights_v, texture_weights_h, lamda):
    ''' all cyclic elements present '''
    r, c = texture_weights_h.shape        
    k = r * c
    texture_weights_h = np.asfortranarray(texture_weights_h.astype('float32'))
    texture_weights_v = np.asfortranarray(texture_weights_v.astype('float32'))
    lamda = np.float32(lamda)

    dh = -lamda * texture_weights_h.ravel(order='F')
    dv = -lamda * texture_weights_v.ravel(order='F')

    texture_weights_h_permuted_cols = np.asfortranarray(np.roll(texture_weights_h,1,axis=1))
    dh_permuted_cols = -lamda * texture_weights_h_permuted_cols.ravel(order='F')
    texture_weights_v_permuted_rows = np.asfortranarray(np.roll(texture_weights_v,1,axis=0))
    dv_permuted_rows = -lamda * texture_weights_v_permuted_rows.ravel(order='F')
       
    texture_weights_h_permuted_cols_head = np.zeros_like(texture_weights_h_permuted_cols, dtype='float32', order='F') 
    texture_weights_h_permuted_cols_head[:,0] = texture_weights_h_permuted_cols[:,0]
    dh_permuted_cols_head = -lamda * texture_weights_h_permuted_cols_head.ravel(order='F')
    
    texture_weights_v_permuted_rows_head = np.zeros_like(texture_weights_v_permuted_rows, dtype='float32', order='F')
    texture_weights_v_permuted_rows_head[0,:] = texture_weights_v_permuted_rows[0,:]
    dv_permuted_rows_head = -lamda * texture_weights_v_permuted_rows_head.ravel(order='F')

    texture_weights_h_no_tail = np.zeros_like(texture_weights_h, dtype='float32', order='F')
    texture_weights_h_no_tail[:,:-1] = texture_weights_h[:,:-1]
    dh_no_tail = -lamda * texture_weights_h_no_tail.ravel(order='F')

    texture_weights_v_no_tail = np.zeros_like(texture_weights_v, dtype='float32', order='F')
    texture_weights_v_no_tail[:-1,:] = texture_weights_v[:-1,:]
    dv_no_tail = -lamda * texture_weights_v_no_tail.ravel(order='F')
    
    Ah = spdiags([dh_permuted_cols_head, dh_no_tail], [-k+r, -r], k, k)
    
    Av = spdiags([dv_permuted_rows_head, dv_no_tail], [-r+1,-1],  k, k)
    
    A = 1 - (dh + dv + dh_permuted_cols + dv_permuted_rows)

    d = spdiags(A, 0, k, k)
    
    A = Ah + Av
    A = A + A.T + d
    return A

def solve_sparse_system(A, B, method='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, x0=None):
    """
    
    Consolidation of two functions into one:
    1) solve_linear_equation(A, B)
    2) solver_sparse(A, B)

    Solves for x = b/A  [[b is vector(B)]]
    A can be sparse (csc or csr) or dense
    b must be dense
    
   """    

    r, c = B.shape
    
    b = B.ravel(order='F').astype(np.float32)
    
    N = A.shape[0]
    if method == 'cg': 
        if CG_prec == 'ILU':
            A_ilu = spilu(A.tocsc(), drop_tol=LU_TOL, fill_factor=FILL)
            M = LinearOperator(shape=(N, N), matvec=A_ilu.solve, dtype='float32')
        else:
            M = None
        if x0 is None:
            x0 = b  # input image more closely resembles its smoothed self than a draw from any distribution
        return cg(A, b, x0=x0, tol=CG_TOL, maxiter=MAX_ITER, M=M)[0].astype(np.float32).reshape(r,c, order='F')

    elif method == 'direct':
        use_solver( useUmfpack = False ) # use single precision
        return spsolve(A, b).astype(np.float32).reshape(r,c, order='F')

def smooth(image_01_maxRGB_reduced, restore_shape, texture_style='I', kernel_shape=(5,1), sharpness=0.001, lamda=0.5, solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, return_texture_weights=False):

    # these outputs are memory-mapped to .npy files
    texture_weights_v, texture_weights_h = calculate_texture_weights(image_01_maxRGB_reduced, restore_shape, kernel_shape=kernel_shape, sharpness=sharpness, texture_style=texture_style)
    
    # smoother requires texture_weights AND lambda value

    if st.session_state.keys_.smoother_output_fullsize_key not in st.session_state.memmapped:        

        print(f'memmapped: NEW KEY [st.session_state.keys_.smoother_output_fullsize_key]: {st.session_state.keys_.smoother_output_fullsize_key}')

        fpath = os.path.join(st.session_state.npy_dir, f'{st.session_state.keys_.smoother_output_fullsize_key}.npy')
        if fpath in st.session_state.mmap_file_wref_lookup:
            if not st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.closed:
                st.session_state.mmap_file_wref_lookup[fpath][0]()._mmap.close()
                _ = st.session_state.mmap_file_wref_lookup.pop(fpath, None)

                print(f'[{timestamp()}] Lingering open file was closed via weakref:')
                print(f'[{timestamp()}]      FILENAME: {fpath}\nMMAP_NAME: {st.session_state.mmap_file_wref_lookup[fpath][1]}')  

        if not os.path.isfile(fpath):
            A = construct_map_cyclic(texture_weights_v, texture_weights_h, lamda)
            
            image_01_maxRGB_reduced_smooth = solve_sparse_system(A, image_01_maxRGB_reduced, method=solver, CG_prec=CG_prec, CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL, x0=None)
            
            image_01_maxRGB_smooth = np.clip(imresize(image_01_maxRGB_reduced_smooth, size=restore_shape),0.,1.)

            impath = os.path.join(st.session_state.image_dir, f'{st.session_state.keys_.smoother_output_fullsize_key}.{st.session_state.input_file_ext}') 
            st.session_state.saved_images[impath] = cv2.imwrite(impath, float32_to_uint8(image_01_maxRGB_smooth))
            save_to_npy(fpath, image_01_maxRGB_smooth)
            st.session_state.keys_to_npy[st.session_state.keys_.smoother_output_fullsize_key] = fpath
            del image_01_maxRGB_smooth, image_01_maxRGB_reduced_smooth

        smoother_output_fullsize = np.lib.format.open_memmap(fpath, mode='r')
        st.session_state.mmap_file_wref_lookup[fpath] =  (weakref.ref(smoother_output_fullsize), 'smoother_output_fullsize')

        st.session_state.memmapped[st.session_state.keys_.smoother_output_fullsize_key] = smoother_output_fullsize

    #return st.session_state.memmapped[smoother_output_fullsize_key]
    return st.session_state.keys_.smoother_output_fullsize_key
