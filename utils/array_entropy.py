import streamlit as st
import numpy as np

def entropy(array, bins=255, lo=0, hi=255):

    if array.dtype.name[:5] == 'float':
        array = (array * 255).astype(np.uint8)
    
    counts = np.histogram(array,bins=bins, range=(lo,hi))[0]
    counts = counts[counts>0]
    N = counts.sum()
    return (np.log2(N) - (np.dot(counts, np.log2(counts)) / N)).astype(np.float32)
    
def xentropy(p, q, bins=255, lo=0, hi=255):
    if p.dtype.name[:5] == 'float':
        p = (p * 255).astype(np.uint8)

    if q.dtype.name[:5] == 'float':
        q = (q * 255).astype(np.uint8)
    
    counts_p = np.histogram(p,bins=bins, range=(lo,hi))[0] + 1e-5
    counts_q = np.histogram(q,bins=bins, range=(lo,hi))[0] + 1e-5
    N = counts_q.sum()
    return (np.log2(N) - (np.dot(counts_p, np.log2(counts_q)) / N)).astype(np.float32)

def kl_divergence(P,Q, bins=255, lo=0, hi=255):
    return xentropy(P,Q, bins=bins, lo=lo, hi=hi) - entropy(P, bins=bins, lo=lo, hi=hi)

def js_divergence(p,q):
    m = 0.5*(p+q)
    return 0.5*(kl_divergence(p,m)+ kl_divergence(q,m))

def bits2U16(bit_hot_array, mask=None):
    base2=2**np.arange(16, dtype=np.uint16)[::-1]
    if mask is not None:
        base2=base2*mask
    return np.dot(bit_hot_array, base2)

def bit_split(array):
    return np.unpackbits(array).reshape(*array.shape, 8)

def bit_concat(p, q):
    if p.dtype != 'uint8':
        if p.dtype.name[:5] == 'float':
            p = 255 * p
        p = p.astype(np.uint8)
    if q.dtype != 'uint8':
        if q.dtype.name[:5] == 'float':
            q = 255 * q
        q = q.astype(np.uint8)
        
    p_ = bit_split(p.ravel())
    q_ = bit_split(q.ravel())
    pq = np.hstack([p_,q_])
    return bits2U16(pq)

def joint_entropy(p,q, bins=256*256, lo=0, hi=256*256):
    pq = bit_concat(p,q)
    pq_counts = np.histogram(pq, bins=bins, range=(lo,hi))[0]
    pq_counts = pq_counts[pq_counts>0]
    N = pq_counts.sum()
    return (np.log2(N) - (np.dot(pq_counts, np.log2(pq_counts)) / N)).astype(np.float32)

def mutual_information(p,q):
    return entropy(p) + entropy(q) - joint_entropy(p,q)

def variation_of_information(p,q, normalized=True):
    joint = joint_entropy(p,q) + 1e-5
    mutual = mutual_information(p,q)
    if normalized:
        return (1. - mutual/joint)
    else:
        return joint - mutual

def normalized_variation_of_information(p,q):
    joint = joint_entropy(p,q) + 1e-5
    mutual = mutual_information(p,q)
    return (1. - mutual/joint)

def conditional_entropy(p,q):
    return entropy(p) - mutual_information(p,q)



