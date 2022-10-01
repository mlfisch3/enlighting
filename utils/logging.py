import numpy as np
from PIL import Image
import datetime
import streamlit as st
import psutil
import os
import gc
import sys


def log_variable_size(var_name, x):
    s = sys.getsizeof(x)/float(2**20)
    print(f'Size of {var_name}: {s:.2f} MB')
    sys.stdout.flush()

def timestamp():
    return datetime.datetime.now().isoformat() #strftime("%Y%m%d_%H%M%S")

def log_memory(ref_id):
    pid = os.getpid()
    mem = psutil.Process(pid).memory_info()[0]/float(2**20)
    virt = psutil.virtual_memory()[3]/float(2**20)
    swap = psutil.swap_memory()[1]/float(2**20)

    print(f'[{timestamp()}]  [{ref_id}|{pid}]    rss: {mem:.2f} MB  (virtual: {virt:.2f} MB, swap: {swap:.2f} MB)')
    sys.stdout.flush()


def runtime(f):
    '''
       
        @runtime
        def test(n):
            j = 0
            for i in range(n):
                j+=1
       
        >>> test(100000)
        runtime: 0.0050120 s

    '''
    def run(*args):
        start = datetime.datetime.now()
        out = f(*args)
        end = datetime.datetime.now()
        elapsed = (end-start).total_seconds()
        print(f'runtime: {elapsed:.7f} s')
        return out
    return run

def view_memory(f):
    '''
    >>> @view_memory
        def test(a):
            return 2*a

    >>> l = test(np.ones((1000000,), dtype='uint8'))
    [2022-01-31T23:46:44.906963]  [test|B|32060]    rss: 200.70 MB  (virtual: 15989.61 MB, swap: 33152.62 MB)
    [2022-01-31T23:46:44.908984]  [test|E|32060]    rss: 201.70 MB  (virtual: 15989.46 MB, swap: 33152.77 MB)
    

    '''
    def run(*args):
        log_memory(f'{f.__name__}|B')
        out = f(*args)
        log_memory(f'{f.__name__}|E')
        return out
    return run
