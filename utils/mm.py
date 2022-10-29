import streamlit as st
import weakref
import os
from utils.logging import timestamp

def pivot_dict(d):
    '''
    Swaps keys and values of a dictionary d
    Each unique value becomes a key, whose value is a list of all keys in d for which it is the value
    
    Example:
    >>> d0 = {'a':2, 'b':2, 'c':1, 'd':2, 'e':3}
    >>> pivot_dict(d0)
    {1: ['c'], 2: ['a', 'b', 'd'], 3: ['e']}

    '''

    d_ = {}
    new_keys = list(set(d.values()))
    for k in new_keys:
        d_[k] = []
    for k,v in d.items():
        d_[v].append(k)
        
    return d_     

def references_dead_object(wref):
  '''
  returns true if object referred to by weak reference is dead
  '''
  #len([u for u in str(_11).replace('>',' ').split(' ') if u == 'dead'])
  return any([u=='dead' for u in str(wref).replace('>',' ').split(' ')])

def clear(cache_checked, data_checked):
 
  print(f'[{timestamp()}] Clearing resources ...')
  if st.session_state.data_checked:
    print(f'[{timestamp()}] Clearing ALL ...')
    print(f'[{timestamp()}] Clearing cache ...')
    clear_cache()
    print(f'[{timestamp()}] Clearing data ...')
    clear_data()
  elif st.session_state.cache_checked:
    print(f'[{timestamp()}] Clearing cache ...')
    clear_cache()

  print('\n')

def clear_data():

  print(f'[{timestamp()}] Deleting locally saved images ...')
  for k,v in st.session_state.saved_images.items():
    if all([os.path.isfile(k), v]):
      print(f'[{timestamp()}] Deleting file {k} ...')
      os.remove(k)
      st.session_state.saved_images[k] = False

  # print(f'[{timestamp()}] Clearing cached arrays ...')
  # clear_cache()


def clear_cache():
  
    print(f'[{timestamp()}] Closing open memory mapped files ...')
    if len(st.session_state.mmap_file_wref_lookup.keys()) > 0:
      # (try) to close any open memory mapped files and unbind the mapped variable
      for k,v in st.session_state.mmap_file_wref_lookup.items():
          if references_dead_object(v[0]):
              print(f'object of weak reference \'{v[1]}\' has already been garbage collected')
          elif not v[0]()._mmap.closed:
              filename = vars(v[0]())['filename']
              print(f'object of weak reference \'{v[1]}\' is bound to open memory mapped file{filename}. closing ...')
              v[0]()._mmap.close()
              if v[0]()._mmap.closed:
                  print(f"Successfully closed memory mapped file {vars(v[0]())['filename']} bound to object \'{v[1]}\'")
              else:
                  print(f"Unable to close memory mapped file {vars(v[0]())['filename']} bound to object \'{v[1]}\'")
  
      print(f'[{timestamp()}] Clearing keys from memory map ...')
      del st.session_state.memmapped
      st.session_state.memmapped = {}

      st.session_state.purge_count += 1
      print('completed memory map purge')
      print(f'app.py||PURGE_COUNT: {st.session_state.purge_count}')

def get_mmaps():
  '''
  gets filename for each memory-mapped variable
  Usage
  >>> mmaps = get_mmaps()
  >>> for k,v in mmaps.items():
        print(k,v)
  '''
  mempaths={}
  globals_copy = globals().copy()
  for name in globals_copy.keys():
      try:
          name_ref = weakref.ref(eval(compile(name,'tmp.txt', 'eval')))
      except (NameError, TypeError):
          continue
      if hasattr(name_ref(), '__dict__'):
          v = vars(name_ref())
          if '_mmap' in v.keys():
              mempaths[name] = v['filename']

  return mempaths



def get_weakrefs():
  '''
  Usage
  >>> weakrefs = get_weakrefs()
  >>> for k,v in weakrefs.items():
        print(k,v)
  '''

  weak={}

  globals_copy = globals().copy()
  for f in globals_copy.keys():
      if type(globals_copy[f])==weakref.ref:
          weak[f]=globals_copy[f]
  return weak


if __name__ == '__main__':
  
  print('\n mmaps:  \n')
  
  mmaps = get_mmaps()

  for k,v in mmaps.items():
    print(k,v)

  print('\n Weak References:  \n')

  weakrefs = get_weakrefs()
  for k,v in weakrefs.items():
        print(k,v)