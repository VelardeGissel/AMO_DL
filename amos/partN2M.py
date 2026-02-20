#GV 25.Sep-2025
from .midi2df2midi import midi_to_dataframe, save_midi_from_df
import pandas as pd
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Union
import joblib
#GV 25.11.2025
#remove part information from a nmat dataframe
def remove_part_inf(nmat):
    nmat.loc[nmat['track number']!=1, 'track number'] = 1
    nmat.loc[nmat['track name']!='Piano', 'track name'] = 'Piano'
    nmat.loc[nmat['channel']!=1, 'channel'] = 1
    nmat.loc[nmat['program']!=1, 'program'] = 1
    return nmat
 
'''
# ---- Example ----
a = np.arange(12).reshape((3, 2, 2))
flat_A = flatten_mnc(a)
print(flat_A)  # [ 0  2  4  6  8 10  1  3  5  7  9 11]

a_rec = unflatten_mnc(flat_A, 3, 2, 2)
print(np.array_equal(a, a_rec))  # True
'''
def flatten_mnc(A: np.ndarray) -> np.ndarray:
    
    """
    gv 16.02.2026
    Flatten an array of shape (m, n, c) into shape (m*n*c,)
    using the order shown in the example:
      A[:,:,0] (ravel C), then A[:,:,1], ..., then A[:,:,c-1]
    """
    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError("Expected a 3D array of shape (m, n, c).")
    # move c to the front, then ravel in C order
    return A.transpose(2, 0, 1).reshape(-1)


def unflatten_mnc(flat: np.ndarray, m: int, n: int, c: int) -> np.ndarray:
    """
    gv 16.02.2026
    Inverse of flatten_mnc. Reconstructs array of shape (m, n, c)
    from a flat vector of length m*n*c.
    """
    flat = np.asarray(flat)
    if flat.size != m * n * c:
        raise ValueError(f"flat has length {flat.size}, expected {m*n*c}.")
    # reshape to (c, m, n) then move c back to last axis
    return flat.reshape(c, m, n).transpose(1, 2, 0)

def createXy(pianoroll_p, samples_per_qn = 12 ):
    #GV. 18.Feb
    #chunking piano and orchestral versions every quarter note.
    #the last onset is quarter notes ->shape*samples_per_qn
    #1. extract the matrix 128*samples_per_qn*parts
    #2. flaten, and stack the flattened piano, or orchestral labels.
    l_onset = int(pianoroll_p.shape[1]/samples_per_qn)
    si = 0
    #for X
    sample_i = pianoroll_p[:,si:si+samples_per_qn,:]
    si = si + samples_per_qn
    X = flatten_mnc(sample_i)
        
    for i in range(1,l_onset):
        sample_i = pianoroll_p[:,si:si+samples_per_qn,:]
        si = si + samples_per_qn
        flat_A = flatten_mnc(sample_i)
        X = np.vstack([X, flat_A])
    return X