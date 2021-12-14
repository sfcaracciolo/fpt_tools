
import numpy as np 

def centered_fpt(fpt, window):
    onsets = fpt[:,5]
    onsets -= window // 2
    NAN = nan_value(fpt)
    cfpt = np.where(fpt != NAN, fpt - onsets[:, np.newaxis], NAN)
    return cfpt

def out_of_window_mask(fpt, window, cols=None):
    """Return rows with some cols fiducials out of window range including nan values."""
    if cols is not None:
        fpt = fpt[:, cols]

    cfpt = centered_fpt(fpt, window)

    full_mask = np.zeros_like(cfpt, dtype=np.bool8)
    np.logical_or(cfpt < 0, cfpt > window-1, out=full_mask)
    
    mask = np.zeros_like(cfpt.shape[0], dtype=np.bool8)
    np.any(full_mask, axis=1, out=mask)

    return mask

def nan_value(fpt):
    if fpt.dtype == np.int32:
        NAN = 2147483647
    elif fpt.dtype == np.float32:
        NAN = np.nan
    else:
        raise ValueError('FPT datatype does not supported.')

    return NAN

def cc_mask(data, filter, cc_thr = .97):

    if filter.ndim == 1:
        filter = filter[:, np.newaxis]

    # make a beat template
    template = np.mean(
        data,
        axis=0,
        where=np.logical_not(filter),
    )

    # compute cc
    cc = np.corrcoef(
        data,
        y = template,
    )[:-1,-1]

    # excluding rows with cc<0.97 
    mask = cc < cc_thr
    # exclude filter from cc_mask
    mask[filter] = False

    return mask
    
def measurements(data, fpt, filter):
    NAN = nan_value(fpt)
    m = np.empty((fpt.shape[0], 9), dtype=NAN.dtype)

    m[:, 0] = np.where(filter, NAN, fpt[:,2] - fpt[:,0]) # P duration
    m[:, 1] = np.where(filter, NAN, fpt[:,3] - fpt[:,0]) # PR interval
    m[:, 2] = np.where(filter, NAN, fpt[:,7] - fpt[:,3]) # QRS duration
    m[:, 3] = np.where(filter, NAN, fpt[:,11] - fpt[:,3]) # QT interval
    m[:, 4] = np.where(filter, NAN, data[fpt[:,1]]) # Ppeak
    m[:, 5] = np.where(filter, NAN, data[fpt[:,4]]) # Qpeak
    m[:, 6] = np.where(filter, NAN, data[fpt[:,5]]) # Rpeak
    m[:, 7] = np.where(filter, NAN, data[fpt[:,6]]) # Speak
    m[:, 8] = np.where(filter, NAN, data[fpt[:,10]]) # Tpeak

    return m