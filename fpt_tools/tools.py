
import numpy as np 

def centered_fpt(fpt, window):
    onsets = fpt[:,5].copy()
    onsets -= window // 2
    NAN = nan_value(fpt)
    cfpt = np.where(fpt != NAN, fpt - onsets[:, np.newaxis], NAN)
    return cfpt

def out_of_window_mask(fpt, window, cols=None):
    """Return rows with some cols fiducials out of window range including nan values."""

    cfpt = centered_fpt(fpt, window)

    if cols is not None:
        cfpt = cfpt[:, cols]

    full_filter = np.zeros_like(cfpt, dtype=np.bool8)
    np.logical_or(cfpt < 0, cfpt > window-1, out=full_filter)
    
    filter = np.zeros(cfpt.shape[0], dtype=np.bool8)
    np.any(full_filter, axis=1, out=filter)

    mask = np.logical_not(filter)
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

    # make a beat template
    template = np.mean(
        data,
        axis=0,
        where=np.logical_not(filter)[:, np.newaxis],
    )

    # compute cc
    cc = np.corrcoef(
        data,
        y = template,
    )[:-1,-1]

    # excluding rows with cc<0.97 
    cc_filter = cc < cc_thr
    
    # exclude filter from cc_mask
    cc_filter[filter] = False

    return np.logical_not(cc_filter)
    
def excursion_mask(fpt, window, filter, cols=None, q_range = (0., 1.)):
    
    cfpt = centered_fpt(fpt, window)

    if cols is not None:
        cfpt = cfpt[:, cols]
    
    pre_mask = np.logical_not(filter)
    fcfpt = cfpt[pre_mask,:]

    q1 = np.quantile(fcfpt, q_range[0], axis=0)
    q3 = np.quantile(fcfpt, q_range[1], axis=0)

    ex_mask = np.ones(cfpt.shape[0], dtype=np.bool8)
    for i in range(cfpt.shape[1]):
        ex_mask[:] = ex_mask & (cfpt[:,i] >= q1[i]) & (cfpt[:,i] <= q3[i])

    return ex_mask


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