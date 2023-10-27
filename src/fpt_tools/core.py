
import numpy as np 
import numpy.ma as ma 
IntNaN = 2147483647

class MatrixRowMasked(ma.MaskedArray):

    def set_masked_rows(self, masked_rows: np.ndarray):
        self.mask[masked_rows,:] = True

    def get_masked_indices(self):
        return np.nonzero(self.get_row_mask())[0]
    
    def get_row_mask(self):
        return self.mask[:, 0]
    
    
class FPTMasked:

    def __init__(self, fpt:np.ndarray, masked_cols: np.ndarray, beat_matrix: np.ndarray) -> None:
        """Table columns: Pon, Peak, Poff, QRSon, Qpeak, Rpeak, Speak, QRSoff, res, Ton, Tpeak, Toff, res
        fpt in samples
        """

        mask = np.zeros_like(fpt, dtype=np.bool_)
        mask[fpt == IntNaN] = True # mask nan
        mask[:, [8, 12]]= True # mask res
        mask[:, masked_cols] = True # custom col mask


        valid_cols = np.ones(13, dtype=np.bool_)
        valid_cols[[8, 12]] = False
        valid_cols[masked_cols] = False
        self.invalid_fiducial_mask = np.any(mask[:, valid_cols], axis=1)
        
        self.fpt = ma.array(
            fpt,
            dtype=np.int32,
            mask=mask,
            fill_value=IntNaN,
            hard_mask=True
            )

        self.nbeats, self.window = beat_matrix.shape
        self.cfpt = self.center_fpt(self.fpt, self.window)

        self.beat_matrix = MatrixRowMasked(
            beat_matrix,
            mask=False,
            fill_value=np.nan,
            hard_mask=True
            )

        self.measurements = MatrixRowMasked(
            np.empty((self.nbeats, 9), dtype=np.float64),
            mask=False, 
            fill_value=np.nan,
            hard_mask=True
            )
    
        self.cfpt = ma.filled(self.cfpt, fill_value=0) # force to zero to eval beat_matrix in set_measurement
        self.ix = np.arange(self.nbeats)

    def set_masked_rows(self, mask):
        self.beat_matrix.set_masked_rows(mask)
        self.measurements.set_masked_rows(mask)

    def set_out_of_window_mask(self) -> np.ndarray:
        """Return rows with some cols fiducials out of window range including nan values."""
        mask = ma.any(ma.logical_or(self.cfpt < 0, self.cfpt > self.window-1), axis=1) # ma.any: Masked values are considered as False during computation.
        self.set_masked_rows(mask)
        return np.count_nonzero(mask)

    def set_invalid_fiducial_mask(self):
        mask = self.invalid_fiducial_mask
        self.set_masked_rows(mask)
        return np.count_nonzero(mask)

    @staticmethod
    def center_fpt(fpt, window) -> np.ndarray:
        r_pos = fpt[:,5].copy()
        r_pos -= window // 2
        return fpt - r_pos[:, np.newaxis]

    def set_lesser_than_cc_mask(self, threshold):
        # compute cc
        cc = np.corrcoef(
            self.beat_matrix.data,
            y = self.template_beat(),
        )[:-1,-1]
        cc[self.beat_matrix.get_row_mask()] = np.inf #  ma.corrcoef is very slow, instead use np.corrcoef and force the mask
        mask = cc < threshold
        self.set_masked_rows(mask)
        return np.count_nonzero(mask)
    
    def set_measurements(self, fs: float):
        self.measurements[:, 0] = (self.fpt[:,2] - self.fpt[:,0]) / fs # P duration
        self.measurements[:, 1] = (self.fpt[:,3] - self.fpt[:,0]) / fs # PR interval
        self.measurements[:, 2] = (self.fpt[:,7] - self.fpt[:,3]) / fs # QRS duration
        self.measurements[:, 3] = (self.fpt[:,11] - self.fpt[:,3]) / fs # QT interval
        self.measurements[:, 4] = self.beat_matrix[self.ix, self.cfpt[:,1]] # Ppeak
        self.measurements[:, 5] = self.beat_matrix[self.ix, self.cfpt[:,4]] # Qpeak
        self.measurements[:, 6] = self.beat_matrix[self.ix, self.cfpt[:,5]] # Rpeak
        self.measurements[:, 7] = self.beat_matrix[self.ix, self.cfpt[:,6]] # Speak
        self.measurements[:, 8] = self.beat_matrix[self.ix, self.cfpt[:,10]] # Tpeak

    def template_beat(self):
        return ma.mean(
            self.beat_matrix,
            axis=0,
        )