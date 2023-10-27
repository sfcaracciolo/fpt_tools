# fpt_tools
Simple tools for FPT (fiducial point table) processing. 

The main idea is to work with FPT and beat matrix simultaneously using numpy MaskedArray to track bad beats specified by different exclusion criteria.

## Usage

```python
from fpt_tools import FPTMasked

fpt = # set FPT np.int32 datatype with the following 13 columns in samples: Pon, Peak, Poff, QRSon, Qpeak, Rpeak, Speak, QRSoff, res, Ton, Tpeak, Toff, res. Where res are reserved columns. Like int dtype has not nan value, put 2147483647 is delineation fail.

beat_matrix = # set the beats matrix with shape (n_beats, windows_size)

# create the model
model = FPTMasked(
    fpt,
    masked_cols, # This allows some columns to be excluded from the analysis. For example, Qpeak does not exist in the WR model.
    beat_matrix
)
ofw_amount = model.set_out_of_window_mask() # excluding beats with some fiducial out of window
ifm_amount = model.set_invalid_fiducial_mask() # excluding beats with some nan in valid cols
cc_amount = model.set_lesser_than_cc_mask(.90) # excluding beats in comparision with template beat.
bad_beats = model.beat_matrix.get_masked_indices() # take indexes of excluded beats

model.set_measurements(fs) # compute measurements with this columns: PR interval, QRS duration, QT interval, Ppeak, Qpeak, Rpeak, Speak, Tpeak.
```