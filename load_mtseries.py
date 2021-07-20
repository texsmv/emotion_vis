from sktime.datasets import load_basic_motions
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np

# x, y = load_from_tsfile_to_dataframe('Wafer/Wafer_TRAIN.ts')
# x, y = load_from_tsfile_to_dataframe('Libras/Libras_TRAIN.ts')
# x, y = load_from_tsfile_to_dataframe('UWaveGestureLibraryAll/UWaveGestureLibraryAll_TRAIN.ts')
# x, y = load_from_tsfile_to_dataframe('StandWalkJump/StandWalkJump_TRAIN.ts')
x, y = load_from_tsfile_to_dataframe('Handwriting/Handwriting_TRAIN.ts')



print(x.shape)
print(y.shape)

print(np.array(x.to_numpy()[1][0]).shape)