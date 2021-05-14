from typing import Tuple, Any

import numpy as np

Label = int
Sample = Tuple[Any, Label]
AnnotatedSample = Tuple[Any, Label, int]
Pair = Tuple[AnnotatedSample, AnnotatedSample]
ROCCurve = Tuple[np.ndarray, np.ndarray, np.ndarray]
