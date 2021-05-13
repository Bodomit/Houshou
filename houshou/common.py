from typing import Tuple, Any

Label = int
Sample = Tuple[Any, Label]
AnnotatedSample = Tuple[Any, Label, int]
Pair = Tuple[AnnotatedSample, AnnotatedSample]
