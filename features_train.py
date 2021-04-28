from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

SETTINGS.CAPTURE_MODE = "sys"  # type: ignore

# Create experiment.
ex = Experiment("houshou")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.automain
def run(results_directory: str, lambda_value: float):
    print(results_directory)
    print(lambda_value)
