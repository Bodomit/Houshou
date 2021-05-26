from argparse import ArgumentParser

from houshou.utils import find_last_epoch_path


def main(experiment_path: str):
    # Get the path to the last checkpoint.
    feature_model_checkpoint_path = find_last_epoch_path(experiment_path)

    print("Experiment Directory: ", experiment_path)
    print("Checkpoint Path; ", feature_model_checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_path")
    args = parser.parse_args()
    main(args.experiment_path)
