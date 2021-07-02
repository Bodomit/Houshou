import argparse
import glob
import itertools
import os
import pickle
import re
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import tqdm
from ruyaml import YAML
from sklearn.metrics import auc

from houshou.utils import get_lambdas, sort_lambdas

TEST_SUBSETS = {
    "faces": ["vggface2_MTCNN", "CelebA_MTCNN"],
    "fullbody": ["Market-1501", "RAP2"],
}

METRICS_COLUMN_NAME_MAP = {
    "lambda": "Lambda",
    "train_auc": "Train AUC",
    "test_auc": "Test AUC",
    "threshold": "Threshold",
    "true_negatives": "TN",
    "false_positives": "FP",
    "false_negatives": "FN",
    "true_positives": "TP",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "far": "FAR",
    "frr": "FRR",
    "binary_accuracy": "Accuracy",
    "loss": "Loss",
    "loss_total": "Loss",
    "val_loss_total": "Val Loss",
}

ROC = Tuple[np.ndarray, np.ndarray, float]
SPLIT_ROCS = List[ROC]


def main(
    input_directory: str,
    output_directory: str,
    skip_verification_rocs: bool = False,
    skip_verification_metrics: bool = False,
    is_fullbody: bool = False,
):

    print("Input Directory: ", input_directory)
    print("Output Directory: ", output_directory)
    os.makedirs(output_directory, exist_ok=True)

    lambda_values = get_lambdas(input_directory)
    print("Lamba Values found: ", lambda_values)

    if is_fullbody:
        test_datasets = TEST_SUBSETS["fullbody"]
    else:
        test_datasets = TEST_SUBSETS["faces"]

    aggregate_feature_tests(
        input_directory,
        lambda_values,
        output_directory,
        skip_verification_rocs,
        skip_verification_metrics,
        test_datasets,
    )

    aggregate_aem_results(
        input_directory, lambda_values, output_directory, test_datasets
    )


def plot_relative_losses(
    output_directory: str, metrics_for_lambda: Dict[str, pd.DataFrame]
):
    relative_losses_per_lambda: Dict[str, pd.Series] = {}
    for lambda_value in metrics_for_lambda:
        lmb = float(lambda_value)

        if lmb == 1.0:
            continue

        metrics = metrics_for_lambda[lambda_value]
        try:
            loss_attribute = metrics["loss_attribute"]
        except KeyError:
            loss_attribute = metrics["loss_penalty"]

        loss_triplet = metrics["loss_triplet"]

        loss_attribute_weighted = loss_attribute * lmb
        loss_triplet_weighted = loss_triplet * (1 - lmb)

        relative_loss = loss_attribute_weighted / loss_triplet_weighted
        relative_loss.name = f"$\\lambda = {lambda_value}$"
        relative_losses_per_lambda[lambda_value] = relative_loss

    # Combine the metrics together.
    sorted_lambdas = sort_lambdas(list(relative_losses_per_lambda.keys()))
    df_rows: List[pd.Series] = []
    for lambda_value in sorted_lambdas:
        df_rows.append(relative_losses_per_lambda[lambda_value])
    df = pd.concat(df_rows, axis=1)

    # Save the full metrics.
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(
        os.path.join(
            output_directory,
            f"relative_losses.csv",
        )
    )

    fig, axs = plt.subplots(figsize=(10, 8))
    df.plot.line(ax=axs, logy=False)

    axs.set_ylabel(
        "Relative Loss (Weighted Attribute Loss or Penalty / Weighted Triplet Loss)"
    )
    axs.set_xlabel("Epoch")
    axs.set_title("Relative Loss vs Epoch")

    fig.set_tight_layout(True)
    fig.savefig(os.path.join(output_directory, "relative_losses.eps"))
    fig.savefig(os.path.join(output_directory, "relative_losses.png"))


def aggregate_aem_results(
    input_directory: str,
    lambda_values: List[str],
    output_directory: str,
    test_datasets: List[str],
):
    output_directory = os.path.join(output_directory, "aems")
    os.makedirs(output_directory, exist_ok=True)

    # Get the sub-dataset combinartions.
    aem_experiments = list(sorted(itertools.product(test_datasets, test_datasets)))

    yaml = YAML(typ="safe")

    # Get the results for each sub-experiment.
    for train_dataset, test_dataset in aem_experiments:
        print("AEM Experiment:", f"{train_dataset} - {test_dataset}")

        # Get the results for each lambda.
        results_for_lambda: Dict[str, pd.Series] = {}
        for lambda_value in tqdm.tqdm(
            lambda_values, desc=f"Loading AEM Results", ascii=True, dynamic_ncols=True
        ):
            results_path = os.path.join(
                input_directory,
                str(lambda_value),
                "aems",
                train_dataset,
                f"{test_dataset}.yaml",
            )

            try:
                with open(results_path, "r") as infile:
                    results_for_lambda[lambda_value] = yaml.load(infile)
            except FileNotFoundError:
                print(f"File not found: {results_path}")
                raise
            results_for_lambda[lambda_value]["lambda"] = lambda_value

        # Combine the metrics together.
        sorted_lambdas = sort_lambdas(lambda_values)
        df_rows: List[pd.Series] = []
        for lambda_value in sorted_lambdas:
            df_rows.append(results_for_lambda[lambda_value])
        df = pd.DataFrame.from_records(df_rows)
        df = df.set_index("lambda")

        # Save the full metrics.
        os.makedirs(output_directory, exist_ok=True)
        df.to_csv(
            os.path.join(
                output_directory,
                f"{train_dataset}_{test_dataset}_results.csv",
            )
        )

        # Give the columns for readable names.
        new_column_names = [n.split("test_")[1] for n in df.columns]
        df.columns = new_column_names

        # Save the full version in markdown.
        path = os.path.join(
            output_directory,
            f"{train_dataset}_{test_dataset}_results.md",
        )
        md_str = df.to_markdown(floatfmt=".3f")
        assert isinstance(md_str, str)
        with open(path, "w") as outfile:
            outfile.write(md_str)


def get_attribute_specific_metric_keys(input_directory: str, test_set: str):
    pattern = os.path.join(
        input_directory,
        "*",
        "feature_tests",
        "verification",
        test_set,
        "*",
        "roc_curves*.pickle",
    )

    filepaths = glob.glob(pattern)

    n_classes: Set[str] = set([])
    keys: Set[str] = set([])
    for filepath in filepaths:
        match = re.search(r"([\d]+)/roc_curves([\w]+).pickle", filepath)
        if match:
            n_classes.add(match.groups()[0])
            keys.add(match.groups()[1])
    return n_classes, keys


def aggregate_feature_tests(
    input_directory: str,
    lambda_values: List[str],
    output_directory: str,
    skip_verification_rocs: bool,
    skip_verification_metrics: bool,
    test_datasets: List[str],
):
    verification_test_output = os.path.join(output_directory, "feature_tests", "verification")
    os.makedirs(verification_test_output, exist_ok=True)

    for test_set in test_datasets:

        n_classes, keys = get_attribute_specific_metric_keys(input_directory, test_set)

        auc_for_lambdas_for_n_classes: Dict[int, Dict[str, float]] = {}
        for n_class in sorted(n_classes, key=lambda s: int(s)):

            print("Test Set: ", test_set)
            print("N Classes: ", n_class)

            aggregate_reid_metrics(
                    input_directory,
                    test_set,
                    lambda_values,
                    output_directory,
                    n_class)

            for key in keys:
                if not skip_verification_metrics:
                    aggregate_verification_metrics(
                        input_directory,
                        test_set,
                        lambda_values,
                        verification_test_output,
                        n_class,
                        key,
                    )

                if not skip_verification_rocs:
                    aucs_for_lambdas_for_n_class = plot_verification_curve(
                        input_directory,
                        test_set,
                        lambda_values,
                        verification_test_output,
                        n_class,
                        key,
                    )
                    auc_for_lambdas_for_n_classes[
                        int(n_class)
                    ] = aucs_for_lambdas_for_n_class
            print()

        plot_aucs_per_lambda_vs_n_classes(
            auc_for_lambdas_for_n_classes, verification_test_output, test_set
        )


def plot_aucs_per_lambda_vs_n_classes(
    auc_for_lambdas_for_n_classes, verification_test_output, test_set
):

    output_path = os.path.join(verification_test_output, f"auc_per_nclass_{test_set}")

    df = pd.DataFrame.from_records(auc_for_lambdas_for_n_classes).T.sort_index()

    column_names = df.columns.astype(str)
    lambda_names = column_names.tolist()
    lambda_names = [f"$\\lambda = {str(x)}$" for x in lambda_names]

    ax = df.plot(figsize=(10, 8))
    fig = ax.get_figure()

    ax.legend(lambda_names)

    ax.set_xscale("log")
    ax.set_xticks(df.index.tolist())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"ROC-AUC vs Number of Classes in Verification Scenario. Test Set: {test_set}"
    )

    fig.savefig(output_path + ".eps")
    fig.savefig(output_path + ".png")
    plt.close()


def aggregate_reid_metrics(
    input_directory: str,
    test_set: str,
    lambda_values: List[str],
    output_directory: str,
    n_classes: str
):
    # Get the metrics for each lambda.
    metrics_for_lambda: Dict[str, pd.Series] = {}
    for lambda_value in tqdm.tqdm(
        sorted(lambda_values),
        desc=f"Loading Reid Metrics",
        ascii=True,
        dynamic_ncols=True,
    ):
        metric_path = os.path.join(
            input_directory,
            str(lambda_value),
            "feature_tests",
            "reid",
            test_set,
            n_classes,
            f"avg_ranks.csv",
        )

        try:
            lambda_metrics = read_metrics(metric_path)["Cumulative Accuracy"]
            lambda_metrics = lambda_metrics.rename(float(lambda_value))
            metrics_for_lambda[lambda_value] = lambda_metrics
        except FileNotFoundError:
            continue

    # Combine the metrics together.
    sorted_lambdas = sort_lambdas(lambda_values)
    df_rows: List[pd.Series] = []
    for lambda_value in sorted_lambdas:
        df_rows.append(metrics_for_lambda[lambda_value])
    df = pd.concat(df_rows, axis=1).T
    df.index.name = "lambda"

    # Save the full metrics.
    os.makedirs(os.path.join(output_directory, "feature_tests", "reid", test_set), exist_ok=True)
    df.to_csv(
        os.path.join(output_directory, "feature_tests", "reid", test_set, f"reid_{n_classes}.csv")
    )


def aggregate_verification_metrics(
    input_directory: str,
    test_set: str,
    lambda_values: List[str],
    output_directory: str,
    n_classes: str,
    key: str,
):
    # Get the metrics for each lambda.
    metrics_for_lambda: Dict[str, pd.Series] = {}
    for lambda_value in tqdm.tqdm(
        lambda_values,
        desc=f"Loading Verification Metrics (key='{key}')",
        ascii=True,
        dynamic_ncols=True,
    ):
        metric_path = os.path.join(
            input_directory,
            str(lambda_value),
            "feature_tests",
            "verification",
            test_set,
            n_classes,
            f"verification{key}.csv",
        )

        split_metrics = read_metrics(metric_path)
        mean_metrics = get_mean_metrics(split_metrics)
        mean_metrics = mean_metrics.rename(float(lambda_value))
        metrics_for_lambda[lambda_value] = mean_metrics

    # Combine the metrics together.
    sorted_lambdas = sort_lambdas(lambda_values)
    df_rows: List[pd.Series] = []
    for lambda_value in sorted_lambdas:
        df_rows.append(metrics_for_lambda[lambda_value])
    df = pd.concat(df_rows, axis=1).T
    df.index.name = "lambda"

    # Save the full metrics.
    os.makedirs(os.path.join(output_directory, test_set), exist_ok=True)
    df.to_csv(
        os.path.join(output_directory, test_set, f"verification_{n_classes}{key}.csv")
    )

    # Give the columns for readable names.
    new_column_names = [METRICS_COLUMN_NAME_MAP[n] for n in df.columns]
    df.columns = new_column_names

    # Round to three decimal places.
    df = df.round(3)

    # Save the full version in markdown.
    path = os.path.join(
        output_directory, test_set, f"verification_full_{n_classes}{key}.md"
    )
    md_str = df.to_markdown()
    assert isinstance(md_str, str)
    with open(path, "w") as outfile:
        outfile.write(md_str)

    # Save a version with certain columns removed.
    del df["F1"]
    del df["Threshold"]
    del df["TN"]
    del df["FP"]
    del df["FN"]
    del df["TP"]

    path = os.path.join(
        output_directory, test_set, f"verification_partial_{n_classes}{key}.md"
    )
    md_str = df.to_markdown()
    assert isinstance(md_str, str)
    with open(path, "w") as outfile:
        outfile.write(md_str)


def read_metrics(metrics_path: str) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path, index_col=0)
    return metrics


def get_mean_metrics(split_metrics: pd.DataFrame) -> pd.Series:
    return split_metrics.mean(axis=0)


def plot_verification_curve(
    input_directory: str,
    test_set: str,
    lambda_values: List[str],
    output_directory: str,
    n_class: str,
    key: str,
) -> Dict[str, float]:

    # Get the roc values for each lambda.
    roc_for_lambda: Dict[str, ROC] = {}
    for lambda_value in tqdm.tqdm(
        lambda_values,
        desc=f"Loading ROCs (key='{key}')",
        ascii=True,
        dynamic_ncols=True,
    ):
        roc_path = os.path.join(
            input_directory,
            str(lambda_value),
            "feature_tests",
            "verification",
            test_set,
            n_class,
            f"roc_curves{key}.pickle",
        )
        split_rocs = read_rocs(roc_path)
        mean_roc = get_mean_roc(split_rocs)
        roc_for_lambda[lambda_value] = mean_roc

    # Plot the figure.
    plt.figure(figsize=(8, 7))
    lw = 2

    aucs_for_lambda: Dict[str, float] = {}
    for lambda_value in sort_lambdas(lambda_values):
        fpr, tpr, auc = roc_for_lambda[lambda_value]
        label = f"$\\lambda = {lambda_value} ({auc:0.2f})$"
        plt.plot(fpr, tpr, lw=lw, alpha=1.0, label=label)
        aucs_for_lambda[lambda_value] = auc

    title = f"ROC Curves per $\\lambda$"

    if key != "" and key != "_all":
        key_name = str.join(", ", key[1:].split("_"))
        title += f", Attribute Pair: ({key_name})"

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    plt.show()

    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(
        os.path.join(output_directory, test_set, f"roc_curves_{n_class}{key}.eps")
    )
    plt.savefig(
        os.path.join(output_directory, test_set, f"roc_curves_{n_class}{key}.png")
    )
    plt.close()

    return aucs_for_lambda


def get_mean_roc(split_rocs: SPLIT_ROCS) -> ROC:
    interp_tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    fpr, tpr, _ = list(zip(*split_rocs))

    for i in range(len(fpr)):
        interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    assert isinstance(mean_tpr, np.ndarray)
    assert isinstance(mean_fpr, np.ndarray)
    assert isinstance(mean_auc, float)

    return mean_fpr, mean_tpr, mean_auc


def read_rocs(roc_path: str) -> SPLIT_ROCS:
    with open(roc_path, "rb") as infile:
        rocs = pickle.load(infile)

    return rocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", metavar="DIR")
    parser.add_argument("output_directory", metavar="DIR")
    parser.add_argument("--skip-verification-rocs", action="store_true")
    parser.add_argument("--skip-verification-metrics", action="store_true")
    parser.add_argument("--fullbody", action="store_true")
    args = parser.parse_args()

    main(
        args.input_directory,
        args.output_directory,
        args.skip_verification_rocs,
        args.skip_verification_metrics,
        args.fullbody,
    )
