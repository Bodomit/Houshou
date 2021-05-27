import pytest

import torch

from torchmetrics.collections import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1
from torchmetrics.functional import stat_scores

from houshou.metrics import BalancedAccuracy


@pytest.fixture
def metrics() -> MetricCollection:
    n_outputs = 2
    metrics = MetricCollection(
        {
            "Accuracy": Accuracy(num_classes=n_outputs),
            "BalancedAccuracy": BalancedAccuracy(num_classes=n_outputs),
            "Precison": Precision(num_classes=n_outputs),
            "Recall": Recall(num_classes=n_outputs),
            "F1": F1(num_classes=n_outputs),
        }
    )

    return metrics


def test_all_positive(metrics):
    y = torch.tensor([1], dtype=torch.int64).repeat(4)
    y_ = torch.tensor([[0.3, 0.7]]).repeat(4, 1)

    output = metrics(y_, y)

    assert output["Accuracy"].item() == 1.0
    assert output["BalancedAccuracy"].item() == 0.5


def test_all_negative(metrics):
    y = torch.tensor([1], dtype=torch.int64).repeat(4)
    y_ = torch.tensor([[0.7, 0.3]]).repeat(4, 1)

    output = metrics(y_, y)

    assert output["Accuracy"].item() == 0
    assert output["BalancedAccuracy"].item() == 0


def test_three_quarters_positive(metrics):
    y = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    y_ = torch.tensor([[0.7, 0.3]]).repeat(4, 1)

    # Expecting accuracy to be 0.75 and balanced accuracy to be 0.5.
    output = metrics(y_, y)

    assert output["Accuracy"].item() == 0.75
    assert output["BalancedAccuracy"].item() == 0.5
