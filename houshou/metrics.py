from collections import defaultdict
from houshou.data.datasets import AttributeDataset
from typing import Any, Dict, List, Optional, Set, Tuple
import itertools

import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, dataloader

from .common import AnnotatedSample, Label, Pair, ROCCurve


class Verifier:
    def __init__(self, dataset: AttributeDataset, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)

        samples, data_map = self._load_data(dataset)
        samples_per_label = self.get_samples_per_label(samples)
        matching_pairs = self.get_matching_pairs(samples_per_label)
        unmatching_pairs = self.get_unmatching_pairs(
            samples_per_label, len(matching_pairs), self.rnd
        )

        self.pairs: List[Pair] = list(set.union(matching_pairs, unmatching_pairs))
        self.pairs_left_id = [p[0][0] for p in self.pairs]
        self.pairs_right_id = [p[1][0] for p in self.pairs]
        self.is_same = [p[0][1] == p[1][1] for p in self.pairs]

        self.attribute_pairs_map = self.map_attribute_pairs(self.pairs)

        assert len(self.pairs_left_id) == len(self.pairs_right_id) == len(self.is_same)

        datamap_dataset = DataMapDataset(data_map)
        self.datamap_dataloader = DataLoader(
            datamap_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
        )

    @staticmethod
    def _load_data(
        dataset: AttributeDataset,
    ) -> Tuple[Set[AnnotatedSample], Dict[int, np.ndarray]]:

        samples: Set[AnnotatedSample] = set()
        data_map: Dict[int, np.ndarray] = {}

        dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)

        for d, (l, a) in dataloader:
            key = len(data_map)
            data_map[key] = d
            samples.add((key, int(l), int(a)))

        return samples, data_map

    @staticmethod
    def get_samples_per_label(
        data: Set[AnnotatedSample],
    ) -> Dict[Label, Set[AnnotatedSample]]:
        samples_per_label: Dict[Label, Set[AnnotatedSample]] = defaultdict(set)
        for sample in data:
            samples_per_label[sample[1]].add(sample)

        return samples_per_label

    @staticmethod
    def get_matching_pairs(
        samples_per_label: Dict[Label, Set[AnnotatedSample]]
    ) -> Set[Pair]:
        matching_pairs: Set[Pair] = set()

        for label in tqdm.tqdm(
            samples_per_label,
            desc="Verifier - Finding Matching Pairs",
            dynamic_ncols=True,
        ):
            samples = samples_per_label[label]
            for pair in itertools.product(samples, samples):
                if pair[0][0] != pair[1][0]:
                    if (pair[1], pair[0]) not in matching_pairs:
                        matching_pairs.add(pair)

        return matching_pairs

    @staticmethod
    def get_unmatching_pairs(
        samples_per_label: Dict[Label, Set[AnnotatedSample]],
        n_pairs: int,
        rnd: np.random.RandomState,
    ) -> Set[Pair]:

        unmatched_pairs = set()
        label_list = list(sorted(samples_per_label))
        sample_list_per_label = {k: list(v) for k, v in samples_per_label.items()}

        for i in tqdm.trange(
            n_pairs,
            desc="Verifier - Finding Non-Matching Pairs",
            dynamic_ncols=True,
        ):

            while True:

                primary_id = label_list[i % len(label_list)]

                while True:
                    secondary_id = rnd.choice(label_list)
                    if primary_id != secondary_id:
                        break

                sample1_idx = rnd.randint(len(sample_list_per_label[primary_id]))
                sample2_idx = rnd.randint(len(sample_list_per_label[secondary_id]))
                sample1 = sample_list_per_label[primary_id][sample1_idx]
                sample2 = sample_list_per_label[secondary_id][sample2_idx]

                if (sample1, sample2) not in unmatched_pairs:
                    break

            unmatched_pairs.add((sample1, sample2))
        return unmatched_pairs

    @staticmethod
    def map_attribute_pairs(pairs: List[Pair]) -> Dict[Tuple[int, int], Set[int]]:
        attribute_pair_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        for i, attribute_pair in enumerate(sorted((p[0][2], p[1][2])) for p in pairs):
            attribute_pair_map[tuple(attribute_pair)].add(i)  # type: ignore

        return attribute_pair_map

    def get_features(self, model: nn.Module, device: torch.device) -> np.ndarray:
        features_per_batch = []
        ids_per_batch = []
        for ids, data in self.datamap_dataloader:
            _, features = model(data.squeeze().to(device))
            features_per_batch.append(features)
            ids_per_batch.append(ids)

        all_features = torch.cat(features_per_batch).cpu().detach().numpy()
        all_ids = torch.cat(ids_per_batch).cpu().detach().numpy()
        all_ids = [int(x) for x in all_ids]

        assert max(all_ids) + 1 == len(all_ids) == len(set(all_ids))
        assert all_ids == sorted(all_ids)

        return all_features

    def get_pair_distances(self, model: nn.Module, device: torch.device):
        features = self.get_features(model, device)

        pairs_left_features = features[self.pairs_left_id]
        pairs_right_features = features[self.pairs_right_id]

        distances = pairs_left_features - pairs_right_features
        distances = np.power(distances, 2)
        distances = np.sum(distances, axis=1)
        distances = np.sqrt(distances)

        return distances

    def get_distances_with_label(
        self, model: nn.Module, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances = self.get_pair_distances(model, device)
        labels = np.array(self.is_same)

        assert isinstance(distances, np.ndarray)
        assert isinstance(labels, np.ndarray)
        return distances, labels

    def roc_auc(
        self, model: nn.Module, device: torch.device
    ) -> Tuple[float, Dict[Tuple[int, int], float]]:
        distances, labels = self.get_distances_with_label(model, device)
        auc = metrics.roc_auc_score(labels, np.negative(distances))  # type: ignore
        assert isinstance(auc, float)

        attribute_pair_aucs: Dict[Tuple[int, int], float] = {}
        for attribute_pair in self.attribute_pairs_map:
            attribute_pair_indx = list(sorted(self.attribute_pairs_map[attribute_pair]))
            distances_ap = distances[attribute_pair_indx]
            labels_ap = labels[attribute_pair_indx]

            auc_ap = metrics.roc_auc_score(
                labels_ap, np.negative(distances_ap)
            )  # type: ignore
            assert isinstance(auc_ap, float)

            attribute_pair_aucs[attribute_pair] = auc_ap

        return auc, attribute_pair_aucs


class CVThresholdingVerifier(Verifier):
    def __init__(
        self,
        dataset: AttributeDataset,
        batch_size: int,
        seed: int = 42,
        n_splits=10,
    ):
        super().__init__(dataset, batch_size, seed)
        self.n_splits = n_splits

    def cv_thresholding_verification(
        self, model: nn.Module, device: torch.device
    ) -> Tuple[
        Tuple[pd.DataFrame, List[ROCCurve]],
        Dict[Tuple[int, int], Tuple[pd.DataFrame, List[ROCCurve]]],
    ]:
        distances, labels = self.get_distances_with_label(model, device)
        kf = KFold(n_splits=self.n_splits, shuffle=True)  # type: ignore

        metrics_ds: Optional[pd.DataFrame] = None
        splits_rocs: List[ROCCurve] = []

        attribute_pair_results: Dict[
            Tuple[int, int], Tuple[pd.DataFrame, List[ROCCurve]]
        ] = {}

        for split_train, split_test in tqdm.tqdm(
            kf.split(distances),
            desc="CV Verification",
            total=self.n_splits,
            dynamic_ncols=True,
        ):
            # Full Results
            split_results, split_rocs = self.thresholding_verification(
                distances[split_train],
                labels[split_train],
                distances[split_test],
                labels[split_test],
            )

            if metrics_ds is None:
                metrics_ds = pd.DataFrame.from_records([split_results])
            else:
                metrics_ds = metrics_ds.append(split_results, ignore_index=True)

            splits_rocs.append(split_rocs)

            # Separate out results for each unique attribute pair.
            split_train_set = set(split_train)
            split_test_set = set(split_test)

            assert len(split_train) == len(split_train_set)
            assert len(split_test) == len(split_test_set)

            for attribute_pair in self.attribute_pairs_map:
                att_pair_indexes = self.attribute_pairs_map[attribute_pair]

                split_train_ap = list(
                    sorted(set.intersection(split_train_set, att_pair_indexes))
                )
                split_test_ap = list(
                    sorted(set.intersection(split_test_set, att_pair_indexes))
                )

                split_results_ap, split_rocs_ap = self.thresholding_verification(
                    distances[split_train_ap],
                    labels[split_train_ap],
                    distances[split_test_ap],
                    labels[split_test_ap],
                )

                if attribute_pair not in attribute_pair_results:
                    metrics_ds_ap = pd.DataFrame.from_records([split_results_ap])
                    splits_rocs_ap = [split_rocs_ap]
                    attribute_pair_results[attribute_pair] = (
                        metrics_ds_ap,
                        splits_rocs_ap,
                    )
                else:
                    metrics_ds_ap = attribute_pair_results[attribute_pair][0]
                    splits_rocs_ap = attribute_pair_results[attribute_pair][1]

                    metrics_ds_ap = metrics_ds_ap.append(
                        split_results_ap, ignore_index=True
                    )
                    splits_rocs_ap.append(split_rocs_ap)

                    attribute_pair_results[attribute_pair] = (
                        metrics_ds_ap,
                        splits_rocs_ap,
                    )

        assert metrics_ds is not None
        return (
            (metrics_ds, splits_rocs),
            attribute_pair_results,
        )

    def thresholding_verification(
        self, train_x, train_y, test_x, test_y
    ) -> Tuple[Dict[str, Any], ROCCurve]:

        train_fpr, train_tpr, train_thresholds = metrics.roc_curve(
            train_y, -train_x  # type: ignore
        )
        auc = metrics.roc_auc_score(train_y, -train_x)  # type: ignore

        assert isinstance(train_fpr, np.ndarray)
        assert isinstance(train_tpr, np.ndarray)
        assert isinstance(train_thresholds, np.ndarray)

        min_total_error = np.abs(train_tpr[0] - (1 - train_fpr[0]))
        optimal_threshold = np.abs(train_thresholds[0])

        for fpr, tpr, threshold in zip(
            train_fpr, train_tpr, train_thresholds
        ):  # type: ignore
            error = np.abs(tpr - (1 - fpr))
            if error < min_total_error:
                min_total_error = error
                optimal_threshold = np.abs(threshold)

        y_true = test_y
        y_pred = test_x <= optimal_threshold

        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true, y_pred  # type: ignore
        ).ravel()

        verification_metrics = {
            "train_auc": auc,
            "threshold": optimal_threshold,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "accuracy": metrics.accuracy_score(y_true, y_pred),  # type: ignore
            "precision": metrics.precision_score(y_true, y_pred),  # type: ignore
            "recall": metrics.recall_score(y_true, y_pred),  # type: ignore
            "f1": metrics.f1_score(y_true, y_pred),  # type: ignore
            "far": fp / (fp + tn),
            "frr": fn / (fn + tp),
        }

        return verification_metrics, (train_fpr, train_tpr, train_thresholds)


class DataMapDataset(Dataset):
    def __init__(self, datamap: Dict[int, np.ndarray]):
        super().__init__()
        self.dataid_with_data = [(k, datamap[k]) for k in sorted(datamap)]

    def __len__(self) -> int:
        return len(self.dataid_with_data)

    def __getitem__(self, index) -> Tuple[int, np.ndarray]:
        return self.dataid_with_data[index]
