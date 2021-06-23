import itertools
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from numpy.random import default_rng
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from torchmetrics.functional import stat_scores
from torchmetrics.utilities.data import to_categorical

from .common import AnnotatedSample, Label, Pair, ROCCurve


class Verifier:
    def __init__(
        self,
        batch_size: int,
        max_n_matching_pairs: Optional[int],
        max_n_classes: Optional[int],
        debug: bool,
        seed: int,
    ):
        self.batch_size = batch_size
        self.max_n_matching_pairs = max_n_matching_pairs
        self.max_n_classes = max_n_classes
        self.rnd = np.random.RandomState(seed)
        self.debug = debug

    def setup(self, dataloader: DataLoader):
        samples, data_map = self._load_data(dataloader)
        samples_per_label = self.get_samples_per_label(samples)

        if self.max_n_classes is not None:
            classes = sorted(samples_per_label)[: self.max_n_classes]
            samples_per_label = {c: samples_per_label[c] for c in classes}

        matching_pairs = self.get_matching_pairs(
            samples_per_label,
            self.debug,
            self.batch_size,
            self.max_n_matching_pairs,
            self.rnd,
        )
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
            datamap_dataset, batch_size=self.batch_size, num_workers=4
        )

    @staticmethod
    def _load_data(
        dataloader: DataLoader,
    ) -> Tuple[Set[AnnotatedSample], Dict[int, np.ndarray]]:

        samples: Set[AnnotatedSample] = set()
        data_map: Dict[int, np.ndarray] = {}

        for db, (lb, ab) in dataloader:
            for d, l, a in zip(db, lb, ab):
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
        samples_per_label: Dict[Label, Set[AnnotatedSample]],
        debug: bool,
        batch_size: int,
        max_n_matching_pairs: Optional[int],
        rnd: np.random.RandomState,
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

                if debug and len(matching_pairs) >= batch_size * 10:
                    break

        if (
            max_n_matching_pairs is not None
            and len(matching_pairs) > max_n_matching_pairs
        ):
            indexs = rnd.choice(
                len(matching_pairs), max_n_matching_pairs, replace=False
            )

            matching_pairs_list = list(matching_pairs)
            matching_pairs = set([matching_pairs_list[i] for i in indexs])

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
            outputs = model(data.to(device).squeeze())

            if isinstance(outputs, torch.Tensor):
                features = outputs
            elif len(outputs) == 2:
                _, features = outputs
            else:
                raise ValueError

            features_per_batch.append(features.cpu().detach().numpy())
            ids_per_batch.append(ids.cpu().detach().numpy())

        all_features = np.concatenate(features_per_batch)
        all_ids = np.concatenate(ids_per_batch)
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

            try:
                auc_ap = metrics.roc_auc_score(labels_ap, np.negative(distances_ap))
            except ValueError as ex:
                warnings.warn(f"Verification testing for ({attribute_pair}) failed. Skipping: " + str(ex))
                continue

            assert isinstance(auc_ap, float)

            attribute_pair_aucs[attribute_pair] = auc_ap

        return auc, attribute_pair_aucs


class CVThresholdingVerifier(Verifier):
    def __init__(
        self,
        batch_size: int,
        max_n_matching_pairs: Optional[int] = 1000000,
        max_n_classes: Optional[int] = None,
        debug=False,
        seed: int = 42,
        n_splits=10,
    ):
        super().__init__(batch_size, max_n_matching_pairs, max_n_classes, debug, seed)
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

        desc = "CV Verification"
        desc += f" (Classes: {self.max_n_classes})" if self.max_n_classes else ""

        for split_train, split_test in tqdm.tqdm(
            kf.split(distances),
            desc,
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

                try:
                    split_results_ap, split_rocs_ap = self.thresholding_verification(
                        distances[split_train_ap],
                        labels[split_train_ap],
                        distances[split_test_ap],
                        labels[split_test_ap],
                    )
                except ValueError as ex:
                    warnings.warn(f"Verification testing for ({attribute_pair}) failed. Skipping: " + str(ex))
                    continue

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

        test_fpr, test_tpr, test_thresholds = metrics.roc_curve(
            test_y, -test_x  # type: ignore
        )

        train_auc = metrics.roc_auc_score(train_y, -train_x)
        test_auc = metrics.roc_auc_score(test_y, -test_x)

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
            "train_auc": train_auc,
            "test_auc": test_auc,
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

        return verification_metrics, (test_fpr, test_tpr, test_thresholds)


class DataMapDataset(Dataset):
    def __init__(self, datamap: Dict[int, np.ndarray]):
        super().__init__()
        self.dataid_with_data = [(k, datamap[k]) for k in sorted(datamap)]

    def __len__(self) -> int:
        return len(self.dataid_with_data)

    def __getitem__(self, index) -> Tuple[int, np.ndarray]:
        return self.dataid_with_data[index]


class BalancedAccuracy(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.add_state(
            "correct", default=torch.tensor(0).repeat(num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0).repeat(num_classes), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_ = to_categorical(preds)
        assert preds_.shape == target.shape

        scores = stat_scores(
            preds_, target, num_classes=self.num_classes, reduce="macro"
        )

        correct_per_class = scores[:, 0]
        total_per_class = scores[:, 4]

        self.correct += correct_per_class
        self.total += total_per_class

    def compute(self):
        use_mask = self.total > 0
        accuracy_per_class = self.correct[use_mask].float() / self.total[use_mask]
        return torch.sum(accuracy_per_class) / self.num_classes


class ReidentificationTester:
    def __init__(
        self,
        batch_size: int,
        max_n_classes: Optional[int],
        debug: bool,
        seed: int,
    ):
        self.batch_size = batch_size
        self.max_n_classes = max_n_classes
        self.rnd = np.random.RandomState(seed)
        self.debug = debug
        self.seed = seed

        self.num_workers = min(32, len(os.sched_getaffinity(0)))

    def setup(self, dataloader: DataLoader):
        samples, data_map = self._load_data(dataloader)
        samples_per_label = self.get_samples_per_label(samples)

        if self.max_n_classes is not None:
            classes = sorted(samples_per_label)[: self.max_n_classes]
            samples_per_label = {c: samples_per_label[c] for c in classes}

        self.gallery, self.probe = self.split_gallery_probe(
            samples_per_label, self.seed)

        self.gallery_dataloader = self.subset_to_dataloader(self.gallery, data_map)
        self.probe_dataloader = self.subset_to_dataloader(self.probe, data_map)

    def reidentification(self, model: nn.Module, device: torch.device):
        # Get the gallery features.
        gallery_features = self.get_gallery_features(model, device)
        avg_ranks = self.get_ranks(model, device, gallery_features)
        return avg_ranks

    def get_ranks(
            self,
            model: nn.Module,
            device: torch.device,
            gallery_features: torch.Tensor) -> np.ndarray:

        gallery_labels = torch.tensor([s[1] for s in self.gallery]).to(device)
        probe_labels = torch.tensor([s[1] for s in self.probe]).to(device)
        n_probes = len(probe_labels)

        sum_ranks_per_batch: List[torch.Tensor] = []

        for ids, data in self.probe_dataloader:
            # Get the probe features and labels for the batch.
            n_probe_labels_remaining = len(probe_labels)
            probe_features_batch = self.get_feature_batch(model, device, data)
            probe_labels_batch = probe_labels[:len(probe_features_batch)]
            probe_labels = probe_labels[len(probe_features_batch)::]

            assert len(ids) == len(data) == len(probe_labels_batch)
            assert len(probe_labels) + len(probe_labels_batch) \
                == n_probe_labels_remaining

            # Get the distances.
            gallery_grid = gallery_features.unsqueeze(dim=0)
            gallery_grid = gallery_grid.expand([len(probe_labels_batch), -1, -1])

            probe_batch_grid = probe_features_batch.unsqueeze(dim=1)
            probe_batch_grid = probe_batch_grid.expand([-1, len(gallery_features), -1])

            distances = gallery_grid - probe_batch_grid
            distances = distances.square().sum(dim=2).sqrt()

            # Get the correct label mask.
            correct_label_mask = \
                probe_labels_batch.unsqueeze(dim=1) == gallery_labels.unsqueeze(dim=0)

            # Sort gallery by shortest distance per probe.
            distance_sort_order = distances.argsort(dim=1)

            # Rank per probe.
            ranks_batch = torch.zeros_like(correct_label_mask)
            for i in range(correct_label_mask.shape[0]):
                ranks_batch[i, :] = correct_label_mask[i].index_select(0, distance_sort_order[i])

            ranks_batch = ranks_batch.cumsum(dim=1)
            sum_ranks_batch_over_probes = ranks_batch.sum(dim=0)
            sum_ranks_per_batch.append(sum_ranks_batch_over_probes)

        # Sum the ranks over all batches and return the average.
        sum_ranks_all_batches = torch.stack(sum_ranks_per_batch, dim=0).sum(dim=0)
        avg_ranks = sum_ranks_all_batches.float() / n_probes

        return avg_ranks.cpu().detach().numpy()

    @staticmethod
    def _load_data(
        dataloader: DataLoader,
    ) -> Tuple[Set[AnnotatedSample], Dict[int, np.ndarray]]:

        samples: Set[AnnotatedSample] = set()
        data_map: Dict[int, np.ndarray] = {}

        for db, (lb, ab) in dataloader:
            for d, l, a in zip(db, lb, ab):
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
    def split_gallery_probe(
            samples_per_label: Dict[Label, Set[AnnotatedSample]],
            seed: int) -> Tuple[List[AnnotatedSample], List[AnnotatedSample]]:
        # Get n random numbers between 0 and 1000 for n labels
        # to index the selected gallery image.
        rng = default_rng(seed)
        idxs = rng.integers(0, 1000, size=len(samples_per_label))

        gallery: Set[AnnotatedSample] = set()
        probe: Set[AnnotatedSample] = set()
        for label, idx in zip(samples_per_label, idxs):
            n_samples = len(samples_per_label[label])
            sample = list(samples_per_label[label])[idx % n_samples]
            gallery.add(sample)

            other_samples = samples_per_label[label] - set([sample])
            probe.update(other_samples)

        gallery_list = list(sorted(gallery))
        probe_list = list(sorted(probe))

        return gallery_list, probe_list

    def subset_to_dataloader(
            self, subset: List[AnnotatedSample], data_map: Dict[int, np.ndarray]):
        subset_sample_ids = [s[0] for s in subset]
        subset_data_map = {k: data_map[k] for k in subset_sample_ids}
        dataset = DataMapDataset(subset_data_map)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

        return dataloader

    def get_gallery_features(
            self,
            model: nn.Module,
            device: torch.device) -> torch.Tensor:
        features_per_batch = []
        ids_per_batch = []
        for ids, data in self.gallery_dataloader:
            features = self.get_feature_batch(model, device, data)
            features_per_batch.append(features)
            ids_per_batch.append(ids)

        all_features = torch.cat(features_per_batch)
        all_ids = torch.cat(ids_per_batch).int()

        assert len(all_ids) == len(set(all_ids))
        assert all_ids.tolist() == sorted(all_ids) == [s[0] for s in self.gallery]

        return all_features

    @staticmethod
    def get_feature_batch(
            model: nn.Module,
            device: torch.device,
            data: torch.Tensor):

        outputs = model(data.to(device).squeeze())

        if isinstance(outputs, torch.Tensor):
            features = outputs
        elif len(outputs) == 2:
            _, features = outputs
        else:
            raise ValueError

        return features
