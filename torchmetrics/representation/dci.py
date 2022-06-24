import math
from random import shuffle as sh
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.special import entr

from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_for_empty_tensors
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import AverageMethod
from torchmetrics.utilities.imports import _SKLEARN_AVAILABLE

if _SKLEARN_AVAILABLE:
    from sklearn.ensemble import GradientBoostingClassifier


def _check_batch_size_dimension(x1: torch.Tensor, x2: torch.Tensor) -> None:
    """Check whether the two input tensors ``x1`` and ``x2`` have the same shape along the first dimension.

    Args:
        x1: first tensor to compare
        x2: second tensor to compare

    Raises:
        RuntimeError:
            If ``x1`` and ``x2`` differ in shape along the first dimension
    """
    if x1.shape[0] != x2.shape[0]:
        raise RuntimeError("Tensor 1 and tensor 2 are expected to have the same shape along dimension 0")


def _check_classification_target(target: torch.Tensor, num_classes: int) -> None:
    """Perform basic validation of target labels.

    Args:
        target: tensor containing the target labels
        num_classes: int indicating the number of classes

    Raises:
        RuntimeError:
            If ``target`` is a floating point tensor
        ValueError:
            If ``target`` is not in [0, ``num_classes``)
    """
    if target.is_floating_point():
        raise RuntimeError("The `target` has to be an integer tensor.")

    if target.min() < 0:
        raise ValueError("The `target` has to be a non-negative tensor.")

    if target.max() >= num_classes:
        raise ValueError("The `target` has to be a smaller than num classes.")


def _split_train_test(
    data: torch.Tensor, target: torch.Tensor, train_percentage: Optional[float] = 0.8, shuffle: Optional[bool] = True
) -> Tuple[torch.Tensor]:
    """Split both data and target into train test subsets.

    Args:
        data: tensor with input data
        target: tensor containing the target
        train_percentage: float indicating the percentage of samples used for training
        shuffle: bool indicating to shuffle the tensor or not, Default True.

    Return:
        train_data: tensor with the training subset of ``data``
        train_target: tensor with the training subset of ``target``
        test_data: tensor with the test subset of ``data``
        test_target: tensor with the test subset of ``target``
    """
    is_empty = _check_for_empty_tensors(data, target)
    if is_empty:
        raise RuntimeError("Cannot split into train and test an empty tensor.")

    _check_batch_size_dimension(data, target)
    if train_percentage <= 0 or train_percentage >= 1:
        raise ValueError("Train percentage should be between 0 and 1.")

    num_samples = data.shape[0]
    num_samples_train = int(math.floor(num_samples * train_percentage))
    if shuffle:
        indexes = list(range(num_samples))
        sh(indexes)
        data = data[indexes]
        target = target[indexes]

    train_data = data[:num_samples_train]
    train_target = target[:num_samples_train]

    test_data = data[num_samples_train:]
    test_target = target[num_samples_train:]
    return train_data, train_target, test_data, test_target


def _compute_disentanglement(relative_importance: torch.Tensor) -> Tuple[torch.Tensor]:
    """Compute disentanglement scores and feature weights of DCI from the relative importance matrix.

    Let Rij be the (i,j) element of the ``relative importance`` matrix.
    ``disentanglement`` is computed as:
    Di = 1 - H(Pi), where H(Pi) = - sum_k(Pik log Pik) where Pik = Rij / sum_k(Rik)
    ``feature weights`` are computed as sum_j(Rij)/sum_ij(Rij).

    Args:
        relative_importance: tensor of dimension ``(D, K)`` where ``D``
        is the number of features and ``K`` is the number of the factors of variation

    Return:
        disentanglement: tensor of dimension ``D`` with features disentanglement
        feature_weights: tensor of dimension ``D`` with weights accounting for relative units
    """
    num_features = relative_importance.shape[0]
    num_factors = relative_importance.shape[1]
    normalized_importance_matrix = relative_importance / (relative_importance.sum(dim=1, keepdim=True) + 1e-11)

    if relative_importance.sum() == 0:
        return torch.zeros(num_features), torch.zeros(num_features)

    entropy = entr(normalized_importance_matrix) / math.log(num_factors)
    entropy = entropy.sum(dim=1)

    disentanglement = 1 - entropy
    feature_weights = relative_importance.sum(dim=1) / relative_importance.sum()

    return disentanglement, feature_weights


def _compute_completeness(relative_importance: torch.Tensor) -> torch.Tensor:
    """Compute completeness scores and factor weights of DCI from the relative importance matrix.

    Let Rij be the (i,j) element of the ``relative importance`` matrix.
    ``completeness`` is computed as:
    Cj = 1 - H(Pj), where H(Pj) = - sum_d(Pdj log Pdj) where Pdj = Rij / sum_d(Rdj)
    ``factor weights`` are computed as sum_i(Rij)/sum_ij(Rij).

    Args:
        relative_importance: tensor of dimension ``(D, K)`` where ``D``
        is the number of features and ``K`` is the number of the factors of variation

    Return:
        completeness: tensor of dimension ``K`` with factor completeness
        factor_weights: tensor of dimension ``K`` with weights accounting for relative factors
    """
    num_features = relative_importance.shape[0]
    num_factors = relative_importance.shape[1]
    normalizes_importance_matrix = relative_importance / (relative_importance.sum(dim=0) + 1e-11)

    if relative_importance.sum() == 0.0:
        return torch.zeros(num_factors), torch.zeros(num_factors)

    entropy = entr(normalizes_importance_matrix) / math.log(num_features)
    entropy = entropy.sum(dim=0)

    completeness = 1 - entropy
    factor_weights = relative_importance.sum(dim=0) / relative_importance.sum()
    return completeness, factor_weights


def _compute_informativeness(
    classifiers: List[GradientBoostingClassifier], features: torch.Tensor, targets: torch.Tensor
) -> int:
    """Compute informativeness score of DCI from.

    Args:
        classifiers: list of ``sklearn.ensemble.GradientBoostingClassifier``.
        Each of them predicting a factor of variation.
        features: tensor with the features to predict factor of variations from
        targets: tensor with target factor of variations

    Return:
        pred_error: list indicating the fraction of errors got by classifiers to compute targets from features
    """
    pred_error = []
    for factor_idx, cl in enumerate(classifiers):
        pred = cl.predict(features.cpu().numpy())
        factor_error = (torch.from_numpy(pred).to(targets.device) != targets[:, factor_idx]).float().mean()
        pred_error.append(factor_error)
    return pred_error


def _fit_factor_classifiers(
    classifiers: List[GradientBoostingClassifier], data: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Fit metric classifiers.

    Args:
        classifiers (list[GradientBoostingClassifier]): list of GradientBoostingClassifier that are used for
            classification of the factors of variation.
        data (torch.Tensor): tensor of dimension ``(N, D)`` used by the classifiers to train. ``N`` represents the
            number of samples while ``D`` the number of features.
        target (torch.Tensor): target labels of dimension ``(N, K)`` used to train classifiers with ``K`` the number
            of factors of variation.
    """

    num_features = data.shape[1]
    num_factors = target.shape[1]
    relative_importance = np.zeros((num_features, num_factors))
    for factor_idx in range(num_factors):
        classifiers[factor_idx].fit(data.cpu().numpy(), target[:, factor_idx].cpu().numpy())

        relative_importance[:, factor_idx] = classifiers[factor_idx].feature_importances_

    relative_importance = torch.from_numpy(relative_importance)

    return relative_importance


class DisentanglementCompletenessInformativeness(Metric):
    """DCI Metric from "[A Framework for the Quantitive Evaluation of Disentangled
    Representations](https://openreview.net/pdf?id=By-7dz-AZ)".

    .. note:: using this metrics requires you to have ``sklearn`` install. Either install as ``pip install
        torchmetrics[image]`` or ``pip install sklearn``

    .. note:: the ``forward`` method can be used but ``compute_on_step`` is disabled by default (oppesit of
        all other metrics) as this metric does not really make sense to calculate on a single batch. This
        means that by default ``forward`` will just call ``update`` underneat.

    Args:
        num_features (int): int indicating the number of features of the representation
        factor_sizes (tuple): K-dimensional tuple with the number of admissible values for each factor of variation. `K`
            the number of factors of variation.

        average (str): defines the reduction that is applied with the different factors of variation.
            Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each factor of variation.
                Disentanglement and completeness are computed respectively averaging according
                to  feature and factor weights. Informativeness is computed as average.
            - ``'none'`` or ``None``: Calculate the metric for each metric, and return
              the metric for every class.
        compute_on_step (bool):
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

    Raises:
        ModuleNotFoundError:
            If ``sklearn`` is not installed
        ValueError:
            If ``factor_sizes`` has less than two elements or an element is smaller than 1
        ValueError:
            If ``average`` is not ``macro``, ``none`` or ``None``
    """

    def __init__(
        self,
        num_features: int,
        factor_sizes: Tuple,
        average: Optional[str] = "macro",
        compute_on_step: Optional[bool] = False,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__(compute_on_step=compute_on_step)

        rank_zero_warn(
            "Metric `DCIScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if not _SKLEARN_AVAILABLE:
            raise ModuleNotFoundError(
                "DCIScore metric requires that `sklearn` is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install sklearn`."
            )
        if not isinstance(factor_sizes, Tuple):
            raise ValueError("factor_sizes is expected to be a Tuple")

        num_invalid_factors = len(list(filter(lambda x: x <= 1, factor_sizes)))
        if len(factor_sizes) <= 1 or num_invalid_factors >= 1:
            raise ValueError("Both the number of factors and factor sizes are required to be > 1.")

        allowed_average = ["macro", "none", None]
        if average not in allowed_average:
            raise ValueError("`average` value is not admissible.")

        self.num_factors = len(factor_sizes)
        self.factor_sizes = factor_sizes
        self.num_features = num_features
        self.average = average

        self.add_state("gathered_features", default=[], dist_reduce_fx=None)
        self.add_state("gathered_targets", default=[], dist_reduce_fx=None)

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the state with provided features.

        Args:
            features (torch.Tensor): features computed from the model
            targets (torch.Tensor): associated targets

        Raises:
            RuntimeError: if features and targets are not 2-dimensional vectors
            RuntimeError: if features and targets diffear in the first dimension
            RuntimeError: if features have different dimension with respect to ``num_features``
            RuntimeError: if targets is a float tensor
            ValueError: if targets are out of factor admissible values
        """

        _check_batch_size_dimension(features, targets)

        if len(features.shape) != 2 or len(targets.shape) != 2:
            raise RuntimeError("Both features and targets should be 2-dimensional vectors")

        if features.shape[1] != self.num_features:
            raise RuntimeError(f"Features should be {self.num_features}-dimensional")
        for factor_idx in range(self.num_factors):
            _check_classification_target(targets[:, factor_idx], self.factor_sizes[factor_idx])

        self.gathered_features.append(features)
        self.gathered_targets.append(targets)

    def compute(self):
        """Compute DCI metric.

        Returns:
            dci_metric: dictionary containing the different components of the DCI metric. ``disentanglement``,
                ``completeness``, ``informativeness_train``, ``informativeness_test``
        """
        gathered_features = dim_zero_cat(self.gathered_features)

        gathered_targets = dim_zero_cat(self.gathered_targets)

        train_data, train_target, test_data, test_target = _split_train_test(
            gathered_features, gathered_targets, train_percentage=0.8
        )
        classifiers = [GradientBoostingClassifier()] * self.num_factors
        relative_importance = _fit_factor_classifiers(classifiers, train_data, train_target)

        disentanglement, feature_weights = _compute_disentanglement(relative_importance)

        completeness, factor_weights = _compute_completeness(relative_importance)

        train_informativeness = _compute_informativeness(classifiers, train_data, train_target)
        test_informativeness = _compute_informativeness(classifiers, test_data, test_target)

        if self.average == AverageMethod.MACRO:
            disentanglement = (disentanglement * feature_weights).sum()
            completeness = (completeness * factor_weights).sum()
            train_informativeness = torch.tensor(train_informativeness).mean()
            test_informativeness = torch.tensor(test_informativeness).mean()

        dci_metric = {
            "disentanglement": disentanglement,
            "completeness": completeness,
            "informativeness_train": train_informativeness,
            "informativeness_test": test_informativeness,
        }
        return dci_metric
