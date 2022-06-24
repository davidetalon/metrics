import numpy as np
import pytest
import torch

from torchmetrics.representation.dci import (
    DisentanglementCompletenessInformativeness,
    _check_batch_size_dimension,
    _check_classification_target,
    _compute_completeness,
    _compute_disentanglement,
    _compute_informativeness,
    _fit_factor_classifiers,
    _split_train_test,
)
from torchmetrics.utilities.imports import _SKLEARN_AVAILABLE

if _SKLEARN_AVAILABLE:
    from sklearn.ensemble import GradientBoostingClassifier


def test_batch_size_checker():
    """Test if the first dimension checker works as expected."""
    x1 = torch.randn(2, 3)
    x2 = torch.randn(3, 4)
    with pytest.raises(
        RuntimeError, match="Tensor 1 and tensor 2 are expected to have the same shape along dimension 0"
    ):
        _ = _check_batch_size_dimension(x1, x2)

    x2 = torch.randn(4)
    with pytest.raises(
        RuntimeError, match="Tensor 1 and tensor 2 are expected to have the same shape along dimension 0"
    ):
        _ = _check_batch_size_dimension(x1, x2)


def test_target_checker():
    """Test if the target checker works as expected."""
    target = torch.randn(4)
    with pytest.raises(RuntimeError, match="The `target` has to be an integer tensor."):
        _ = _check_classification_target(target, num_classes=5)

    target = torch.full((4,), -1)
    with pytest.raises(ValueError, match="The `target` has to be a non-negative tensor."):
        _ = _check_classification_target(target, num_classes=5)

    target = torch.full((4,), 5)
    with pytest.raises(ValueError, match="The `target` has to be a smaller than num classes."):
        _ = _check_classification_target(target, num_classes=5)


def test_split_train_test():
    """Check train_test split function."""
    data = torch.randn(10, 4)
    target = torch.randint(0, 5, (10,))
    train_data, train_target, test_data, test_target = _split_train_test(data, target, train_percentage=0.85)
    assert train_data.shape[0] == train_target.shape[0] and test_data.shape[0] == test_target.shape[0]
    assert train_data.shape[0] == 8 and test_data.shape[0] == 2

    train_data, train_target, test_data, test_target = _split_train_test(
        data, target, train_percentage=0.95, shuffle=False
    )
    assert torch.equal(train_data, data[:9]) and torch.equal(train_target, target[:9])
    assert torch.equal(test_data, data[-1].unsqueeze(0)) and torch.equal(test_target, target[-1].unsqueeze(0))

    with pytest.raises(ValueError, match="Train percentage should be between 0 and 1."):
        _ = _split_train_test(data, target, train_percentage=1.0)

    target = torch.randint(0, 5, (11,))
    with pytest.raises(
        RuntimeError, match="Tensor 1 and tensor 2 are expected to have the same shape along dimension 0"
    ):
        _ = _split_train_test(data, target, train_percentage=0.8)

    data = torch.randn(0)
    target = torch.randn(0)
    with pytest.raises(RuntimeError, match="Cannot split into train and test an empty tensor."):
        _ = _split_train_test(data, target, train_percentage=0.8)


def test_dci_raises_errors_and_warnings():

    with pytest.warns(
        UserWarning,
        match="Metric `DCIScore` will save all extracted features in buffer."
        " For large datasets this may lead to large memory footprint.",
    ):
        _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 2))

    if _SKLEARN_AVAILABLE:
        with pytest.raises(ValueError, match="factor_sizes is expected to be a Tuple"):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4))

        with pytest.raises(ValueError, match="Both the number of factors and factor sizes are required to be > 1."):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4,))

        with pytest.raises(ValueError, match="Both the number of factors and factor sizes are required to be > 1."):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 1))

        with pytest.raises(ValueError, match="`average` value is not admissible."):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 2), average="micro")
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="DCIScore metric requires that `sklearn` is installed."
            " Either install as `pip install torchmetrics[image]` or `pip install sklearn`.",
        ):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 2))


def test_update():
    num_features = 6
    metric = DisentanglementCompletenessInformativeness(num_features=num_features, factor_sizes=(4, 4))

    data = torch.randn(4, 6)
    target = torch.randint(0, 4, (4, 2))
    metric.update(data, target)

    assert len(metric.gathered_features) == 1
    assert list(metric.gathered_features[0].shape) == [4, num_features]

    metric.reset()

    assert len(metric.gathered_features) == 0

    with pytest.raises(RuntimeError, match="Both features and targets should be 2-dimensional vectors"):
        metric.update(torch.randn(4, 6, 7), target)

    with pytest.raises(RuntimeError, match=f"Features should be {num_features}-dimensional"):
        metric.update(torch.randn(4, 7), target)


def test_dci_disentanglement_score():
    """Test DCI disentanglement."""

    # perfect disentanglement
    disentanglement_score, feature_weights = _compute_disentanglement(torch.eye(4))
    assert torch.allclose(disentanglement_score, torch.ones_like(disentanglement_score))
    assert torch.allclose(feature_weights, torch.full((4,), 0.25))

    # dead component
    relative_importance = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, torch.ones_like(disentanglement_score))
    assert torch.allclose(feature_weights, torch.tensor([0.5, 0.5, 0.0]))

    # empty relative importance
    relative_importance = torch.zeros(5, 5)
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, torch.zeros_like(disentanglement_score))
    assert torch.allclose(feature_weights, torch.zeros(5))

    # redundant code
    relative_importance = torch.tensor(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, torch.ones_like(disentanglement_score))
    assert torch.allclose(feature_weights, torch.full((4,), 0.25))

    # missed factor
    relative_importance = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, torch.ones_like(disentanglement_score))
    assert torch.allclose(feature_weights, torch.full((2,), 0.5))

    # one code two factors
    relative_importance = torch.eye(5)
    relative_importance = torch.hstack([relative_importance, relative_importance])
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, torch.full((5,), 1.0 - np.log(2) / np.log(10)))
    assert torch.allclose(feature_weights, torch.full((5,), 1.0 / 5.0))


def test_completeness_score():
    """Test Completeness score."""

    # perfect disentanglement
    relative_importance = torch.eye(5)
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, torch.ones_like(completeness_score))
    assert torch.allclose(feature_weights, torch.full((5,), 1.0 / 5.0))

    # dead code
    relative_importance = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, torch.ones_like(completeness_score))
    assert torch.allclose(feature_weights, torch.full((2,), 0.5))

    # zero relative importance
    relative_importance = torch.zeros(5, 5)
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, torch.zeros_like(completeness_score))
    assert torch.allclose(feature_weights, torch.zeros(5))

    # test redundant
    relative_importance = torch.eye(5, 5)
    relative_importance = torch.vstack([relative_importance, relative_importance])
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, torch.full((5,), 1.0 - np.log(2) / np.log(10)))
    assert torch.allclose(feature_weights, torch.full((5,), 1.0 / 5.0))

    # test missed factor
    relative_importance = torch.eye(5, 5)
    completeness_score, feature_weights = _compute_completeness(relative_importance[:, 2:])
    assert torch.allclose(completeness_score, torch.ones_like(completeness_score))
    assert torch.allclose(feature_weights, torch.full((3,), 1.0 / 3.0))

    # one code, two factors
    relative_importance = torch.eye(5, 5)
    relative_importance = torch.hstack([relative_importance, relative_importance])
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, torch.ones_like(completeness_score))
    assert torch.allclose(feature_weights, torch.full((10,), 1.0 / 10.0))


def test_classifiers_fitting():
    """Test fitting of classifiers."""
    num_factors = 3
    num_features = 5
    factor_size = 4
    num_samples = factor_size
    classifiers = [GradientBoostingClassifier()] * num_factors

    data = torch.randint(0, 5, (num_samples, num_features))
    target = torch.arange(0, factor_size).unsqueeze(1).repeat(1, num_factors)
    relative_importance = _fit_factor_classifiers(classifiers, data, target)
    assert list(relative_importance.shape) == [num_features, num_factors]


def test_informativeness_score():
    """Test informativeness score."""
    num_factors = 3
    factor_size = 5
    num_samples = factor_size
    classifiers = [GradientBoostingClassifier()] * num_factors
    target = torch.arange(0, factor_size).unsqueeze(1).repeat(1, num_factors)
    data = torch.randint(0, 5, (num_samples, num_factors))
    _ = _fit_factor_classifiers(classifiers, data, target)

    informativeness = _compute_informativeness(classifiers, data, data)
    assert len(informativeness) == num_factors


def test_compute():
    """Test compute DCI."""
    num_features = 6
    num_factors = 3
    num_samples = 5
    factor_size = 5
    metric = DisentanglementCompletenessInformativeness(
        num_features=num_features, factor_sizes=(factor_size,) * num_factors
    )

    data = torch.randn(num_samples, num_features)
    target = torch.arange(0, factor_size).unsqueeze(1).repeat(1, num_factors)
    metric.update(data, target)
    dci = metric.compute()
    scalar_shape = torch.tensor(1.0).shape
    assert dci["disentanglement"].shape == scalar_shape
    assert dci["completeness"].shape == scalar_shape
    assert dci["informativeness_train"].shape == scalar_shape
    assert dci["informativeness_test"].shape == scalar_shape

    metric = DisentanglementCompletenessInformativeness(
        num_features=num_features, factor_sizes=(factor_size,) * num_factors, average=None
    )
    metric.update(data, target)
    dci = metric.compute()
    assert dci["disentanglement"].shape[0] == num_features
    assert dci["completeness"].shape[0] == num_factors
    assert len(dci["informativeness_train"]) == num_factors
    assert len(dci["informativeness_test"]) == num_factors
