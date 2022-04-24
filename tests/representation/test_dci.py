from torchmetrics.representation.dci import *
import torch

def test_batch_size_checker():
    """Test if the first dimension checker works as expected"""
    x1 = torch.randn(2, 3)
    x2 = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="Tensor 1 and tensor 2 are expected to have the same shape along dimension 0"):
        _ = _check_batch_size_dimension(x1, x2)
    
    x2 = torch.randn(4)
    with pytest.raises(RuntimeError, match="Tensor 1 and tensor 2 are expected to have the same shape along dimension 0"):
        _ = _check_batch_size_dimension(x1, x2)

def test_target_checker():
    """Test if the target checker works as expected"""
    target = torch.randn(4)
    with pytest.raises(RuntimeError, match="The `target` has to be an integer tensor."):
        _ = _check_classification_target(target, num_classes=5)

    target = torch.full(4, -1)
    with pytest.raises(ValueError, match="The `target` has to be a non-negative tensor."):
        _ = _check_classification_target(target, num_classes=5)
    
    target = torch.full(4, 5)
    with pytest.raises(ValueError, match="The `target` has to be a smaller than num classes."):
        _ = _check_classification_target(target, num_classes=5)

def test_split_train_test():
    """Check train_test split function"""
    data = torch.randn(10, 4)
    target = torch.randint(0, 5, (10))
    train_data, train_target, test_data, test_target = _split_train_test(data, target, train_percentage=0.85)
    assert train_data.shape[0] == train_target.shape[0] and test_data.shape[0] == test_target.shape[0]
    assert train_data.shape[0] != 9 and test_data.shape[0] != 1
    
    train_data, train_target, test_data, test_target = _split_train_test(data, target, train_percentage=0.85, shuffle=False)
    assert torch.equal(train_data, data[:9]) and torch.equal(train_target, target[:9])
    assert torch.equal(test_data, data[-1]) and torch.equal(test_target, target[-1])

    with pytest.raises(ValueError, match="Train percentage should be in (0, 1)."):
        _ = _split_train_test(data, target, train_percentage=1.0)
    
    target = torch.randint(0, 5, (11))
    with pytest.raises(RuntimeError, match="Tensor 1 and tensor 2 are expected to have the same shape along dimension 0"):
        _ = _split_train_test(data, target, train_percentage=0.8)

    
    data = torch.randn(0)
    target = torch.randn(0)
    with pytest.raises(RuntimeError, match="Cannot split into train and test an empty tensor."):
        _ = _split_train_test(data, target, train_percentage=0.8)

def test_dci_raises_errors_and_warnings():

    with pytest.warns(
        UserWarning,
        match="Metric `DCIScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.,
    ):
        _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 2))

    if _SKLEARN_AVAILABLE:
        with pytest.raises(ValueError, match="Both the number of factors and factor sizes are required to be > 1."):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4))
        
        with pytest.raises(ValueError, match="Both the number of factors and factor sizes are required to be > 1."):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 1))
        
        with pytest.raises(ValueError, match="`average` value is not admissible."):
            _ = DisentanglementCompletenessInformativeness(num_features=6, factor_sizes=(4, 2), average='micro')
    else:
        with pytest.raises(
            ModuleNotFoundError,
            match="DCIScore metric requires that `sklearn` is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install sklearn`.
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

    with pytest.raises(ValueError, match="Both features and targets should be 2-dimensional vectors"):
        metric.update(torch.randn(4, 6, 7), target)     
    
    with pytest.raises(RuntimeError, match=f"Features should be {num_features}-dimensional")    
        metric.update(torch.randn(4, 7), target)

def test_dci_disentanglement_score():
    """Test DCI disentanglement"""

    # perfect disentanglement
    disentanglement_score, feature_weights = _compute_disentanglement(torch.eye(4))
    assert torch.allclose(disentanglement_score, 1.0)
    assert torch.allclose(feature_weights, torch.ones_like(feature_weights)) 
    
    # dead component
    relative_importance = torch.tensor([[1., 0.], [0., 1.], [0., 0.])
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, 1.0)
    assert torch.allcose(disentanglement_score, torch.tensor([1., 1., 0.]))

    # empty relative importance
    relative_importance = torch.zeros(5, 5)
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, 0.)
    assert torch.allclose(disentanglement_score, torch.zeros(5))

    # redundant code
    relative_importance = torch.tensor([[1., 0., 0.,], [1., 0., 0.,], [0., 1., 0.], [0., 0., 1.]])
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, 1.)
    assert torch.allclose(disentanglement_score, torch.full(4, 0.25))

    # missed factor
    relative_importance = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, 1.)
    assert torch.allclose(disentanglement_score, torch.full(2, 0.5))

    # one code two factors
    relative_importance = torch.eye(5)
    relative_importance = torch.hstack(relative_importance, relative_importance)
    disentanglement_score, feature_weights = _compute_disentanglement(relative_importance)
    assert torch.allclose(disentanglement_score, 1. - np.log(2)/np.log(10))
    assert torch.allclose(disentanglement_score, torch.full(5, 1./5.))

def test_completeness_score():
    """Test Completeness score"""

    # perfect disentanglement
    relative_importance = torch.eye(5)
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, 1.)
    assert torch.allclose(completeness_score, torch.ones(5))

    # dead code
    relative_importance = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, 1.)
    assert torch.allclose(completeness_score, torch.full(2, .5))

    # zero relative importance
    relative_importance = torch.zeros(5, 5)
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, 0.)
    assert torch.allclose(completeness_score, torch.zeros(5))

    # test redundant
    relative_importance = torch.zeros(5, 5)
    relative_importance = torch.vstack(relative_importance, relative_importance)
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, 1. - np.log(2)/np.log(10))
    assert torch.allclose(completeness_score, torch.full(5, 1./5.))

    # test missed factor
    relative_importance = torch.zeros(5, 5)
    completeness_score, feature_weights = _compute_completeness(relative_importance[:, 2:])
    assert torch.allclose(completeness_score, 1.)
    assert torch.allclose(completeness_score, torch.full(5, .25))

    # one code, two factors
    relative_importance = torch.zeros(5, 5)
    relative_importance = torch.hstack(relative_importance, relative_importance)
    completeness_score, feature_weights = _compute_completeness(relative_importance)
    assert torch.allclose(completeness_score, 1.)
    assert torch.allclose(completeness_score, torch.full(10, 1./5.))

# informativeness
# fit factor classifiers

