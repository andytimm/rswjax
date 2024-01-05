import pytest
import numpy as np
import pandas as pd
import rswjax

@pytest.fixture
def data_frame():
    np.random.seed(5)
    n = 10
    age = np.random.randint(20, 30, size=n) * 1.
    sex = np.random.choice([0., 1.], p=[.4, .6], size=n)
    height = np.random.normal(5, 1, size=n)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "height": height
    })

@pytest.fixture
def losses():
    return [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5), rswjax.EqualityLoss(5.3)]

@pytest.fixture
def regularizer():
    return rswjax.EntropyRegularizer()

@pytest.fixture
def funs():
    return [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]

def test_nan_from_admm_exception(data_frame, funs, losses, regularizer):
    with pytest.raises(ValueError) as e:
        rswjax.rsw(data_frame, funs, losses, regularizer, .000001, verbose=True)

def test_warn_too_many_losses(capsys, data_frame, regularizer):
    losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5),
              rswjax.EqualityLoss(5.3), rswjax.EqualityLoss(5.3)]
    
    rswjax.rsw(data_frame, None, losses, regularizer, .01, eps_abs=1e-8, maxiter=1, verbose=True)

    captured = capsys.readouterr()
    assert "more losses are passed" in captured.out

def test_warn_too_few_losses(capsys, data_frame, regularizer):
    losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5)]
    
    rswjax.rsw(data_frame, None, losses, regularizer, .01, eps_abs=1e-8, maxiter=1, verbose=True)

    captured = capsys.readouterr()
    assert "A loss is not defined for all columns" in captured.out
