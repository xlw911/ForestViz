import pytest
import numpy as np
import pandas as pd
from main import train_model, evaluate_model, save_results
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


@pytest.fixture
def sample_data():
    # 创建测试数据
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = X.dot(np.array([1, 2, 3, 4, 5])) + np.random.normal(0, 0.1, 100)
    cols = [f'feature_{i}' for i in range(5)]
    return X, y, cols


def test_train_model(sample_data):
    X, y, _ = sample_data
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    assert isinstance(model, RandomForestRegressor)
    assert X_train.shape[0] == 80  # 默认test_size=0.2
    assert X_test.shape[0] == 20
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20


def test_evaluate_model(sample_data):
    X, y, _ = sample_data
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics, y_train_pred, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    assert 'train' in metrics
    assert 'test' in metrics
    assert isinstance(metrics['train']['r2'], float)
    assert isinstance(metrics['test']['r2'], float)
    assert len(y_train_pred) == len(y_train)
    assert len(y_pred) == len(y_test)


def test_save_results(sample_data, tmp_path):
    X, y, cols = sample_data
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics, y_train_pred, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    model_path = tmp_path / "test_model.pkl"
    results_path = tmp_path / "test_results.xlsx"

    feature_importances = save_results(
        model, metrics, y_train, y_train_pred, y_test, y_pred, cols,
        str(model_path), str(results_path)
    )

    assert os.path.exists(model_path)
    assert os.path.exists(results_path)
    assert len(feature_importances) == X.shape[1]

    # 检查保存的模型是否可以加载
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, RandomForestRegressor)

    # 检查结果文件
    df_train = pd.read_excel(results_path, sheet_name='train')
    df_test = pd.read_excel(results_path, sheet_name='test')
    assert len(df_train) == len(y_train)
    assert len(df_test) == len(y_test)


def test_model_performance(sample_data):
    X, y, _ = sample_data
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics, _, _ = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 检查R2分数在合理范围内
    assert -1 <= metrics['train']['r2'] <= 1
    assert -1 <= metrics['test']['r2'] <= 1

    # 检查MSE和RMSE是非负数
    assert metrics['train']['mse'] >= 0
    assert metrics['train']['rmse'] >= 0
    assert metrics['test']['mse'] >= 0
    assert metrics['test']['rmse'] >= 0