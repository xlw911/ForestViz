from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score, mean_squared_error


def train_model(X, y, test_size=0.2, model_params=None):
    """训练随机森林模型并返回模型和评估结果"""
    if model_params is None:
        model_params = {
            'criterion': 'friedman_mse',
            'max_depth': 7,
            'n_estimators': 1000,
            'n_jobs': -1
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    rf = RandomForestRegressor(**model_params)
    rf.fit(X_train, y_train)

    return rf, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """评估模型并返回结果"""
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    metrics = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    }

    return metrics, y_train_pred, y_pred


def save_results(model, metrics, y_train, y_train_pred, y_test, y_pred, feature_names, model_path, results_path):
    """保存模型和结果"""
    # 保存模型
    joblib.dump(model, model_path)

    # 保存结果到Excel
    df_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': y_train_pred})
    df_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_pred})

    with pd.ExcelWriter(results_path) as writer:
        df_train.to_excel(writer, sheet_name='train', index=None)
        df_test.to_excel(writer, sheet_name='test', index=None)

    # 打印特征重要性
    feature_importances = model.feature_importances_
    print("\nFeature Importances:")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {feature_importances[i]}")

    return feature_importances


if __name__ == "__main__":
    # 示例用法
    df = pd.read_excel(r'./data.xlsx', index_col=0)
    cols = df.columns[:-1]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    rf, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics, y_train_pred, y_pred = evaluate_model(rf, X_train, X_test, y_train, y_test)

    # 打印评估指标
    print("Training Set Metrics:")
    print(f"R-squared (R^2): {metrics['train']['r2']}")
    print(f"Mean Squared Error (MSE): {metrics['train']['mse']}")
    print(f"Root Mean Squared Error (RMSE): {metrics['train']['rmse']}")

    print("\nTesting Set Metrics:")
    print(f"R-squared (R^2): {metrics['test']['r2']}")
    print(f"Mean Squared Error (MSE): {metrics['test']['mse']}")
    print(f"Root Mean Squared Error (RMSE): {metrics['test']['rmse']}")

    save_results(rf, metrics, y_train, y_train_pred, y_test, y_pred, cols,
                 './rfmodel.pkl', './rf_results_DXT.xlsx')