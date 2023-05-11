from typing import List

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap


def get_shap_values(reg, X):

    print("shap...")
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X)
    print("done.")

    return explainer, shap_values


def model_interpretation(
    model: lgb.LGBMClassifier,
    to_train: pd.DataFrame,
    to_drop: List,
) -> None:
    """Analyse results of the lightgbm classifier through :
    - ROC curve
    - shap feature importance
    - shap partial dependency plots
    Args:
        model (lgb.LGBMClassifier): [full classifier lgb]
        train_data (pd.DataFrame): [data used to train the model]
        predictions_2021 (pd.DataFrame): [full predictions of the 5 fold]
        configs (Dict): [model parameters]
    """

    # # SHAP VALUES
    # !!! takes quite a bit of time
    explainer, shap_values = get_shap_values(model, to_train.drop(to_drop, axis=1))

    plt.figure(figsize=(35, 35))
    shap.summary_plot(
        shap_values[0],
        to_train.drop(to_drop, axis=1),
        plot_size=(12, 12),
        plot_type="bar",
    )
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, to_train.drop(to_drop, axis=1))

