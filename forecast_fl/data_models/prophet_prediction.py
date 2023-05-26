#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forecast_fl.data_evaluation.evaluate_results import mean_absolute_percentage_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

logger = logging.getLogger("cmdstanpy")
logger.setLevel(logging.ERROR)

# " to remove the stan printing"
class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def create_holidays_prophet(special_holidays):

    holidays = pd.DataFrame()

    for special_days in special_holidays["FERIER"].unique():
        if special_days != "None":
            list_special_days = special_holidays.loc[special_holidays["FERIER"] == special_days, "DATE"].unique()
            df_to_add = pd.DataFrame(
                {
                    "holiday": special_days,
                    "ds": list_special_days,
                    "lower_window": 0,
                    "upper_window": 1,
                }
            )

            holidays = pd.concat((holidays, df_to_add))

    return holidays


def deduce_mape(test, split_date=None):

    if split_date:
        test = test.loc[test["ds"] >= split_date]

    erreur_test = mean_absolute_percentage_error(
        np.exp(test.loc[(~test["y"].isnull()) & (test["y"] > 0)]["y"].tolist()) - 1,
        np.exp(test.loc[(~test["y"].isnull()) & (test["y"] > 0)]["PREDICTION"]) - 1,
    )

    return erreur_test


def fitting_prophet(df, holidays, prophet_config):

    with suppress_stdout_stderr():
        model = (
            Prophet(
                stan_backend="CMDSTANPY",
                interval_width=0.95,
                growth="linear",
                holidays_prior_scale=prophet_config["holidays_prior_scale"],
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                n_changepoints=prophet_config["n_changepoints"],
                changepoint_prior_scale=prophet_config["changepoint_prior_scale"],
                changepoint_range=prophet_config["changepoint_range"],
                holidays=holidays,
            )
            .add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=prophet_config["yearly_fourier_order"],
            )
            .add_seasonality(
                name="monthly",
                period=30.5,
                fourier_order=prophet_config["monthly_fourier_order"],
            )
            .add_seasonality(
                name="weekly",
                period=7,
                fourier_order=prophet_config["weekly_fourier_order"],
            )
        )

        model.fit(df, iter=prophet_config["n_iterations"])

    return model


def predicting_prophet(model, df):

    predictions = model.predict(df)
    df["PREDICTION"] = predictions["yhat"].clip(0, None).values

    # remove the artificial negative value used to force prophet to predict 0 when
    # needed instead of oscillating around it
    df["y"] = np.where(df["y"] < 0, 0, df["y"])

    return df, predictions


def prophet_predict(
    full_predict: pd.DataFrame,
    split_date: str,
    special_holidays: pd.DataFrame,
    ub_n: str,
    configs: Dict[str, Any],
    base_path: Path,
    granularity: str,
):

    parameters_from_config = configs.load.parameters_training_model
    model_name = "prophet"
    prophet_config = configs.load.config_prophet

    train = full_predict.loc[full_predict["ds"] < split_date].reset_index(drop=True)
    test = full_predict.loc[full_predict["ds"] >= split_date].reset_index(drop=True)

    # create prophet holidays format
    holidays = create_holidays_prophet(special_holidays)

    # train / to remove the stan printing
    #################################### train / test model
    model = fitting_prophet(train, holidays, prophet_config)
    test, _ = predicting_prophet(model, test)
    train, _ = predicting_prophet(model, train)

    erreur_test = deduce_mape(test)
    logging.info(f"UB NAME = {ub_n}; - PROPHET ON TEST SET: {erreur_test}")

    ################################# train fully on overall data
    model = fitting_prophet(full_predict, holidays, prophet_config)
    full_predict, predictions = predicting_prophet(model, full_predict)

    erreur_test = deduce_mape(full_predict, split_date)
    logging.info(f"UB NAME = {ub_n}; - PROPHET ERROR OVERFITTED: {erreur_test}")

    full_predict["MODEL"] = model_name
    full_predict.columns = [x.upper() for x in full_predict.columns]
    train.columns = [x.upper() for x in train.columns]
    test.columns = [x.upper() for x in test.columns]

    if parameters_from_config["save_plots"]:

        # plot preds vs signal
        model.plot_components(predictions)
        path = base_path / str(ub_n) / "plots" / "time_series"
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(path.absolute())
            + f"/plot_timeseries_model_{model_name}_ub_{ub_n}_codesite_{granularity}_prophet_seasonality.png"
        )
        plt.close()

        full_predict.set_index("DS")[["PREDICTION", "Y"]].plot(alpha=0.5, title=ub_n, figsize=(15, 10))
        plt.savefig(
            str(path.absolute()) + f"/FULL_plot_timeseries_model_{model_name}_ub_{ub_n}_codesite_{granularity}.png"
        )
        plt.close()

    if prophet_config["prophet_cross_val"]:
        df_cv = cross_validation(model, initial="100 days", period="2 days", horizon="2 days")

        df_p = performance_metrics(df_cv)
        erreur_2D = df_p.loc[df_p["horizon"] == "2 days", "mape"].values[0]
        erreur_test = mean_absolute_percentage_error(train["y"], train["PREDICTION"])

        logging.info(f"UB NAME = {ub_n}; site {granularity} MAPE +2D = {erreur_2D}; - TEST: {erreur_test}")
        plot_cross_validation_metric(df_cv, metric="mape")

    return full_predict, train, test, model
