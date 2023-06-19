# -*- coding: utf-8 -*-
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error
from IPython.display import display


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = "Score", show_description: str = True
) -> pd.DataFrame:
    """Generate a report for a regression model.

    Args:
        y_true (np.ndarray): the true labels of the data
        y_pred (np.ndarray): the predicted labels of the data (must be binary)
        label (str, optional): tailor score name. useful if you want to compare multiple models. Defaults to 'Score'.
        show_description (str, optional): Provide a description of each metric to the report. Defaults to True.

    Returns:
        pd.DataFrame: a dataframe containing the classification report

    Example:
        >>> import numpy as np
        >>> from r4_helpers import regression_report
        >>> y_true = np.array([1,   0.1, 0.5, 6, -2, 0, 1.1, -0.5, 12])
        >>> y_pred = np.array([0.5, 0.3, 0.2, 3, -1, 1, 1.5, -1, 1])
        >>> regression_report(y_true, y_pred)
    """

    dict_clf = {
        "r-squared": [
            explained_variance_score(y_true, y_pred),
            (
                "Explained variance regression score function. Best possible score is 1.0, lower"
                " values are worse."
            ),
        ],
        "RMSE": [
            mean_squared_error(y_true, y_pred, squared=False),
            "Root Mean Squared Error (RMSE)",
        ],
        "MAE": [
            mean_absolute_error(y_true, y_pred),
            "Mean Absolute Error (MAE)",
        ],
        "r-value output": [
            np.corrcoef(y_true, y_pred)[0, 1],
            "Correlation between y_true and y_pred",
        ],
    }

    # Convert Dictionary to DataFrame
    df_report = pd.DataFrame.from_dict(
        dict_clf, orient="index", columns=[label, "Description"]
    ).round(3)

    if show_description is False:
        df_report.drop(columns=["Description"], inplace=True)

    return df_report


def normlize_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, quantitative_features: list
) -> pd.DataFrame:
    """Normalize numerical features using StandardScaler

    Args:
        X_train (pd.DataFrame): The train set coming from the train_test_split function
        X_test (pd.DataFrame): The test set coming from the train_test_split function
        quantitative_list (List): List of numerical features

    Returns:
        pd.DataFrame: The train and test set with the numerical features normalized
    """

    # initialize scaler
    scaler = StandardScaler()
    # fit scaler on train data
    _ = scaler.fit(X_train[quantitative_features])
    # transform train and test data
    X_train_scaled = pd.DataFrame(
        data=scaler.transform(X_train[quantitative_features]),
        columns=[x + "_scaled" for x in quantitative_features],
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        data=scaler.transform(X_test[quantitative_features]),
        columns=[x + "_scaled" for x in quantitative_features],
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled


def encode_categorical_features(X_train, X_test, categorical_features):
    """Encode categorical features using sklearn OneHotEncoder
    Args:
        X_train (pd.DataFrame): The train set coming from the train_test_split function
        X_test (pd.DataFrame): The test set coming from the train_test_split function
        categorical_features (list): list of categorical features

    Returns:
        X_train_cat (pd.DataFrame): Train data with encoded categorical features
        X_test_cat (pd.DataFrame): Test data with encoded categorical features
    """

    # initialize encoder
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    # fit encoder on train data
    _ = encoder.fit(X_train[categorical_features])

    # transform train and test data
    X_train_cat = pd.DataFrame(
        data=encoder.transform(X_train[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
        index=X_train.index,
    )

    X_test_cat = pd.DataFrame(
        data=encoder.transform(X_test[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
        index=X_test.index,
    )

    return X_train_cat, X_test_cat


def get_variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the variance inflation factors (VIF) for the features in the dataframe X
    :param X: dataframe contraining the features
    :return: dataframe with the variance inflation factors
    """

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return vif_data


# flake8: noqa: C901, F722
def diagnostic_plots(
    model_fit: "fitted linear model" = None,  # type: ignore
    X: Union[pd.DataFrame, None] = None,
    y: Union[pd.Series, None] = None,
    figsize: tuple = (12, 12),
    limit_cooks_plot: bool = False,
    subplot_adjust_args: dict = {"wspace": 0.3, "hspace": 0.3},
):
    """Plot the diagnostic information for Regression models.

    This is done to include the use of models defined using *statsmodels*, and *sklearn*.
    Specifically the currently accepted models are:

    - sklearn.linear_model._base.LinearRegression

    - sklearn.linear_model._coordinate_descent.Lasso

    - sklearn.linear_model._ridge.Ridge

    - statsmodels.regression.linear_model.RegressionResultsWrapper

    - statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper

    The sklearn models listed have been tried, although in theory all variants of the sklearn linear regression
    models should work with this function.

    Parameters
    ----------
    model_fit:
        A fitted linear regression model, ideally one of the ones listed above.
    X:
        The array of predictors used to fit the model. This is only required if the inputted model is an sklearn
        model. For all other models just the fitted model is sufficient.
    y:
        The target array used to fit the model. This is only required if the inputted model is an sklearn
        model. For all other models just the fitted model is sufficient.
    figsize: tuple
        Width and height of the figure in inches
    limit_cooks_plot: bool
        Whether to apply a y-limit to the cooks distances plot (i.e. would you like to see the cooks distances
        better, or the individual scatter points better?)
    subplot_adjust_args: dict
        A dictionary of arguments to change the dimensions of the subplots. This is useful to include if the chosen
        figsize is making the plot labels overlap.

    Returns
    -------
    diagplot: matplotlib.figure.Figure
        Figure with four diagnostic plots: residuals vs fitted, QQplot, scale location, residuals vs leverage
    axes: np.array
        An array of the associated axes of the four subplots.

    Examples
    --------
    >>> ## sklearn example
    >>> import seaborn as sns
    >>> from sklearn.linear_model import LinearRegression
    >>> from r4_helpers import diagnostic_plots
    >>> data = sns.load_dataset(name="mpg")
    >>> X = data[['cylinders', 'displacement', 'weight', 'acceleration']]
    >>> y = data["mpg"]
    >>> model_fit = LinearRegression().fit(X, y)
    >>> fig, axs = diagnostic_plots(model_fit=model_fit,
    ...                                         X=X,
    ...                                         y=y,
    ...                                         figsize = (8,8),
    ...                                         limit_cooks_plot = False,
    ...                                         subplot_adjust_args={"wspace": 0.3, "hspace": 0.3}
    ...                                        )
    >>> ## statsmodels example
    >>> import seaborn as sns
    >>> import statsmodels.api as sm
    >>> from r4_helpers import diagnostic_plots
    >>> data = sns.load_dataset(name="mpg")
    >>> X = data[['cylinders', 'displacement', 'weight', 'acceleration']]
    >>> y = data["mpg"]
    >>> model_fit = sm.OLS(endog=y, exog=X).fit()
    >>> fig, axs = diagnostic_plots(model_fit=model_fit,
    ...                                         X=None,
    ...                                         y=None,
    ...                                         figsize = (8,8),
    ...                                         limit_cooks_plot = False,
    ...                                         subplot_adjust_args={"wspace": 0.3, "hspace": 0.3}
    ...                                        )

    """

    def _get_model_fit(model_fit):
        source = str(type(model_fit))

        if not any(substring in source for substring in ["sklearn", "statsmodels"]):
            raise ValueError(f"The model {source}, is currently not supported by this function.")

        if "sklearn" in source:
            if X is None and not any(
                substring in str(type(X)) for substring in ["DataFrame", "Series", "ndarray"]
            ):
                raise ValueError(
                    f"""The fitted model specified was an sklearn model which requires X to not be None and one of
                    either a pandas DataFrame or Series, or a numpy ndarray. The X supplied was of type: {str(type(X))}"""
                )

            if y is None and not any(
                substring in str(type(y)) for substring in ["Series", "ndarray"]
            ):
                raise ValueError(
                    f"""The fitted model specified was an sklearn model which requires y to not be None and one of
                    either a pandas Series, or a numpy ndarray. The y supplied was of type: {str(type(y))}"""
                )

            model_fitted_y = model_fit.predict(X)
            model_residuals = y - model_fit.predict(X)
            model_norm_residuals = model_residuals / model_residuals.std()

            model_abs_resid = np.abs(model_residuals)
            model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        if "statsmodels" in source:
            # model values
            model_fitted_y = model_fit.fittedvalues
            # model residuals
            model_residuals = model_fit.resid
            # absolute residuals
            model_abs_resid = np.abs(model_residuals)

            if "regression.mixed_linear_model" in source:
                # studentized residuals
                model_norm_residuals = model_residuals / model_residuals.std()
                # absolute square root residuals
                model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

            if "regression.linear_model" in source:
                # normalized residuals
                model_norm_residuals = pd.Series(
                    model_fit.get_influence().resid_studentized_internal
                )
                model_norm_residuals.index.name = "index"
                model_norm_residuals.name = "model_norm_residuals"

                # root square absolute normalized residuals
                model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        return (
            model_fitted_y,
            model_residuals,
            model_norm_residuals,
            model_abs_resid,
            model_norm_residuals_abs_sqrt,
        )

    def _get_model_leverage(model_fit):
        source = str(type(model_fit))

        if "statsmodels.regression.linear" in source:
            # leverage, from statsmodels internals
            model_leverage = pd.Series(model_fit.get_influence().hat_matrix_diag.transpose())

        else:
            if "sklearn" in source:
                temp = X.values
            if "statsmodels" in source:
                # get the data used to predict the y (i.e. X or the exogenous variables)
                temp = model_fit.model.data.exog

            hat_temp = temp.dot(np.linalg.inv(temp.T.dot(temp)).dot(temp.T))
            hat_temp_diag = np.diagonal(hat_temp)
            model_leverage = pd.Series(hat_temp_diag)

        model_leverage.name = "model_leverage"
        model_leverage.index.name = "index"

        return model_leverage

    def _get_num_of_parameters(model_fit):
        source = str(type(model_fit))

        if "sklearn" in source:
            number_of_parameters = len(model_fit.coef_)

        if "statsmodels" in source:
            number_of_parameters = len(model_fit.params)

        return number_of_parameters

    (
        model_fitted_y,
        model_residuals,
        model_norm_residuals,
        model_abs_resid,
        model_norm_residuals_abs_sqrt,
    ) = _get_model_fit(model_fit)
    model_leverage = _get_model_leverage(model_fit)
    number_of_parameters = _get_num_of_parameters(model_fit)

    ## Generate plots

    # create figure with 4 subplots
    diagplot, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    #     _ = plt.subplots_adjust(wspace=0.6, hspace=0.6)
    _ = plt.subplots_adjust(**subplot_adjust_args)
    _ = plt.suptitle("Model diagnostics")
    # First plot: Residuals vs fitted
    _ = sns.regplot(
        x=model_fitted_y,
        y=model_residuals,
        scatter=True,
        lowess=True,
        line_kws={"color": "#e58038", "lw": 1, "alpha": 0.8},
        scatter_kws={"alpha": 0.3},
        ax=diagplot.axes[0],
    )
    x_range = np.linspace(min(model_fitted_y), max(model_fitted_y), 50)
    _ = diagplot.axes[0].plot(
        x_range,
        np.repeat(0, len(x_range)),
        lw=1,
        ls=":",
        color="#808080",
    )
    _ = diagplot.axes[0].set_title("Residuals vs Fitted")
    _ = diagplot.axes[0].set_xlabel("Fitted values")
    _ = diagplot.axes[0].set_ylabel("Residuals")
    margin_res = 0.10 * (max(model_residuals) - min(model_residuals))
    _ = diagplot.axes[0].set_ylim(
        min(model_residuals) - margin_res, max(model_residuals) + margin_res
    )

    # annotations: top 3 absolute residuals
    abs_resid_top_3 = model_abs_resid.sort_values(ascending=False)[:3]
    for i in abs_resid_top_3.index:
        _ = diagplot.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    res = stats.probplot(model_norm_residuals, dist="norm", plot=None, rvalue=True)
    ordered_theoretical_quantiles = res[0][0]
    ordered_residuals = res[0][1]
    slope = res[1][0]
    intercept = res[1][1]
    r_value = res[1][2]

    _ = diagplot.axes[1].scatter(ordered_theoretical_quantiles, ordered_residuals, alpha=0.3)
    _ = diagplot.axes[1].plot(
        ordered_theoretical_quantiles,
        slope * ordered_theoretical_quantiles + intercept,
        "#e58038",
    )
    _ = diagplot.axes[1].plot([], [], ls="", label=f"$R^2={round(r_value ** 2, 3)}$")
    _ = diagplot.axes[1].legend(loc="lower right")

    # _ = diagplot.axes[1].get_lines()[1].set_markerfacecolor("#e58038")
    _ = diagplot.axes[1].set_title("Normal Q-Q")
    _ = diagplot.axes[1].set_xlabel("Theoretical Quantiles")
    _ = diagplot.axes[1].set_ylabel("Standardized Residuals")

    abs_norm_resid_top_3 = np.abs(model_norm_residuals).sort_values(ascending=False)[:3]
    norm_resid_top_3 = model_norm_residuals[abs_norm_resid_top_3.index]
    ordered_df = pd.DataFrame(
        {
            "ordered_residuals": ordered_residuals,
            "ordered_theoretical_quantiles": ordered_theoretical_quantiles,
        }
    )

    for i in abs_norm_resid_top_3.index:
        #         index = np.where(pd.Series(ordered_residuals).index == i)[0][0]

        _ = diagplot.axes[1].annotate(
            i,
            xy=(
                ordered_df.loc[
                    ordered_df["ordered_residuals"] == norm_resid_top_3[i],
                    "ordered_theoretical_quantiles",
                ].values[0],
                ordered_df.loc[
                    ordered_df["ordered_residuals"] == norm_resid_top_3[i],
                    "ordered_residuals",
                ].values[0],
            ),
        )

    # Third plot: scale location
    _ = sns.regplot(
        x=model_fitted_y,
        y=model_norm_residuals_abs_sqrt,
        scatter=True,
        ci=False,
        lowess=True,
        line_kws={"color": "#e58038", "lw": 1, "alpha": 0.8},
        scatter_kws={"alpha": 0.3},
        ax=diagplot.axes[2],
    )
    _ = diagplot.axes[2].set_title("Scale-Location")
    _ = diagplot.axes[2].set_xlabel("Fitted values")
    _ = diagplot.axes[2].set_ylabel(r"$\sqrt{|Standardized Residuals|}$")
    # annotations: top 3 absolute normalized residuals
    for i in abs_norm_resid_top_3.index:
        _ = diagplot.axes[2].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))

    # Fourth plot: residuals vs leverages
    _ = sns.regplot(
        x=model_leverage,
        y=model_norm_residuals,
        scatter=True,
        ci=False,
        lowess=True,
        line_kws={"color": "#e58038", "lw": 1, "alpha": 0.8},
        scatter_kws={"alpha": 0.3},
        ax=diagplot.axes[3],
    )
    _ = diagplot.axes[3].set_xlim(0, max(model_leverage) + 0.01)

    if limit_cooks_plot:
        _ = diagplot.axes[3].set_ylim(
            min(model_norm_residuals) - 0.5, max(model_norm_residuals) + 0.5
        )
    _ = diagplot.axes[3].set_title("Residuals vs Leverage")
    _ = diagplot.axes[3].set_xlabel("Leverage")
    _ = diagplot.axes[3].set_ylabel("Standardized Residuals")

    # annotations: top 3 levarages
    leverage_top_3 = model_leverage.sort_values(ascending=False)[:3]
    for i in leverage_top_3.index:
        _ = diagplot.axes[3].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))
    # extra lines to indicate Cook's distances
    x_range = np.linspace(0.001, max(model_leverage), 50)

    def cooksdistances(boundary):
        return lambda x: np.sqrt((boundary * number_of_parameters * (1 - x)) / x)

    for line in [0.5, 1]:
        l_formula = cooksdistances(line)
        for place in [1, -1]:
            cooks_line = plt.plot(
                x_range,
                place * l_formula(x_range),
                lw=1,
                ls="--",
                color="#e58038",
            )
            y_text = place * l_formula(max(model_leverage) + 0.01)
            if min(model_norm_residuals) - 0.5 < y_text < max(model_norm_residuals) + 0.5:
                _ = plt.text(
                    max(model_leverage) + 0.01,
                    y_text,
                    str(line),
                    color="#e58038",
                )
    _ = diagplot.axes[3].legend(
        cooks_line[:2], ["Cook's distance"], handlelength=3, loc="lower right"
    )

    return diagplot, axes
