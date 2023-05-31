# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.base import is_classifier, is_regressor
from sklearn.feature_selection import RFECV

import matplotlib.pyplot as plt
from IPython.display import display


def robust_rfecv(
    X,
    y,
    model_list,
    step=1,
    cv=KFold(3),
    scoring=None,
    preprocessing_pipe=None,
    groups=None,
    show=False,
    plot=True,
    njobs=-4,
    **kwargs,
):
    """Run Recursive Feature Elimination (RFE) using multiple Machine learning models.
    The function is suitable for both classification and regression problem.
    Parameters
    ----------
    X: pd.DataFrame
        Contains training vectors.
    y: pd.Series
        Target values.
    model_list: List
        Estimators that have either coef. or feature importance. object.
    step: int, optional (default=1)
        It refers to the (integer) number of features to remove at each iteration.
    cv: cross-validation generator (default: KFold(3))
        Determines the cross-validation splitting strategy.
        For more details see https://scikit-learn.org/stable/modules/cross_validation.html
    scoring: string, callable or None, optional
        A string or a scorer callable object / function with signature.
        For more details see https://scikit-learn.org/stable/modules/model_evaluation.html
    preprocessing_pipe: Pipeline object, or Nones
        Preprocessing steps needed before the estimator (e.g. StandardScaler)
    groups: pd.Series, or Nones
        Group labels for the samples used while splitting the dataset into train/test set.
        Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).
        For more details see https://scikit-learn.org/stable/modules/cross_validation.html
        section 3.1.2.2. Cross-validation iterators with stratification based on class labels
    show: Boolean
        Whether to print the results.
    plot: Boolean
        Whether to plot the rfe summary plot.
    njobs: int
        The amount of cores to use when running the RFECV worker function.
    kwargs: key, value pairings
        Additional keyword arguments for plot setting matplotlib style. Available setting:
            figsize  : tuple (default: (7, 5))
            textsize : int (default: 15) adjust size for xticks label, title and legend
            xlim : tuple. Set the x limits of the current axes.
            ylim : tuple. Set the y limits of the current axes.
    Returns
    -------
    df_summary: pd.DataFrame
        Dataframe which show they number of optimal feature,
        score value and score type e.g., r2, auc.
    df_feature:
        DataFrame containing the selected features for each ML model
        and the corresponding intersection, results will be printed if show = True
    fig: matplotlib.figure.Figure
        matplotlib object, plots will be printed if plot = True
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from neuropy.dimensionality_reduction import robust_rfecv
    >>> from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> # load dataset
    >>> cancer = load_breast_cancer()
    >>> df = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']],
    ...                   columns = np.append(cancer['feature_names'], ['target']))
    >>> X = df.drop(columns=['target'])
    >>> y = df['target']
    >>> # prepare pipeline and classifies
    >>> pipe = Pipeline([
    ...    ('inputer', SimpleImputer(strategy='median')),
    ...    ('scaler', StandardScaler())
    ... ])
    >>> clf = [AdaBoostClassifier(n_estimators=20),
    ...        RandomForestClassifier(n_estimators=20,max_depth=5),
    ...        LinearDiscriminantAnalysis()]
    >>> df_summary, df_feature, fig = robust_rfecv(X,
    ...                                            y,
    ...                                            model_list=clf,
    ...                                            preprocessing_pipe=pipe,
    ...                                            scoring='roc_auc')
    """
    X = X.copy()
    y = y.copy()

    if not model_list:
        raise AttributeError("No ML model found as input.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X should be of type pd.DataFrame, currently the function cannot handle np.array"
        )

    # detect if regression of classification problem based on the type of model
    check_model_type = pd.DataFrame(columns=["clf", "is_classifier", "is_regressor"])
    for m in model_list:
        check_model_type = pd.concat(
            [
                check_model_type,
                pd.DataFrame(
                    {
                        "clf": m.__class__.__name__,
                        "is_classifier": is_classifier(m),
                        "is_regressor": is_regressor(m),
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        ).astype({"is_classifier": bool, "is_regressor": bool})

    if len(check_model_type["is_classifier"]) == check_model_type["is_classifier"].sum():
        # all model are classifiers
        if scoring is None:
            scoring = "roc_auc"  # assign default scoring
        print(
            f"rfecv for classification problem with {len(model_list)} ML models"
            f" with {scoring} score"
        )
        title_plot = "Recursive Feature Elimination (Classification)"
    elif len(check_model_type["is_regressor"]) == check_model_type["is_regressor"].sum():
        # all model are regressors
        if scoring is None:
            scoring = "neg_root_mean_squared_error"  # assign default scoring
        print(
            f"rfecv for regression problem with {len(model_list)} ML model mo"
            f" with {scoring} score"
        )
        title_plot = "Recursive Feature Elimination (Regression)"
    else:
        raise ValueError("'model_list' can contain either only classifiers or only regressors")

    if plot:
        # set figure dimension based on kwargs
        if "figsize" in kwargs.keys():
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=kwargs["figsize"])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

        # set figure dimension based on kwargs
        if "textsize" in kwargs.keys():
            textsize = kwargs["textsize"]
        else:
            textsize = 10

    else:
        fig = None

    scores = []
    features = []
    n_features = []
    classifier_name = []

    if preprocessing_pipe:
        pipe_names_list = (
            np.array(
                [[x.lower(), y.__class__.__name__.lower()] for x, y in preprocessing_pipe.steps]
            )
            .flatten()
            .tolist()
        )

        if any("impute" in s for s in pipe_names_list):
            X = pd.DataFrame(preprocessing_pipe.fit_transform(X), columns=X.columns)
        else:
            X = pd.DataFrame(
                preprocessing_pipe.fit_transform(X.dropna(axis=0)),
                columns=X.dropna(axis=0).columns,
            )

    if X.shape != X.dropna(axis=0).shape:
        X = X.dropna(axis=0)
        warnings.warn(
            f"""The dataframe passed as X included nan's
            and there was either no pipe passed or the pipe
                      did not include an imputer, therefore
                      the nan's have been dropped row wise. The shape of
                      the resulting data frame is {X.shape}"""
        )

    # select the same rows in y that are in X
    y = y[X.index]

    for clf in model_list:
        # extract the name of the classifier
        name = clf.__class__.__name__
        classifier_name.append(name)

        # rfecv for model_list
        rfecv = RFECV(estimator=clf, step=step, cv=cv, scoring=scoring, n_jobs=njobs)

        _ = rfecv.fit(X, y, groups=groups)

        # make list from chosen features
        features_rfe = [f for f, s in zip(X.columns, rfecv.support_) if s]
        features.append(features_rfe)

        # make list from best grid scores
        score = round(np.mean(rfecv.cv_results_["mean_test_score"]), 2)
        scores.append(score)

        # make list from optimal n. of features
        n_feature = rfecv.n_features_
        n_features.append(n_feature)

        if plot:
            try:
                x_range = np.arange(X.shape[1], 0, -step)[::-1]
                plt.plot(
                    x_range,
                    rfecv.cv_results_["mean_test_score"],
                    label=name,
                    marker="o",
                    markersize=4,
                )
            except ValueError:
                x_range = np.arange(X.shape[1], 0, -step)
                x_range = np.append(x_range, 1)[::-1]
                plt.plot(
                    x_range,
                    rfecv.cv_results_["mean_test_score"],
                    label=name,
                    marker="o",
                    markersize=4,
                )

    df_summary = pd.DataFrame(
        data={
            "optimal_n_features": np.array(n_features).astype(int),
            "highest_performance": np.array(scores),
            "score": scoring,
        },
        index=[classifier_name],
    ).sort_values(by=["highest_performance"], ascending=False)

    # compute intersection features
    classifier_name.append("intersection")
    intersection = set(features[0]).intersection(*features)
    features.append(list(intersection))

    # create dataframe which contains the feature list
    dict_feature = dict()
    for name_col, feature_col in zip(classifier_name, features):
        feature_col.sort()  # arrange values alphabetically
        dict_feature[name_col] = feature_col
    df_feature = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_feature.items()]))

    if plot:
        _ = ax.set_xlabel("Number of features selected")
        _ = ax.set_ylabel(f"cv mean {scoring}")
        _ = ax.set_title(title_plot)
        _ = ax.legend()
        _ = plt.setp(ax.get_xticklabels(), fontsize=textsize)
        _ = plt.setp(ax.get_yticklabels(), fontsize=textsize)
        _ = ax.yaxis.label.set_size(textsize)
        _ = ax.xaxis.label.set_size(textsize)
        _ = ax.title.set_size(textsize + 2)
        if "xlim" in kwargs.keys():
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs.keys():
            plt.ylim(kwargs["ylim"])

        _ = plt.axvline(
            x=len(df_feature["intersection"].dropna()),
            linestyle="-.",
            color="grey",
            linewidth=1.5,
            label="optimal number of features",
        )  #
        _ = ax.legend(loc="lower right", fontsize=textsize - 1)

    if show:
        display(df_summary)

    return df_summary, df_feature, fig


def check_grid_search_hyperparameters(gridCV: object, param_grid: dict) -> None:
    """Check if the best hyperparameter is on the edge of the param_grid.

    Args:
        gridCV (object): a trained sklearn.model_selection.GridSearchCV object
        param_grid (dict): dictionary of hyperparameters used in the grid search

    Returns:
        None: print warning if the best hyperparameter is on the edge of the param_grid

    Examples:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from c4_helpers import check_grid_search_hyperparameter
        >>> # load dataset
        >>> cancer = load_breast_cancer()
        >>> X, y = cancer.data, cancer.target
        >>> # prepare grid search
        >>> param_grid = {'n_estimators': [10, 20, 30],
        ...               'max_depth': [3, 5, 7]}
        >>> gridCV = GridSearchCV(RandomForestClassifier(),
        ...                       param_grid=param_grid,
        ...                       cv=3,
        ...                       scoring='roc_auc')
        >>> gridCV.fit(X, y)
        >>> # check if the best hyperparameter is on the edge of the param_grid
        >>> check_grid_search_hyperparameter(gridCV, param_grid)
        Warning: max_depth=7 is on the maximum edge of the param_grid. Please Enlarge the param_grid of this parameter.
    """

    # check if GridSearchCV has been fitted
    if not gridCV.best_params_:
        raise ValueError("The GridSearchCV object has not been fitted yet.")

    # check if param_grid is a dictionary
    if not isinstance(param_grid, dict):
        raise TypeError("param_grid should be a dictionary.")

    # Get min and max values of param_grid with numerical values only
    param_grid_num = {k: v for k, v in param_grid.items() if isinstance(v[0], (int, float))}
    param_grid_num_min = {k: min(v) for k, v in param_grid_num.items()}
    param_grid_num_max = {k: max(v) for k, v in param_grid_num.items()}
    param_grid_num_min, param_grid_num_max

    # check if the values of the grid search are on the edges of the param_grid
    for k, v in param_grid_num_min.items():
        if v == gridCV.best_params_[k]:
            print(
                f"Warning: {k}={gridCV.best_params_[k]} is on the minimum edge of the param_grid."
                " Please Enlarge the param_grid of this parameter."
            )
    for k, v in param_grid_num_max.items():
        if v == gridCV.best_params_[k]:
            print(
                f"Warning: {k}={gridCV.best_params_[k]} is on the maximum edge of the param_grid."
                " Please Enlarge the param_grid of this parameter."
            )
