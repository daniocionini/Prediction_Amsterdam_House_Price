# -*- coding: utf-8 -*-
from itertools import combinations, product
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder


def chain_snap(data, fn=lambda x: x.shape, msg=None):
    r"""Print things in method chaining, leaving the dataframe untouched.
    Parameters
    ----------
    data: pandas.DataFrame
        the initial data frame for which the functions will be applied to in the pipe
    fn: lambda
        function that takes a pandas.DataFrame and that creates output to be printed
    msg: str or None
        optional message to be printed above output of the function
    Examples
    --------
    >>> from neuropy.utils import chain_snap
    >>> import pandas as pd
    >>> df = pd.DataFrame({'letter': ['a', 'b', 'c', 'c', 'd', 'c', 'a'],
    ...                    'number': [5,4,6,3,8,1,5]})
    >>> df = df.pipe(chain_snap, msg='Shape of the dataframe:')
    >>> df = df.pipe(chain_snap,
    ...              fn = lambda df: df['letter'].value_counts(),
    ...              msg="Frequency of letters:")
    >>> df = df.pipe(chain_snap,
    ...              fn = lambda df: df.loc[df['letter']=='c'],
    ...              msg="Dataframe where letter is c:")
    """
    if msg:
        print(msg + ": " + str(fn(data)))
    else:
        print(fn(data))
    return data

def normal_check(data: pd.DataFrame) -> pd.DataFrame:
    """Compare the distribution of numeric variables to a normal distribution using the Kolmogrov-Smirnov test.
    Wrapper for `scipy.stats.kstest`: the empircal data is compared to a normally distributed variable with the
    same mean and standard deviation. A significant result (p < 0.05) in the goodness of fit test means that the
    data is not normally distributed.
    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe including the columns of interest
    Returns
    -------
    df_normality_check: pd.DataFrame
        Dataframe with column names, p-values and an indication of normality
    Examples
    --------
    >>> tips = sns.load_dataset("tips")
    >>> df_normality_check = normal_check(tips)
    """
    # Select numeric columns only
    num_features = data.select_dtypes(include="number").columns.tolist()
    # Compare distribution of each feature to a normal distribution with given mean and std
    df_normality_check = data[num_features].apply(
        lambda x: stats.kstest(
            x.dropna(),
            stats.norm.cdf,
            args=(np.nanmean(x), np.nanstd(x)),
            N=len(x),
        )[1],
        axis=0,
    )

    # create a label that indicates whether a feature has a normal distribution or not
    df_normality_check = pd.DataFrame(df_normality_check).reset_index()
    df_normality_check.columns = ["feature", "p-value"]
    df_normality_check["normality"] = df_normality_check["p-value"] >= 0.05

    return df_normality_check

def correlation_analysis(
    data: pd.DataFrame,
    col_list=None,
    row_list=None,
    check_norm=False,
    method: str = "pearson",
    dropna: str = "pairwise",
) -> dict:
    r"""Run correlations for numerical features and return output in different formats.
    Different methods to compute correlations and to handle missing values are implemented.
    Inspired by `researchpy.corr_case` and `researchpy.corr_pair`.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    row_list: list or None (default: None)
        List with names of columns in `data` that should be in the rows of the correlogram.
        If None, all columns are used but only every unique combination.
    col_list: list or None (default: None)
        List with names of columns in `data` that should be in the columns of the correlogram.
        If None, all columns are used and only every unique combination.
    check_norm: bool (default: False)
        If True, normality will be checked for columns in `data` using `normal_check`.
        This influences the used method for correlations, i.e. Pearson
        or Spearman. Note: normality check ignores missing values.
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau,
        implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`
        Will be ignored if check_norm=True. Instead, Person's r is used
        for every combination of normally distributed
        columns and Spearman's rho is used for all other combinations.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over the complete
        `data` ('listwise') or for every correlation
        separately ('pairwise')


    Returns
    -------
    result_dict: dict
    Dictionary containing with the following keys:
    info: pandas.DataFrame
        Description of correlation method, missing values handling
        and number of observations
    r-values: pandas.DataFrame
        Dataframe with correlation coefficients. Indices and columns
        are column names from `data`. Only lower
        triangle is filled.
    p-values: pandas.DataFrame
        Dataframe with p-values. Indices and columns are column names
        from `data`. Only lower triangle is filled.
    N: pandas.DataFrame
        Dataframe with numbers of observations. Indices and columns
        are column names from `data`. Only lower
        triangle is filled. If dropna ='listwise', every correlation
        will have the same number of observations.
    summary: pandas.DataFrame
        Dataframe with columns ['analysis', 'feature1', 'feature2',
        'r-value', 'p-value', 'N', 'stat-sign']
        which indicate the type of test used for the correlation,
        the pair of columns, the correlation coefficient,
        the p-value, the number of observations for each combination
        of columns in `data` and whether the r-value is
        statistically significant.

    Examples
    --------
    >>> import seaborn as sns
    >>> iris = sns.load_dataset('iris')
    >>> dict_results = correlation_analysis(iris,
    ...                                     method='pearson',
    ...                                     dropna='listwise',
    ...                                     check_norm=True)
    >>> dict_results['summary']
    References
    ----------
    Bryant, C (2018). researchpy's documentation [Revision 9ae5ed63]. Retrieved from
    https://researchpy.readthedocs.io/en/latest/
    """

    # Settings test
    if method == "pearson":
        test, test_name = stats.pearsonr, "Pearson"
    elif method == "spearman":
        test, test_name = stats.spearmanr, "Spearman Rank"
    elif method == "kendall":
        test, test_name = stats.kendalltau, "Kendall's Tau-b"
    else:
        raise ValueError("method not in {'pearson', 'kendall', 'spearman'}")

    # Copy numerical data from the original data
    data = data.copy().select_dtypes("number")

    # Get correct lists
    if col_list and not row_list:
        row_list = data.select_dtypes("number").drop(col_list, axis=1).columns.tolist()
    elif row_list and not col_list:
        col_list = data.select_dtypes("number").drop(row_list, axis=1).columns.tolist()

    # Initializing dataframes to store results
    info = pd.DataFrame()
    summary = pd.DataFrame()
    if not col_list and not row_list:
        r_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        p_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        n_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        iterator = combinations(data.columns, 2)  # type: ignore
    else:
        r_vals = pd.DataFrame(columns=col_list, index=row_list)
        p_vals = pd.DataFrame(columns=col_list, index=row_list)
        n_vals = pd.DataFrame(columns=col_list, index=row_list)
        iterator = product(col_list, row_list)  # type: ignore

    if dropna == "listwise":
        # Remove rows with missing values
        data = data.dropna(how="any", axis="index")
        info = pd.concat(
            [
                info,
                pd.DataFrame(
                    {
                        f"{test_name} correlation test using {dropna} deletion": (
                            f"Total observations used = {len(data)}"
                        )
                    },
                    index=[0],
                ),
            ]
        )
    elif dropna == "pairwise":
        info = pd.concat(
            [
                info,
                pd.DataFrame(
                    {
                        f"{test_name} correlation test using {dropna} deletion": (
                            f"Observations in the data = {len(data)}"
                        )
                    },
                    index=[0],
                ),
            ]
        )
    else:
        raise ValueError("dropna not in {'listwise', 'pairwise'}")

    if check_norm:
        # Check normality of all columns in the data
        df_normality = normal_check(data)
        norm_names = df_normality.loc[df_normality["normality"], "feature"].tolist()

    # Iterating through the Pandas series and performing the correlation
    for col1, col2 in iterator:
        if dropna == "pairwise":
            # Remove rows with missing values in the pair of columns
            test_data = data[[col1, col2]].dropna()
        else:
            test_data = data

        if check_norm:
            # Select Pearson's r only if both columns are normally distributed
            if (col1 in norm_names) and (col2 in norm_names):
                test, test_name = stats.pearsonr, "Pearson"
            else:
                test, test_name = stats.spearmanr, "Spearman Rank"

        # Run correlations
        r_value, p_value = test(test_data.loc[:, col1], test_data.loc[:, col2])
        n_value = len(test_data)

        # Store output in matrix format
        try:
            r_vals.loc[col2, col1] = r_value
            p_vals.loc[col2, col1] = p_value
            n_vals.loc[col2, col1] = n_value
        except KeyError:
            r_vals.loc[col1, col2] = r_value
            p_vals.loc[col1, col2] = p_value
            n_vals.loc[col1, col2] = n_value

        # Store output in dataframe format
        dict_summary = {
            "analysis": test_name,
            "feature1": col1,
            "feature2": col2,
            "r-value": r_value,
            "p-value": p_value,
            "stat-sign": (p_value < 0.05),
            "N": n_value,
        }

        summary = pd.concat(
            [summary, pd.DataFrame(data=dict_summary, index=[0])],
            axis=0,
            ignore_index=True,
            sort=False,
        )

    # Embed results within a dictionary
    result_dict = {
        "r-value": r_vals,
        "p-value": p_vals,
        "N": n_vals,
        "info": info,
        "summary": summary,
    }

    return result_dict

def one_way_ANOVA(
    data: pd.DataFrame,
    feature: str,
    grouping_var: str,
    groups_of_interest: list,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
    col_wrap=None,
):
    """Run one-way ANOVAs using `scipy.stats.f_oneway` and check homogeneity of variances with Levenes test using `scipy.stats.levene`.

    `one_way_ANOVA` assumes equal variances within the groups and will not give a warning if show=False.

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature` and `grouping_var` in columns
    feature: str
        Name of the feature
    grouping_var: str
        Name of the  column with grouping labels in `data`
    groups_of_interest: list
        Names (str) of labels in `data[grouping_var]`
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    col_wrap: int or None (default: None)
        If int, number of subplots that are allowed in a single row

    Returns
    -------
    df_result: pd.DataFrame
    df_descriptive: pd.DataFrame
    distplot: Figure
        Figure if plot == True, else None
    boxplot: Figure
        Figure if plot == True, else None

    Examples
    --------
    >>> import seaborn as sns
    >>> tips = sns.load_dataset("tips")
    >>> _, _, _, _ = one_way_ANOVA(tips, 'tip', 'day', ['Sat','Sun','Thur'], show = True, plot = False)

    """
    # select the 'feature' and 'grouping_var' columns and remove row if any nan present
    data = data.copy()
    data = data[[feature, grouping_var]].dropna(axis=0, how="any")

    # Raise error if feature is not numeric
    if feature not in data.select_dtypes("number").columns:
        raise TypeError(f"Feature {feature} should be numeric")

    # select the groups of interest and remove any not used category from the categorical index
    data = data.loc[data[grouping_var].isin(groups_of_interest), :]
    if data[grouping_var].dtype.name == "category":
        data[grouping_var] = data[grouping_var].cat.remove_unused_categories()

    # get descriptive values, keep only interested rows
    df_descriptive = data.groupby(grouping_var, observed=True)[feature].describe()
    _ = df_descriptive.reset_index(inplace=True)

    # Raise warning if groups of interest not in the dataframe
    if not all(grp in df_descriptive[grouping_var].values.tolist() for grp in groups_of_interest):
        warnings.warn(
            f"One of the groups did not have any observations for {feature}",
            stacklevel=2,
        )

    values_per_group = {
        grp_label: values
        for grp_label, values in data.groupby(grouping_var, observed=True)[feature]
    }

    # Check assumption: homogeneity of variances
    (levene, levene_p_value) = stats.levene(*values_per_group.values())

    if levene_p_value > 0.05:
        # Equal variances:
        variance_outcome = "Equal"
        trust_results = "trustworthy"
    else:
        # Unequal variances: ANOVA cannot be trusted
        variance_outcome = "Unequal"
        trust_results = "untrustworthy"

    # Run one way ANOVA
    (f_value, p_value) = stats.f_oneway(*values_per_group.values())

    # Lakens, D.(2013).Calculating and reporting effect sizes to facilitate cumulative science:
    # a practical primer for t - tests and ANOVAs.Frontiers in psychology, 4, 863.
    # eta_squared = ((f * df_effect) / ((f * df_effect) + df_error))

    df_effect = len(groups_of_interest) - 1
    df_error = data[feature].count() - df_effect
    eta_squared = (f_value * df_effect) / ((f_value * df_effect) + df_error)

    if show:
        print(
            f"=== One-way anova: variable = *{feature}* | groups ="
            f" *{', '.join(groups_of_interest)}* defined in *{grouping_var}*"
            " ===\n"
        )
        print("Missing values are dropped\n")

        # Describe the samples
        print(df_descriptive)
        print("\n")

        # Print results Levenes test
        print("Levenes test for homogeneity of variances (H0 = homogeneity):")
        print(f"- W = {levene:.2f}")
        print(f"- p-value = {levene_p_value:.3f}")

        if levene_p_value > 0.05:
            # Equal variances:
            print("- Equal variances detected \n")
        else:
            print(
                "- Unequal variances detected by Levenes test, so ANOVA results"
                " might be untrustworthy"
            )

        # Print results ANOVA
        print("Outcome ANOVA: ")
        print(f"- F-value = {f_value:.2f}")
        print(f"- df_effect = {df_effect}")
        print(f"- df_error = {df_error}")
        print(f"- p-value = {p_value:.3f}")

        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")
        print("\n")

    distplot = None
    boxplot = None

    if plot:
        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(ax=ax, x=grouping_var, y=feature, data=data)
        _ = sns.swarmplot(ax=ax, x=grouping_var, y=feature, data=data, color=".25", alpha=0.50, size=2)
        _ = ax.set_title(f"Boxplot {feature} across {grouping_var}")
        plt.xticks(rotation=90)

    dict_result = {
        "test-type": "one way ANOVA",
        "feature": feature,
        "group-var": grouping_var,
        "f-value": round(f_value, 3),
        "eta-squared": round(eta_squared, 3),
        "df-effect": int(df_effect),
        "df-error": int(df_error),
        "p-value": round(p_value, 3),
        "stat-sign": (p_value < 0.05),
        "variance": variance_outcome,
        "results": trust_results,
    }

    df_result = pd.DataFrame(data=dict_result, index=[0])

    return df_result, df_descriptive, distplot, boxplot

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
        >>> # import classification_report
        >>> y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1])
        >>> y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        >>> classification_report(y_true, y_pred)

    """

    # Retrieve Confusion Matrix
    df_outcome = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).assign(
        **{
            "TP": lambda d: np.where((d["y_true"] == 1) & (d["y_pred"] == 1), 1, 0),
            "TN": lambda d: np.where((d["y_true"] == 0) & (d["y_pred"] == 0), 1, 0),
            "FP": lambda d: np.where((d["y_true"] == 0) & (d["y_pred"] == 1), 1, 0),
            "FN": lambda d: np.where((d["y_true"] == 1) & (d["y_pred"] == 0), 1, 0),
        }
    )

    # generate True Positive (TP),  True Negative (TN), False Positive (FP), False Negative (FN)
    TP = df_outcome["TP"].sum()
    TN = df_outcome["TN"].sum()
    FP = df_outcome["FP"].sum()
    FN = df_outcome["FN"].sum()
    N = len(df_outcome)

    assert TN + TP + FN + FP == N, "TN + TP + FP + FN does not match the total amount of entries"

    dict_clf = {
        "True Positive (TP)": [
            TP,
            (
                "The number of real positive cases in the data that are correctly classified as"
                " positive by the model"
            ),
        ],
        "True Negative (TN)": [
            TN,
            (
                "The number of real negative cases in the data that are correctly classified as"
                " negative by the model"
            ),
        ],
        "False Positive (FP)": [
            FP,
            (
                "The number of real negative cases in the data that are incorrectly classified as"
                " positive by the model"
            ),
        ],
        "False Negative (FN)": [
            FN,
            (
                "The number of real positive cases in the data that are incorrectly classified as"
                " negative by the model"
            ),
        ],
        "True Positive (TP) %": [
            round(TP / N * 100, 2),
            (
                "The percentage of real positive cases in the data that are correctly classified"
                " as positive by the model"
            ),
        ],
        "True Negative (TN) %": [
            round(TN / N * 100, 2),
            (
                "The percentage of real negative cases in the data that are correctly classified"
                " as negative by the model"
            ),
        ],
        "False Positive (FP) %": [
            round(FP / N * 100, 2),
            (
                "The percentage of real negative cases in the data that are incorrectly"
                " classified as positive by the model"
            ),
        ],
        "False Negative (FN) %": [
            round(FN / N * 100, 2),
            (
                "The percentage of real positive cases in the data that are incorrectly"
                " classified as negative by the model"
            ),
        ],
        "Condition Positive (P)": [TP + FN, "The number of real positive cases in the data"],
        "Condition Negative (N)": [TN + FP, "The number of negative positive cases in the data"],
        "Accuracy": [
            (TP + TN) / (TP + TN + FP + FN),
            "Number of cases correct divided by the number of total cases",
        ],
        "Sensitivity": [
            TP / (TP + FN),
            (
                "Sensitivity a.k.a. Recall, True Positive Rate (TPR). If y_true=1 what's the"
                " probability that y_pred=1 ?"
            ),
        ],
        "Specificity": [
            TN / (FP + TN),
            (
                "Specificity a.k.a. True Negative Rate (TNR). If y_true=0 what's the probability"
                " that y_pred=0 ?"
            ),
        ],
        "Precision": [
            TP / (TP + FP),
            (
                "Precision a.k.a. Predicted Positive Value (PPV). If y_pred=1 what's the"
                " probability that y_true=1 ?"
            ),
        ],
        "Negative Predicted Value (NPV)": [
            TN / (TN + FN),
            "If y_pred=0 what's the probability that y_true=0 ?",
        ],
        "Prevalence": [
            (TP + FN) / (TN + TP + FN + FP),
            (
                "Prevalence a.k.a. Prior Probability. If y_true=1 what's the probability that"
                " y_pred=1 ?"
            ),
        ],
        "Matthews Correlation Coefficient (MCC)": [
            (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
            "MCC is a correlation coefficient between the observed and predicted binary",
        ],
    }

    # check consistency of the metrics
    assert dict_clf["Accuracy"][0] == (TP + TN) / N
    assert dict_clf["Precision"][0] == TP / (TP + FP)
    assert round(dict_clf["Accuracy"][0], 6) == round(
        dict_clf["Sensitivity"][0] * dict_clf["Prevalence"][0]
        + dict_clf["Specificity"][0] * (1 - dict_clf["Prevalence"][0]),
        6,
    )

    # Add F1 score.
    dict_clf["f1-score"] = [
        (
            2
            * (dict_clf["Precision"][0] * dict_clf["Sensitivity"][0])
            / (dict_clf["Precision"][0] + dict_clf["Sensitivity"][0])
        ),
        "the harmonic mean of precision and sensitivity",
    ]

    # Add balanced Accuracy
    dict_clf["Balanced Accuracy"] = [
        (dict_clf["Sensitivity"][0] + dict_clf["Specificity"][0]) / 2,
        "the average of sensitivity and specificity",
    ]

    # Convert Dictionary to DataFrame
    df_report = pd.DataFrame.from_dict(
        dict_clf, orient="index", columns=[label, "Description"]
    ).round(2)

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
    encoder = OneHotEncoder(drop="first", sparse=False)
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
