# -*- coding: utf-8 -*-
import os
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def tSNE(
    data: pd.DataFrame,
    n_components: int = 2,
    normalize: bool = True,
    hue: Union[str, None] = None,
    tag: Union[str, None] = None,
    label_fontsize: int = 14,
    figsize: tuple = (11.7, 8.27),
    generate_plot: bool = False,
    colors_3D: LinearSegmentedColormap = ["#e58038", "#F7F1F0", "#306fbe"],
    **kwargs,
):
    """Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) Analysis.

    More info : https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe which contains some numerical feature
    n_components : int, optional (default: 2)
        Dimension of the embedded space (2D or 3D).
    normalize : bool, optional (default: True)
        Normalize data prior tSNE.
    hue: string, optional
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric, although color mapping will behave
        differently in latter case.
    tag: string, optional
        Tag each point with the value relative to the corresponding column (only 2D currently)
    label_fontsize: int, optional (default: 14)
        Font size for the `tag`
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    generate_plot  : bool, optional (default: False)
        Save figure in .jpg format and store in working directory
    kwargs: key, value pairings
        Additional keyword arguments relative to tSNE() function. Additional info:
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Graph with clusters in embedded space

    Examples
    --------
    >>> import seaborn as sns
    >>> from r4_helpers import tSNE
    >>> data = sns.load_dataset("mpg")
    >>> fig = tSNE(data, n_components=2, hue='origin', tag='name', generate_plot = False)

    """
    # check hue input
    if hue is None:
        warnings.warn("A hue has not been set, hence it shall not be shown in the plot.")

    if hue is not None:
        if not isinstance(hue, str):
            raise ValueError("hue input needs to be a string")

        if hue not in data.columns:
            raise ValueError(f"hue='{hue}' is not contained in dataframe")

    # check tag input
    if tag is not None:
        if not isinstance(tag, str):
            raise ValueError("tag needs to be str type")
        if tag not in data.columns:
            raise ValueError(f"tag='{tag}' is not contained in dataframe")
    else:
        pass

    # t-SNE takes into account only numerical feature
    data_num = data.select_dtypes(include="number")
    if hue in data_num.columns:
        data_num = data_num.drop(hue, axis=1)
    data_obj = data.select_dtypes(exclude="number")
    if hue and hue not in data_obj.columns:
        # Add hue column if it is numerical
        data_obj = pd.concat([data_obj, data[[hue]]], axis=1)

    # remove any row with NaNs and normalize data with z-score
    data_num = data_num.dropna(axis="index", how="any")

    if normalize:
        # get z-score to treat different dimensions with equal importance
        data_num = StandardScaler().fit_transform(data_num)

    # Apply t-SNE to normalized_movements: normalized_data
    tsne_features = TSNE(n_components=n_components, **kwargs).fit_transform(data_num)

    # show t-SNE cluster
    if n_components == 2:
        # combine tsne feature with categorical and/or object variables
        df_tsne = pd.DataFrame(data=tsne_features, columns=["t-SNE (x)", "t-SNE (y)"])
        df_tsne = pd.concat([df_tsne, data_obj], axis=1)
        # plot 2D
        fig = plt.figure(figsize=figsize)
        _ = plt.title("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        _ = sns.scatterplot(
            x="t-SNE (x)",
            y="t-SNE (y)",
            hue=hue,
            legend="full",
            data=df_tsne,
            alpha=0.8,
        )

    elif n_components == 3:
        # combine tsne feature with categorical and/or object variables
        df_tsne = pd.DataFrame(
            data=tsne_features, columns=["t-SNE (x)", "t-SNE (y)", "t-SNE (z)"]
        )
        df_tsne = pd.concat([df_tsne, data_obj], axis=1)
        # plot 3D
        fig = plt.figure(figsize=figsize)
        _ = plt.title("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        ax = fig.add_subplot(111, projection="3d")
        if hue:
            i = ax.scatter(
                df_tsne["t-SNE (x)"],
                df_tsne["t-SNE (y)"],
                df_tsne["t-SNE (z)"],
                c=df_tsne[hue],
                cmap=colors_3D,
                s=60,
                alpha=0.8,
            )
            fig.colorbar(i)
        else:
            ax.scatter(
                df_tsne["t-SNE (x)"],
                df_tsne["t-SNE (y)"],
                df_tsne["t-SNE (z)"],
                s=60,
                alpha=0.8,
            )
        _ = ax.view_init(30, 185)

    else:
        raise ValueError("n_components can be either 2 or 3")

    # tag each point
    if tag is not None and n_components == 2:
        for x, y, tag in zip(df_tsne["t-SNE (x)"], df_tsne["t-SNE (y)"], df_tsne[tag]):
            plt.annotate(tag, (x, y), fontsize=label_fontsize, alpha=0.75)

    # save in working directory
    if generate_plot:
        _ = fig.savefig("t-SNE.png", dpi=600, bbox_inches="tight")
        print(f".png has been saved in {os.getcwd()}")

    return df_tsne, fig


def PCA_analysis(data: pd.DataFrame, n_components: int = 2, plot: bool = True, **kwargs):
    """Perform Principal Component Analysis (PCA).

    Analysis is done only using numerical features.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe which contains some numerical feature
    n_components : int (default: 2)
        Number of principal components to obtain with PCA
    hue: string, optional
        Grouping variable that will produce points with different colors
    plot  : bool, optional (default: True)
        plot PCA analysis
    kwargs: key, value pairings
        Additional keyword arguments for plot setting matplolib style. Available setting:
            figsize : tuple (default: (20, 10))
            textsize : int (default: 15) adjust size for xticks label, title and legend

    Returns
    -------
    pca_summary: pandas.DataFrame
        Nrovide info relative to the explained variance (and cumulative) for each pca component

    pca_features: pandas.DataFrame
        Number of PCA component defined by n_components

    summary_input: pandas.DataFrame
        Info about the number of features feed to PCA with Nan values, outliers

    fig: matplotlib.figure.Figure
        plot object if plot==True. Left: Plot with the first 2 PCA components, Right: Scree plot

    Examples
    --------
    >>> import seaborn as sns
    >>> from r4_helpers import PCA_analysis
    >>> data = sns.load_dataset(name="iris")
    >>> (pca_summary,
    ...  pca_features,
    ...  summary_input,
    ...  pca_components,
    ...  fig) = PCA_analysis(data, n_components=2, plot=True)

    """
    if not isinstance(n_components, int):
        raise ValueError("n_components must be an integer.")

    data = data.copy()

    # PCA takes into account only numerical feature
    data_num = data.select_dtypes(include="number")

    # store info relative to input PCA
    summary_input = (
        pd.DataFrame(
            data={
                "PCA_inputs": data_num.columns,
                "nsample_input": data_num.count().values,
                "Nan_values": data_num.isna().sum(),
            }
        )
        .reset_index(drop=True)
        .round(2)
    )

    # remove any row with NaNs and add info
    data_num = data_num.dropna(axis="index", how="any")
    summary_input["nsample_pca"] = data_num.count().values

    # standardize features and info
    data_num_normalized = StandardScaler().fit_transform(data_num)
    temp = pd.DataFrame(data=data_num_normalized)
    summary_input["noutliers"] = (np.abs(temp) > 3).sum()

    # get all PCA components
    pca_analysis = PCA(n_components=None).fit(data_num_normalized)
    pca_summary = pd.DataFrame(
        data={
            "pca_component": np.array(range(pca_analysis.n_components_)) + 1,
            "explained_variance_pct": 100 * pca_analysis.explained_variance_ratio_,
            "cumulative_explained_variance_pct": 100
            * np.cumsum(pca_analysis.explained_variance_ratio_),
        }
    ).round(2)

    # retrieve k_components eigenvector and z is the compressed features via PCA.
    pca_components = pd.DataFrame(
        data=pca_analysis.components_,
        columns=data_num.columns,
        index=["comp" + str(x) for x in np.array(range(pca_analysis.n_components_)) + 1],
    )

    if plot:
        # set figure dimension based on kwargs
        if "figsize" in kwargs.keys():
            fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=kwargs["figsize"])
        else:
            fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

        # Plot explained variance plot (elbow plot)
        _ = ax1.bar(
            pca_summary["pca_component"],
            pca_summary["explained_variance_pct"],
            alpha=0.5,
            align="center",
            label="individual explained variance",
            width=1.0,
            edgecolor="grey",
            linewidth=2,
        )
        _ = ax1.step(
            x=pca_summary["pca_component"],
            y=pca_summary["cumulative_explained_variance_pct"],
            where="mid",
            linewidth=3,
            label="cumulative explained variance",
        )
        _ = ax1.hlines(
            y=90,
            xmin=0.5,
            xmax=pca_summary["pca_component"].max() + 0.5,
            linestyles="dashdot",
            colors="grey",
            label="90% threshold",
        )

        _ = ax1.set(
            xlim=[0.5, pca_summary["pca_component"].max() + 0.5],
            ylim=[0, 105],
            title="Scree plot",
            xlabel="PCA Components",
            ylabel="Explained Variance (%)",
            xticks=pca_summary["pca_component"],
        )
        _ = ax1.legend(loc="best")
        # _ = plt.show()
    else:
        fig = None

    # Compress features using PCA. Note: z = pca_feature
    pca_features = pd.DataFrame(
        data=PCA(n_components=n_components).fit_transform(data_num_normalized),
        columns=["PCA_" + str(i) for i in range(1, n_components + 1)],
    )

    return (pca_summary, pca_features, summary_input, pca_components, fig)
