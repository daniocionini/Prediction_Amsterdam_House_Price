from configurations_dataprep import * # import the configuration file created
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


# -------------------------------------- Train and Test CSV
data_train = data_train.dropna() # Remove NAN values

# -------------------------------------- Data Type
data_type_df = pd.DataFrame.from_dict(
    {
        "Price": "continuous", # target feature
        "Address": "nominal",
        "Zip": "nominal",
        "Area": "continuous",
        "Room": "discrete",
        "Lon": "continuous",
        "Lat": "continuous",
    },
    orient="index",
    columns=["data_type"],
)
data_type_df.transpose()

list_nominal = data_type_df.loc[lambda x: x["data_type"] == "nominal"].index 
list_ordinal = data_type_df.loc[lambda x: x["data_type"] == "ordinal"].index 
list_discrete = data_type_df.loc[lambda x: x["data_type"] == "discrete"].index
list_continuous = data_type_df.loc[lambda x: x["data_type"] == "continuous"].index  

# --------------------------------------- Outliers Analysis
df_outlier = {
    'mean': [3.54, 605929.67, 93.97, 4.89, 52.36],
    '3xstd': [4.50, 1456688.11, 157.02, 0.16, 0.07],
    'n outlier (3 x std)': [9, 11, 12, 0, 0],
    '% outlier (3 x std)': ['1.0%', '1.0%', '2.0%', '0.0%', '0.0%']
}

df_outlier = pd.DataFrame(df_outlier, index=['Room', 'Price', 'Area', 'Lon', 'Lat'])