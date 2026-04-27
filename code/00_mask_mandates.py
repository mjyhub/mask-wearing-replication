#  Libraries
import pandas as pd

#  Load ans subset data
df = pd.read_csv("../raw_data/OxCGRT_AUS_latest.csv")

col_subsets = ["RegionName", "RegionCode", "Date", "H6M_Facial Coverings"]

df.index = pd.to_datetime(df["Date"], format="%Y%m%d")

df = df.loc[:, col_subsets]

#  Find rolling averages
rolling_days = 14
df_rolling = df.loc[:, ["RegionName", "H6M_Facial Coverings"]].groupby(
    "RegionName").rolling(window=rolling_days).mean()

#  Find first time mandates are consistently put in place
mandate_limit = 3

df_mandates = df_rolling[df_rolling["H6M_Facial Coverings"]
                         >= mandate_limit].groupby("RegionName").head(1)

#  Save data

df_mandates.to_csv("../data/mandate_start_dates.csv")
