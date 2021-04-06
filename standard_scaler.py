import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler

# 読み込み
speak_data = pd.read_csv(
    "/Users/fuyan/Documents/siraisi_lab/B4/40_program/csv/.csv",
    encoding="utf-8",
)
df = pd.DataFrame(speak_data, columns=speak_columns)

stdsc = StandardScaler()

df_std = stdsc.fit_transform(df)

with open(
    "/Users/fuyan/Documents/siraisi_lab/B4/40_program/feature/feature_value_std.csv",
    "w",
) as f:
    writer = csv.writer(f)
    writer.writerow(speak_columns)
    writer.writerows(df_std)
