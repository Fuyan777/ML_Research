import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler

speak_columns = [
    # "speak",
    "ave_gaze_x", "std_gaze_x", "min_gaze_x", "max_gaze_x", "median_gaze_x",
    "ave_gaze_y", "std_gaze_y", "min_gaze_y", "max_gaze_y", "median_gaze_y",
    "ave_poze_x", "std_poze_x", "min_poze_x", "max_poze_x", "median_poze_x",
    "ave_poze_y", "std_poze_y", "min_poze_y", "max_poze_y", "median_poze_y",
    "ave_poze_z", "std_poze_z", "min_poze_z", "max_poze_z", "median_poze_z",
    "ave_mouth", "std_mouth", "min_mouth", "max_mouth", "median_mouth",
]

# 読み込み
speak_data = pd.read_csv("/Users/fuyan/Documents/siraisi_lab/B4/40_program/csv/c-output_feature_value_non.csv", encoding="utf-8")
df = pd.DataFrame(speak_data, columns=speak_columns)

stdsc = StandardScaler()

df_std = stdsc.fit_transform(df)

with open('/Users/fuyan/Documents/siraisi_lab/B4/40_program/feature/feature_value_std.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(speak_columns)
    writer.writerows(df_std)
