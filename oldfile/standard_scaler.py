import numpy as np
import pandas as pd
import csv
import file_path
import common
from sklearn.preprocessing import StandardScaler

# パス設定
date = "20210115"
user = "a"
# speak = "keep-v2"
# speak = "change-v2"
# speak = "non"

# 読み込み
# speak_data = pd.read_csv(
#     file_path.feature_path + "/%s-output/feature_value_%s.csv" % (user, speak)
# )

speak_data = pd.read_csv(
    file_path.feature_path + "/%s-output/feature_value_change_w16.csv" % (user)
)

df = pd.DataFrame(speak_data, columns=common.speak_columns)

stdsc = StandardScaler()

df_std = stdsc.fit_transform(df)

with open(
    file_path.feature_path + "/%s-output/w16_std/feature_value_change_std.csv" % (user),
    "w",
) as f:
    writer = csv.writer(f)
    writer.writerow(common.speak_columns)
    writer.writerows(df_std)
