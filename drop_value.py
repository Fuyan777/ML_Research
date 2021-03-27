# -*- coding: utf-8 -*-
import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv(
    "/Users/fuyan/Documents/siraisi_lab/B4/40_program/csv/20201015_ver2.0.csv"
)
# 欠損値処理
drop_index = df.index[df[" face_id"] == 1]
drop_d = df.drop(drop_index)
drop_d.to_csv(
    "/Users/fuyan/Documents/siraisi_lab/B4/40_program/csv/20201015_ver2.0_new.csv"
)
