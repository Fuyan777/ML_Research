import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import pandas as pd
import csv
import warnings
from pprint import pprint
import file_path
import Common

# パス設定
date = "20201015"
user = "c"
# user = 'c'
# speak = 'keep'
# speak = 'change'
speak = "non"

# 特徴量の抽出
# turn-keepかtakingでpathを変える必要あり
time_speak_path = file_path.csv_path + "/speak-20201015/face_%s_speak_%s.csv" % (
    user,
    speak,
)

# 発話・非発話区間・発話タイミングのデータ抽出処理（秒単位）
df_sp = pd.read_csv(time_speak_path)
start_time = df_sp["start"]
end_time = df_sp["end"]