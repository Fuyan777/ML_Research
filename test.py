import pandas as pd
import pprint
import collections


top20 = [
    "max_mouth",
    "AU25_r_std",
    "AU25_r_max",
    "std_mouth",
    "min_mouth",
    "AU06_r_max",
    "AU06_r_min",
    "AU07_r_max",
    "AU04_r_max",
    "AU17_r_min",
    "AU09_r_max",
    "AU09_r_min",
    "AU10_r_max",
    "min_pose_Rx",
    "std_pose_Tx",
    "min_gaze_x",
    "min_pose_Ry",
    "ave_pose_Ry",
    "max_pose_Tz",
    "ave_pose_Tz",
    "med_mouth"]

top10 = [
    'AU06_r_max',
    'AU04_r_max', 'AU17_r_min',
    'std_mouth', 'max_mouth',
    'max_mouth', 'AU25_r_max',
    'std_pose_Tx', 'max_mouth',
    'max_pose_Tz', 'ave_pose_Tz',
    'max_mouth',
    'AU25_r_std',
    'max_mouth', 'AU25_r_max'
]

top10_variable = [
    'AU04_r_max', 'AU17_r_min', 'AU09_r_max', 'AU09_r_min',
    'AU04_r_max', 'AU17_r_min', 'AU09_r_max', 'AU09_r_min',
    'max_mouth', 'med_mouth', 'std_mouth',
    'AU10_r_max', 'AU25_r_max', 'AU25_r_std',
    'max_mouth', 'std_pose_Tx', 'min_pose_Ry', 'min_gaze_x',
    'AU05_r_max', 'AU05_r_std', 'AU17_r_min',
    'max_mouth', 'med_mouth',
    'AU17_r_max', 'AU25_r_std',
    'max_mouth', 'ave_mouth', 'med_mouth'
    'AU17_r_max', 'AU25_r_std'
]

c = collections.Counter(top10_variable)


for i in range(len(top10_variable)):
    key = c.most_common()[i][0]
    value = c.most_common()[i][1]
    print(key+","+str(value))
