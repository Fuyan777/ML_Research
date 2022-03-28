import pandas as pd
import pprint
import collections


top20 =[
'AU07_r_max', 'max_mouth', 'AU06_r_max', 'AU06_r_min',
'AU04_r_max', 'AU12_r_std', 'AU09_r_max', 'AU10_r_std', 'duration_of_non_speak_other',
'duration_of_non_speak_other', 'std_mouth', 'duration_of_speak_other',
'AU10_r_max', 'AU25_r_max', 'med_mouth', 'max_mouth',
'duration_of_non_speak_other', 'isSpeak_other', 'duration_of_speak_other',
'ave_pose_Tz', 'duration_of_non_speak_other', 'max_mouth', 'AU05_r_max',
'std_mouth', 'max_mouth', 'AU25_r_max',
'AU25_r_max', 'AU17_r_max',
'AU25_r_max', 'duration_of_non_speak_other', 'duration_of_speak_other'
]

c = collections.Counter(top20)


for i in range(len(top20)):
    key = c.most_common()[i][0]
    value = c.most_common()[i][1]
    print(key+","+str(value))
