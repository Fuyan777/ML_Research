import pandas as pd
import pprint
import collections
import opensmile
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyworld as pw

sampling_rate = 16000
signal = np.zeros(sampling_rate)
csv_path = "/Volumes/mac-ssd/face_csv/a-20211201.csv"
voice_wav_path = "/Volumes/mac-ssd/movie/converted_data/g3_20220330_screen_off/h-20220330_screen_off.wav"

# 特徴量算出用のパラメタ
frame_length = 2048 # 特徴量を１つ算出するのに使うサンプル数
hop_length   = 512 # 何サンプルずらして特徴量を算出するかを決める変数
n_mfcc = 20

y, sr = librosa.load(voice_wav_path)
rms   = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
print(rms)

y = y.astype(np.float)
_f0, _time = pw.dio(y, sr)
f0 = pw.stonemask(y, _f0, _time, sr)
plt.plot(f0, linewidth=3, color="green", label="F0 contour")
plt.legend(fontsize=10)
plt.show()

# rms = librosa.feature.rms(y=wave) #音量の計算
# times = librosa.times_like(rms, sr=fs) #時間軸の生成
# plt.plot(times, rms[0]*2**(1/2)) #rms➡振幅に変換

print('Sampling rate (Hz): %d' % sr)
print('Audio length (seconds): %.2f' % (len(y) / sr))

# smile = opensmile.Smile(
#     feature_set='my.conf',
#     feature_level='lld',
# )
# smile.process_signal(
#     signal,
#     sampling_rate
# )
# output_features = smile.process_file(voice_wav_path)
# print(output_features.shape)
# output_features.to_csv("/Users/fuyan/LocalDocs/ml-research/" + "opensmile_output.csv")

# face_feature_data = pd.read_csv(csv_path)
# print(face_feature_data)


# top20 =[
# 'AU07_r_max', 'max_mouth', 'AU06_r_max', 'AU06_r_min',
# 'AU04_r_max', 'AU12_r_std', 'AU09_r_max', 'AU10_r_std', 'duration_of_non_speak_other',
# 'duration_of_non_speak_other', 'std_mouth', 'duration_of_speak_other',
# 'AU10_r_max', 'AU25_r_max', 'med_mouth', 'max_mouth',
# 'duration_of_non_speak_other', 'isSpeak_other', 'duration_of_speak_other',
# 'ave_pose_Tz', 'duration_of_non_speak_other', 'max_mouth', 'AU05_r_max',
# 'std_mouth', 'max_mouth', 'AU25_r_max',
# 'AU25_r_max', 'AU17_r_max',
# 'AU25_r_max', 'duration_of_non_speak_other', 'duration_of_speak_other'
# ]

# c = collections.Counter(top20)


# for i in range(len(top20)):
#     key = c.most_common()[i][0]
#     value = c.most_common()[i][1]
#     print(key+","+str(value))