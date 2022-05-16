from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv


# group
# "g1_20210128", "g2_20220106", "g3_20220105", "g4_20220212"
group_path = "g3_20220330_screen_off"

user = ["g", "h", "i"]
file_path = "20220330_screen_off"


def extraction_speak(
    group_path,
    user,
    file_path
):
    # inaSpeechSegmenter
    input_file = "/Volumes/mac-ssd/movie/converted_data/%s/%s-%s.wav" % (
        group_path, user, file_path)
    seg = Segmenter(vad_engine='smn', detect_gender=False)
    segmentation = seg(input_file)
    print(segmentation)
    seg2csv(segmentation, './elan_output_csv/%s-%s.csv' % (user, file_path))


for _user in user:
    extraction_speak(
        group_path,
        _user,
        file_path
    )
