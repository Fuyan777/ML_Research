import pympi
from resources import resources
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv


class Data_collection:
    def __init__(self):
        print("Data_collection")

    def extraction_speak(self):
        file_group = "g2_20220106"
        file_path = "f-20220106"

        # inaSpeechSegmenter
        input_file = "/Volumes/mac-ssd/movie/converted_data/%s/%s.wav" % (
            file_group, file_path)
        seg = Segmenter(vad_engine='smn', detect_gender=False)
        segmentation = seg(input_file)
        print(segmentation)
        seg2csv(segmentation, './elan_output_csv/%s.csv' % (file_path))

        # for segment in segmentation:
        #     segment_label = segment[0]

        #     if (segment_label == 'speech'):  # 音声区間

        #         # 区間の開始時刻の単位を秒からミリ秒に変換
        #         start_time = segment[1] * 1000
        #         end_time = segment[2] * 1000

        # AudioSegment

        # pympi sample
        # elan_file_path = '/Users/fuyan/Documents/ELAN/test.eaf'
        # eaf = pympi.Elan.Eaf(elan_file_path)
        # eaf.add_tier('tier1')
        # eaf.add_annotation("tier1", 1.0, 2.3, value="x")
        # eaf.to_file(elan_file_path.replace('.eaf', '_fto.eaf'))
