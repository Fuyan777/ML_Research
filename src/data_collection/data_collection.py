import pympi
from resources import resources
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv


class Data_collection:
    def __init__(self):
        print("Data_collection")

    def extraction_speak(self):

        # inaSpeechSegmenter
        # input_file = "/Users/fuyan/Documents/a-20210128.wav"
        # seg = Segmenter(vad_engine='smn', detect_gender=False)
        # segmentation = seg(input_file)
        # print(segmentation)
        # seg2csv(segmentation, './elan_output_csv/myseg.csv')

        # pympi sample
        # elan_file_path = '/Users/fuyan/Documents/ELAN/test.eaf'
        # eaf = pympi.Elan.Eaf(elan_file_path)
        # eaf.add_tier('tier1')
        # eaf.add_annotation("tier1", 1.0, 2.3, value="x")
        # eaf.to_file(elan_file_path.replace('.eaf', '_fto.eaf'))
