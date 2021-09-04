from learning_flow import preprocessing


class Model:
    def __init__(self):
        print("Model")

    def learn_random_forest(self):
        print("learn_random_forest")
        pre = preprocessing.Preprocessing()
        pre.extraction_speak_features()
