from learning_flow import model


class Output:
    def __init__(self):
        print("output")

    def out_predict_speak(self):
        print("out_predict_speak")
        m = model.Model()
        m.set_machine_learning_model()
