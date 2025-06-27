class MaskDetectionBase:
    def load_model(self, model_path):
        raise NotImplementedError("load_model method must be implemented by subclass")

    def predict(self, processed_input):
        raise NotImplementedError("predict method must be implemented by subclass")