import Layer
import numpy as np


class HeadShoulderDetector:
    layers = list(Layer())

    def predict(self, x: np.array()):
        for layer in self.layers:
            if layer.predict(x) > -1:
                continue
            else:
                return -1
        return 1
