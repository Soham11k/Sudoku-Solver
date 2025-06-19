import numpy as np
from tensorflow.keras.models import load_model

model = load_model("digit_model.h5")
dummy_input = np.zeros((1, 28, 28, 1))  # All black image
prediction = model.predict(dummy_input)
print("Prediction:", np.argmax(prediction), "Confidence:", np.max(prediction))
