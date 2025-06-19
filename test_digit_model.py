from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import random

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
img_rows, img_cols = x_test.shape[1], x_test.shape[2]
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype("float32") / 255.0
y_test_cat = np_utils.to_categorical(y_test)

# Load model
model_file_path = input('Enter a .h5 model file path: ').strip()

try:
    model = load_model(model_file_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Predict
predictions = model.predict(x_test, verbose=0)

# Interactive test viewer
while True:
    index = random.randint(0, len(x_test) - 1)
    pred_label = np.argmax(predictions[index])
    actual_label = y_test[index]
    confidence = np.max(predictions[index]) * 100

    print(f"\n🧠 Model prediction: {pred_label} (Confidence: {confidence:.2f}%)")
    print(f"✅ Actual label: {actual_label}")

    plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {pred_label}, Actual: {actual_label}")
    plt.axis('off')
    plt.show()

    cont = input("Press Enter to continue or type 'q' to quit: ").strip().lower()
    if cont == 'q':
        print("🔚 Exiting test viewer.")
        break
