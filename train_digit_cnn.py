from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam  # changed to Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load and visualize
(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0], cmap='gray')
plt.title("Sample training image")
plt.axis('off')
plt.show()

# Preprocessing
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# Compile and print summary
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
early_stop = EarlyStopping(patience=2, restore_best_weights=True)
train = model.fit(
    x_train, y_train,
    batch_size=35,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
model_file_path = input('Enter a file path (with .h5 extension) where the model will be saved: ').strip()
if not model_file_path.endswith('.h5'):
    model_file_path += '.h5'
model.save(model_file_path)

# Load and re-evaluate to confirm
loaded_model = load_model(model_file_path)
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("✅ Loaded model test loss:", score[0])
print("✅ Loaded model accuracy:", score[1])
print("✅ Code is done, so everything works fine!")
