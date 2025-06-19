print('Importing code and libraries from other files...')

from webcam import *
import os
import tensorflow as tf
import cv2 as cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

def main():
    try:
        model = tf.keras.models.load_model('models/handwritten_cnn.h5')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    webcam_width, webcam_height = 1920, 1080
    webcam = cv.VideoCapture(0)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, webcam_width)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, webcam_height)

    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    solver = WebcamSudokuSolver(model)

    print('Logs:')
    print("Press any key in the video window to exit.")

    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        output_frame = solver.solve(frame)
        cv.imshow('Webcam Sudoku Solver', output_frame)

        if cv.waitKey(1) >= 0:
            break

    cv.destroyAllWindows()
    webcam.release()

if __name__ == "__main__":
    main()

print('Code is done, so everything works fine!')
