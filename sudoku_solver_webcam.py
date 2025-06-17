import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model("digit_model.h5")


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh


def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                best_cnt = approx
                max_area = area
    return best_cnt


def reorder(points):
    points = points.reshape((4, 2))
    new = np.zeros((4, 2), dtype=np.float32)

    add = points.sum(1)
    diff = np.diff(points, axis=1)

    new[0] = points[np.argmin(add)]
    new[2] = points[np.argmax(add)]
    new[1] = points[np.argmin(diff)]
    new[3] = points[np.argmax(diff)]

    return new


def warp_image(img, points):
    reordered = reorder(points)
    pts1 = np.float32(reordered)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (450, 450))
    return img_warp, matrix


def extract_digits(img_warp):
    cells = []
    side = img_warp.shape[0] // 9
    gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
    for y in range(9):
        row = []
        for x in range(9):
            x1, y1 = x * side, y * side
            cell = gray[y1:y1 + side, x1:x1 + side]
            digit = recognize_digit(cell)
            row.append(digit)
        cells.append(row)
    return cells


def recognize_digit(cell):
    cell = cv2.resize(cell, (28, 28))
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh / 255.0
    thresh = np.expand_dims(thresh, axis=-1)
    thresh = np.expand_dims(thresh, axis=0)
    pred = model.predict(thresh)
    return int(np.argmax(pred)) if np.max(pred) > 0.9 else 0


def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num or \
                board[row // 3 * 3 + i // 3][col // 3 * 3 + i % 3] == num:
            return False
    return True


def solve(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve(board):
                            return True
                        board[i][j] = 0
                return False
    return True


def overlay_solution(img, board, solution, warp_mat_inv):
    side = 50
    for y in range(9):
        for x in range(9):
            if board[y][x] == 0:
                val = str(solution[y][x])
                pt = np.array([[[x * side + 25, y * side + 35]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(pt, warp_mat_inv)
                x_out, y_out = int(dst[0][0][0]), int(dst[0][0][1])
                cv2.putText(img, val, (x_out - 10, y_out + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_processed = 0
solved_board = None
original_board = None
warp_matrix = None
inv_matrix = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - last_processed > 2:
        img = frame.copy()
        processed = preprocess(img)
        contour = find_largest_contour(processed)

        if contour is not None:
            warp, warp_matrix = warp_image(img, contour)
            inv_matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [450, 0], [0, 450], [450, 450]]), reorder(contour))
            original_board = extract_digits(warp)
            solved_board = [row[:] for row in original_board]
            solve(solved_board)
        last_processed = time.time()

    if solved_board is not None and original_board is not None and inv_matrix is not None:
        overlay_solution(frame, original_board, solved_board, inv_matrix)

    cv2.imshow("Sudoku Solver", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
