import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("digit_classifier.h5")

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh

def find_largest_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area, best_cnt = 0, None
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
    s = points.sum(1)
    diff = np.diff(points, axis=1)
    new[0] = points[np.argmin(s)]
    new[2] = points[np.argmax(s)]
    new[1] = points[np.argmin(diff)]
    new[3] = points[np.argmax(diff)]
    return new

def warp(img, points):
    ordered = reorder(points)
    pts1 = np.float32(ordered)
    pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (450, 450))
    return warped, matrix

def recognize_digit_cnn(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm = resized.astype("float32") / 255.0
    input_data = norm.reshape(1, 28, 28, 1)
    pred = model.predict(input_data, verbose=0)
    digit = np.argmax(pred)
    return digit if np.max(pred) > 0.8 else 0

def extract_digits(warped):
    grid = []
    side = warped.shape[0] // 9
    for y in range(9):
        row = []
        for x in range(9):
            x1, y1 = x * side, y * side
            cell = warped[y1:y1+side, x1:x1+side]
            digit = recognize_digit_cnn(cell)
            row.append(digit)
        grid.append(row)
    return grid

def is_valid(board, r, c, n):
    for i in range(9):
        if board[r][i] == n or board[i][c] == n or board[r//3*3+i//3][c//3*3+i%3] == n:
            return False
    return True

def solve(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if is_valid(board, r, c, num):
                        board[r][c] = num
                        if solve(board): return True
                        board[r][c] = 0
                return False
    return True

def overlay_solution(img, original, solved, matrix_inv):
    side = 50
    for y in range(9):
        for x in range(9):
            if original[y][x] == 0:
                pt = np.array([[[x*side + 25, y*side + 35]]], dtype='float32')
                dst = cv2.perspectiveTransform(pt, matrix_inv)
                x_out, y_out = int(dst[0][0][0]), int(dst[0][0][1])
                cv2.putText(img, str(solved[y][x]), (x_out-10, y_out+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# Main loop
cap = cv2.VideoCapture(0)
frame_count = 0
sudoku_contour = None
cached_board = None
solved_board = None
matrix_inv = None

while True:
    ret, frame = cap.read()
    if not ret: break
    display = frame.copy()
    frame_count += 1

    if frame_count % 10 == 0:
        thresh = preprocess(frame)
        sudoku_contour = find_largest_contour(thresh)

        if sudoku_contour is not None:
            warped, matrix = warp(frame, sudoku_contour)
            board = extract_digits(warped)
            cached_board = [row[:] for row in board]
            solved_board = [row[:] for row in board]

            if solve(solved_board):
                matrix_inv = cv2.getPerspectiveTransform(
                    np.float32([[0,0],[450,0],[0,450],[450,450]]),
                    reorder(sudoku_contour)
                )

    if cached_board and solved_board and matrix_inv is not None:
        overlay_solution(display, cached_board, solved_board, matrix_inv)

    cv2.imshow("Sudoku Solver", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
