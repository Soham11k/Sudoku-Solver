import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("digit_model.h5")

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh

def find_biggest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    ordered = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(1)
    diff = np.diff(points, axis=1)
    ordered[0] = points[np.argmin(add)]
    ordered[2] = points[np.argmax(add)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered

def warp(img, pts):
    ordered = reorder(pts)
    pts1 = np.float32(ordered)
    pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv = cv2.getPerspectiveTransform(pts2, pts1)
    return cv2.warpPerspective(img, matrix, (450,450)), inv

def recognize_digit(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = thresh.shape
    thresh = thresh[4:h-4, 4:w-4]
    if thresh.shape[0] == 0 or thresh.shape[1] == 0:
        return 0
    thresh = cv2.resize(thresh, (28, 28))
    thresh = thresh.astype("float32") / 255.0
    thresh = np.expand_dims(thresh, axis=(0, -1))
    prediction = model.predict(thresh, verbose=0)
    digit = np.argmax(prediction)
    confidence = prediction[0][digit]
    return digit if confidence > 0.75 else 0

def extract_grid(img_warp):
    cells = []
    side = img_warp.shape[0] // 9
    for y in range(9):
        row = []
        for x in range(9):
            x1, y1 = x * side, y * side
            cell = img_warp[y1:y1+side, x1:x1+side]
            row.append(recognize_digit(cell))
        cells.append(row)
    return cells

def is_valid(board, r, c, num):
    for i in range(9):
        if board[r][i] == num or board[i][c] == num or \
           board[r//3*3+i//3][c//3*3+i%3] == num:
            return False
    return True

def solve(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1,10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve(board): return True
                        board[i][j] = 0
                return False
    return True

def overlay_solution(frame, board, solved, inv_matrix):
    side = 450 // 9
    for y in range(9):
        for x in range(9):
            if board[y][x] == 0:
                val = solved[y][x]
                pt = np.array([[[x * side + 20, y * side + 35]]], dtype="float32")
                pt_out = cv2.perspectiveTransform(pt, inv_matrix)
                x_out, y_out = int(pt_out[0][0][0]), int(pt_out[0][0][1])
                cv2.putText(frame, str(val), (x_out, y_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
frame_count = 0
cached_board = None
solved_board = None
matrix_inv = None

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    display = frame.copy()
    frame_count += 1

    if frame_count % 10 == 0:
        try:
            thresh = preprocess(frame)
            contour = find_biggest_contour(thresh)
            if contour is not None:
                warp_img, matrix_inv = warp(frame, contour)
                board = extract_grid(warp_img)
                board_copy = [row[:] for row in board]
                if solve(board_copy):
                    cached_board = board
                    solved_board = board_copy
        except:
            pass

    if cached_board and solved_board and matrix_inv is not None:
        overlay_solution(display, cached_board, solved_board, matrix_inv)

    cv2.imshow("Real-Time Sudoku Solver", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
