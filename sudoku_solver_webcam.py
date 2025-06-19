import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ====== CONFIGURATION ======
MODEL_PATH = "digit_model.h5"
GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9

# ====== LOAD MODEL ======
model = load_model(MODEL_PATH)

def preprocess_cell(cell):
    cell = cv2.resize(cell, (32, 32))
    cell = cell / 255.0
    cell = cell.reshape(1, 32, 32, 1)
    return cell

def find_biggest_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    biggest = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

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

def is_valid(board, y, x, n):
    for i in range(9):
        if board[y][i] == n or board[i][x] == n:
            return False
    x0 = (x // 3) * 3
    y0 = (y // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[y0 + i][x0 + j] == n:
                return False
    return True

def solve(board):
    for y in range(9):
        for x in range(9):
            if board[y][x] == 0:
                for n in range(1, 10):
                    if is_valid(board, y, x, n):
                        board[y][x] = n
                        if solve(board):
                            return True
                        board[y][x] = 0
                return False
    return True

def draw_solution(img, original, solved):
    for y in range(9):
        for x in range(9):
            if original[y][x] == 0:
                cv2.putText(
                    img,
                    str(solved[y][x]),
                    (x * CELL_SIZE + 10, (y + 1) * CELL_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    biggest = find_biggest_contour(thresh)
    if biggest is not None:
        reordered = reorder(biggest)
        pts1 = np.float32(reordered)
        pts2 = np.float32([[0, 0], [GRID_SIZE, 0], [GRID_SIZE, GRID_SIZE], [0, GRID_SIZE]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(img, matrix, (GRID_SIZE, GRID_SIZE))
        warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

        board = []
        original = []

        for y in range(9):
            row = []
            orig_row = []
            for x in range(9):
                cell = warp_gray[y*CELL_SIZE:(y+1)*CELL_SIZE, x*CELL_SIZE:(x+1)*CELL_SIZE]
                cell_blur = cv2.GaussianBlur(cell, (3,3), 0)
                _, cell_thresh = cv2.threshold(cell_blur, 128, 255, cv2.THRESH_BINARY_INV)
                if cv2.countNonZero(cell_thresh) < 50:
                    row.append(0)
                    orig_row.append(0)
                else:
                    pred = model.predict(preprocess_cell(cell_thresh))
                    class_id = np.argmax(pred)
                    confidence = np.max(pred)
                    if confidence > 0.8:
                        row.append(class_id)
                        orig_row.append(class_id)
                    else:
                        row.append(0)
                        orig_row.append(0)
            board.append(row)
            original.append(orig_row)

        solved_board = [row[:] for row in board]
        if solve(solved_board):
            warp_color = warp.copy()
            solution_img = draw_solution(warp_color, original, solved_board)
            inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            unwarp = cv2.warpPerspective(solution_img, inv_matrix, (frame.shape[1], frame.shape[0]))

            mask = cv2.cvtColor(unwarp, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            solution_fg = cv2.bitwise_and(unwarp, unwarp, mask=mask)
            final = cv2.add(frame_bg, solution_fg)
            cv2.imshow("Real Time Sudoku Solver", final)
        else:
            cv2.imshow("Real Time Sudoku Solver", frame)
    else:
        cv2.imshow("Real Time Sudoku Solver", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
