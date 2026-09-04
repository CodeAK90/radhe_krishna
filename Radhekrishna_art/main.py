import cv2
import numpy as np
import random

IMAGE = "radhe_krishna.jpeg"
OUTPUT = "radhe_krishna_reel.mp4"

W, H = 540, 960       # 9:16
FPS = 30
DURATION = 14

random.seed(10)


def load_image():
    img = cv2.imread(IMAGE)
    if img is None:
        raise FileNotFoundError(
            f"'{IMAGE}' nahi mili. Image ko main.py ke same folder me rakho."
        )
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def extract_lines(img):
    # Black background wali neon image ke liye:
    # brightness se drawing lines detect karo.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: dark background ignore, bright strokes retain
    _, mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

    # Bahut tiny noise hatao, lekin lines ko blur MAT karo
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    strokes = []

    for contour in contours:
        length = cv2.arcLength(contour, False)

        # Tiny particles/noise skip
        if length < 18:
            continue

        # Contour simplify, but preserve details
        epsilon = max(0.7, length * 0.0015)
        approx = cv2.approxPolyDP(contour, epsilon, False)

        if len(approx) < 2:
            continue

        pts = approx.reshape(-1, 2)

        # Very long contours are split into manageable strokes
        for start in range(0, len(pts) - 1, 12):
            segment = pts[start:start + 13]

            if len(segment) >= 2:
                y_bottom = int(np.max(segment[:, 1]))
                strokes.append((y_bottom, segment.copy()))

    # Bottom -> top
    strokes.sort(key=lambda s: s[0], reverse=True)

    return strokes


def get_color(img, x, y):
    x = int(np.clip(x, 0, W - 1))
    y = int(np.clip(y, 0, H - 1))
    b, g, r = img[y, x]

    # Keep original neon color, slightly brighten
    return (
        min(255, int(b * 1.15)),
        min(255, int(g * 1.15)),
        min(255, int(r * 1.15)),
    )


def draw_stroke(canvas, glow, img, pts, progress):
    if len(pts) < 2:
        return

    # Draw only part of the current stroke
    count = max(2, int((len(pts) - 1) * progress) + 1)
    current = pts[:count]

    for i in range(len(current) - 1):
        p1 = tuple(map(int, current[i]))
        p2 = tuple(map(int, current[i + 1]))

        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        color = get_color(img, mx, my)

        # Main sharp line
        cv2.line(
            canvas, p1, p2, color, 1, cv2.LINE_AA
        )

        # VERY subtle glow only (not blur on the actual drawing)
        cv2.line(
            glow, p1, p2, color, 2, cv2.LINE_AA
        )


def main():
    img = load_image()
    strokes = extract_lines(img)

    print("Drawing strokes:", len(strokes))

    if not strokes:
        raise RuntimeError(
            "Image se lines detect nahi hui. Bright neon outline image use karo."
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (W, H))

    if not writer.isOpened():
        raise RuntimeError("VideoWriter start nahi hua.")

    window = "CodeAK - Radha Krishna Sketch"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 450, 800)

    total_frames = FPS * DURATION

    # First 10 sec: actual drawing
    draw_frames = FPS * 10

    # Small star field
    stars = []
    for _ in range(45):
        stars.append([
            random.randint(20, W - 20),
            random.randint(20, H - 20),
            random.randint(1, 2)
        ])

    for frame in range(total_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        glow = np.zeros_like(canvas)

        progress = min(1.0, frame / draw_frames)

        # Smooth timing
        eased = progress * progress * (3 - 2 * progress)

        visible_float = eased * len(strokes)
        visible = int(visible_float)

        # Completed strokes
        for i in range(visible):
            draw_stroke(canvas, glow, img, strokes[i][1], 1.0)

        # Current stroke is drawn progressively
        if visible < len(strokes):
            current_progress = visible_float - visible
            draw_stroke(
                canvas,
                glow,
                img,
                strokes[visible][1],
                current_progress
            )

        # Tiny drawing cursor/brush glow at current stroke end
        if visible < len(strokes):
            pts = strokes[visible][1]
            idx = min(
                len(pts) - 1,
                max(0, int((len(pts) - 1) *
                           (visible_float - visible)))
            )

            x, y = map(int, pts[idx])
            color = get_color(img, x, y)

            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)

            # Only a small glow around the pencil point
            cv2.circle(glow, (x, y), 5, color, 1, cv2.LINE_AA)

        # Subtle glow behind lines
        glow = cv2.GaussianBlur(glow, (0, 0), 1.8)
        canvas = cv2.addWeighted(canvas, 1.0, glow, 0.28, 0)

        # Background stars
        for x, y, size in stars:
            if random.random() < 0.015:
                cv2.circle(
                    canvas, (x, y), size,
                    (80, 70, 110), -1, cv2.LINE_AA
                )

        writer.write(canvas)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

        if frame % FPS == 0:
            print(f"Rendering {frame // FPS}/{DURATION} sec")

    writer.release()
    cv2.destroyAllWindows()

    print("\n================================")
    print("DONE 🔥")
    print("Video:", OUTPUT)
    print("================================")


if __name__ == "__main__":
    main()
