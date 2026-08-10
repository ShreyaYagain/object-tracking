import cv2
import time
import numpy as np

# ── Pick your tracker here ─────────────────────────────────
# Change this to "KCF" or "Meanshift" to switch
TRACKER_NAME = "Meanshift"

# ── Load video ─────────────────────────────────────────────
video = cv2.VideoCapture("test_video.mp4")
ok, frame = video.read()
if not ok:
    print("ERROR: Could not open video")
    exit()

# ── User selects object ────────────────────────────────────
print("Draw a box around the object, then press SPACE or ENTER")
bbox = cv2.selectROI("Select Object", frame, False)
cv2.destroyAllWindows()

x, y, w, h = [int(v) for v in bbox]

# ── Initialize tracker based on choice ────────────────────
if TRACKER_NAME == "CSRT":
    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(frame, bbox)
    use_meanshift = False

elif TRACKER_NAME == "KCF":
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, bbox)
    use_meanshift = False

elif TRACKER_NAME == "Meanshift":
    use_meanshift = True
    tracker = None

    # Meanshift needs a colour histogram of the selected region
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    track_window = (x, y, w, h)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    prev_center = (x + w//2, y + h//2)
    still_frames = 0

# ── Trail storage ──────────────────────────────────────────
trail_points = []

# ── FPS tracking ───────────────────────────────────────────
fps = 0
frame_count = 0

print(f"Tracking started with {TRACKER_NAME}. Press Q to quit.")

while True:
    ok, frame = video.read()
    if not ok:
        print("Video ended.")
        break

    frame_count += 1
    success = False

    # ── CSRT or KCF update ─────────────────────────────────
    if not use_meanshift:
        t1 = time.time()
        success, bbox = tracker.update(frame)
        t2 = time.time()
        fps = 1.0 / (t2 - t1 + 0.0001)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = x + w//2, y + h//2

    # ── Meanshift update ───────────────────────────────────
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        t1 = time.time()
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        t2 = time.time()
        fps = 1.0 / (t2 - t1 + 0.001)

        x, y, w, h = track_window
        cx, cy = x + w//2, y + h//2

        # check if stuck
        movement = ((cx - prev_center[0])**2 +
                    (cy - prev_center[1])**2) ** 0.5
        if movement < 2:
            still_frames += 1
        else:
            still_frames = 0

        prev_center = (cx, cy)
        success = still_frames <= 10  # treat stuck as lost

    # ── Draw results ───────────────────────────────────────
    if success:
        # Add to trail
        trail_points.append((cx, cy))
        if len(trail_points) > 40:
            trail_points.pop(0)

        # Draw bounding box
        color = (255, 0, 0) if use_meanshift else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw trail
        for i in range(1, len(trail_points)):
            thickness = max(1, int(i / 4))
            brightness = int(255 * i / len(trail_points))
            cv2.line(frame,
                     trail_points[i-1],
                     trail_points[i],
                     (0, brightness, brightness), thickness)

        # Draw center dot
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        status = "Tracking"
        cv2.putText(frame, f"{TRACKER_NAME} | {status}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        if use_meanshift:
            cv2.putText(frame, "Colour-based tracker — may drift",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255), 1)
    else:
        trail_points.clear()
        cv2.putText(frame, f"{TRACKER_NAME} | LOST / STUCK",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    # ── FPS and frame counter ──────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f"\nDone! Tracked {frame_count} frames with {TRACKER_NAME}")
