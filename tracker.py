import cv2

# ── 1. Load video ──────────────────────────────────────────
video = cv2.VideoCapture("test_video.mp4")

if not video.isOpened():
    print("ERROR: Could not open video file")
    exit()

# ── 2. Read first frame ────────────────────────────────────
ok, frame = video.read()
if not ok:
    print("ERROR: Could not read frame")
    exit()

# ── 3. User draws bounding box ─────────────────────────────
print("Draw a box around the object, then press ENTER or SPACE")
bbox = cv2.selectROI("Select Object", frame, False)
cv2.destroyAllWindows()

# ── 4. Create CSRT tracker ─────────────────────────────────
tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, bbox)
print("Tracker initialized!")

# ── 5. Loop through video frames ───────────────────────────
while True:
    ok, frame = video.read()
    if not ok:
        print("Video ended")
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    if ok:
        # Draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("CSRT Tracker", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()