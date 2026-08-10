import cv2
import time

# ── Load video and select ROI once ────────────────────────
video = cv2.VideoCapture("test_video.mp4")
ok, first_frame = video.read()
video.release()

print("Draw a box around the object, then press SPACE or ENTER")
bbox = cv2.selectROI("Select Object", first_frame, False)
cv2.destroyAllWindows()
print(f"ROI selected: {bbox}")


# ── Function to run CSRT or KCF ───────────────────────────
def run_tracker(tracker_name, tracker_fn, bbox):
    video = cv2.VideoCapture("test_video.mp4")
    ok, frame = video.read()

    tracker = tracker_fn()
    tracker.init(frame, bbox)

    fps_list = []
    frame_count = 0
    success_count = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        start = time.time()
        ok, new_bbox = tracker.update(frame)
        end = time.time()

        fps = 1.0 / (end - start + 0.0001)
        fps_list.append(fps)
        frame_count += 1
        if ok:
            success_count += 1

        if ok:
            x, y, w, h = [int(v) for v in new_bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{tracker_name} | FPS: {fps:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"{tracker_name} | LOST",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.imshow(f"{tracker_name} Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    accuracy = (success_count / frame_count * 100) if frame_count else 0
    print(f"\n{tracker_name} Results:")
    print(f"  Average FPS : {avg_fps:.2f}")
    print(f"  Accuracy    : {accuracy:.1f}% frames tracked successfully")
    return avg_fps, accuracy


# ── Function to run Meanshift ──────────────────────────────
def run_meanshift(bbox):
    video = cv2.VideoCapture("test_video.mp4")
    ok, frame = video.read()

    x, y, w, h = [int(v) for v in bbox]
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    track_window = (x, y, w, h)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    fps_list = []
    frame_count = 0
    stuck_count = 0
    prev_center = (x + w//2, y + h//2)
    still_frames = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame_count += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        start = time.time()
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        end = time.time()

        x, y, w, h = track_window
        curr_center = (x + w//2, y + h//2)

        # measure how much box moved this frame
        movement = ((curr_center[0] - prev_center[0])**2 +
                    (curr_center[1] - prev_center[1])**2) ** 0.5

        # if box hasnt moved more than 2 pixels = its stuck
        if movement < 2:
            still_frames += 1
        else:
            still_frames = 0  # reset if it moves

        # if stuck for more than 10 frames in a row = lost
        stuck = still_frames > 10
        if stuck:
            stuck_count += 1

        prev_center = curr_center

        fps = 1.0 / ((end - start) + 0.001)
        fps_list.append(fps)

        # red if stuck, blue if moving
        color = (0, 0, 255) if stuck else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        status = "STUCK / LOST" if stuck else "Tracking"
        cv2.putText(frame, f"Meanshift | {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(frame, "WARNING: Meanshift tracks colour not shape",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 165, 255), 1)

        cv2.imshow("Meanshift Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    accuracy = ((frame_count - stuck_count) / frame_count * 100) if frame_count else 0

    print(f"\nMeanshift Results:")
    print(f"  Average FPS : {avg_fps:.2f}")
    print(f"  Accuracy    : {accuracy:.1f}% (stuck-frame based estimate)")
    print(f"  Stuck frames: {stuck_count} out of {frame_count} total frames")
    print(f"  Note        : Meanshift tracks colour not shape — gets stuck easily")
    return avg_fps, accuracy


# ── Run all 3 trackers ─────────────────────────────────────
print("\n--- Running CSRT ---")
csrt_fps, csrt_acc = run_tracker("CSRT", cv2.legacy.TrackerCSRT_create, bbox)

print("\n--- Running KCF ---")
kcf_fps, kcf_acc = run_tracker("KCF", cv2.legacy.TrackerKCF_create, bbox)

print("\n--- Running Meanshift ---")
ms_fps, ms_acc = run_meanshift(bbox)

# ── Print final comparison table ──────────────────────────
print("\n" + "="*50)
print(f"{'Algorithm':<12} {'Avg FPS':<15} {'Accuracy':<12}")
print("="*50)
print(f"{'CSRT':<12} {csrt_fps:<15.2f} {csrt_acc:<12.1f}%")
print(f"{'KCF':<12} {kcf_fps:<15.2f} {kcf_acc:<12.1f}%")
print(f"{'Meanshift':<12} {ms_fps:<15.2f} {ms_acc:<12.1f}%")
print("="*50)
print("\nNote: Higher FPS = faster. Higher Accuracy = more frames tracked correctly.")
print("CSRT: best accuracy | KCF: best speed | Meanshift: colour-based, unreliable")