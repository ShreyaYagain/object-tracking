import cv2
import time
import numpy as np

# ── Load video and select ROI once ────────────────────────
video = cv2.VideoCapture("test_video.mp4")
ok, first_frame = video.read()
video.release()

print("Draw a box around the object, then press SPACE or ENTER")
bbox = cv2.selectROI("Select Object", first_frame, False)
cv2.destroyAllWindows()

x, y, w, h = [int(v) for v in bbox]
print(f"ROI selected: {bbox}")

# ── Initialize CSRT ────────────────────────────────────────
csrt = cv2.legacy.TrackerCSRT_create()
csrt.init(first_frame, bbox)

# ── Initialize KCF ─────────────────────────────────────────
kcf = cv2.legacy.TrackerKCF_create()
kcf.init(first_frame, bbox)

# ── Initialize Meanshift ───────────────────────────────────
roi = first_frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
track_window = (x, y, w, h)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
ms_prev_center = (x + w//2, y + h//2)
ms_still_frames = 0

# ── Trail storage ──────────────────────────────────────────
trail_csrt = []
trail_kcf  = []
trail_ms   = []

# ── Stats history for info box ─────────────────────────────
frame_count  = 0
csrt_status  = "Tracking"
kcf_status   = "Tracking"
ms_status    = "Tracking"
fps_csrt     = 0
fps_kcf      = 0
fps_ms       = 0

# ── Panel size ─────────────────────────────────────────────
pw, ph = 620, 380

print("Running all 3 trackers. Press Q to quit.")

video = cv2.VideoCapture("test_video.mp4")
ok, frame = video.read()  # skip first frame

while True:
    ok, frame = video.read()
    if not ok:
        print("Video ended.")
        break

    frame_count += 1

    frame_csrt = frame.copy()
    frame_kcf  = frame.copy()
    frame_ms   = frame.copy()

    # ── CSRT ───────────────────────────────────────────────
    t1 = time.time()
    ok_csrt, bbox_csrt = csrt.update(frame)
    t2 = time.time()
    fps_csrt = 1.0 / (t2 - t1 + 0.0001)

    if ok_csrt:
        csrt_status = "Tracking"
        cx, cy_c, cw, ch = [int(v) for v in bbox_csrt]
        center_csrt = (cx + cw//2, cy_c + ch//2)
        trail_csrt.append(center_csrt)
        if len(trail_csrt) > 40:
            trail_csrt.pop(0)
        cv2.rectangle(frame_csrt, (cx, cy_c),
                      (cx+cw, cy_c+ch), (0, 255, 0), 2)
        for i in range(1, len(trail_csrt)):
            b = int(255 * i / len(trail_csrt))
            cv2.line(frame_csrt, trail_csrt[i-1],
                     trail_csrt[i], (0, b, b), max(1, i//8))
        cv2.circle(frame_csrt, center_csrt, 4, (0, 255, 255), -1)
        cv2.putText(frame_csrt, f"CSRT | FPS: {fps_csrt:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame_csrt, "Most Accurate",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
    else:
        csrt_status = "LOST"
        trail_csrt.clear()
        cv2.putText(frame_csrt, "CSRT | LOST",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    # ── KCF ────────────────────────────────────────────────
    t1 = time.time()
    ok_kcf, bbox_kcf = kcf.update(frame)
    t2 = time.time()
    fps_kcf = 1.0 / (t2 - t1 + 0.0001)

    if ok_kcf:
        kcf_status = "Tracking"
        kx, ky, kw, kh = [int(v) for v in bbox_kcf]
        center_kcf = (kx + kw//2, ky + kh//2)
        trail_kcf.append(center_kcf)
        if len(trail_kcf) > 40:
            trail_kcf.pop(0)
        cv2.rectangle(frame_kcf, (kx, ky),
                      (kx+kw, ky+kh), (0, 255, 0), 2)
        for i in range(1, len(trail_kcf)):
            b = int(255 * i / len(trail_kcf))
            cv2.line(frame_kcf, trail_kcf[i-1],
                     trail_kcf[i], (0, b, b), max(1, i//8))
        cv2.circle(frame_kcf, center_kcf, 4, (0, 255, 255), -1)
        cv2.putText(frame_kcf, f"KCF  | FPS: {fps_kcf:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame_kcf, "Fastest Speed",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
    else:
        kcf_status = "LOST"
        trail_kcf.clear()
        cv2.putText(frame_kcf, "KCF | LOST",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    # ── Meanshift ──────────────────────────────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    t1 = time.time()
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    t2 = time.time()
    fps_ms = 1.0 / (t2 - t1 + 0.001)

    mx, my, mw, mh = track_window
    center_ms = (mx + mw//2, my + mh//2)
    movement = ((center_ms[0] - ms_prev_center[0])**2 +
                (center_ms[1] - ms_prev_center[1])**2) ** 0.5
    if movement < 2:
        ms_still_frames += 1
    else:
        ms_still_frames = 0
    ms_prev_center = center_ms
    ms_stuck = ms_still_frames > 10

    if not ms_stuck:
        ms_status = "Tracking"
        trail_ms.append(center_ms)
        if len(trail_ms) > 40:
            trail_ms.pop(0)
        cv2.rectangle(frame_ms, (mx, my),
                      (mx+mw, my+mh), (255, 0, 0), 2)
        for i in range(1, len(trail_ms)):
            b = int(255 * i / len(trail_ms))
            cv2.line(frame_ms, trail_ms[i-1],
                     trail_ms[i], (0, b, b), max(1, i//8))
        cv2.circle(frame_ms, center_ms, 4, (0, 255, 255), -1)
        cv2.putText(frame_ms, f"Meanshift | FPS: {fps_ms:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)
        cv2.putText(frame_ms, "Colour-based tracker",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 165, 255), 1)
    else:
        ms_status = "STUCK/LOST"
        trail_ms.clear()
        cv2.putText(frame_ms, "Meanshift | STUCK / LOST",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.putText(frame_ms, "Colour-based -- lost the object",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 165, 255), 1)

    # ── Info box (4th panel) ───────────────────────────────
    info = np.zeros((ph, pw, 3), dtype=np.uint8)
    info[:] = (25, 25, 25)

    # each row block is ph/4 tall
    block = ph // 4

    # ── Title ──────────────────────────────────────────────
    cv2.putText(info, "Live Stats",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    cv2.line(info, (20, block - 5), (pw - 20, block - 5),
             (70, 70, 70), 1)

    # ── CSRT block ─────────────────────────────────────────
    csrt_col = (0, 255, 0) if csrt_status == "Tracking" else (0, 0, 255)
    by = block  # start y of this block
    cv2.putText(info, "CSRT",
                (20, by + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)
    cv2.putText(info, "Highest accuracy",
                (20, by + 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (130, 130, 130), 1)
    cv2.putText(info, f"FPS: {fps_csrt:.0f}",
                (20, by + 78), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    cv2.putText(info, csrt_status,
                (pw // 2, by + 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, csrt_col, 2)
    cv2.line(info, (20, by + block - 5), (pw - 20, by + block - 5),
             (60, 60, 60), 1)

    # ── KCF block ──────────────────────────────────────────
    kcf_col = (0, 255, 0) if kcf_status == "Tracking" else (0, 0, 255)
    by = block * 2
    cv2.putText(info, "KCF",
                (20, by + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)
    cv2.putText(info, "Fastest speed",
                (20, by + 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (130, 130, 130), 1)
    cv2.putText(info, f"FPS: {fps_kcf:.0f}",
                (20, by + 78), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    cv2.putText(info, kcf_status,
                (pw // 2, by + 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, kcf_col, 2)
    cv2.line(info, (20, by + block - 5), (pw - 20, by + block - 5),
             (60, 60, 60), 1)

    # ── Meanshift block ────────────────────────────────────
    ms_col = (0, 255, 0) if ms_status == "Tracking" else (0, 0, 255)
    by = block * 3
    cv2.putText(info, "Meanshift",
                (20, by + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)
    cv2.putText(info, "Colour-based tracker",
                (20, by + 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (130, 130, 130), 1)
    cv2.putText(info, f"FPS: {fps_ms:.0f}",
                (20, by + 78), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    cv2.putText(info, ms_status,
                (pw // 2, by + 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, ms_col, 2)

    # ── Frame counter ──────────────────────────────────────
    cv2.putText(info, f"Frame: {frame_count}",
                (20, ph - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80, 80, 80), 1)

    # ── Resize all panels to same size ─────────────────────
    f1 = cv2.resize(frame_csrt, (pw, ph))
    f2 = cv2.resize(frame_kcf,  (pw, ph))
    f3 = cv2.resize(frame_ms,   (pw, ph))
    f4 = info  # already pw x ph

    # ── Dividers ───────────────────────────────────────────
    v_div = np.full((ph, 4, 3), 60, dtype=np.uint8)
    h_div = np.full((4, pw*2+4, 3), 60, dtype=np.uint8)

    # ── Build 2x2 grid ─────────────────────────────────────
    top_row    = np.hstack([f1, v_div, f2])
    bottom_row = np.hstack([f3, v_div, f4])

    # ── Title bar ──────────────────────────────────────────
    title = np.zeros((45, pw*2+4, 3), dtype=np.uint8)
    cv2.putText(title,
                "Object Tracking Comparison: CSRT vs KCF vs Meanshift",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)

    final = np.vstack([title, top_row, h_div, bottom_row])

    cv2.imshow("Tracker Comparison", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f"\nDone! Processed {frame_count} frames.")