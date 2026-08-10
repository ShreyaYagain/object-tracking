import cv2
import time

video_path = "/home/xilinx/jupyter_notebooks/test_video.mp4"

def run_tracker(name, tracker_fn, bbox):
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()
    tracker = tracker_fn()
    tracker.init(frame, bbox)
    fps_list, frame_count, success_count = [], 0, 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        t1 = time.time()
        ok, _ = tracker.update(frame)
        t2 = time.time()
        fps_list.append(1.0 / (t2 - t1 + 0.0001))
        frame_count += 1
        if ok:
            success_count += 1
    video.release()
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    accuracy = (success_count / frame_count * 100) if frame_count else 0
    print(f"{name}: FPS={avg_fps:.1f}, Accuracy={accuracy:.1f}%")
    return avg_fps, accuracy

def run_meanshift(bbox):
    video = cv2.VideoCapture(video_path)
    ok, frame = video.read()
    x, y, w, h = [int(v) for v in bbox]
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    track_window = (x, y, w, h)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    prev_center = (x + w//2, y + h//2)
    still_frames = 0
    fps_list, frame_count, stuck_count = [], 0, 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_count += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        t1 = time.time()
        _, track_window = cv2.meanShift(dst, track_window, term_crit)
        t2 = time.time()
        fps_list.append(1.0 / (t2 - t1 + 0.001))
        mx, my, mw, mh = track_window
        curr = (mx + mw//2, my + mh//2)
        move = ((curr[0]-prev_center[0])**2 + (curr[1]-prev_center[1])**2)**0.5
        still_frames = still_frames + 1 if move < 2 else 0
        if still_frames > 10:
            stuck_count += 1
        prev_center = curr
    video.release()
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    accuracy = ((frame_count - stuck_count) / frame_count * 100) if frame_count else 0
    print(f"Meanshift: FPS={avg_fps:.1f}, Accuracy={accuracy:.1f}%")
    return avg_fps, accuracy

# Fixed bbox — same as used on Mac for direct comparison
bbox = (3, 179, 68, 237)

print("="*45)
print("PYNQ-Z2 Hardware Benchmark")
print("ARM Cortex-A9 @ 650MHz")
print("="*45)
print()
csrt_fps, csrt_acc = run_tracker("CSRT", cv2.legacy.TrackerCSRT_create, bbox)
kcf_fps,  kcf_acc  = run_tracker("KCF",  cv2.legacy.TrackerKCF_create,  bbox)
ms_fps,   ms_acc   = run_meanshift(bbox)

print()
print("="*45)
print(f"{'Algorithm':<12} {'FPS':<10} {'Accuracy'}")
print("="*45)
print(f"{'CSRT':<12} {csrt_fps:<10.1f} {csrt_acc:.1f}%")
print(f"{'KCF':<12} {kcf_fps:<10.1f} {kcf_acc:.1f}%")
print(f"{'Meanshift':<12} {ms_fps:<10.1f} {ms_acc:.1f}%")
print("="*45)
print()
print("Mac vs PYNQ Comparison:")
print(f"CSRT:      Mac=52fps  vs  PYNQ={csrt_fps:.1f}fps")
print(f"KCF:       Mac=121fps vs  PYNQ={kcf_fps:.1f}fps")
print(f"Meanshift: Mac=984fps vs  PYNQ={ms_fps:.1f}fps")
print()
print("PYNQ benchmark complete!")