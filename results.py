import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ── Load video and select ROI ──────────────────────────────
video = cv2.VideoCapture("test_video.mp4")
ok, first_frame = video.read()
video.release()

print("Draw a box around the object, then press SPACE or ENTER")
bbox = cv2.selectROI("Select Object", first_frame, False)
cv2.destroyAllWindows()
x, y, w, h = [int(v) for v in bbox]


# ── Run CSRT or KCF ───────────────────────────────────────
def run_tracker(name, tracker_fn, bbox):
    video = cv2.VideoCapture("test_video.mp4")
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
    avg_fps  = sum(fps_list) / len(fps_list) if fps_list else 0
    accuracy = (success_count / frame_count * 100) if frame_count else 0
    print(f"{name}: FPS={avg_fps:.1f}, Accuracy={accuracy:.1f}%")
    return avg_fps, accuracy, frame_count


# ── Run Meanshift ──────────────────────────────────────────
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
        move = ((curr[0]-prev_center[0])**2 +
                (curr[1]-prev_center[1])**2) ** 0.5
        still_frames = still_frames + 1 if move < 2 else 0
        if still_frames > 10:
            stuck_count += 1
        prev_center = curr

    video.release()
    avg_fps  = sum(fps_list) / len(fps_list) if fps_list else 0
    accuracy = ((frame_count - stuck_count) / frame_count * 100) if frame_count else 0
    print(f"Meanshift: FPS={avg_fps:.1f}, Accuracy={accuracy:.1f}%")
    return avg_fps, accuracy, frame_count


# ── Run all 3 ──────────────────────────────────────────────
print("\nRunning CSRT (no window — just measuring)...")
csrt_fps, csrt_acc, total_frames = run_tracker(
    "CSRT", cv2.legacy.TrackerCSRT_create, bbox)

print("Running KCF...")
kcf_fps, kcf_acc, _ = run_tracker(
    "KCF", cv2.legacy.TrackerKCF_create, bbox)

print("Running Meanshift...")
ms_fps, ms_acc, _ = run_meanshift(bbox)

# ── Save results to JSON ───────────────────────────────────
results = {
    "total_frames": total_frames,
    "CSRT":      {"fps": round(csrt_fps, 2), "accuracy": round(csrt_acc, 2)},
    "KCF":       {"fps": round(kcf_fps,  2), "accuracy": round(kcf_acc,  2)},
    "Meanshift": {"fps": round(ms_fps,   2), "accuracy": round(ms_acc,   2)},
}

os.makedirs("results", exist_ok=True)
with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/results.json")

# ── Print table ────────────────────────────────────────────
print("\n" + "="*50)
print(f"{'Algorithm':<12} {'Avg FPS':<15} {'Accuracy':<12}")
print("="*50)
print(f"{'CSRT':<12} {csrt_fps:<15.1f} {csrt_acc:<12.1f}%")
print(f"{'KCF':<12} {kcf_fps:<15.1f} {kcf_acc:<12.1f}%")
print(f"{'Meanshift':<12} {ms_fps:<15.1f} {ms_acc:<12.1f}%")
print("="*50)

# ── Plot FPS comparison bar chart ─────────────────────────
algorithms = ["CSRT", "KCF", "Meanshift"]
fps_values = [csrt_fps, kcf_fps, ms_fps]
acc_values = [csrt_acc, kcf_acc, ms_acc]
colors_fps = ["#2ecc71", "#3498db", "#e74c3c"]
colors_acc = ["#27ae60", "#2980b9", "#c0392b"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e")

# FPS chart
ax1.set_facecolor("#16213e")
bars1 = ax1.bar(algorithms, fps_values, color=colors_fps,
                width=0.5, edgecolor="white", linewidth=0.5)
ax1.set_title("Average FPS (Higher = Faster)",
              color="white", fontsize=13, pad=12)
ax1.set_ylabel("Frames Per Second", color="white")
ax1.tick_params(colors="white")
ax1.spines["bottom"].set_color("#444")
ax1.spines["left"].set_color("#444")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
for bar, val in zip(bars1, fps_values):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(fps_values)*0.02,
             f"{val:.0f}", ha="center", color="white",
             fontweight="bold", fontsize=11)

# Accuracy chart
ax2.set_facecolor("#16213e")
bars2 = ax2.bar(algorithms, acc_values, color=colors_acc,
                width=0.5, edgecolor="white", linewidth=0.5)
ax2.set_title("Tracking Accuracy (Higher = Better)",
              color="white", fontsize=13, pad=12)
ax2.set_ylabel("% Frames Tracked Successfully", color="white")
ax2.set_ylim(0, 115)
ax2.tick_params(colors="white")
ax2.spines["bottom"].set_color("#444")
ax2.spines["left"].set_color("#444")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
for bar, val in zip(bars2, acc_values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             f"{val:.1f}%", ha="center", color="white",
             fontweight="bold", fontsize=11)

plt.suptitle("Object Tracking Algorithm Comparison\nCSRT vs KCF vs Meanshift",
             color="white", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("results/comparison_chart.png",
            dpi=150, bbox_inches="tight",
            facecolor="#1a1a2e")
print("Chart saved to results/comparison_chart.png")
plt.show()