# Object Tracking using OpenCV

### What this project does
Real-time object tracking using three algorithms:
- CSRT: highest accuracy (56 FPS, 100% accuracy)
- KCF: fastest reliable tracker (132 FPS, 100% accuracy)  
- Meanshift: colour-based, unreliable (984 FPS, 41.9% accuracy)

### How to run
1. Install dependencies:
   pip3 install opencv-contrib-python numpy matplotlib

2. Run single tracker:
   python3 final_tracker.py
   (change TRACKER_NAME in line 7 to switch between CSRT/KCF/Meanshift)

3. Run side by side comparison:
   python3 side_by_side.py

4. Generate results chart:
   python3 results.py

### Hardware
Tested on MacBook, deployed on PYNQ-Z1 board.
ARM processor runs Python/OpenCV tracking code.
FPGA fabric handles video I/O pipeline via base overlay.

### Files
- tracker.py       : basic single tracker
- compare.py       : FPS + accuracy comparison
- final_tracker.py : polished tracker with trail visualization
- side_by_side.py  : 2x2 live comparison window
- results.py       : saves results and generates chart