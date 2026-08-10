# Object Tracking in a Video on PYNQ-Z2

This project compares three tracking algorithms (CSRT, KCF, and Meanshift) implemented using OpenCV and deployed on both a laptop and PYNQ-Z2 hardware platform.

## Methodology

- **Video Input & ROI Selection**: Pre-recorded video files with manual bounding box selection using `cv2.selectROI()`
- **Tracking Algorithms Evaluated**:
  - **CSRT**: Channel and Spatial Reliability Tracking - uses discriminative object features
  - **KCF**: Kernel Correlation Filter - exploits redundancy for faster computation
  - **Meanshift**: Shifts search window toward highest color density
- **Performance Metrics**: FPS (Frames Per Second) and Tracking Accuracy (% frames with correct bounding box)
- **Hardware Platforms**: Laptop (baseline) and PYNQ-Z2 (ARM Cortex-A9 at 650MHz)

---

## Results Summary

### 1. Running Man Video (Slow, Predictable Motion)

#### Desktop Performance
| Algorithm | FPS | Accuracy |
|-----------|-----|----------|
| CSRT      | 63  | 100.0%   |
| KCF       | 195 | 100.0%   |
| Meanshift | 979 | 58.1%    |

#### PYNQ-Z2 Performance
| Algorithm | FPS | Accuracy |
|-----------|-----|----------|
| CSRT      | 1.0 | 100.0%   |
| KCF       | 3.6 | 100.0%   |
| Meanshift | 505.4 | 61.7% |

**Key Findings**:
- ✅ CSRT and KCF both achieved perfect 100% accuracy for slow, predictable motion
- ✅ KCF was significantly faster (121 FPS improvement over CSRT)
- ❌ Meanshift struggled with only 41.9% accuracy because the person's clothing color appeared in background regions, causing tracker drift

---

### 2. Flying Bird Video (High-Speed Motion)

#### Desktop Performance
| Algorithm | FPS  | Accuracy |
|-----------|------|----------|
| CSRT      | 126  | 100.0%   |
| KCF       | 1263 | 0.8%     |
| Meanshift | 997  | 39.5%    |

#### PYNQ-Z2 Performance
| Algorithm | FPS   | Accuracy |
|-----------|-------|----------|
| CSRT      | 2.5   | 100.0%   |
| KCF       | 23.2  | 7.0%     |
| Meanshift | 849.6 | 23.3%    |

**Key Findings**:
- ✅ CSRT maintained 100% accuracy, demonstrating robustness to small, fast objects
- ❌ KCF achieved 1755 FPS but only 0.8% accuracy - lost the bird immediately and tracked empty sky
- ❌ Meanshift tracked correctly only 32.6% of the time due to bird's color similarity to sky/background

---

### 3. Running Cheetah Video (Fast, Complex Motion)

#### Desktop Performance
| Algorithm | FPS  | Accuracy |
|-----------|------|----------|
| CSRT      | 126  | 100.0%   |
| KCF       | 1263 | 0.8%     |
| Meanshift | 997  | 39.5%    |

#### PYNQ-Z2 Performance
| Algorithm | FPS   | Accuracy |
|-----------|-------|----------|
| CSRT      | 0.4   | 100.0%   |
| KCF       | 0.6   | 69.4%    |
| Meanshift | 85.3  | 100.0%   |

**Key Findings**:
- ✅ CSRT maintained 100% accuracy across fast motion
- ❌ KCF dropped to 41.7% accuracy - fixed search window cannot keep up with large frame-to-frame displacements
- ⚠️ Meanshift reported 100% accuracy but visually drifted to background regions with similar yellow-brown color and remained stuck (color-only tracking limitation)

---

## Performance Analysis

### Algorithm Comparison

| Aspect | CSRT | KCF | Meanshift |
|--------|------|-----|-----------|
| **Slow Motion** | ✅ 100% | ✅ 100% | ❌ 59% |
| **Fast Motion** | ✅ 100% | ❌ 0.8% | ❌ 40% |
| **Small Objects** | ✅ 100% | ❌ 0.8% | ❌ 40% |
| **Speed (Desktop)** | Moderate | Fast | Very Fast |
| **Robustness** | Excellent | Poor | Poor |
| **Reliability** | Excellent | Fair | Poor |

### Hardware Deployment Impact

**PYNQ-Z2 Slowdown Compared to Laptop**:
- **CSRT**: ~52x slowdown
  - Desktop: ~63-126 FPS
  - PYNQ-Z2: 0.4-2.5 FPS
  
- **KCF**: ~210x slowdown
  - Desktop: ~1263 FPS
  - PYNQ-Z2: 3.6-23.2 FPS
  
- **Meanshift**: ~10x slowdown
  - Desktop: ~979-997 FPS
  - PYNQ-Z2: 85.3-849.6 FPS

**Observations**:
- Correlation filter-based algorithms (CSRT, KCF) incur significant computational costs on embedded processors
- Meanshift is most suitable for resource-constrained embedded systems
- FPGA fabric successfully handled video I/O through base overlay, confirming software pipeline portability

---

## Conclusions

### Algorithm Selection Recommendations

1. **For Reliable Real-World Tracking**: **CSRT**
   - Maintains 100% accuracy across all scenarios
   - Handles slow, fast, small, and large objects equally well
   - Recommended for surveillance, robotics, and autonomous systems

2. **For Speed-Critical, Controlled Scenarios**: **KCF**
   - Best performance on slow, predictable objects
   - Good for applications with limited motion variability
   - Not suitable for high-speed or complex motion

3. **For Embedded/Resource-Constrained Systems**: **Meanshift**
   - Fastest on PYNQ-Z2 (85-849 FPS)
   - Trade-off: Lower accuracy and unreliable tracking
   - Only viable when speed is critical and accuracy is secondary

### Key Takeaways

✅ **CSRT is the most robust choice** for practical applications requiring reliable tracking

⚡ **Algorithm selection must be driven by application requirements** - there is no one-size-fits-all solution

🔧 **Hardware constraints significantly impact performance** - embedded processors impose 10-210x slowdowns depending on algorithm complexity

🎯 **Motion characteristics matter** - fast-moving objects require more sophisticated feature extraction than color-only approaches

---

## Tools & Technologies

**Software**:
- Python 3
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

**Hardware**:
- PYNQ-Z2 Board (ARM Cortex-A9 @ 650MHz + Xilinx FPGA)
- Base overlay for video I/O

