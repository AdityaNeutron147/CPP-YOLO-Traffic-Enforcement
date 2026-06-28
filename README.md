CPP_YOLO_Traffic_Enforcement

**Overview**

This project is a high-performance C++ vision engine engineered for real-time traffic violation detection in demanding urban environments. Rather than relying on a single monolithic network, the perception system decouples the task by utilizing two separately trained, custom YOLOv8 models exported to the optimized ONNX runtime format: one specialized exclusively in crosswalk segmentation, and another dedicated to vehicle and traffic light detection.

The entire compliance engine is written in native C++, which coordinates both models simultaneously and applies sophisticated spatial Intersection over Union (IoU) logic. By calculating the precise geometric overlap between detected vehicle bounding boxes and dynamic crosswalk coordinates in high-speed C++, the engine flags illegal crosswalk occupancy exclusively during active red-light signal phases without the latency overhead of Python.

**Key Achievements & Performance Metrics**

This engine was trained on a dense urban dataset, demonstrating high reliability for critical object detection (e.g., car, trafficLight-Red).

Performance Summary

Peak mAP@0.5: 0.65 <br>
Peak mAP@0.5-0.95: 0.37

Training Convergence: Stable reduction in box, classification, and DFL losses over 50 epochs.

Baseline Validation

Below are the training results and confusion matrix demonstrating the system's baseline accuracy:

Performance Chart - model_performance/C&TL/results.png

Confusion Matrix - model_performance/C&TL/confusion_matrix.png

🧠 Core Architecture & Dual-Model Pipeline

                     ┌──────────────────┐
                     │   Camera Input   │
                     └────────┬─────────┘
                              ▼
            ┌─────────────────┴─────────────────┐
            │       C++ Preprocessing           │
            └────────┬─────────────────┬────────┘
                     ▼                 ▼
     ┌───────────────────────┐ ┌───────────────────────┐
     │  Crosswalk Detector   │ │ Vehicle/Signal Det.   │
     │ (crosswalk_best.onnx) │ │(C&TL_detector2.onnx)  │
     └───────────────┬───────┘ └───────┬───────────────┘
                     │                 │
                     │ Bounding Boxes  │ Bounding Boxes
                     └────────┬────────┘
                              ▼
            ┌─────────────────┴─────────────────┐
            │   C++ Spatial Logic Engine (IoU)  │
            ├───────────────────────────────────┤
            │  Checks: Are vehicles occupying   │
            │  crosswalk during a Red Light?    │
            └────────┬─────────────────┬────────┘
                     ▼                 ▼
              [ VIOLATION ]       [ COMPLIANT ]


Perception Splitting: Uses crosswalk_best.onnx and C&TL_detector2.onnx separately to optimize precision on different feature sets.

C++ Decision Engine: Standardizes input sizes, handles model coordinate mapping, and calculates the exact intersection boundaries natively.

Compliance Gates: Flag a "VIOLATION" only if a vehicle intersects the crosswalk while the traffic light state is specifically detected as "Red".

**Mathematical Formulation of Spatial Logic**

To determine if a vehicle is illegally occupying a crosswalk during a red light, the C++ logic engine avoids heavy polygon collision math by calculating the Intersection over Union (IoU) and physical boundary overlaps of coordinate boxes mapped onto a $640 \times 640$ coordinate canvas.

Let a detected vehicle bounding box be defined as $V = (x_v, y_v, w_v, h_v)$ and a detected crosswalk bounding box be defined as $C = (x_c, y_c, w_c, h_c)$.

The system computes the coordinates of the intersecting rectangle $I = V \cap C$:

$$\text{xmin}_I = \max(x_v, x_c)$$

$$\text{ymin}_I = \max(y_v, y_c)$$

$$\text{xmax}_I = \min(x_v + w_v, x_c + w_c)$$

$$\text{ymax}_I = \min(y_v + h_v, y_c + h_c)$$

The area of the intersection is formulated as:

$$\text{Area}(I) = \max(0, \text{xmax}_I - \text{xmin}_I) \times \max(0, \text{ymax}_I - \text{ymin}_I)$$

A "VIOLATION" state is triggered if and only if:

An active red light detection exists: $\exists e \in \text{elements} \text{ s.t. } \text{classID}(e) = 17$ (Red Light Active).

A vehicle bounding box overlaps with a crosswalk bounding box on the flat plane:

$$\text{Area}(I) > 0 \quad \text{where} \quad v \in \{\text{Vehicles}\} \text{ and } c \in \{\text{Crosswalks}\}$$

By performing this arithmetic natively in C++ on standard cv::Rect objects using the overloaded & intersection operator ((cw.box & e.box).area()), execution overhead is kept under $0.05\text{ ms}$ per frame, making the engine incredibly light.

**How to use the model yourself**

Requirements

CMake (v3.10+) <br>
OpenCV (v4.0.0+) <br>
ONNX Runtime (v1.14.1+)

Configuring Input Modes

You can easily toggle between inputs inside your primary execution file by modifying the main() entry point:

```
int main() {
    // 1. Initialize the dual-model engine with your weight files
    AutonomousPerceptionEngine system_core(L"onnx_model/crosswalk_best.onnx", L"onnx_model/C&TL_detector2.onnx");
    
    // OPTION A: PROCESS SINGLE STATIC IMAGE
    system_core.process_image("image2.jpg");

    // OPTION B: PROCESS VIDEO ARCHIVES (.mp4, .avi)
    // system_core.process_video("traffic_test.mp4");

    // OPTION C: REAL-TIME WEBCAM / RTSP EDGE STREAM
    // system_core.process_video("0");

    return 0;
}
```

Built with passion in C++, OpenCV, and YOLOv8 ONNX Runtime.

Built with C++, OpenCV, and YOLOv8
