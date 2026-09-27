<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=2500&pause=800&color=00D2FF&center=true&vCenter=true&width=700&lines=TinyML+Gesture+Recognition+%F0%9F%A4%9A;Train+in+Python+%F0%9F%90%8D;Deploy+on+STM32+%F0%9F%94%A9;Wave+%E2%9E%9C+Sup+%E2%9E%9C+Predict+%E2%9A%A1" alt="Typing SVG" />

### On-device gesture recognition — from raw accelerometer data to a quantized neural net running on a microcontroller.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00D2FF,100:6C63FF&height=160&section=header&text=&fontSize=0" width="100%"/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-int8-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![STM32](https://img.shields.io/badge/STM32F446RE-Cortex--M4-03234B?style=for-the-badge&logo=stmicroelectronics&logoColor=white)
![X--CUBE--AI](https://img.shields.io/badge/X--CUBE--AI-enabled-00979D?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

</div>

---

## ✨ The Idea

> **Can a $5 microcontroller understand a hand gesture in real time — with no cloud, no GPU, and no internet?**

This project answers **yes**. It's an end-to-end **TinyML** pipeline that:

1. 📊 **Captures** motion sensor data (accelerometer/gyro windows) for two gestures — `hi` 👋 and `sup` 🤙
2. 🧠 **Trains** a compact 1D-CNN in TensorFlow/Keras to tell them apart
3. 🗜️ **Quantizes** the model to full `int8` TensorFlow Lite — shrinking it to fit on a microcontroller
4. 🚀 **Deploys** it straight into firmware using **ST's X-CUBE-AI**, running live inference on an **STM32F446RE (Cortex-M4)**

```
   Sensor Stream          Training (Python)              Deployment (STM32)
 ┌────────────────┐     ┌───────────────────┐        ┌────────────────────────┐
 │ hi.csv/sup.csv │ ──▶ │  preprocess.py     │        │  X-CUBE-AI runtime     │
 │ (raw motion)   │     │  ↓ windows (50x6)  │        │  gesture_model_data.c  │
 └────────────────┘     │  train.py → .h5    │  ──▶   │  ↓                     │
                        │  export_tflite.py  │        │  UART @115200 → 👋/🤙  │
                        │  ↓ model.tflite     │        └────────────────────────┘
                        └───────────────────┘
```

## 🎬 How It Flows

<div align="center">

| Step | What happens | File |
|:---:|:---|:---|
| 1️⃣ | Raw sensor readings are windowed into `(50 samples × 6 channels)` chunks | `python/preprocess.py` |
| 2️⃣ | A 1D-CNN (`Conv1D → Pool → Conv1D → Pool → Dense`) learns gesture patterns | `python/train.py` |
| 3️⃣ | Model is converted & quantized to `int8` TFLite for tiny footprint | `python/export_tflite.py` |
| 4️⃣ | STM32CubeIDE + X-CUBE-AI compiles the model into C and flashes the board | `stm32/tinyml/` |
| 5️⃣ | The board classifies gestures **live** and reports results over UART | 🔴 real-time |

</div>

<div align="center">

```
 👋  hi   ──▶  [ CNN ]  ──▶  0  ──▶  UART: "Gesture: HI"
 🤙  sup  ──▶  [ CNN ]  ──▶  1  ──▶  UART: "Gesture: SUP"
```
</div>

---

## 📂 Project Layout

```
tinyml-gesture-stm32/
├── python/                   🐍 Model training & conversion
│   ├── preprocess.py         →  CSV → normalized, windowed .npy arrays
│   ├── train.py              →  Trains the 1D-CNN, saves gesture_model.h5
│   ├── export_tflite.py      →  Quantizes to int8 model.tflite
│   └── model_data.h          →  C header for embedding the model
│
└── stm32/                    🔩 Embedded firmware
    └── tinyml/                →  STM32CubeIDE project (STM32F446RE)
        ├── Core/              →  HAL init, main loop, UART
        └── X-CUBE-AI/         →  Auto-generated inference engine
```

---

## 🚀 Quick Start

### 1. Train the model

```bash
cd python
pip install tensorflow pandas scikit-learn numpy

python preprocess.py        # data/hi.csv + data/sup.csv → windowed .npy files
python train.py             # trains CNN → gesture_model.h5
python export_tflite.py     # quantizes → model.tflite
```

### 2. Flash the firmware

```bash
# Open stm32/tinyml in STM32CubeIDE
# Build & flash to an STM32F446RE (Nucleo-F446RE or similar)
```

### 3. Watch it work ⚡

Open a serial monitor at **115200 baud** on USART2 and start gesturing — predictions stream out live.

---

## 🧠 Model Architecture

```
Input (50 timesteps × 6 channels)
        │
   Conv1D(64, k=3) → ReLU
        │
   MaxPool1D(2)
        │
   Conv1D(128, k=3) → ReLU
        │
   MaxPool1D(2)
        │
   Flatten → Dense(128) → Dropout(0.5)
        │
   Dense(1) → Sigmoid  →  0 = "hi" 👋 | 1 = "sup" 🤙
```

---

## 🛠️ Tech Stack

<div align="center">

![TensorFlow](https://skillicons.dev/icons?i=tensorflow)
![Python](https://skillicons.dev/icons?i=python)
![C](https://skillicons.dev/icons?i=c)
![STM32](https://img.shields.io/badge/-STM32-03234B?style=flat-square&logo=stmicroelectronics&logoColor=white)

</div>

## 📜 License

This project is licensed under the **MIT License**. The bundled `X-CUBE-AI` runtime is subject to ST's own license terms (see `LICENSE_X-CUBE-AI.txt`).

---

<div align="center">

**Made with 🧠 + ⚙️ — proving that Machine Learning doesn't need a data center.**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:00D2FF&height=100&section=footer" width="100%"/>

</div>
