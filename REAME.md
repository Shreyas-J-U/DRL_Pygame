# DRL-Based Autonomous Driving Prototype (Prediction-Aware)

## 📌 Overview

This project demonstrates a **Deep Reinforcement Learning (DRL)** based decision-making system for autonomous driving in crowded environments.

The key idea is:

> The agent uses **predicted pedestrian trajectories** to make **anticipatory driving decisions**, instead of reacting only to current positions.

This serves as a **scaled prototype** of our full pipeline:

```
YOLOv8 → DeepSORT → Trajectron++ → PPO (DRL) → Control
```

---

## 🎯 Features

* Custom **Gymnasium-compatible environment**
* **Pygame-based simulation**
* PPO agent using **Stable-Baselines3**
* **Prediction-aware state representation**
* Toggle between:

  * Prediction OFF (reactive)
  * Prediction ON (anticipatory)
* Visualized **future pedestrian trajectories**

---

## 🧠 Key Concept

Unlike traditional approaches, the agent observes:

* Current pedestrian positions
* **Predicted future trajectories (mock Trajectron++)**

This allows:

* Early braking
* Safer navigation
* Reduced collisions

---

## 📁 Project Structure

```
drl_prediction_demo/
│
├── main.py              # Run trained model (visual demo)
├── train.py             # Train PPO agent
├── config.py            # Parameters
│
├── env/
│   ├── car_env.py
│   ├── entities.py
│   ├── renderer.py
│
├── utils/
│   ├── state.py
│   ├── reward.py
│   ├── prediction.py
│
├── models/              # Saved models
├── logs/                # Training logs
```

---

## ⚙️ Installation

### 1. Clone repository

```
git clone <your-repo-url>
cd drl_prediction_demo
```

### 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate     # Windows
# OR
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```
python train.py
```

This will:

* Train PPO agent
* Save model in `models/`

---

## ▶️ Running the Demo

### Run multiple episodes

```
python demo.py --episodes 3
```

### Load specific checkpoint

```
python demo.py --model models/ppo_model_predTrue_50000_steps.zip
```

---

## 🔁 Prediction Modes

The environment supports two modes:

| Mode           | Behavior             |
| -------------- | -------------------- |
| Prediction OFF | Reactive driving     |
| Prediction ON  | Anticipatory driving |

Toggle this in config or environment settings.

---

## 📊 Expected Behavior

### Without Prediction

* Late braking
* Higher collision rate

### With Prediction

* Early slowing
* Smooth navigation
* Improved safety

---

## 🧪 Evaluation Metrics (Optional)

* Episode reward
* Collision rate
* Success rate (goal reached)
* Smoothness (acceleration penalty)

---

## 🔗 Relation to Full Project

This prototype simulates:

* Detection → simulated
* Tracking → simulated
* Prediction → linear model (mock Trajectron++)
* Decision → PPO (DRL)
* Environment → Pygame (instead of CARLA)

---

