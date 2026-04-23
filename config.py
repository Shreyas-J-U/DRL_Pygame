# ============================================================================
# Configuration for Prediction-Aware Autonomous Driving Simulation
# ============================================================================

# === ENVIRONMENT ===
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
LANE_Y = 300  # Road center line
ROAD_HEIGHT = 100  # Total road height (lane_y - 50 to lane_y + 50)

# === SIMULATION ===
FPS = 30
DT = 1 / FPS  # Fixed timestep
MAX_STEPS = 500
MAX_SIMULATION_TIME = 20  # seconds

# === CAR DYNAMICS ===
CAR_INITIAL_X = 50
CAR_INITIAL_Y = LANE_Y
CAR_WIDTH = 40
CAR_HEIGHT = 20
CAR_MAX_VELOCITY = 100  # pixels/frame at DT
CAR_ACCELERATION = 5  # pixels/frame^2
CAR_BRAKE_DECELERATION = 7  # pixels/frame^2
CAR_LANE_CHANGE_SPEED = 200  # pixels/frame

# === PEDESTRIAN DYNAMICS ===
NUM_PEDESTRIANS = 5
PED_RADIUS = 10
PED_SPEED_MIN = 40  # pixels/frame
PED_SPEED_MAX = 80
PED_SPAWN_X_MIN = 200
PED_SPAWN_X_MAX = 800
PED_CROSS_SPEED_DEVIATION = 0.2  # Random variation

# === PREDICTION ===
PREDICTION_STEPS = 5  # Number of future positions to predict
PREDICTION_HORIZON = PREDICTION_STEPS * DT  # Time horizon in seconds

# === ACTION SPACE ===
# Discrete actions:
# 0: accelerate
# 1: brake
# 2: move up (change lane)
# 3: move down (change lane)
NUM_ACTIONS = 4

# === OBSERVATION SPACE ===
# Car state: x, y, velocity (3 features)
# Current pedestrian state: num_pedestrians * (x, y) (2*num_pedestrians features)
# Predicted pedestrian state: num_pedestrians * prediction_steps * (x, y)
CAR_STATE_DIM = 3
PED_CURRENT_STATE_DIM = 2 * NUM_PEDESTRIANS
PED_PREDICTED_STATE_DIM = 2 * NUM_PEDESTRIANS * PREDICTION_STEPS
OBSERVATION_DIM = CAR_STATE_DIM + PED_CURRENT_STATE_DIM + PED_PREDICTED_STATE_DIM

# === REWARD WEIGHTS ===
COLLISION_PENALTY = -100
PROGRESS_REWARD = 1.0  # per pixel moved forward
SAFE_DISTANCE_REWARD = 0.5  # per timestep maintaining safe distance
BRAKE_PENALTY = -1  # per braking action
PREDICTION_BONUS = 0.1  # Bonus when using predictions effectively

# === SAFETY ===
COLLISION_THRESHOLD = 25  # pixels (car radius + ped radius)
SAFE_DISTANCE = 80  # pixels

# === TRAINING ===
LEARNING_RATE = 3e-4
TIMESTEPS_TRAINING = 50000
BATCH_SIZE = 128
N_STEPS = 2048  # PPO n_steps parameter
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda

# === DEBUG/RENDER ===
SHOW_PREDICTIONS = True  # Draw predicted trajectories
PREDICTION_LINE_COLOR = (0, 0, 255)  # Blue
PREDICTION_POINT_COLOR = (100, 100, 255)  # Light blue
CAR_COLOR = (0, 255, 0)  # Green
PED_COLOR = (255, 0, 0)  # Red
BACKGROUND_COLOR = (30, 30, 30)
ROAD_COLOR = (50, 50, 50)
FONT_SIZE = 24
