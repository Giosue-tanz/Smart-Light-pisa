# smartlight_live_video_dqn.py
# Smart Light Pisa - v1.8 | Video Live + DQN Real-time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import random

# ==============================
# CONFIGURAZIONE
# ==============================
MIN_GREEN, MAX_GREEN = 10, 40
ACTION_SIZE = MAX_GREEN - MIN_GREEN + 1

# ==============================
# ESECUZIONE
# ==============================
if __name__ == "__main__":
    run_live()