from typing import List, Dict
import math

# A dictionary that maps sign words to the movement instructions used by the 3D model.
SIGN_MOVEMENT_LIBRARY = {
    "HELLO": {
        "hand_pose": "open_palm",
        "hand_rotation": [15, 0, 0],
        "finger_bend": [0, 0, 0, 0, 0],
        "facial_expression": "smile"
    },
    "YOU": {
        "hand_pose": "point",
        "hand_rotation": [0, 10, 0],
        "finger_bend": [0, 1, 1, 1, 1],
        "facial_expression": "neutral"
    },
    # Add more words, etc.
}

def convert_sign_text_to_movements(sign_text: str) -> List[Dict]:
    """
    Convert sign_text into a list of frames describing how each sign
    is animated. Each recognized token from SIGN_MOVEMENT_LIBRARY
    is turned into a 'frame' at a certain time offset.
    """
    frames = build_time_based_frames(sign_text.upper())
    return frames

def build_time_based_frames(sign_text: str) -> List[Dict]:
    tokens = sign_text.split()
    frames = []
    current_time = 0.0

    for token in tokens:
        movement = SIGN_MOVEMENT_LIBRARY.get(token, None)
        if movement:
            joints_dict = {}

            if "hand_rotation" in movement:
                joints_dict["RightHand"] = {
                    "rotation": degrees_to_radians(movement["hand_rotation"]),
                    "position": [0, 0, 0]
                }

            if "finger_bend" in movement:
                joints_dict["RightFingers"] = {
                    "rotation": [0.0, 0.0, 0.0],  # or more complex logic
                    "position": [0, 0, 0]
                }

            # Add a frame
            frames.append({
                "time": current_time,
                "joints": joints_dict
            })
            current_time += 1.0  # naive 1 second per token

    return frames

def degrees_to_radians(deg_list):
    return [d * math.pi / 180.0 for d in deg_list]