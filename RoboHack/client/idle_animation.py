import threading
import time
import numpy as np


class IdleAnimationThread(threading.Thread):

    def __init__(self, env, stop_event):
        super().__init__()
        self.env = env
        self.stop_event = stop_event
        self.daemon = True

        # --- Animation Parameters (Tune these) ---
        self.start_time = time.time()

        self.nod_frequency = 0.4

        self.nod_amplitude = 0.15

        self.jaw_frequency = 1.0

        self.jaw_amplitude = 0.5
        self.action_dim = self.env.config.action_dim
        self.action_shape = (self.action_dim,)

    def run(self):
        print("Starting idle animation routine...")

        try:
            # We must get one observation to start, though we won't use it
            obs = self.env.get_observation()
            info = {}

            while not self.stop_event.is_set():
                current_time = time.time() - self.start_time


                # Calculate nod (up/down) value
                nod_value = self.nod_amplitude * np.sin(current_time * 2 * np.pi * self.nod_frequency)

                # Calculate jaw (open/close) value
                jaw_value = self.jaw_amplitude * np.sin(current_time * 2 * np.pi * self.jaw_frequency)

                # Create a base action vector of all zeros
                action = np.zeros(self.action_shape)


                action[4] = nod_value
                action[6] = jaw_value

                # Clip the final action to be safe
                action = np.clip(action, -1.0, 1.0)

                # --- Step the environment ---
                # We send the action but don't need the new observation
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    obs = self.env.get_observation()
                    info = {}

                time.sleep(0.02)  # ~50 Hz loop

        finally:
            print("Stopping idle animation.")