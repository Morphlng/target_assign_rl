import os

import numpy as np
import pygame
from PIL import Image


class PygameRecord:
    def __init__(self, filename: str, fps: int):
        self.fps = fps
        self.filename = filename
        self.frames = []

        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)

    def add_frame(self):
        curr_surface = pygame.display.get_surface()
        x3 = pygame.surfarray.array3d(curr_surface)
        x3 = np.moveaxis(x3, 0, 1)
        array = Image.fromarray(np.uint8(x3))
        self.frames.append(array)

    def save(self):
        self.frames[0].save(
            self.filename,
            save_all=True,
            optimize=False,
            append_images=self.frames[1:],
            loop=0,
            duration=int(1000 / self.fps),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred: {exc_value}")
        self.save()
        # Return False if you want exceptions to propagate, True to suppress them
        return False


if __name__ == "__main__":
    from target_assign_rl import RuleAgent, raw_env

    env = raw_env(dict(render_config={"pause_at_end": False}))
    policy = RuleAgent(env.num_threats)

    for i in range(1, 11):
        with PygameRecord(f"output_{i}.gif", 30) as recorder:
            env.reset()
            for agent in env.agents:
                obs, reward, te, tr, info = env.last()
                if te or tr:
                    action = None
                else:
                    action = policy.predict(obs)
                env.step(action)
                env.render()
                recorder.add_frame()

            # Make the last frame last longer
            for _ in range(30):
                recorder.add_frame()
            recorder.save()
