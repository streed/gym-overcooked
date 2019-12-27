import os

import cv2
import gym
import numpy as np
import sounddevice as sd
import soundfile as sf

from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces

HEIGHT = 320
WIDTH = 640
N_CHANNELS = 3


DURATIONS = 5
FS = 44100

sd.default.samplerate = FS
sd.default.channels = 1

def load_sound(path):
    data, fs = sf.read(path, dtype='float32')
    return (data, fs)

SOUNDS = {
    0: None,
    1: load_sound(os.path.dirname(__file__) + '/sounds/scientist/sneeze.wav'),
    2: load_sound(os.path.dirname(__file__) + '/sounds/scientist/ok2.wav'),
    3: load_sound(os.path.dirname(__file__) + '/sounds/scientist/please.wav'),
    4: load_sound(os.path.dirname(__file__) + '/sounds/scientist/lowervoice.wav'),
    5: load_sound(os.path.dirname(__file__) + '/sounds/scientist/nooo.wav'),
    6: load_sound(os.path.dirname(__file__) + '/sounds/scientist/yesletsgo.wav'),
    7: load_sound(os.path.dirname(__file__) + '/sounds/scientist/ridiculous.wav'),
    8: load_sound(os.path.dirname(__file__) + '/sounds/scientist/yawn.wav'),
    9: load_sound(os.path.dirname(__file__) + '/sounds/scientist/c1a0_sci_bigday.wav'),
    10: load_sound(os.path.dirname(__file__) + '/sounds/scientist/pain1.wav'),
    11: load_sound(os.path.dirname(__file__) + '/sounds/scientist/pain9.wav'),
    12: load_sound(os.path.dirname(__file__) + '/sounds/cs1.6/mktoasty.wav'),
    13: load_sound(os.path.dirname(__file__) + '/sounds/cs1.6/mkwelldone.wav'),
    14: load_sound(os.path.dirname(__file__) + '/sounds/cs1.6/mkstnothing.wav'),
    15: load_sound(os.path.dirname(__file__) + '/sounds/cs1.6/mksupurb.wav'),
    16: load_sound(os.path.dirname(__file__) + '/sounds/scientist/leadtheway.wav'),
    17: None
}

N_DISCRETE_ACTIONS = len(SOUNDS.keys())

class OvercookedEnv(gym.Env):
    metadata = {'render.modes': ['sound']}

    def __init__(self):
        super(OvercookedEnv, self).__init__()

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(HEIGHT, WIDTH, N_CHANNELS),
                                            dtype=np.uint8)
        self.cap = cv2.VideoCapture(0)

        self.image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self.current_sound = None

    def step(self, action):
        self.current_sound = SOUNDS[action]
        self._play_sound()
        reward = self._record_sound()

        ret, captured_image = self.cap.read()
        resized_image = cv2.resize(captured_image, (WIDTH, HEIGHT))

        return resized_image, reward, False, {}

    def reset(self):
        self.image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self.current_sound = None

    def render(self, mode='sound', close=False):
        pass

    def _play_sound(self):
        if self.current_sound is not None:
            data, fs = self.current_sound
            sd.play(data, fs)
            sd.wait()

    def _record_sound(self):
        recording = sd.rec(int(DURATIONS * FS))
        sd.wait()
        volumn_norm = np.linalg.norm(recording)*10
        return volumn_norm
