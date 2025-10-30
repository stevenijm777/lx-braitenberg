#!/usr/bin/env python3

import math
from asyncio import AbstractEventLoop

import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from dtps import context, ContextConfig, DTPSContext
from dtps_http import RawData
from duckietown_messages.sensors.compressed_image import CompressedImage
from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from duckietown_messages.utils.exceptions import DataDecodingError
from dt_robot_utils import get_robot_name


from solution.connections import get_motor_left_matrix, get_motor_right_matrix
from solution.preprocessing import preprocess


def rescale(a: float, L: float, U: float):
    if np.allclose(L, U):
        return 0.0
    return (a - L) / (U - L)

def rgb_from_jpg(filename: str) -> np.ndarray:
    """
    Load a JPG image and convert it to a numpy array in RGB format.

    Args:
        filename (str): Path to the JPG file.

    Returns:
        np.ndarray: H x W x 3 array with dtype=np.uint8
    """
    with Image.open(filename) as img:
        img = img.convert("RGB")  # Ensure 3 channels
        arr = np.array(img, dtype=np.uint8)
    return arr

def posneg(value, max_value=None, skim=0, nan_color=(0.5, 0.5, 0.5), zero_color=(1.0, 1.0, 1.0)):
    """
    Converts a 2D float value to a RGB representation, where
    red is positive, blue is negative, white is zero.

    :param value: The field to represent.
     :type value: array[HxW]

    :param max_value:  Maximum of absolute value (if None, detect).
     :type max_value:  float,>0

    :param skim:       Fraction to skim (in percent).
     :type skim:       float,>0,<100

    :param nan_color:  Color to give for regions of NaN and Inf.
     :type nan_color:  color

    :return: posneg: A RGB image.
     :rtype: array[HxWx3](uint8)

    """

    # TODO: put this in reprep
    value = value.copy()
    if value.ndim > 2:
        value = value.squeeze()

    if value.dtype == np.dtype("uint8"):
        value = value.astype("float32")

    if len(value.shape) != 2:
        raise Exception("I expected a H x W image, got shape %s." % str(value.shape))

    isfinite = np.isfinite(value)
    isnan = np.logical_not(isfinite)
    # set nan to 0
    value[isnan] = 0

    if max_value is None:
        abs_value = abs(value)
        # if skim != 0:
        #     abs_value = skim_top(abs_value, skim)

        max_value = np.max(abs_value)

        if max_value == 0:
            result = np.zeros((value.shape[0], value.shape[1], 3), dtype="uint8")
            for i in range(3):
                result[:, :, i] = zero_color[i] * 255
            return result

    assert np.isfinite(max_value)

    positive = np.minimum(np.maximum(value, 0), max_value) / max_value
    negative = np.maximum(np.minimum(value, 0), -max_value) / -max_value
    positive_part = (positive * 255).astype("uint8")
    negative_part = (negative * 255).astype("uint8")

    result = np.zeros((value.shape[0], value.shape[1], 3), dtype="uint8")

    anysign = np.maximum(positive_part, negative_part)
    R = 255 - negative_part[:, :]
    G = 255 - anysign
    B = 255 - positive_part[:, :]

    # remember the nans
    R[isnan] = nan_color[0] * 255
    G[isnan] = nan_color[1] * 255
    B[isnan] = nan_color[2] * 255

    result[:, :, 0] = R
    result[:, :, 1] = G
    result[:, :, 2] = B

    return result

@dataclass
class BraitenbergAgentConfig:
    gain: float = 0.3
    const: float = 0.3


class BraitenbergAgent:
    config = BraitenbergAgentConfig()

    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    rgb: Optional[np.ndarray]
    l_max: float
    r_max: float
    l_min: float
    r_min: float

    def __init__(self):
        self.rgb = None
        self.l_max = -500000.0
        self.r_max = -500000.0
        self.l_min = math.inf
        self.r_min = math.inf
        self.left = None
        self.right = None

        self.is_shutdown = False

        self._camera_name = "front_center"
        self._wheels_name = "base"

        self._robot_name = get_robot_name()

        self._pwm: Optional[DTPSContext] = None
        self._loop: Optional[AbstractEventLoop] = None




    def compute_commands(self) -> Tuple[float, float]:
        """Returns the commands (pwm_left, pwm_right)"""
        # If we have not received any image, we don't move
        if self.rgb is None:
            return 0.0, 0.0

        if self.left is None:
            # if it is the first time, we initialize the structures
            shape = self.rgb.shape[0], self.rgb.shape[1]
            self.left = get_motor_left_matrix(shape)
            self.right = get_motor_right_matrix(shape)

        from matplotlib import pyplot as plt
        plt.imshow(self.rgb, interpolation='nearest')
        plt.show()
        # let's take only the intensity of RGB
        P = preprocess(self.rgb)
        # now we just compute the activation of our sensors
        l = float(np.sum(P * self.left))
        r = float(np.sum(P * self.right))
        print(f"l = {l}")
        print(f"r = {r}")
        # These are big numbers -- we want to normalize them.
        # We normalize them using the history

        # first, we remember the high/low of these raw signals
        self.l_max = max(l, self.l_max)
        self.r_max = max(r, self.r_max)
        self.l_min = min(l, self.l_min)
        self.r_min = min(r, self.r_min)

        print(f"l_max = {self.l_max}")
        print(f"r_max = {self.r_max}")
        print(f"l_min = {self.l_min}")
        print(f"r_min = {self.r_min}")

        # now rescale from 0 to 1
        ls = rescale(l, self.l_min, self.l_max)
        print(f"ls = {ls}")
        rs = rescale(r, self.r_min, self.r_max)
        print(f"rs = {rs}")
        gain = self.config.gain
        const = self.config.const
        pwm_left = const + ls * gain
        pwm_right = const + rs * gain

        return pwm_left, pwm_right



    async def on_received_image(self, data: RawData):
        try:
            jpeg: CompressedImage = CompressedImage.from_rawdata(data)
        except DataDecodingError as e:
            self.logerr(f"Failed to decode an incoming message: {e.message}")
            return

        if self.rgb is None:
            print("received first observations")

        self.rgb = jpeg.to_rgb()
        pwm_left, pwm_right = self.compute_commands()
        data = DifferentialPWM(left=pwm_left, right=pwm_right)
        try:
            await self._pwm.publish(data.to_rawdata())
        except Exception as e:
            print(f"Failed to publish last command. {e}")

    async def worker(self):
        # create switchboard context
        switchboard = (await context("switchboard")).navigate(self._robot_name)
        # wait for camera to be ready
        jpeg = await (switchboard / "sensor" / "camera" / self._camera_name / "jpeg").until_ready()
        self._pwm = await (switchboard / "actuator" / "wheels" / self._wheels_name / "pwm").until_ready()
        # Enable dynamic reconnection to the topic
        jpeg = jpeg.configure(ContextConfig(patient=True))

        # subscribe
        await jpeg.subscribe(self.on_received_image)
        # ---
        await self.join()

    async def join(self):
        while not self.is_shutdown:
            await asyncio.sleep(1)

    def spin(self):
        try:
            asyncio.run(self.worker())
        except RuntimeError:
            if not self.is_shutdown:
                self.logerr("An error occurred while running the event loop")
                raise

    def on_shutdown(self):
        if self._loop is not None:
            self.loginfo("Shutting down the event loop")
            self._loop.stop()
        self.is_shutdown = True


if __name__ == "__main__":
    # initialize the node
    agent_node = BraitenbergAgent()
    # keep the node alive
    agent_node.spin()


