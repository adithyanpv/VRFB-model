# sensor_model.py

import numpy as np


class SensorModel:

    def __init__(self, config):

        self.cfg = config

        # Sensor noise levels
        self.voltage_noise_std = config.voltage_noise_std
        self.current_noise_std = config.current_noise_std
        self.temperature_noise_std = config.temperature_noise_std

        # Optional constant bias
        self.current_bias = 0.0

    # ------------------------------------------------------------
    def set_current_bias(self, bias):
        self.current_bias = bias

    # ------------------------------------------------------------
    def measure(self, true_outputs):

        measured_voltage = (
            true_outputs["voltage_stack"]
            + np.random.normal(0, self.voltage_noise_std)
        )

        measured_current = (
            true_outputs.get("current", 0.0)
            + self.current_bias
            + np.random.normal(0, self.current_noise_std)
        )

        measured_temperature = (
            true_outputs["temperature_stack"]
            + np.random.normal(0, self.temperature_noise_std)
        )

        return {
            "voltage": measured_voltage,
            "current": measured_current,
            "temperature": measured_temperature
        }