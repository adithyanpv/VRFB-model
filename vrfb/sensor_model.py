# sensor_model.py

import numpy as np


class SensorModel:
    """
    Models all physical sensors on a real VRFB system.

    Sensors modelled
    ----------------
    voltage          : stack voltage transducer          (±20 mV std)
    current          : current shunt / Hall sensor        (±0.2 A std + optional bias)
    temperature_stack: stack thermocouple                 (±0.3 K std)
    temperature_tank : tank thermocouple                  (±0.5 K std)
    flow_rate        : electromagnetic flow meter         (±0.5 LPM std)

    All five are standard sensors on any VRFB system from lab to grid scale.
    temperature_tank and flow_rate were previously passed as clean physics
    values (out["temperature_tank"], out["flow_rate"]) — this fix adds
    realistic measurement noise so training data and inference both reflect
    what a real sensor suite would deliver.
    """

    # Electromagnetic flow meter: ±0.5 LPM typical at 1 Hz sampling
    FLOW_NOISE_LPM = 0.5

    def __init__(self, config):

        self.cfg = config

        # Noise standard deviations
        self.voltage_noise_std      = config.voltage_noise_std       # V
        self.current_noise_std      = config.current_noise_std       # A
        self.temperature_noise_std  = config.temperature_noise_std   # K
        # Tank thermocouple slightly noisier than stack (longer cable, more EMI)
        self.temp_tank_noise_std    = 0.5                            # K
        self.flow_noise_std         = (
            self.FLOW_NOISE_LPM * config.LPM_to_m3s                 # m³/s
        )

        # Optional current sensor DC bias
        self.current_bias = 0.0

    # ------------------------------------------------------------------
    def set_current_bias(self, bias):
        self.current_bias = bias

    # ------------------------------------------------------------------
    def measure(self, true_outputs: dict) -> dict:
        """
        Returns a dict of noisy sensor readings from true physics outputs.

        All five REN input features are measured here so that both
        dataset_gen.py and server.py pull from the same noisy source —
        not from a mixture of noisy (V, I, T_stack) and clean (T_tank, Q).
        """
        measured_voltage = (
            true_outputs["voltage_stack"]
            + np.random.normal(0, self.voltage_noise_std)
        )

        measured_current = (
            true_outputs.get("current", 0.0)
            + self.current_bias
            + np.random.normal(0, self.current_noise_std)
        )

        measured_temp_stack = (
            true_outputs["temperature_stack"]
            + np.random.normal(0, self.temperature_noise_std)
        )

        measured_temp_tank = (
            true_outputs["temperature_tank"]
            + np.random.normal(0, self.temp_tank_noise_std)
        )

        measured_flow = (
            true_outputs["flow_rate"]
            + np.random.normal(0, self.flow_noise_std)
        )
        # Flow rate cannot be negative
        measured_flow = max(measured_flow, 0.0)

        return {
            "voltage"         : measured_voltage,
            "current"         : measured_current,
            "temperature"     : measured_temp_stack,   # stack thermocouple
            "temperature_tank": measured_temp_tank,    # tank thermocouple
            "flow_rate"       : measured_flow,         # flow meter
        }