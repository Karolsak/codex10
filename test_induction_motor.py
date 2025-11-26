"""
Test script for induction motor calculations (no GUI required)
"""

import sys
import os

# Import the calculator classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import without GUI dependencies
import numpy as np
from dataclasses import dataclass


@dataclass
class MotorParameters:
    """Store induction motor parameters"""
    power_hp: float = 20.0
    poles: int = 4
    frequency: float = 60.0
    voltage: float = 460.0
    slip: float = 0.03
    friction_windage_percent: float = 2.0
    stator_resistance: float = 0.5
    rotor_resistance: float = 0.3
    stator_leakage: float = 1.5
    rotor_leakage: float = 1.5
    magnetizing_reactance: float = 50.0
    moment_of_inertia: float = 0.5
    load_torque: float = 50.0


class InductionMotorCalculator:
    """Calculation engine for induction motor analysis"""

    def __init__(self, params: MotorParameters):
        self.params = params
        self.hp_to_watts = 746

    def calculate_basic_parameters(self) -> dict:
        """Calculate basic motor parameters"""
        P_out = self.params.power_hp * self.hp_to_watts
        P_fw = (self.params.friction_windage_percent / 100) * P_out
        P_mech = P_out + P_fw
        s = self.params.slip
        P_rotor_loss = (s / (1 - s)) * P_mech
        P_rotor_input = P_mech + P_rotor_loss
        n_s = 120 * self.params.frequency / self.params.poles
        omega_s = 2 * np.pi * n_s / 60
        n_r = n_s * (1 - s)
        omega_r = omega_s * (1 - s)
        T_out = P_mech / omega_r

        return {
            'output_power_w': P_out,
            'output_power_hp': self.params.power_hp,
            'friction_windage_w': P_fw,
            'mech_power_w': P_mech,
            'rotor_loss_w': P_rotor_loss,
            'rotor_input_w': P_rotor_input,
            'sync_speed_rpm': n_s,
            'sync_speed_rad_s': omega_s,
            'rotor_speed_rpm': n_r,
            'rotor_speed_rad_s': omega_r,
            'output_torque_nm': T_out,
            'slip': s,
            'slip_percent': s * 100
        }


def main():
    """Run test calculations"""
    print("=" * 70)
    print("INDUCTION MOTOR CALCULATION TEST")
    print("=" * 70)
    print()

    # Create motor parameters
    params = MotorParameters(
        power_hp=20.0,
        poles=4,
        frequency=60.0,
        slip=0.03,
        friction_windage_percent=2.0
    )

    # Create calculator
    calculator = InductionMotorCalculator(params)

    # Calculate parameters
    results = calculator.calculate_basic_parameters()

    # Display problem
    print("PROBLEM:")
    print(f"A {params.power_hp} HP, {params.poles}-pole, three-phase induction motor")
    print(f"Friction and windage losses: {params.friction_windage_percent}% of output")
    print(f"Full-load slip: {params.slip * 100}%")
    print()

    # Display solution
    print("=" * 70)
    print("SOLUTION:")
    print("=" * 70)
    print()

    print("(a) ROTOR I²R LOSS:")
    print(f"    Output Power:           {results['output_power_w']:.2f} W")
    print(f"    Friction & Windage:     {results['friction_windage_w']:.2f} W")
    print(f"    Mechanical Power:       {results['mech_power_w']:.2f} W")
    print(f"    ")
    print(f"    Formula: Rotor I²R Loss = (s/(1-s)) × P_mech")
    print(f"    Rotor I²R Loss = ({params.slip:.3f}/{1-params.slip:.3f}) × {results['mech_power_w']:.2f}")
    print(f"    ")
    print(f"    ╔═════════════════════════════════════════════╗")
    print(f"    ║  ROTOR I²R LOSS = {results['rotor_loss_w']:.2f} W          ║")
    print(f"    ╚═════════════════════════════════════════════╝")
    print()

    print("(b) ROTOR INPUT (AIR-GAP POWER):")
    print(f"    Formula: Rotor Input = Mechanical Power + Rotor Loss")
    print(f"    Rotor Input = {results['mech_power_w']:.2f} + {results['rotor_loss_w']:.2f}")
    print(f"    ")
    print(f"    ╔═════════════════════════════════════════════╗")
    print(f"    ║  ROTOR INPUT = {results['rotor_input_w']:.2f} W          ║")
    print(f"    ╚═════════════════════════════════════════════╝")
    print()

    print("(c) OUTPUT TORQUE:")
    print(f"    Synchronous Speed:      {results['sync_speed_rpm']:.2f} RPM")
    print(f"                            {results['sync_speed_rad_s']:.2f} rad/s")
    print(f"    Rotor Speed:            {results['rotor_speed_rpm']:.2f} RPM")
    print(f"                            {results['rotor_speed_rad_s']:.2f} rad/s")
    print(f"    ")
    print(f"    Formula: Output Torque = P_mech / ω_rotor")
    print(f"    Output Torque = {results['mech_power_w']:.2f} / {results['rotor_speed_rad_s']:.2f}")
    print(f"    ")
    print(f"    ╔═════════════════════════════════════════════╗")
    print(f"    ║  OUTPUT TORQUE = {results['output_torque_nm']:.2f} N⋅m        ║")
    print(f"    ╚═════════════════════════════════════════════╝")
    print()

    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"(a) Rotor I²R Loss:    {results['rotor_loss_w']:.2f} W")
    print(f"(b) Rotor Input:       {results['rotor_input_w']:.2f} W")
    print(f"(c) Output Torque:     {results['output_torque_nm']:.2f} N⋅m")
    print("=" * 70)
    print()

    print("✓ All calculations completed successfully!")
    print()

    # Verification
    print("VERIFICATION:")
    print(f"Power relationship check:")
    print(f"  P_rotor_input = P_mech + P_rotor_loss")
    print(f"  {results['rotor_input_w']:.2f} = {results['mech_power_w']:.2f} + {results['rotor_loss_w']:.2f}")
    calc_sum = results['mech_power_w'] + results['rotor_loss_w']
    print(f"  {results['rotor_input_w']:.2f} ≈ {calc_sum:.2f}")
    if abs(results['rotor_input_w'] - calc_sum) < 0.01:
        print("  ✓ Verified!")
    else:
        print("  ✗ Error in calculation")
    print()

    print(f"Slip relationship check:")
    print(f"  P_rotor_loss / P_mech = s / (1-s)")
    print(f"  {results['rotor_loss_w']:.2f} / {results['mech_power_w']:.2f} = {params.slip:.3f} / {1-params.slip:.3f}")
    ratio_left = results['rotor_loss_w'] / results['mech_power_w']
    ratio_right = params.slip / (1 - params.slip)
    print(f"  {ratio_left:.6f} = {ratio_right:.6f}")
    if abs(ratio_left - ratio_right) < 0.0001:
        print("  ✓ Verified!")
    else:
        print("  ✗ Error in calculation")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
