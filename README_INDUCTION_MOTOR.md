# Advanced Induction Motor Simulator

## Overview
A comprehensive Python + Tkinter application for analyzing three-phase induction motors, featuring both static parameter calculation and dynamic simulation with real-time ODE solvers.

## Problem Solution

### Given Problem:
A 20 HP, 4-pole, three-phase induction motor has friction and windage losses of 2% of the output. The full-load slip is 3%. Calculate for full-load:
- (a) The rotor I²R-loss
- (b) The rotor input
- (c) The output torque

### Solution:

**Given:**
- Output Power: 20 HP = 14,920 W
- Poles: 4
- Slip: 3% (0.03)
- Friction & Windage: 2% of output = 298.4 W

**(a) Rotor I²R Loss:**
```
Mechanical Power = Output + Friction & Windage
P_mech = 14,920 + 298.4 = 15,218.4 W

Rotor I²R Loss = (s/(1-s)) × P_mech
Rotor I²R Loss = (0.03/0.97) × 15,218.4
**Rotor I²R Loss = 470.36 W**
```

**(b) Rotor Input (Air-Gap Power):**
```
Rotor Input = Mechanical Power + Rotor Copper Loss
Rotor Input = 15,218.4 + 470.36
**Rotor Input = 15,688.76 W**
```

**(c) Output Torque:**
```
Synchronous Speed: n_s = 120f/p = 120(60)/4 = 1800 RPM
ω_s = 2πn_s/60 = 188.5 rad/s

Rotor Speed: ω_r = (1-s)ω_s = 0.97 × 188.5 = 182.85 rad/s

Output Torque = P_mech / ω_r
Output Torque = 15,218.4 / 182.85
**Output Torque = 83.24 N⋅m**
```

## Features

### 1. User Interface (Tkinter GUI)
- **Main Menu**: File, Analysis, and Help menus
- **Tabbed Interface**: Static Analysis, Dynamic Simulation, and Motor Parameters
- **Input Parameters**: Text entries for quick parameter input
- **Control Sliders**: Real-time adjustment of motor parameters
- **Auto-scaling**: Automatic window resize and plot adjustment

### 2. Calculation Modules (Mathematical Modeling)

#### Static Analysis:
- Basic motor parameters calculation
- Rotor losses and efficiency
- Torque calculations
- Equivalent circuit analysis

#### Dynamic Simulation:
- **Differential Equations**:
  ```
  dω/dt = (T_e - T_load - Bω) / J
  dθ/dt = ω
  ```
  Where:
  - ω = rotor angular velocity
  - θ = rotor position
  - T_e = electromagnetic torque
  - T_load = load torque
  - B = damping coefficient
  - J = moment of inertia

- **ODE Solvers**:
  - **RK45 (Runge-Kutta 4th order)**: High accuracy, adaptive stepping
  - **Euler Method**: Simple, fast for quick simulations

### 3. Results Visualization

#### Static Analysis Plots:
- Torque-Speed Characteristic Curve
- Efficiency vs Slip Curve

#### Dynamic Simulation Plots (Real-time):
- Rotor Speed vs Time
- Electromagnetic Torque vs Time
- Rotor Current vs Time
- Mechanical Power vs Time
- Slip vs Time
- Torque-Speed Trajectory

#### Control Buttons:
- **Start**: Begin dynamic simulation
- **Stop**: Pause simulation
- **Reset**: Clear simulation data and restart

### 4. Advanced Features for Electrical Engineering

#### Equivalent Circuit Analysis:
- Thevenin equivalent calculation
- Rotor current computation
- Air-gap power calculation
- Copper loss analysis

#### Parameter Studies:
- Adjustable stator and rotor resistances
- Variable reactances (leakage and magnetizing)
- Mechanical inertia effects
- Load torque variations

#### Performance Metrics:
- Efficiency calculation
- Power factor estimation
- Speed regulation
- Starting characteristics

## Installation

### Requirements:
```bash
pip install numpy matplotlib
```

For GUI (Tkinter):
- **Linux**: `sudo apt-get install python3-tk`
- **macOS**: Included with Python
- **Windows**: Included with Python

### Running the Application:
```bash
python3 induction_motor_simulator.py
```

## Usage Guide

### 1. Static Analysis Tab:
1. Enter motor parameters (Power, Poles, Frequency, Slip, Friction & Windage)
2. Click "Calculate" to solve the problem
3. View detailed results in the text area
4. Use Analysis menu for Torque-Speed and Efficiency curves

### 2. Dynamic Simulation Tab:
1. Select ODE solver method (RK45 recommended for accuracy)
2. Click "Start" to begin simulation
3. Observe real-time plots showing motor startup behavior
4. Click "Stop" to pause, "Reset" to restart

### 3. Motor Parameters Tab:
1. Use sliders to adjust motor parameters in real-time
2. Modify electrical parameters (resistances, reactances)
3. Change mechanical parameters (inertia, load torque)
4. Click "Update Calculator" to apply changes

## Technical Details

### Mathematical Model:

**Power Flow in Induction Motor:**
```
Stator Input → Stator Copper Loss → Air-Gap Power → Rotor Copper Loss →
Mechanical Power → Friction & Windage → Output Power
```

**Key Relationships:**
- P_airgap / s = P_rotor_copper
- P_mech = P_airgap × (1 - s)
- T_e = P_airgap / ω_s
- η = P_out / P_in

**Slip:**
```
s = (n_s - n_r) / n_s
```

**Synchronous Speed:**
```
n_s = 120f / p (RPM)
```

### Dynamic Model:

The motor dynamics are governed by:
```
J(dω/dt) = T_e - T_load - Bω
```

Where electromagnetic torque varies with slip:
```
T_e = f(slip, motor_parameters)
```

## Practical Applications

1. **Motor Selection**: Evaluate motor performance for specific applications
2. **Starting Analysis**: Study startup transients and inrush currents
3. **Load Matching**: Optimize motor-load pairing
4. **Efficiency Studies**: Analyze efficiency at various operating points
5. **Parameter Sensitivity**: Understand impact of design parameters
6. **Educational Tool**: Learn induction motor theory and behavior

## Code Structure

```
induction_motor_simulator.py
├── MotorParameters (dataclass)
│   └── Stores all motor parameters
├── InductionMotorCalculator
│   ├── calculate_basic_parameters()
│   ├── calculate_equivalent_circuit()
│   ├── calculate_torque_speed_curve()
│   └── calculate_efficiency()
├── DynamicSimulator
│   ├── motor_dynamics() - Differential equations
│   ├── rk45_step() - RK45 integration
│   ├── euler_step() - Euler integration
│   └── simulate_step() - Main simulation loop
└── InductionMotorGUI
    ├── create_menu()
    ├── create_static_analysis_tab()
    ├── create_dynamic_simulation_tab()
    ├── create_parameters_tab()
    └── start/stop/reset_simulation()
```

## Example Calculations

### Default Motor (20 HP, 4-pole, 60 Hz):
- Synchronous Speed: 1800 RPM
- Full-Load Speed: 1746 RPM (3% slip)
- Output Torque: 83.24 N⋅m
- Rotor I²R Loss: 470.36 W
- Rotor Input: 15,688.76 W

### Startup Simulation:
- Initial current: ~6-7× rated current
- Startup time: ~0.5-2 seconds (depends on inertia and load)
- Peak torque: ~2-3× rated torque

## Troubleshooting

### Issue: GUI doesn't display
**Solution**: Ensure Tkinter is installed:
```bash
# Linux
sudo apt-get install python3-tk

# Test
python3 -c "import tkinter"
```

### Issue: Plots not updating
**Solution**: Check matplotlib backend:
```python
import matplotlib
matplotlib.use('TkAgg')
```

### Issue: Simulation too fast/slow
**Solution**: Adjust time step (dt) in code or simulation speed multiplier

## Future Enhancements

- [ ] Variable frequency drive (VFD) simulation
- [ ] Voltage dip/unbalance analysis
- [ ] Multi-motor systems
- [ ] Data export to CSV/Excel
- [ ] Custom load torque profiles
- [ ] Thermal modeling
- [ ] Harmonic analysis

## References

1. "Electric Machinery Fundamentals" - Stephen Chapman
2. "Analysis of Electric Machinery and Drive Systems" - Paul Krause
3. IEEE Standards for Induction Motors
4. IEC 60034 Motor Standards

## License

Educational and research use.

## Author

Created for electrical engineering education and practical motor analysis.

---

**Note**: This simulator provides accurate results for typical induction motor analysis. For critical applications, always verify results with manufacturer data and field measurements.
