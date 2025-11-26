"""
Advanced Induction Motor Simulator with Dynamic Analysis
Comprehensive Python + Tkinter Application for Electrical Engineering
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
from dataclasses import dataclass
from typing import Tuple, List
import threading
import time


@dataclass
class MotorParameters:
    """Store induction motor parameters"""
    power_hp: float = 20.0  # Output power in HP
    poles: int = 4
    frequency: float = 60.0  # Hz
    voltage: float = 460.0  # Line voltage (V)
    slip: float = 0.03  # Full-load slip
    friction_windage_percent: float = 2.0  # % of output
    stator_resistance: float = 0.5  # Ohms per phase
    rotor_resistance: float = 0.3  # Ohms per phase (referred to stator)
    stator_leakage: float = 1.5  # Ohms
    rotor_leakage: float = 1.5  # Ohms
    magnetizing_reactance: float = 50.0  # Ohms
    moment_of_inertia: float = 0.5  # kg⋅m²
    load_torque: float = 50.0  # N⋅m


class InductionMotorCalculator:
    """Calculation engine for induction motor analysis"""

    def __init__(self, params: MotorParameters):
        self.params = params
        self.hp_to_watts = 746  # Conversion factor

    def calculate_basic_parameters(self) -> dict:
        """Calculate basic motor parameters for the given problem"""
        # Convert HP to Watts
        P_out = self.params.power_hp * self.hp_to_watts

        # Friction and windage losses
        P_fw = (self.params.friction_windage_percent / 100) * P_out

        # Mechanical power developed
        P_mech = P_out + P_fw

        # Rotor copper losses (I²R loss)
        s = self.params.slip
        P_rotor_loss = (s / (1 - s)) * P_mech

        # Rotor input (air-gap power)
        P_rotor_input = P_mech + P_rotor_loss

        # Synchronous speed
        n_s = 120 * self.params.frequency / self.params.poles  # RPM
        omega_s = 2 * np.pi * n_s / 60  # rad/s

        # Rotor speed
        n_r = n_s * (1 - s)  # RPM
        omega_r = omega_s * (1 - s)  # rad/s

        # Output torque
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

    def calculate_equivalent_circuit(self, slip: float) -> dict:
        """Calculate equivalent circuit parameters at given slip"""
        params = self.params

        # Thevenin equivalent
        X_m = params.magnetizing_reactance
        R_s = params.stator_resistance
        X_s = params.stator_leakage
        X_r = params.rotor_leakage
        R_r = params.rotor_resistance

        # Thevenin voltage
        Z_m = 1j * X_m
        Z_s = R_s + 1j * X_s
        V_th = (params.voltage / np.sqrt(3)) * Z_m / (Z_s + Z_m)

        # Thevenin impedance
        Z_th = Z_s * Z_m / (Z_s + Z_m)
        R_th = Z_th.real
        X_th = Z_th.imag

        # Rotor current
        if slip > 0.0001:
            Z_rotor = R_r / slip + 1j * X_r
            Z_total = Z_th + Z_rotor
            I_rotor = abs(V_th) / abs(Z_total)
        else:
            I_rotor = 0.0

        # Power calculations
        if slip > 0.0001:
            P_airgap = 3 * I_rotor**2 * R_r / slip
            P_mech = P_airgap * (1 - slip)
            P_rotor_copper = P_airgap * slip
        else:
            P_airgap = 0.0
            P_mech = 0.0
            P_rotor_copper = 0.0

        # Torque
        omega_s = 2 * np.pi * params.frequency * 2 / params.poles
        if omega_s > 0:
            torque = P_airgap / omega_s
        else:
            torque = 0.0

        return {
            'slip': slip,
            'rotor_current': I_rotor,
            'airgap_power': P_airgap,
            'mech_power': P_mech,
            'rotor_copper_loss': P_rotor_copper,
            'torque': torque,
            'V_th': abs(V_th),
            'R_th': R_th,
            'X_th': X_th
        }

    def calculate_torque_speed_curve(self, slip_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate torque-speed characteristic curve"""
        torques = []
        speeds = []

        n_s = 120 * self.params.frequency / self.params.poles

        for s in slip_range:
            result = self.calculate_equivalent_circuit(s)
            torques.append(result['torque'])
            speeds.append(n_s * (1 - s))

        return np.array(speeds), np.array(torques)

    def calculate_efficiency(self, slip: float) -> float:
        """Calculate motor efficiency at given slip"""
        result = self.calculate_equivalent_circuit(slip)

        # Stator copper loss (approximation)
        P_stator_copper = 3 * result['rotor_current']**2 * self.params.stator_resistance

        # Core loss (approximation - typically 2-5% of rating)
        P_core = 0.03 * self.params.power_hp * self.hp_to_watts

        # Friction and windage
        P_fw = (self.params.friction_windage_percent / 100) * result['mech_power']

        # Total input power
        P_in = result['airgap_power'] + P_stator_copper + P_core

        # Output power
        P_out = result['mech_power'] - P_fw

        if P_in > 0:
            efficiency = (P_out / P_in) * 100
        else:
            efficiency = 0.0

        return max(0, min(100, efficiency))


class DynamicSimulator:
    """Real-time ODE solver for dynamic motor simulation"""

    def __init__(self, params: MotorParameters):
        self.params = params
        self.calculator = InductionMotorCalculator(params)
        self.reset_simulation()

    def reset_simulation(self):
        """Reset simulation state"""
        self.time = 0.0
        self.omega = 0.0  # Initial rotor speed (rad/s)
        self.theta = 0.0  # Rotor position (rad)
        self.history = {
            'time': [],
            'omega': [],
            'torque': [],
            'current': [],
            'power': [],
            'slip': []
        }

    def motor_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Differential equations for induction motor dynamics
        state = [omega, theta]
        domega/dt = (T_e - T_load - B*omega) / J
        dtheta/dt = omega
        """
        omega, theta = state

        # Calculate synchronous speed
        omega_s = 2 * np.pi * self.params.frequency * 2 / self.params.poles

        # Calculate slip
        if omega_s > 0:
            slip = (omega_s - omega) / omega_s
        else:
            slip = 1.0

        # Limit slip to reasonable range
        slip = np.clip(slip, -0.5, 1.5)

        # Calculate electromagnetic torque
        result = self.calculator.calculate_equivalent_circuit(slip)
        T_e = result['torque']

        # Load torque (can be speed-dependent)
        T_load = self.params.load_torque

        # Damping coefficient (approximation)
        B = 0.1

        # Moment of inertia
        J = self.params.moment_of_inertia

        # Differential equations
        domega_dt = (T_e - T_load - B * omega) / J
        dtheta_dt = omega

        return np.array([domega_dt, dtheta_dt])

    def rk45_step(self, dt: float):
        """Runge-Kutta 45 integration step"""
        state = np.array([self.omega, self.theta])

        # RK45 coefficients
        k1 = self.motor_dynamics(self.time, state)
        k2 = self.motor_dynamics(self.time + dt/2, state + dt*k1/2)
        k3 = self.motor_dynamics(self.time + dt/2, state + dt*k2/2)
        k4 = self.motor_dynamics(self.time + dt, state + dt*k3)

        # Update state
        state_new = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        self.omega = state_new[0]
        self.theta = state_new[1]
        self.time += dt

    def euler_step(self, dt: float):
        """Euler integration step"""
        state = np.array([self.omega, self.theta])
        dstate = self.motor_dynamics(self.time, state)

        self.omega += dstate[0] * dt
        self.theta += dstate[1] * dt
        self.time += dt

    def simulate_step(self, dt: float, method: str = 'rk45'):
        """Perform one simulation step"""
        if method == 'rk45':
            self.rk45_step(dt)
        else:
            self.euler_step(dt)

        # Calculate current motor parameters
        omega_s = 2 * np.pi * self.params.frequency * 2 / self.params.poles
        slip = (omega_s - self.omega) / omega_s if omega_s > 0 else 1.0
        slip = np.clip(slip, -0.5, 1.5)

        result = self.calculator.calculate_equivalent_circuit(slip)

        # Store history
        self.history['time'].append(self.time)
        self.history['omega'].append(self.omega * 60 / (2 * np.pi))  # Convert to RPM
        self.history['torque'].append(result['torque'])
        self.history['current'].append(result['rotor_current'])
        self.history['power'].append(result['mech_power'] / 1000)  # kW
        self.history['slip'].append(slip * 100)  # Percent

        # Limit history length
        max_points = 500
        for key in self.history:
            if len(self.history[key]) > max_points:
                self.history[key].pop(0)


class InductionMotorGUI:
    """Main GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Induction Motor Simulator")
        self.root.geometry("1400x900")

        # Parameters
        self.params = MotorParameters()
        self.calculator = InductionMotorCalculator(self.params)
        self.simulator = DynamicSimulator(self.params)

        # Simulation control
        self.is_running = False
        self.sim_thread = None
        self.dt = 0.01  # Time step (seconds)
        self.sim_method = tk.StringVar(value='rk45')

        # Create UI
        self.create_menu()
        self.create_main_layout()

        # Auto-scaling support
        self.root.bind('<Configure>', self.on_window_resize)

        # Initial calculation
        self.calculate_static_parameters()

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reset Parameters", command=self.reset_parameters)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Calculate Static Parameters", command=self.calculate_static_parameters)
        analysis_menu.add_command(label="Show Torque-Speed Curve", command=self.show_torque_speed_curve)
        analysis_menu.add_command(label="Show Efficiency Curve", command=self.show_efficiency_curve)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_main_layout(self):
        """Create main application layout"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Static Analysis
        self.static_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.static_frame, text="Static Analysis")
        self.create_static_analysis_tab()

        # Tab 2: Dynamic Simulation
        self.dynamic_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dynamic_frame, text="Dynamic Simulation")
        self.create_dynamic_simulation_tab()

        # Tab 3: Motor Parameters
        self.params_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.params_frame, text="Motor Parameters")
        self.create_parameters_tab()

    def create_static_analysis_tab(self):
        """Create static analysis tab"""
        # Left panel - Input
        left_frame = ttk.LabelFrame(self.static_frame, text="Input Parameters", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Basic parameters
        row = 0
        ttk.Label(left_frame, text="Output Power (HP):").grid(row=row, column=0, sticky='w', pady=2)
        self.power_entry = ttk.Entry(left_frame, width=15)
        self.power_entry.insert(0, str(self.params.power_hp))
        self.power_entry.grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(left_frame, text="Number of Poles:").grid(row=row, column=0, sticky='w', pady=2)
        self.poles_entry = ttk.Entry(left_frame, width=15)
        self.poles_entry.insert(0, str(self.params.poles))
        self.poles_entry.grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(left_frame, text="Frequency (Hz):").grid(row=row, column=0, sticky='w', pady=2)
        self.freq_entry = ttk.Entry(left_frame, width=15)
        self.freq_entry.insert(0, str(self.params.frequency))
        self.freq_entry.grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(left_frame, text="Full-Load Slip (%):").grid(row=row, column=0, sticky='w', pady=2)
        self.slip_entry = ttk.Entry(left_frame, width=15)
        self.slip_entry.insert(0, str(self.params.slip * 100))
        self.slip_entry.grid(row=row, column=1, pady=2)

        row += 1
        ttk.Label(left_frame, text="Friction & Windage (%):").grid(row=row, column=0, sticky='w', pady=2)
        self.fw_entry = ttk.Entry(left_frame, width=15)
        self.fw_entry.insert(0, str(self.params.friction_windage_percent))
        self.fw_entry.grid(row=row, column=1, pady=2)

        row += 1
        ttk.Button(left_frame, text="Calculate", command=self.calculate_static_parameters).grid(
            row=row, column=0, columnspan=2, pady=10)

        # Right panel - Results
        right_frame = ttk.LabelFrame(self.static_frame, text="Calculation Results", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(right_frame, width=60, height=30, wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True)

        # Configure grid weights
        self.static_frame.grid_columnconfigure(0, weight=1)
        self.static_frame.grid_columnconfigure(1, weight=2)
        self.static_frame.grid_rowconfigure(0, weight=1)

    def create_dynamic_simulation_tab(self):
        """Create dynamic simulation tab"""
        # Control panel
        control_frame = ttk.LabelFrame(self.dynamic_frame, text="Simulation Controls", padding=10)
        control_frame.pack(side='top', fill='x', padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side='left', padx=5)

        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_simulation, width=12)
        self.start_btn.pack(side='left', padx=2)

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, width=12, state='disabled')
        self.stop_btn.pack(side='left', padx=2)

        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self.reset_simulation, width=12)
        self.reset_btn.pack(side='left', padx=2)

        # Method selection
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(side='left', padx=20)

        ttk.Label(method_frame, text="Solver:").pack(side='left', padx=5)
        ttk.Radiobutton(method_frame, text="RK45", variable=self.sim_method, value='rk45').pack(side='left')
        ttk.Radiobutton(method_frame, text="Euler", variable=self.sim_method, value='euler').pack(side='left')

        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Ready", foreground='green')
        self.status_label.pack(side='right', padx=10)

        # Plots frame
        plots_frame = ttk.Frame(self.dynamic_frame)
        plots_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 8))

        self.ax1 = self.fig.add_subplot(3, 2, 1)
        self.ax2 = self.fig.add_subplot(3, 2, 2)
        self.ax3 = self.fig.add_subplot(3, 2, 3)
        self.ax4 = self.fig.add_subplot(3, 2, 4)
        self.ax5 = self.fig.add_subplot(3, 2, 5)
        self.ax6 = self.fig.add_subplot(3, 2, 6)

        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Initialize plots
        self.initialize_plots()

    def create_parameters_tab(self):
        """Create motor parameters tab with sliders"""
        # Create scrollable frame
        canvas = tk.Canvas(self.params_frame)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Basic parameters
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Parameters", padding=10)
        basic_frame.pack(fill='x', padx=10, pady=5)

        self.create_slider(basic_frame, "Output Power (HP)", 1, 500, self.params.power_hp,
                          lambda v: setattr(self.params, 'power_hp', v))

        self.create_slider(basic_frame, "Voltage (V)", 100, 1000, self.params.voltage,
                          lambda v: setattr(self.params, 'voltage', v))

        self.create_slider(basic_frame, "Frequency (Hz)", 30, 100, self.params.frequency,
                          lambda v: setattr(self.params, 'frequency', v))

        self.create_slider(basic_frame, "Full-Load Slip (%)", 0.1, 15, self.params.slip * 100,
                          lambda v: setattr(self.params, 'slip', v / 100), resolution=0.1)

        # Electrical parameters
        elec_frame = ttk.LabelFrame(scrollable_frame, text="Electrical Parameters", padding=10)
        elec_frame.pack(fill='x', padx=10, pady=5)

        self.create_slider(elec_frame, "Stator Resistance (Ω)", 0.1, 5, self.params.stator_resistance,
                          lambda v: setattr(self.params, 'stator_resistance', v), resolution=0.1)

        self.create_slider(elec_frame, "Rotor Resistance (Ω)", 0.1, 5, self.params.rotor_resistance,
                          lambda v: setattr(self.params, 'rotor_resistance', v), resolution=0.1)

        self.create_slider(elec_frame, "Stator Leakage Reactance (Ω)", 0.5, 10, self.params.stator_leakage,
                          lambda v: setattr(self.params, 'stator_leakage', v), resolution=0.1)

        self.create_slider(elec_frame, "Rotor Leakage Reactance (Ω)", 0.5, 10, self.params.rotor_leakage,
                          lambda v: setattr(self.params, 'rotor_leakage', v), resolution=0.1)

        self.create_slider(elec_frame, "Magnetizing Reactance (Ω)", 10, 100, self.params.magnetizing_reactance,
                          lambda v: setattr(self.params, 'magnetizing_reactance', v))

        # Mechanical parameters
        mech_frame = ttk.LabelFrame(scrollable_frame, text="Mechanical Parameters", padding=10)
        mech_frame.pack(fill='x', padx=10, pady=5)

        self.create_slider(mech_frame, "Moment of Inertia (kg⋅m²)", 0.1, 5, self.params.moment_of_inertia,
                          lambda v: setattr(self.params, 'moment_of_inertia', v), resolution=0.1)

        self.create_slider(mech_frame, "Load Torque (N⋅m)", 0, 500, self.params.load_torque,
                          lambda v: setattr(self.params, 'load_torque', v))

        self.create_slider(mech_frame, "Friction & Windage (%)", 0, 10, self.params.friction_windage_percent,
                          lambda v: setattr(self.params, 'friction_windage_percent', v), resolution=0.1)

        # Update button
        ttk.Button(scrollable_frame, text="Update Calculator",
                  command=self.update_calculator).pack(pady=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_slider(self, parent, label, from_, to, initial, command, resolution=1):
        """Create a labeled slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=5)

        ttk.Label(frame, text=label, width=35).pack(side='left')

        value_label = ttk.Label(frame, text=f"{initial:.2f}", width=10)
        value_label.pack(side='right')

        def on_change(val):
            value_label.config(text=f"{float(val):.2f}")
            command(float(val))

        slider = ttk.Scale(frame, from_=from_, to=to, orient='horizontal',
                          command=on_change, value=initial)
        slider.pack(side='right', fill='x', expand=True, padx=5)

    def update_calculator(self):
        """Update calculator with new parameters"""
        self.calculator = InductionMotorCalculator(self.params)
        self.simulator = DynamicSimulator(self.params)
        messagebox.showinfo("Success", "Calculator updated with new parameters!")

    def calculate_static_parameters(self):
        """Calculate and display static motor parameters"""
        try:
            # Update parameters from entries
            self.params.power_hp = float(self.power_entry.get())
            self.params.poles = int(self.poles_entry.get())
            self.params.frequency = float(self.freq_entry.get())
            self.params.slip = float(self.slip_entry.get()) / 100
            self.params.friction_windage_percent = float(self.fw_entry.get())

            # Update calculator
            self.calculator = InductionMotorCalculator(self.params)

            # Calculate
            results = self.calculator.calculate_basic_parameters()

            # Display results
            self.results_text.delete(1.0, tk.END)

            output = "=" * 60 + "\n"
            output += "INDUCTION MOTOR ANALYSIS - STATIC PARAMETERS\n"
            output += "=" * 60 + "\n\n"

            output += "PROBLEM STATEMENT:\n"
            output += f"A {self.params.power_hp} HP, {self.params.poles}-pole, three-phase induction motor\n"
            output += f"Friction and windage losses: {self.params.friction_windage_percent}% of output\n"
            output += f"Full-load slip: {self.params.slip * 100}%\n\n"

            output += "=" * 60 + "\n"
            output += "SOLUTION:\n"
            output += "=" * 60 + "\n\n"

            output += "(a) ROTOR I²R LOSS:\n"
            output += f"    Output Power:           {results['output_power_w']:.2f} W\n"
            output += f"    Friction & Windage:     {results['friction_windage_w']:.2f} W\n"
            output += f"    Mechanical Power:       {results['mech_power_w']:.2f} W\n"
            output += f"    \n"
            output += f"    Rotor I²R Loss = (s/(1-s)) × P_mech\n"
            output += f"    Rotor I²R Loss = ({self.params.slip:.3f}/{1-self.params.slip:.3f}) × {results['mech_power_w']:.2f}\n"
            output += f"    ╔═══════════════════════════════════════════╗\n"
            output += f"    ║  Rotor I²R Loss = {results['rotor_loss_w']:.2f} W        ║\n"
            output += f"    ╚═══════════════════════════════════════════╝\n\n"

            output += "(b) ROTOR INPUT (AIR-GAP POWER):\n"
            output += f"    Rotor Input = Mechanical Power + Rotor Loss\n"
            output += f"    Rotor Input = {results['mech_power_w']:.2f} + {results['rotor_loss_w']:.2f}\n"
            output += f"    ╔═══════════════════════════════════════════╗\n"
            output += f"    ║  Rotor Input = {results['rotor_input_w']:.2f} W        ║\n"
            output += f"    ╚═══════════════════════════════════════════╝\n\n"

            output += "(c) OUTPUT TORQUE:\n"
            output += f"    Synchronous Speed:      {results['sync_speed_rpm']:.2f} RPM\n"
            output += f"    Rotor Speed:            {results['rotor_speed_rpm']:.2f} RPM\n"
            output += f"    Rotor Speed (rad/s):    {results['rotor_speed_rad_s']:.2f} rad/s\n"
            output += f"    \n"
            output += f"    Output Torque = P_mech / ω_rotor\n"
            output += f"    Output Torque = {results['mech_power_w']:.2f} / {results['rotor_speed_rad_s']:.2f}\n"
            output += f"    ╔═══════════════════════════════════════════╗\n"
            output += f"    ║  Output Torque = {results['output_torque_nm']:.2f} N⋅m      ║\n"
            output += f"    ╚═══════════════════════════════════════════╝\n\n"

            output += "=" * 60 + "\n"
            output += "ADDITIONAL PARAMETERS:\n"
            output += "=" * 60 + "\n"
            output += f"Slip:                     {results['slip_percent']:.2f}%\n"
            output += f"Synchronous Speed:        {results['sync_speed_rpm']:.2f} RPM\n"
            output += f"                          {results['sync_speed_rad_s']:.2f} rad/s\n"
            output += f"Rotor Speed:              {results['rotor_speed_rpm']:.2f} RPM\n"
            output += f"                          {results['rotor_speed_rad_s']:.2f} rad/s\n"
            output += f"Power Factor:             ~0.85 (typical)\n"

            # Calculate efficiency
            efficiency = self.calculator.calculate_efficiency(self.params.slip)
            output += f"Estimated Efficiency:     {efficiency:.2f}%\n"

            self.results_text.insert(1.0, output)

        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {str(e)}")

    def show_torque_speed_curve(self):
        """Display torque-speed characteristic curve"""
        slip_range = np.linspace(0.001, 1.0, 100)
        speeds, torques = self.calculator.calculate_torque_speed_curve(slip_range)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(speeds, torques, 'b-', linewidth=2)
        ax.axvline(x=speeds[len(speeds)//20], color='r', linestyle='--', label='Operating Point')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Speed (RPM)', fontsize=12)
        ax.set_ylabel('Torque (N⋅m)', fontsize=12)
        ax.set_title('Induction Motor Torque-Speed Characteristic', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def show_efficiency_curve(self):
        """Display efficiency vs slip curve"""
        slip_range = np.linspace(0.01, 0.2, 50)
        efficiencies = [self.calculator.calculate_efficiency(s) for s in slip_range]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(slip_range * 100, efficiencies, 'g-', linewidth=2)
        ax.axvline(x=self.params.slip * 100, color='r', linestyle='--', label='Rated Slip')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Slip (%)', fontsize=12)
        ax.set_ylabel('Efficiency (%)', fontsize=12)
        ax.set_title('Motor Efficiency vs Slip', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def initialize_plots(self):
        """Initialize dynamic simulation plots"""
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Speed (RPM)')
        self.ax1.set_title('Rotor Speed vs Time')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Torque (N⋅m)')
        self.ax2.set_title('Electromagnetic Torque vs Time')
        self.ax2.grid(True, alpha=0.3)

        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Current (A)')
        self.ax3.set_title('Rotor Current vs Time')
        self.ax3.grid(True, alpha=0.3)

        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Power (kW)')
        self.ax4.set_title('Mechanical Power vs Time')
        self.ax4.grid(True, alpha=0.3)

        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('Slip (%)')
        self.ax5.set_title('Slip vs Time')
        self.ax5.grid(True, alpha=0.3)

        self.ax6.set_xlabel('Speed (RPM)')
        self.ax6.set_ylabel('Torque (N⋅m)')
        self.ax6.set_title('Torque-Speed Trajectory')
        self.ax6.grid(True, alpha=0.3)

        self.canvas.draw()

    def update_plots(self):
        """Update dynamic simulation plots"""
        if len(self.simulator.history['time']) < 2:
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()

        # Plot data
        time = self.simulator.history['time']

        self.ax1.plot(time, self.simulator.history['omega'], 'b-', linewidth=2)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Speed (RPM)')
        self.ax1.set_title('Rotor Speed vs Time')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.plot(time, self.simulator.history['torque'], 'r-', linewidth=2)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Torque (N⋅m)')
        self.ax2.set_title('Electromagnetic Torque vs Time')
        self.ax2.grid(True, alpha=0.3)

        self.ax3.plot(time, self.simulator.history['current'], 'g-', linewidth=2)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Current (A)')
        self.ax3.set_title('Rotor Current vs Time')
        self.ax3.grid(True, alpha=0.3)

        self.ax4.plot(time, self.simulator.history['power'], 'm-', linewidth=2)
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Power (kW)')
        self.ax4.set_title('Mechanical Power vs Time')
        self.ax4.grid(True, alpha=0.3)

        self.ax5.plot(time, self.simulator.history['slip'], 'c-', linewidth=2)
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('Slip (%)')
        self.ax5.set_title('Slip vs Time')
        self.ax5.grid(True, alpha=0.3)

        self.ax6.plot(self.simulator.history['omega'], self.simulator.history['torque'], 'b-', linewidth=2)
        self.ax6.set_xlabel('Speed (RPM)')
        self.ax6.set_ylabel('Torque (N⋅m)')
        self.ax6.set_title('Torque-Speed Trajectory')
        self.ax6.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def start_simulation(self):
        """Start dynamic simulation"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="Status: Running", foreground='green')

            # Start simulation thread
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()

    def stop_simulation(self):
        """Stop dynamic simulation"""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Stopped", foreground='red')

    def reset_simulation(self):
        """Reset simulation"""
        was_running = self.is_running
        if was_running:
            self.stop_simulation()
            time.sleep(0.2)

        self.simulator.reset_simulation()
        self.initialize_plots()
        self.status_label.config(text="Status: Reset", foreground='blue')

    def run_simulation(self):
        """Simulation loop (runs in separate thread)"""
        update_interval = 0.05  # Update plots every 50ms
        last_update = time.time()

        while self.is_running:
            # Simulate one step
            self.simulator.simulate_step(self.dt, method=self.sim_method.get())

            # Update plots periodically
            current_time = time.time()
            if current_time - last_update > update_interval:
                self.root.after(0, self.update_plots)
                last_update = current_time

            # Control simulation speed
            time.sleep(self.dt / 10)  # Run faster than real-time

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.params = MotorParameters()
        self.calculator = InductionMotorCalculator(self.params)
        self.simulator = DynamicSimulator(self.params)

        # Update entry fields
        self.power_entry.delete(0, tk.END)
        self.power_entry.insert(0, str(self.params.power_hp))

        self.poles_entry.delete(0, tk.END)
        self.poles_entry.insert(0, str(self.params.poles))

        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, str(self.params.frequency))

        self.slip_entry.delete(0, tk.END)
        self.slip_entry.insert(0, str(self.params.slip * 100))

        self.fw_entry.delete(0, tk.END)
        self.fw_entry.insert(0, str(self.params.friction_windage_percent))

        messagebox.showinfo("Reset", "Parameters reset to default values")

    def show_about(self):
        """Show about dialog"""
        about_text = """
Advanced Induction Motor Simulator
Version 1.0

Features:
• Static parameter calculation
• Dynamic motor simulation
• Real-time ODE solvers (RK45, Euler)
• Torque-speed characteristics
• Efficiency analysis
• Interactive parameter adjustment

Developed for electrical engineering education
and practical motor analysis.
        """
        messagebox.showinfo("About", about_text)

    def on_window_resize(self, event):
        """Handle window resize for auto-scaling"""
        if hasattr(self, 'canvas'):
            # Redraw canvas to fit new size
            self.canvas.draw()


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = InductionMotorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
