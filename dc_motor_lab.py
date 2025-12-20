"""
DC Motor Laboratory
--------------------
A comprehensive Python + Tkinter application for analyzing a 4-pole lap wound DC motor
with educational controls, steady-state calculations, and dynamic simulation.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox

matplotlib.use("Agg")


@dataclass
class DCMotorInputs:
    """Nameplate and configuration inputs for the lap-wound DC motor."""

    poles: int = 4
    conductors: int = 540
    flux_per_pole_wb: float = 0.025  # 25 mWb
    speed_rpm: float = 1000.0
    supply_voltage: float = 230.0
    armature_resistance: float = 0.8
    mechanical_inertia: float = 0.35
    viscous_friction: float = 0.01
    inductance: float = 0.05
    torque_constant: float = 1.1
    load_torque: float = 8.0


@dataclass
class SteadyStateResults:
    """Steady-state open calculations for the DC motor."""

    induced_emf: float
    armature_current: float
    stray_losses: float
    lost_torque: float


@dataclass
class SimulationState:
    """Container for a simulation sample point."""

    time: float
    speed_rpm: float
    armature_current: float
    torque_nm: float


class Integrator:
    """Basic integration helpers supporting Euler and RK45 (Bogacki-Shampine)."""

    @staticmethod
    def euler(step: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float) -> np.ndarray:
        return y + h * step(t, y)

    @staticmethod
    def rk45(step: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float) -> np.ndarray:
        # Bogacki–Shampine 3(2) method (cash-karp style lightweight)
        k1 = step(t, y)
        k2 = step(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = step(t + 0.75 * h, y + 0.75 * h * k2)
        return y + h * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)


class DCMotorModel:
    """Dynamic model and steady-state solver for the DC motor."""

    def __init__(self, inputs: DCMotorInputs):
        self.inputs = inputs

    def compute_steady_state(self) -> SteadyStateResults:
        data = self.inputs
        area_paths = data.poles  # lap winding => A = poles
        induced_emf = (
            data.poles
            * data.flux_per_pole_wb
            * data.conductors
            * data.speed_rpm
            / (60 * area_paths)
        )
        armature_current = (data.supply_voltage - induced_emf) / data.armature_resistance
        electrical_power = data.supply_voltage * armature_current
        copper_loss = (armature_current ** 2) * data.armature_resistance
        stray_losses = electrical_power - copper_loss
        omega = 2 * math.pi * data.speed_rpm / 60
        lost_torque = stray_losses / omega if omega else 0.0
        return SteadyStateResults(
            induced_emf=induced_emf,
            armature_current=armature_current,
            stray_losses=stray_losses,
            lost_torque=lost_torque,
        )

    @staticmethod
    def solve_textbook_shunt_case():
        """Solve the requested 4-pole shunt motor example.

        The motor takes 22 A from a 220 V supply, has 0.5 Ω armature resistance,
        100 Ω shunt field, 300 lap-connected conductors, and 20 mWb/pole.
        Returns speed (rpm) and developed torque (N·m).
        """

        supply_voltage = 220.0
        total_current = 22.0
        armature_resistance = 0.5
        shunt_resistance = 100.0
        poles = 4
        conductors = 300
        flux = 0.02  # 20 mWb
        paths = poles  # lap wound

        field_current = supply_voltage / shunt_resistance
        armature_current = total_current - field_current
        back_emf = supply_voltage - armature_current * armature_resistance

        speed_rpm = 60 * paths * back_emf / (poles * flux * conductors)
        torque_nm = (poles * conductors * flux * armature_current) / (2 * math.pi * paths)
        return speed_rpm, torque_nm

    def state_derivative(self, t: float, state: np.ndarray, supply: float, load_torque: float) -> np.ndarray:
        # state[0] = armature current (A)
        # state[1] = angular speed (rad/s)
        data = self.inputs
        i_a, omega = state
        k = data.torque_constant * data.flux_per_pole_wb
        back_emf = k * omega
        di_dt = (supply - back_emf - data.armature_resistance * i_a) / data.inductance
        torque = k * i_a
        domega_dt = (torque - load_torque - data.viscous_friction * omega) / data.mechanical_inertia
        return np.array([di_dt, domega_dt])


class SimulationEngine:
    """Threaded simulation engine for real-time plotting."""

    def __init__(self, model: DCMotorModel):
        self.model = model
        self.running = False
        self.samples: List[SimulationState] = []
        self.integrator = Integrator.rk45
        self.solver_name = "RK45"
        self.step_size = 0.01
        self.max_time = 5.0
        self._thread: threading.Thread | None = None

    def set_integrator(self, name: str):
        if name.lower() == "euler":
            self.integrator = Integrator.euler
            self.solver_name = "Euler"
        else:
            self.integrator = Integrator.rk45
            self.solver_name = "RK45"

    def start(self, supply: float, load_torque: float, callback: Callable[[SimulationState], None]):
        if self.running:
            return
        self.running = True
        self.samples.clear()
        self._thread = threading.Thread(
            target=self._run, args=(supply, load_torque, callback), daemon=True
        )
        self._thread.start()

    def stop(self):
        self.running = False

    def reset(self):
        self.stop()
        self.samples.clear()

    def _run(self, supply: float, load_torque: float, callback: Callable[[SimulationState], None]):
        t = 0.0
        state = np.array([0.0, 0.0])
        while self.running and t <= self.max_time:
            deriv = lambda current_t, current_state: self.model.state_derivative(
                current_t, current_state, supply, load_torque
            )
            state = self.integrator(deriv, t, state, self.step_size)
            t += self.step_size
            speed_rpm = state[1] * 60 / (2 * math.pi)
            torque = self.model.inputs.torque_constant * self.model.inputs.flux_per_pole_wb * state[0]
            sample = SimulationState(time=t, speed_rpm=speed_rpm, armature_current=state[0], torque_nm=torque)
            self.samples.append(sample)
            callback(sample)
            time.sleep(self.step_size * 0.5)
        self.running = False


class DCMotorLabApp(tk.Tk):
    """Tkinter front-end for steady-state and dynamic DC motor exploration."""

    def __init__(self):
        super().__init__()
        self.title("DC Motor Laboratory")
        self.geometry("1200x800")
        self.minsize(900, 700)

        self.inputs = DCMotorInputs()
        self.model = DCMotorModel(self.inputs)
        self.engine = SimulationEngine(self.model)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._build_layout()
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.bind("<Configure>", self._on_resize)

    def _build_layout(self):
        self._build_calculator_tab()
        self._build_simulation_tab()
        self._build_advanced_tab()

    # --- Calculator Tab ---
    def _build_calculator_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Main Menu & Calculator")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        inputs_frame = ttk.LabelFrame(tab, text="Input Parameters")
        outputs_frame = ttk.LabelFrame(tab, text="Calculated Results")
        controls_frame = ttk.LabelFrame(tab, text="Controls & Solved Example")

        inputs_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        outputs_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        fields = [
            ("Poles", "poles", self.inputs.poles),
            ("Conductors", "conductors", self.inputs.conductors),
            ("Flux per pole (Wb)", "flux_per_pole_wb", self.inputs.flux_per_pole_wb),
            ("Speed (rpm)", "speed_rpm", self.inputs.speed_rpm),
            ("Supply voltage (V)", "supply_voltage", self.inputs.supply_voltage),
            ("Armature resistance (Ohm)", "armature_resistance", self.inputs.armature_resistance),
        ]
        self.input_vars = {}
        for idx, (label, attr, value) in enumerate(fields):
            ttk.Label(inputs_frame, text=label).grid(row=idx, column=0, sticky="w", padx=5, pady=5)
            var = tk.StringVar(value=str(value))
            self.input_vars[attr] = var
            ttk.Entry(inputs_frame, textvariable=var, width=20).grid(row=idx, column=1, sticky="ew", padx=5, pady=5)
        inputs_frame.columnconfigure(1, weight=1)

        self.output_labels = {}
        for idx, label in enumerate([
            "Induced EMF (V)",
            "Armature Current (A)",
            "Stray Losses (W)",
            "Lost Torque (N·m)",
        ]):
            ttk.Label(outputs_frame, text=label).grid(row=idx, column=0, sticky="w", padx=5, pady=5)
            value_label = ttk.Label(outputs_frame, text="-")
            value_label.grid(row=idx, column=1, sticky="e", padx=5, pady=5)
            self.output_labels[label] = value_label
        outputs_frame.columnconfigure(1, weight=1)

        calc_button = ttk.Button(controls_frame, text="Compute", command=self._on_compute)
        calc_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        reset_button = ttk.Button(controls_frame, text="Reset", command=self._reset_inputs)
        reset_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        example_button = ttk.Button(
            controls_frame, text="Load Textbook Case", command=self._apply_textbook_case
        )
        example_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self._create_sliders(controls_frame)
        controls_frame.columnconfigure(4, weight=1)

        example_speed, example_torque = DCMotorModel.solve_textbook_shunt_case()
        ttk.Label(
            controls_frame,
            text=(
                f"Solved example (4-pole shunt, 220 V/22 A):\n"
                f"Speed ≈ {example_speed:.0f} rpm | Torque ≈ {example_torque:.2f} N·m"
            ),
            justify="left",
        ).grid(row=2, column=0, columnspan=5, sticky="w", padx=5, pady=5)

    def _create_sliders(self, frame: ttk.Frame):
        ttk.Label(frame, text="Adjust Supply Voltage").grid(row=1, column=0, sticky="w")
        self.supply_slider = tk.Scale(frame, from_=180, to=260, orient=tk.HORIZONTAL, resolution=1,
                                      command=self._update_supply_from_slider)
        self.supply_slider.set(self.inputs.supply_voltage)
        self.supply_slider.grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, text="Adjust Load Torque").grid(row=2, column=0, sticky="w")
        self.load_slider = tk.Scale(frame, from_=0, to=20, orient=tk.HORIZONTAL, resolution=0.5,
                                    command=self._update_load_from_slider)
        self.load_slider.set(self.inputs.load_torque)
        self.load_slider.grid(row=2, column=1, sticky="ew")

    def _update_supply_from_slider(self, value: str):
        self.input_vars["supply_voltage"].set(value)

    def _update_load_from_slider(self, value: str):
        try:
            self.inputs.load_torque = float(value)
        except ValueError:
            pass

    def _reset_inputs(self):
        self.inputs = DCMotorInputs()
        for key, var in self.input_vars.items():
            var.set(str(getattr(self.inputs, key)))
        self.supply_slider.set(self.inputs.supply_voltage)
        self.load_slider.set(self.inputs.load_torque)
        for label in self.output_labels.values():
            label.config(text="-")
        self.model = DCMotorModel(self.inputs)
        self.engine = SimulationEngine(self.model)

    def _apply_textbook_case(self):
        """Populate fields with the provided 4-pole shunt motor test case and solve it."""

        example_inputs = {
            "poles": 4,
            "conductors": 300,
            "flux_per_pole_wb": 0.02,
            "speed_rpm": 2100.0,
            "supply_voltage": 220.0,
            "armature_resistance": 0.5,
        }
        for key, value in example_inputs.items():
            self.input_vars[key].set(str(value))
        self.inputs.load_torque = 10.0
        self.supply_slider.set(example_inputs["supply_voltage"])
        self.load_slider.set(self.inputs.load_torque)
        self._on_compute()

    def _on_compute(self):
        try:
            for key, var in self.input_vars.items():
                setattr(self.inputs, key, float(var.get()))
            self.model = DCMotorModel(self.inputs)
        except ValueError:
            messagebox.showerror("Input error", "Please provide numeric inputs")
            return
        results = self.model.compute_steady_state()
        self.output_labels["Induced EMF (V)"].config(text=f"{results.induced_emf:.2f}")
        self.output_labels["Armature Current (A)"].config(text=f"{results.armature_current:.2f}")
        self.output_labels["Stray Losses (W)"].config(text=f"{results.stray_losses:.2f}")
        self.output_labels["Lost Torque (N·m)"].config(text=f"{results.lost_torque:.2f}")

    # --- Simulation Tab ---
    def _build_simulation_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dynamic Simulation")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        main_frame = ttk.Frame(tab)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_speed = self.fig.add_subplot(311)
        self.ax_current = self.fig.add_subplot(312)
        self.ax_torque = self.fig.add_subplot(313)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        control_frame = ttk.LabelFrame(main_frame, text="Simulation Controls")
        control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        for i in range(4):
            control_frame.grid_rowconfigure(i, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="Supply (V)").grid(row=0, column=0, sticky="w")
        self.sim_supply_var = tk.DoubleVar(value=self.inputs.supply_voltage)
        ttk.Entry(control_frame, textvariable=self.sim_supply_var).grid(row=0, column=1, sticky="ew")

        ttk.Label(control_frame, text="Load Torque (N·m)").grid(row=1, column=0, sticky="w")
        self.sim_load_var = tk.DoubleVar(value=self.inputs.load_torque)
        ttk.Entry(control_frame, textvariable=self.sim_load_var).grid(row=1, column=1, sticky="ew")

        ttk.Label(control_frame, text="Step size (s)").grid(row=2, column=0, sticky="w")
        self.step_var = tk.DoubleVar(value=self.engine.step_size)
        ttk.Entry(control_frame, textvariable=self.step_var).grid(row=2, column=1, sticky="ew")

        ttk.Label(control_frame, text="Integrator").grid(row=3, column=0, sticky="w")
        self.integrator_choice = ttk.Combobox(control_frame, values=["RK45", "Euler"], state="readonly")
        self.integrator_choice.set("RK45")
        self.integrator_choice.grid(row=3, column=1, sticky="ew")

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Button(button_frame, text="Start", command=self._start_simulation).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.engine.stop).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Reset", command=self._reset_simulation).grid(row=0, column=2, padx=5)

    def _start_simulation(self):
        try:
            step = float(self.step_var.get())
            self.engine.step_size = max(0.001, step)
            supply = float(self.sim_supply_var.get())
            load = float(self.sim_load_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Simulation parameters must be numeric")
            return

        self.engine.set_integrator(self.integrator_choice.get())
        self._reset_plots()

        def callback(sample: SimulationState):
            self._update_plots(sample)

        self.engine.start(supply, load, callback)

    def _reset_simulation(self):
        self.engine.reset()
        self._reset_plots()

    def _reset_plots(self):
        for ax in [self.ax_speed, self.ax_current, self.ax_torque]:
            ax.clear()
        self.ax_speed.set_ylabel("Speed (rpm)")
        self.ax_current.set_ylabel("Current (A)")
        self.ax_torque.set_ylabel("Torque (N·m)")
        self.ax_torque.set_xlabel("Time (s)")
        self.canvas.draw()

    def _update_plots(self, sample: SimulationState):
        times = [s.time for s in self.engine.samples]
        speeds = [s.speed_rpm for s in self.engine.samples]
        currents = [s.armature_current for s in self.engine.samples]
        torques = [s.torque_nm for s in self.engine.samples]
        self.ax_speed.clear()
        self.ax_current.clear()
        self.ax_torque.clear()
        self.ax_speed.plot(times, speeds, color="blue")
        self.ax_current.plot(times, currents, color="green")
        self.ax_torque.plot(times, torques, color="red")
        self.ax_speed.set_ylabel("Speed (rpm)")
        self.ax_current.set_ylabel("Current (A)")
        self.ax_torque.set_ylabel("Torque (N·m)")
        self.ax_torque.set_xlabel("Time (s)")
        self.canvas.draw_idle()

    # --- Advanced Tab ---
    def _build_advanced_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Advanced Analysis")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        info = (
            "Advanced electrical engineering insights:\n"
            "- Real-time differential equations using adjustable integrators (RK45 or Euler).\n"
            "- Lap wound parallel paths assumed equal to number of poles.\n"
            "- Torque constant multiplies with flux to estimate electromagnetic torque.\n"
            "- Vary inductance, friction, and inertia for practical what-if studies.\n"
            "- Textbook shunt-motor case solved automatically for quick validation."
        )
        ttk.Label(tab, text=info, justify="left").grid(row=0, column=0, sticky="w", padx=10, pady=10)

        data_frame = ttk.LabelFrame(tab, text="Deep Dive Parameters")
        data_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        data_frame.grid_columnconfigure(1, weight=1)

        advanced_fields = [
            ("Mechanical inertia (kg·m²)", "mechanical_inertia"),
            ("Viscous friction (N·m·s)", "viscous_friction"),
            ("Armature inductance (H)", "inductance"),
            ("Torque constant", "torque_constant"),
            ("Load torque (N·m)", "load_torque"),
        ]
        for idx, (label, attr) in enumerate(advanced_fields):
            ttk.Label(data_frame, text=label).grid(row=idx, column=0, sticky="w", padx=5, pady=5)
            var = tk.DoubleVar(value=getattr(self.inputs, attr))
            entry = ttk.Entry(data_frame, textvariable=var)
            entry.grid(row=idx, column=1, sticky="ew", padx=5, pady=5)
            entry.bind("<FocusOut>", lambda e, key=attr, v=var: self._update_advanced(key, v))

        eq_frame = ttk.LabelFrame(tab, text="Dynamic Equations & Time Constants")
        eq_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        eq_frame.grid_columnconfigure(0, weight=1)

        equations = (
            "Electrical: L di/dt = V - E_b - R_a i,  E_b = k_e ω\n"
            "Mechanical: J dω/dt = k_t Φ i - T_load - B ω\n"
            "Torque: T = k_t Φ i_a   |   Speed: n = 60 ω / (2π)"
        )
        self.eq_label = ttk.Label(eq_frame, text=equations, justify="left")
        self.eq_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.time_constants_var = tk.StringVar()
        ttk.Label(eq_frame, textvariable=self.time_constants_var, justify="left").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        self._update_time_constants()

    def _update_advanced(self, key: str, var: tk.DoubleVar):
        try:
            setattr(self.inputs, key, float(var.get()))
            self.model = DCMotorModel(self.inputs)
            self.engine = SimulationEngine(self.model)
            self._update_time_constants()
        except ValueError:
            messagebox.showerror("Input error", "Advanced parameter must be numeric")

    def _update_time_constants(self):
        electrical_tau = self.inputs.inductance / self.inputs.armature_resistance
        mechanical_tau = self.inputs.mechanical_inertia / max(self.inputs.viscous_friction, 1e-6)
        self.time_constants_var.set(
            f"Electrical τ ≈ {electrical_tau:.4f} s | Mechanical τ ≈ {mechanical_tau:.4f} s"
        )

    def _on_resize(self, event):
        width = max(self.winfo_width(), 900)
        height = max(self.winfo_height(), 700)
        # Keep the matplotlib figure responsive to window changes
        if hasattr(self, "fig"):
            self.fig.set_size_inches(width / 180, height / 240, forward=True)
            self.fig.tight_layout()
            if hasattr(self, "canvas"):
                self.canvas.draw_idle()


if __name__ == "__main__":
    app = DCMotorLabApp()
    app.mainloop()
