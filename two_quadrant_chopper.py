#!/usr/bin/env python3
"""
Two-Quadrant DC Chopper Motor Drive — Comprehensive Simulator
==============================================================
A complete educational and engineering tool for analysing a 2-quadrant
chopper-controlled DC motor with regenerative braking.

GIVEN:
  Kb  = 0.1 V/rpm    motor constant
  Ra  = 0.5 Ω        armature resistance
  f   = 100 Hz       chopping frequency  (T = 10 ms)
  La  = very large   ripple-free armature current
  Vs  = 230 V        supply voltage

Part (a) Motoring  [Th1 + D1]  N = 1500 rpm, Ia = 10 A
  Eb   = 0.1 × 1500 = 150 V
  Va   = Eb + Ia·Ra = 155 V
  α    = Va/Vs = 155/230 = 0.6739   ton = 6.739 ms
  Pmech = Eb·Ia = 1500 W   P_Ra = 50 W   Ps = 1550 W

Part (b) Regeneration  [Th2 + D2]  N = 1400 rpm, Ia = −10 A
  Eb   = 0.1 × 1400 = 140 V
  Va   = Eb + Ia·Ra = 140 − 5 = 135 V
  α2   = 1 − Va/Vs = 1 − 135/230 = 0.4130   ton2 = 4.130 ms
  Pmech(braking) = 1400 W   P_Ra = 50 W   P_returned = 1350 W
"""

import math
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
PI = math.pi


def rpm2rad(rpm: float) -> float:
    return rpm * PI / 30.0


def rad2rpm(rad: float) -> float:
    return rad * 30.0 / PI


def kb_to_si(kb_rpm: float) -> float:
    """V/rpm → V·s/rad  (also N·m/A as motor/generator constant)."""
    return kb_rpm * 30.0 / PI


# ===========================================================================
# CALCULATOR
# ===========================================================================

class ChopperCalculator:
    """All analytical calculations for the 2-quadrant chopper/motor system."""

    def __init__(self, vs=230.0, kb=0.1, ra=0.5, f=100.0,
                 la=0.05, j=0.05, b=0.002):
        self.vs = vs
        self.kb = kb          # V/rpm
        self.ra = ra
        self.f = f
        self.T = 1.0 / f      # period  [s]
        self.la = la
        self.j = j
        self.b = b
        self.km = kb_to_si(kb)  # V·s/rad = N·m/A

    # ------------------------------------------------------------------
    def motoring(self, n, ia):
        eb = self.kb * n
        va = eb + ia * self.ra
        alpha = va / self.vs
        ton = alpha * self.T * 1000.0          # ms
        p_mech = eb * ia
        p_ra = ia ** 2 * self.ra
        p_source = va * ia
        omega = rpm2rad(n)
        torque = p_mech / omega if omega > 0 else 0.0
        eff = (p_mech / p_source * 100.0) if p_source > 0 else 0.0
        return dict(eb=eb, va=va, alpha=alpha, alpha_pct=alpha * 100,
                    ton=ton, p_mech=p_mech, p_ra=p_ra,
                    p_source=p_source, torque=torque, efficiency=eff)

    # ------------------------------------------------------------------
    def regeneration(self, n, ia_mag):
        """ia_mag = |Ia|  (the current magnitude; direction = from motor)."""
        eb = self.kb * n
        ia = -ia_mag
        va = eb + ia * self.ra               # va = eb − ia_mag·Ra
        alpha2 = 1.0 - va / self.vs
        ton2 = alpha2 * self.T * 1000.0      # ms
        p_mech = eb * ia_mag                 # mechanical braking power
        p_ra = ia_mag ** 2 * self.ra
        p_returned = va * ia_mag             # power fed back to supply
        omega = rpm2rad(n)
        torque = p_mech / omega if omega > 0 else 0.0
        reff = (p_returned / p_mech * 100.0) if p_mech > 0 else 0.0
        return dict(eb=eb, va=va, alpha2=alpha2, alpha2_pct=alpha2 * 100,
                    ton2=ton2, p_mech=p_mech, p_ra=p_ra,
                    p_returned=p_returned, torque=torque,
                    regen_efficiency=reff)

    # ------------------------------------------------------------------
    def waveform_motoring(self, alpha, cycles=4):
        N = 2000 * cycles
        t = np.linspace(0, cycles * self.T, N)
        va = np.where((t % self.T) < alpha * self.T, self.vs, 0.0)
        ia_ripple = self._ripple_current(t, alpha, "motoring")
        return t * 1000, va, ia_ripple   # t in ms

    def waveform_regen(self, alpha2, cycles=4):
        N = 2000 * cycles
        t = np.linspace(0, cycles * self.T, N)
        va = np.where((t % self.T) < alpha2 * self.T, 0.0, self.vs)
        ia_ripple = self._ripple_current(t, 1.0 - alpha2, "regen")
        return t * 1000, va, ia_ripple

    def _ripple_current(self, t, alpha_eff, mode):
        """Approximate ripple current around mean (finite La)."""
        ia_mean = 10.0
        tau = self.la / self.ra
        ripple = np.zeros_like(t)
        T = self.T
        for i, ti in enumerate(t):
            tc = ti % T
            if tc < alpha_eff * T:
                ripple[i] = (self.vs - ia_mean * self.ra) / self.la * (
                    alpha_eff * T / 2 - tc)
            else:
                ripple[i] = -ia_mean * self.ra / self.la * (
                    (1 - alpha_eff) * T / 2 - (tc - alpha_eff * T))
        if mode == "regen":
            return -ia_mean + ripple * 0.1
        return ia_mean + ripple * 0.1

    # ------------------------------------------------------------------
    def fault_current_transient(self, t_arr, ia0=0.0):
        """Armature short-circuit fault: Va=0 while Eb=const=150V (1500rpm)."""
        # Bus fault: Va = Vs suddenly applied to shorted armature (worst case)
        i_ss = self.vs / self.ra
        tau = self.la / self.ra
        return i_ss * (1 - np.exp(-t_arr / tau))

    # ------------------------------------------------------------------
    def thermal_rise(self, ia, t_arr, r_th=0.8, c_th=200.0):
        tau_th = r_th * c_th
        p_loss = ia ** 2 * self.ra
        t_ss = p_loss * r_th
        return t_ss * (1 - np.exp(-t_arr / tau_th))

    # ------------------------------------------------------------------
    def harmonic_spectrum(self, alpha, n_harmonics=20):
        """Fourier coefficients of chopper output voltage (square wave)."""
        # V_dc = alpha * Vs
        # V_n  = (Vs/(n*pi)) * sqrt(2*(1 - cos(2*pi*n*alpha)))  for n >= 1
        harmon = np.arange(1, n_harmonics + 1)
        v_dc = alpha * self.vs
        v_n = (self.vs / (harmon * PI)) * np.sqrt(
            2 * (1 - np.cos(2 * PI * harmon * alpha)))
        return v_dc, harmon, v_n

    def thd(self, v_n):
        v1 = v_n[0] if len(v_n) > 0 else 1.0
        if v1 < 1e-6:
            return 0.0
        return np.sqrt(np.sum(v_n[1:] ** 2)) / v1 * 100.0


# ===========================================================================
# RK4 INTEGRATION  +  MOTOR ODE
# ===========================================================================

def motor_ode(state, va, km, ra, la, j, b, tl):
    """Returns [dIa/dt, domega/dt]."""
    ia, omega = state
    dia = (va - km * omega - ra * ia) / la
    dom = (km * ia - tl - b * omega) / j
    return np.array([dia, dom])


def rk4(state, va, km, ra, la, j, b, tl, dt):
    k1 = motor_ode(state, va, km, ra, la, j, b, tl)
    k2 = motor_ode(state + 0.5 * dt * k1, va, km, ra, la, j, b, tl)
    k3 = motor_ode(state + 0.5 * dt * k2, va, km, ra, la, j, b, tl)
    k4 = motor_ode(state + dt * k3, va, km, ra, la, j, b, tl)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ===========================================================================
# PID CONTROLLER
# ===========================================================================

class PIDController:
    def __init__(self, kp=2.0, ki=0.8, kd=0.05, dt=1e-3,
                 out_min=0.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.out_min, self.out_max = out_min, out_max
        self._int = 0.0
        self._prev_err = 0.0

    def step(self, error):
        self._int = np.clip(self._int + error * self.dt,
                            self.out_min / max(self.ki, 1e-9),
                            self.out_max / max(self.ki, 1e-9))
        der = (error - self._prev_err) / self.dt
        u = self.kp * error + self.ki * self._int + self.kd * der
        self._prev_err = error
        return float(np.clip(u, self.out_min, self.out_max))

    def reset(self):
        self._int = 0.0
        self._prev_err = 0.0


# ===========================================================================
# FUZZY CONTROLLER  (Mamdani, triangular MFs, COG defuzz)
# ===========================================================================

def _trimf(x, a, b, c):
    if b == a:
        left = 1.0 if x >= a else 0.0
    else:
        left = max(0.0, min(1.0, (x - a) / (b - a)))
    if b == c:
        right = 1.0 if x <= c else 0.0
    else:
        right = max(0.0, min(1.0, (c - x) / (c - b)))
    return min(left, right)


class FuzzyController:
    """Simplified 5×5 Mamdani fuzzy speed controller."""

    _LABELS = ["NL", "NS", "Z", "PS", "PL"]
    _E_CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]   # normalised error
    _DE_CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    _OUT_CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]  # normalised output

    # Rule table [e index][de index] → output label index
    _RULES = [
        [0, 0, 0, 1, 2],  # e=NL
        [0, 0, 1, 2, 3],  # e=NS
        [0, 1, 2, 3, 4],  # e=Z
        [1, 2, 3, 4, 4],  # e=PS
        [2, 3, 4, 4, 4],  # e=PL
    ]

    def __init__(self, e_range=1.0, de_range=1.0, out_scale=0.5):
        self.e_range = e_range
        self.de_range = de_range
        self.out_scale = out_scale

    def _membership(self, val, centers):
        mu = []
        for i, c in enumerate(centers):
            a = centers[i - 1] if i > 0 else c - (centers[1] - centers[0])
            b_c = c
            d = centers[i + 1] if i < len(centers) - 1 else c + (c - centers[-2])
            mu.append(_trimf(val, a, b_c, d))
        return mu

    def step(self, error, derror, dt=1e-3):
        en = np.clip(error / self.e_range, -1.5, 1.5)
        den = np.clip(derror / self.de_range, -1.5, 1.5)
        mu_e = self._membership(en, self._E_CENTERS)
        mu_de = self._membership(den, self._DE_CENTERS)
        num = 0.0
        denom = 0.0
        for i in range(5):
            for j in range(5):
                w = min(mu_e[i], mu_de[j])
                out_center = self._OUT_CENTERS[self._RULES[i][j]]
                num += w * out_center
                denom += w
        if denom < 1e-12:
            return 0.0
        return float(np.clip((num / denom) * self.out_scale, -0.5, 0.5))


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================

class TwoQuadrantApp:
    # ------------------------------------------------------------------
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("2-Quadrant DC Chopper Motor Drive – Comprehensive Simulator")
        self.root.geometry("1400x860")
        self.root.minsize(1024, 640)

        self._sim_running = False
        self._sim_thread = None
        self._sim_data = {}

        # Default parameters as tk variables
        self._vs = tk.DoubleVar(value=230.0)
        self._kb = tk.DoubleVar(value=0.1)
        self._ra = tk.DoubleVar(value=0.5)
        self._f = tk.DoubleVar(value=100.0)
        self._la = tk.DoubleVar(value=0.05)
        self._j = tk.DoubleVar(value=0.05)
        self._b_fric = tk.DoubleVar(value=0.002)
        self._n_m = tk.DoubleVar(value=1500.0)
        self._ia_m = tk.DoubleVar(value=10.0)
        self._n_r = tk.DoubleVar(value=1400.0)
        self._ia_r = tk.DoubleVar(value=10.0)
        self._tl = tk.DoubleVar(value=9.55)
        self._nref = tk.DoubleVar(value=1500.0)
        self._kp = tk.DoubleVar(value=2.0)
        self._ki = tk.DoubleVar(value=0.8)
        self._kd = tk.DoubleVar(value=0.05)

        self._calc = self._make_calc()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab", font=("Helvetica", 9, "bold"))

        self._nb = ttk.Notebook(root)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._build_tabs()

    # ------------------------------------------------------------------
    def _make_calc(self) -> ChopperCalculator:
        return ChopperCalculator(
            vs=self._vs.get(), kb=self._kb.get(), ra=self._ra.get(),
            f=self._f.get(), la=self._la.get(),
            j=self._j.get(), b=self._b_fric.get())

    # ------------------------------------------------------------------
    def _build_tabs(self):
        self._tab_overview()
        self._tab_parameters()
        self._tab_waveforms()
        self._tab_dynamic()
        self._tab_fault()
        self._tab_protection()
        self._tab_speed_ctrl()
        self._tab_thermal()
        self._tab_harmonic()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _scrollable(parent):
        cont = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cont.yview)
        frame = ttk.Frame(cont)
        frame.bind("<Configure>",
                   lambda e: cont.configure(scrollregion=cont.bbox("all")))
        cont.create_window((0, 0), window=frame, anchor="nw")
        cont.configure(yscrollcommand=sb.set)
        cont.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return frame

    @staticmethod
    def _embed_figure(fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return canvas

    @staticmethod
    def _slider_row(parent, label, var, lo, hi, row, fmt=".2f"):
        ttk.Label(parent, text=label, anchor="e", width=28).grid(
            row=row, column=0, sticky="e", padx=4, pady=2)
        sl = ttk.Scale(parent, variable=var, from_=lo, to=hi,
                       orient="horizontal", length=200)
        sl.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        lbl = ttk.Label(parent, width=10, anchor="w")
        lbl.grid(row=row, column=2, sticky="w")
        def _upd(*_):
            lbl.config(text=format(var.get(), fmt))
        var.trace_add("write", _upd)
        _upd()
        return sl

    # ==================================================================
    # TAB 1 – OVERVIEW
    # ==================================================================
    def _tab_overview(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="📋 Overview")
        sf = self._scrollable(frm)
        ttk.Label(sf, text="2-Quadrant DC Chopper Motor Drive",
                  font=("Helvetica", 15, "bold")).pack(pady=(12, 4))

        theory = (
            "═══════════════════════ PROBLEM STATEMENT ═══════════════════════\n"
            "\n"
            "A 2-quadrant chopper controls the speed of a DC motor and enables\n"
            "regenerative braking.\n"
            "\n"
            "GIVEN DATA\n"
            "  Motor constant (Kb)   : 0.1 V/rpm  =  0.9549 V·s/rad\n"
            "  Chopping frequency (f): 100 Hz  →  Period T = 10 ms\n"
            "  Armature resistance   : Ra = 0.5 Ω\n"
            "  Inductance (La)       : Very large  →  ripple-free Ia\n"
            "  Supply voltage        : Vs = 230 V\n"
            "\n"
            "══════════════════════ CIRCUIT DESCRIPTION ═══════════════════════\n"
            "\n"
            "  ┌──────────────────────────────────────────────────────────────┐\n"
            "  │  +Vs ──┬──[Th1]──────────────────┬── Va ──[Ra]──[La]──+    │\n"
            "  │        │                          │                    │    │\n"
            "  │       [D2]       Motor armature   Eb(N)               [Th2] │\n"
            "  │        │                          │                    │    │\n"
            "  │   0V ──┴──[D1]──────────────────┴────────────────────┘    │\n"
            "  └──────────────────────────────────────────────────────────────┘\n"
            "\n"
            "  MOTORING  (Th1 + D1 — Quadrant I):  Va > 0, Ia > 0\n"
            "    • Th1 ON  → Va = Vs = 230 V  (current builds up)\n"
            "    • Th1 OFF → D1 freewheels, Va = 0  (current decays through D1)\n"
            "    • Average Va = α × Vs,  α = ton / T\n"
            "\n"
            "  REGENERATION  (Th2 + D2 — Quadrant II):  Va > 0, Ia < 0\n"
            "    • Th2 ON  → Va = 0  (motor shorted, Ia builds up via Eb)\n"
            "    • Th2 OFF → D2 conducts, Va = Vs  (energy returned to supply)\n"
            "    • Average Va = (1 − α₂) × Vs,  α₂ = ton2 / T\n"
            "\n"
            "══════════════════════════ KEY EQUATIONS ════════════════════════\n"
            "\n"
            "  Back EMF          : Eb = Kb × N\n"
            "  KVL (motor)       : Va = Eb + Ia · Ra  (steady state, large La)\n"
            "  Duty cycle (mot.) : α  = Va / Vs\n"
            "  Duty cycle (regen): α₂ = 1 − Va / Vs\n"
            "  On-time           : ton = α × T    ton2 = α₂ × T\n"
            "\n"
            "  Power (motoring) :\n"
            "    Pmech  = Eb × Ia   [mechanical power developed]\n"
            "    P_Ra   = Ia² × Ra  [copper losses]\n"
            "    Ps     = Va × Ia   [power from supply]\n"
            "    Ps     = Pmech + P_Ra  ✓\n"
            "\n"
            "  Power (regen) :\n"
            "    Pmech  = Eb × |Ia|   [mechanical braking power input]\n"
            "    P_Ra   = Ia² × Ra    [copper losses]\n"
            "    P_ret  = Va × |Ia|   [power returned to supply]\n"
            "    Pmech  = P_Ra + P_ret  ✓\n"
            "\n"
            "═══════════════════════ DYNAMIC MODEL (ODEs) ════════════════════\n"
            "\n"
            "  Electrical:  La · dIa/dt = Va(t) − Km · ω − Ra · Ia\n"
            "  Mechanical:  J  · dω/dt  = Km · Ia − TL − B · ω\n"
            "\n"
            "  Km = Kb_SI = Kb × 60/(2π)  [V·s/rad = N·m/A]\n"
            "  TL = load torque  [N·m],   B = viscous friction  [N·m·s/rad]\n"
            "\n"
            "══════════════════════════ SOLUTIONS ════════════════════════════\n"
            "\n"
            "  Part (a) — MOTORING  (Th1 + D1)\n"
            "    N = 1500 rpm,  Ia = 10 A\n"
            "    Eb   = 0.1 × 1500 = 150 V\n"
            "    Va   = 150 + 10 × 0.5 = 155 V\n"
            "    α    = 155/230 = 0.6739      (67.39 %)\n"
            "    ton  = 0.6739 × 10 ms = 6.739 ms\n"
            "\n"
            "    (ii) Powers:\n"
            "         Pmech  = Eb × Ia  = 150 × 10  = 1500 W\n"
            "         P_Ra   = Ia² × Ra = 100 × 0.5 =   50 W\n"
            "         Ps     = Va × Ia  = 155 × 10  = 1550 W\n"
            "         Check: 1500 + 50 = 1550 W  ✓\n"
            "\n"
            "  Part (b) — REGENERATION  (Th2 + D2)\n"
            "    N = 1400 rpm,  Ia = −10 A\n"
            "    Eb    = 0.1 × 1400 = 140 V\n"
            "    Va    = 140 + (−10) × 0.5 = 135 V\n"
            "    α₂   = 1 − 135/230 = 0.4130  (41.30 %)\n"
            "    ton2  = 0.4130 × 10 ms = 4.130 ms\n"
            "\n"
            "    Powers:\n"
            "         Pmech(braking) = 140 × 10 = 1400 W\n"
            "         P_Ra           = 100 × 0.5 =  50 W\n"
            "         P_returned     = 135 × 10  = 1350 W\n"
            "         η_regen        = 1350/1400 = 96.43 %\n"
            "         Check: 50 + 1350 = 1400 W  ✓\n"
        )
        txt = tk.Text(sf, wrap="word", font=("Courier", 10), bg="#f5f5f5",
                      relief=tk.FLAT, height=55)
        txt.insert("1.0", theory)
        txt.config(state="disabled")
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    # ==================================================================
    # TAB 2 – PARAMETERS & RESULTS
    # ==================================================================
    def _tab_parameters(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="⚙ Parameters & Results")

        top = ttk.Frame(frm)
        top.pack(fill=tk.X, padx=6, pady=4)

        # ---- Parameter sliders ----
        pf = ttk.LabelFrame(top, text="System Parameters")
        pf.grid(row=0, column=0, sticky="nw", padx=4, pady=4)
        pf.columnconfigure(1, weight=1)

        rows = [
            ("Supply voltage Vs (V)",          self._vs,    50, 500),
            ("Motor constant Kb (V/rpm)",       self._kb,     0.05, 0.5),
            ("Armature resistance Ra (Ω)",      self._ra,     0.1, 5.0),
            ("Chopping frequency f (Hz)",       self._f,     25, 500),
            ("Inductance La (H)",               self._la,     0.001, 1.0),
            ("Inertia J (kg·m²)",               self._j,      0.001, 1.0),
            ("Viscous friction B (N·m·s/rad)",  self._b_fric, 0.0, 0.1),
        ]
        for i, (label, var, lo, hi) in enumerate(rows):
            self._slider_row(pf, label, var, lo, hi, i)
            var.trace_add("write", lambda *_: self._refresh_results())

        # ---- Motoring inputs ----
        mf = ttk.LabelFrame(top, text="Part (a) Motoring — Th1+D1")
        mf.grid(row=0, column=1, sticky="nw", padx=4, pady=4)
        mf.columnconfigure(1, weight=1)
        self._slider_row(mf, "Speed N (rpm)", self._n_m, 100, 3000, 0, ".0f")
        self._slider_row(mf, "Armature current Ia (A)", self._ia_m, 1, 50, 1)
        for v in (self._n_m, self._ia_m):
            v.trace_add("write", lambda *_: self._refresh_results())

        # ---- Regen inputs ----
        rf = ttk.LabelFrame(top, text="Part (b) Regeneration — Th2+D2")
        rf.grid(row=0, column=2, sticky="nw", padx=4, pady=4)
        rf.columnconfigure(1, weight=1)
        self._slider_row(rf, "Speed N (rpm)", self._n_r, 100, 3000, 0, ".0f")
        self._slider_row(rf, "|Ia| (A)", self._ia_r, 1, 50, 1)
        for v in (self._n_r, self._ia_r):
            v.trace_add("write", lambda *_: self._refresh_results())

        # ---- Results display ----
        res_outer = ttk.Frame(frm)
        res_outer.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self._res_mot = tk.Text(res_outer, width=50, height=18,
                                font=("Courier", 10), bg="#eef7ee")
        self._res_mot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self._res_reg = tk.Text(res_outer, width=50, height=18,
                                font=("Courier", 10), bg="#f0eef7")
        self._res_reg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self._refresh_results()

    def _refresh_results(self):
        self._calc = self._make_calc()
        rm = self._calc.motoring(self._n_m.get(), self._ia_m.get())
        rr = self._calc.regeneration(self._n_r.get(), self._ia_r.get())

        def write(widget, text):
            widget.config(state="normal")
            widget.delete("1.0", tk.END)
            widget.insert("1.0", text)
            widget.config(state="disabled")

        mot_txt = (
            "══ MOTORING RESULTS (Th1 + D1) ══\n\n"
            "  Input conditions:\n"
            f"    Speed N          = {self._n_m.get():.0f} rpm\n"
            f"    Armature Ia      = {self._ia_m.get():.2f} A\n\n"
            "  Calculated:\n"
            f"    Back EMF Eb      = {rm['eb']:.3f} V\n"
            f"    Average Va       = {rm['va']:.3f} V\n"
            f"    Duty cycle α     = {rm['alpha']:.4f}  ({rm['alpha_pct']:.2f} %)\n"
            f"    Turn-on time ton = {rm['ton']:.4f} ms\n\n"
            "  Power Analysis:\n"
            f"    Pmech (motor)    = {rm['p_mech']:.2f} W\n"
            f"    P_Ra  (copper)   = {rm['p_ra']:.2f} W\n"
            f"    Ps (from source) = {rm['p_source']:.2f} W\n"
            f"    Check: Pmech+P_Ra= {rm['p_mech']+rm['p_ra']:.2f} W\n\n"
            f"    Torque           = {rm['torque']:.4f} N·m\n"
            f"    Efficiency       = {rm['efficiency']:.2f} %\n\n"
            "  Interpretation:\n"
            "    Motor accelerates/runs at constant speed.\n"
            "    Source supplies motoring + copper losses.\n"
            f"    Voltage ripple ≈ 0 (large La assumed)."
        )

        reg_txt = (
            "══ REGENERATION RESULTS (Th2 + D2) ══\n\n"
            "  Input conditions:\n"
            f"    Speed N          = {self._n_r.get():.0f} rpm\n"
            f"    |Ia|             = {self._ia_r.get():.2f} A  (Ia = −{self._ia_r.get():.2f} A)\n\n"
            "  Calculated:\n"
            f"    Back EMF Eb      = {rr['eb']:.3f} V\n"
            f"    Average Va       = {rr['va']:.3f} V\n"
            f"    Duty cycle α₂    = {rr['alpha2']:.4f}  ({rr['alpha2_pct']:.2f} %)\n"
            f"    Turn-on ton2     = {rr['ton2']:.4f} ms\n\n"
            "  Power Analysis:\n"
            f"    Pmech (braking)  = {rr['p_mech']:.2f} W\n"
            f"    P_Ra  (copper)   = {rr['p_ra']:.2f} W\n"
            f"    P returned       = {rr['p_returned']:.2f} W\n"
            f"    Check: P_Ra+Pret = {rr['p_ra']+rr['p_returned']:.2f} W\n\n"
            f"    Torque (braking) = {rr['torque']:.4f} N·m\n"
            f"    Regen efficiency = {rr['regen_efficiency']:.2f} %\n\n"
            "  Interpretation:\n"
            "    Motor decelerates, acting as generator.\n"
            "    Energy returned to supply (regenerative).\n"
            f"    α₂ = ton2/T gives Th2 on-time per cycle."
        )

        write(self._res_mot, mot_txt)
        write(self._res_reg, reg_txt)

    # ==================================================================
    # TAB 3 – WAVEFORMS
    # ==================================================================
    def _tab_waveforms(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="📈 Waveforms")

        ctrl = ttk.Frame(frm)
        ctrl.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(ctrl, text="Refresh Waveforms",
                   command=self._draw_waveforms).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="(Uses current Parameters & Results settings)",
                  font=("Helvetica", 9, "italic")).pack(side=tk.LEFT)

        self._wave_fig = Figure(figsize=(12, 6), dpi=90, tight_layout=True)
        self._wave_canvas = self._embed_figure(self._wave_fig, frm)
        self._draw_waveforms()

    def _draw_waveforms(self):
        self._calc = self._make_calc()
        rm = self._calc.motoring(self._n_m.get(), self._ia_m.get())
        rr = self._calc.regeneration(self._n_r.get(), self._ia_r.get())

        t_m, va_m, ia_m = self._calc.waveform_motoring(rm["alpha"])
        t_r, va_r, ia_r = self._calc.waveform_regen(rr["alpha2"])

        fig = self._wave_fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_m, va_m, "b-", linewidth=1.2)
        ax1.axhline(rm["va"], color="r", linestyle="--", linewidth=1,
                    label=f"Va_avg = {rm['va']:.1f} V")
        ax1.set_title("Motoring – Armature Voltage Va(t)", fontsize=9)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Voltage (V)")
        ax1.legend(fontsize=8)
        ax1.set_ylim(-20, self._vs.get() * 1.15)
        ax1.grid(True, alpha=0.4)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t_m, ia_m, "g-", linewidth=1.2)
        ax2.axhline(self._ia_m.get(), color="r", linestyle="--", linewidth=1,
                    label=f"Ia_avg = {self._ia_m.get():.1f} A")
        ax2.set_title("Motoring – Armature Current Ia(t)", fontsize=9)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Current (A)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.4)

        ax3 = fig.add_subplot(gs[0, 1])
        ax3.plot(t_r, va_r, "b-", linewidth=1.2, label="Va(t)")
        ax3.axhline(rr["va"], color="r", linestyle="--", linewidth=1,
                    label=f"Va_avg = {rr['va']:.1f} V")
        ax3.set_title("Regeneration – Armature Voltage Va(t)", fontsize=9)
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Voltage (V)")
        ax3.legend(fontsize=8)
        ax3.set_ylim(-20, self._vs.get() * 1.15)
        ax3.grid(True, alpha=0.4)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t_r, ia_r, "m-", linewidth=1.2)
        ax4.axhline(-self._ia_r.get(), color="r", linestyle="--", linewidth=1,
                    label=f"Ia_avg = −{self._ia_r.get():.1f} A")
        ax4.set_title("Regeneration – Armature Current Ia(t)", fontsize=9)
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("Current (A)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.4)

        self._wave_canvas.draw()

    # ==================================================================
    # TAB 4 – DYNAMIC SIMULATION
    # ==================================================================
    def _tab_dynamic(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="🔄 Dynamic Simulation")

        ctrl = ttk.LabelFrame(frm, text="Simulation Controls")
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        self._alpha_var = tk.DoubleVar(value=0.674)
        self._tl_var = tk.DoubleVar(value=9.55)
        self._mode_var = tk.StringVar(value="motoring")

        cf = ttk.Frame(ctrl)
        cf.pack(side=tk.LEFT, padx=8, pady=4)
        ttk.Label(cf, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(cf, text="Motoring", variable=self._mode_var,
                        value="motoring").grid(row=0, column=1)
        ttk.Radiobutton(cf, text="Regeneration", variable=self._mode_var,
                        value="regen").grid(row=0, column=2)

        sf2 = ttk.Frame(ctrl)
        sf2.pack(side=tk.LEFT, padx=8)
        self._slider_row(sf2, "Duty cycle α (or 1−α₂)", self._alpha_var,
                         0.01, 0.99, 0)
        self._slider_row(sf2, "Load torque TL (N·m)", self._tl_var,
                         0.0, 50.0, 1)

        btns = ttk.Frame(ctrl)
        btns.pack(side=tk.RIGHT, padx=8)
        ttk.Button(btns, text="▶ Start",
                   command=self._sim_start).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="⏹ Stop",
                   command=self._sim_stop).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="↺ Reset",
                   command=self._sim_reset).pack(side=tk.LEFT, padx=2)

        self._dyn_fig = Figure(figsize=(12, 5), dpi=90, tight_layout=True)
        self._dyn_canvas = self._embed_figure(self._dyn_fig, frm)

        self._sim_t = []
        self._sim_ia = []
        self._sim_n = []
        self._sim_va = []

    def _sim_start(self):
        if self._sim_running:
            return
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._sim_worker,
                                            daemon=True)
        self._sim_thread.start()
        self._poll_sim()

    def _sim_stop(self):
        self._sim_running = False

    def _sim_reset(self):
        self._sim_running = False
        time.sleep(0.05)
        self._sim_t.clear()
        self._sim_ia.clear()
        self._sim_n.clear()
        self._sim_va.clear()
        self._dyn_fig.clear()
        self._dyn_canvas.draw()

    def _sim_worker(self):
        calc = self._make_calc()
        vs = calc.vs
        ra = calc.ra
        la = calc.la
        j = calc.j
        b = calc.b
        km = calc.km

        mode = self._mode_var.get()
        alpha = self._alpha_var.get()
        tl = self._tl_var.get()
        dt = 5e-4   # 0.5 ms

        state = np.array([0.0, rpm2rad(0.0)])
        t_now = 0.0
        t_end = 4.0  # 4 s simulation

        t_list = []
        ia_list = []
        n_list = []
        va_list = []

        while self._sim_running and t_now < t_end:
            if mode == "motoring":
                va = alpha * vs
            else:
                va = (1.0 - alpha) * vs
            state = rk4(state, va, km, ra, la, j, b, tl, dt)
            # clamp to prevent runaway
            state[0] = np.clip(state[0], -500, 500)
            state[1] = np.clip(state[1], -500, 1000)

            t_list.append(t_now)
            ia_list.append(state[0])
            n_list.append(rad2rpm(state[1]))
            va_list.append(va)
            t_now += dt
            time.sleep(0)   # release GIL to allow thread switching

        self._sim_data = dict(t=t_list, ia=ia_list, n=n_list, va=va_list)
        self._sim_running = False

    def _poll_sim(self):
        if not self._sim_running and not self._sim_data:
            return
        if self._sim_data:
            self._draw_dynamic()
            if not self._sim_running:
                self._sim_data = {}
                return
        self.root.after(300, self._poll_sim)

    def _draw_dynamic(self):
        d = self._sim_data
        if not d:
            return
        t = np.array(d["t"])
        ia = np.array(d["ia"])
        n = np.array(d["n"])
        va = np.array(d["va"])

        fig = self._dyn_fig
        fig.clear()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(t, n, "b")
        ax1.set_title("Speed (rpm)")
        ax1.set_xlabel("Time (s)")
        ax1.grid(True, alpha=0.4)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(t, ia, "g")
        ax2.set_title("Armature Current (A)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.4)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(t, n * ia * 0.1 / 1000.0, "r",
                 label="Pmech (kW)")
        ax3.set_title("Mechanical Power (kW)")
        ax3.set_xlabel("Time (s)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.4)

        self._dyn_canvas.draw()

    # ==================================================================
    # TAB 5 – FAULT CURRENT
    # ==================================================================
    def _tab_fault(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="⚡ Fault Current")

        info = ttk.LabelFrame(frm, text="Fault Analysis — Short-Circuit Transient")
        info.pack(fill=tk.X, padx=6, pady=4)

        self._fault_fault_type = tk.StringVar(value="terminal")
        ttk.Label(info, text="Fault type:").grid(row=0, column=0, padx=4)
        ttk.Radiobutton(info, text="Motor-terminal SC",
                        variable=self._fault_fault_type,
                        value="terminal").grid(row=0, column=1)
        ttk.Radiobutton(info, text="Bus fault (Vs applied)",
                        variable=self._fault_fault_type,
                        value="bus").grid(row=0, column=2)
        ttk.Button(info, text="Compute & Plot",
                   command=self._draw_fault).grid(row=0, column=3, padx=8)

        self._fault_res = tk.Text(info, height=5, width=60,
                                  font=("Courier", 9), bg="#fff3cd")
        self._fault_res.grid(row=1, column=0, columnspan=4, padx=4, pady=4,
                             sticky="ew")

        self._fault_fig = Figure(figsize=(11, 4), dpi=90, tight_layout=True)
        self._fault_canvas = self._embed_figure(self._fault_fig, frm)
        self._draw_fault()

    def _draw_fault(self):
        calc = self._make_calc()
        t = np.linspace(0, 0.5, 2000)   # 0–500 ms
        tau = calc.la / calc.ra

        if self._fault_fault_type.get() == "terminal":
            # Va = 0 suddenly, motor has back-EMF Eb0
            eb0 = calc.kb * 1500.0
            i_ss = eb0 / calc.ra
            ia0 = 10.0
            ia = i_ss + (ia0 - i_ss) * np.exp(-t / tau)
            label = "Terminal SC (Va=0)"
        else:
            # Full supply applied to shorted armature
            i_ss = calc.vs / calc.ra
            ia0 = 0.0
            ia = i_ss * (1 - np.exp(-t / tau))
            label = "Bus fault (Va=Vs)"

        i_pk = float(np.max(np.abs(ia)))
        t_pk = float(t[np.argmax(np.abs(ia))] * 1000)

        res_txt = (
            f"  Fault type       : {label}\n"
            f"  Ra               : {calc.ra:.2f} Ω,   La : {calc.la:.4f} H\n"
            f"  Time constant τ  : {tau*1000:.2f} ms\n"
            f"  Steady-state Isc : {i_ss:.1f} A  ({i_ss/calc.ra:.0f}×Ra)\n"
            f"  Peak fault curr. : {i_pk:.1f} A  at t = {t_pk:.1f} ms"
        )
        self._fault_res.config(state="normal")
        self._fault_res.delete("1.0", tk.END)
        self._fault_res.insert("1.0", res_txt)
        self._fault_res.config(state="disabled")

        fig = self._fault_fig
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(t * 1000, ia, "r-", linewidth=1.5, label=label)
        ax1.axhline(i_ss, color="gray", linestyle="--",
                    label=f"Isc_ss = {i_ss:.1f} A")
        ax1.set_title("Fault Current Transient", fontsize=9)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Current (A)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)

        ax2 = fig.add_subplot(1, 2, 2)
        # I²t curve
        i2t = np.cumsum(ia ** 2) * (t[1] - t[0])
        ax2.plot(t * 1000, i2t, "b-", linewidth=1.5)
        ax2.set_title("I²t Energy (Protection Index)", fontsize=9)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("I²t  (A²·s)")
        ax2.grid(True, alpha=0.4)

        self._fault_canvas.draw()

    # ==================================================================
    # TAB 6 – PROTECTION COORDINATION
    # ==================================================================
    def _tab_protection(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="🛡 Protection")

        ctrl = ttk.LabelFrame(frm, text="Relay Settings")
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        self._ip1 = tk.DoubleVar(value=12.0)   # pickup current relay 1
        self._ip2 = tk.DoubleVar(value=20.0)   # pickup current relay 2
        self._slider_row(ctrl, "Relay 1 pickup I (A)", self._ip1, 5, 50, 0)
        self._slider_row(ctrl, "Relay 2 pickup I (A)", self._ip2, 5, 100, 1)
        ttk.Button(ctrl, text="Plot TCC",
                   command=self._draw_protection).grid(row=2, column=0,
                                                        columnspan=3, pady=4)

        self._prot_fig = Figure(figsize=(11, 5), dpi=90, tight_layout=True)
        self._prot_canvas = self._embed_figure(self._prot_fig, frm)
        self._draw_protection()

    def _draw_protection(self):
        ip1 = self._ip1.get()
        ip2 = self._ip2.get()

        I = np.linspace(1.01, 500, 2000)

        def si_curve(I, Ip, K=0.14, alpha=0.02, tms=1.0):
            m = I / Ip
            m = np.where(m > 1.001, m, np.nan)
            return tms * K / (m ** alpha - 1)

        def vi_curve(I, Ip, tms=1.0):
            m = I / Ip
            m = np.where(m > 1.001, m, np.nan)
            return tms * 13.5 / (m - 1)

        def ei_curve(I, Ip, tms=1.0):
            m = I / Ip
            m = np.where(m > 1.001, m, np.nan)
            return tms * 80.0 / (m ** 2 - 1)

        fig = self._prot_fig
        fig.clear()

        ax = fig.add_subplot(1, 2, 1)
        ax.loglog(I, si_curve(I, ip1), "b-", lw=1.5,
                  label=f"SI  Relay1 (Ip={ip1:.0f}A)")
        ax.loglog(I, vi_curve(I, ip1), "b--", lw=1.5,
                  label=f"VI  Relay1 (Ip={ip1:.0f}A)")
        ax.loglog(I, ei_curve(I, ip1), "b:", lw=1.5,
                  label=f"EI  Relay1 (Ip={ip1:.0f}A)")
        ax.loglog(I, si_curve(I, ip2, tms=0.5), "r-", lw=1.5,
                  label=f"SI  Relay2 (Ip={ip2:.0f}A, TMS=0.5)")
        ax.axvline(460, color="k", linestyle="--",
                   label="Max fault Isc=460A")
        ax.set_xlim(max(ip1 * 0.8, 1), 1000)
        ax.set_ylim(0.01, 100)
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Trip time (s)")
        ax.set_title("Time–Current Characteristics (TCC)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.4)

        # Coordination margin
        ax2 = fig.add_subplot(1, 2, 2)
        i_range = np.linspace(ip1 * 1.1, 400, 500)
        t1 = si_curve(i_range, ip1)
        t2 = si_curve(i_range, ip2, tms=0.5)
        ax2.plot(i_range, t1, "b-", lw=1.5, label="Relay 1 (upstream)")
        ax2.plot(i_range, t2, "r-", lw=1.5, label="Relay 2 (downstream)")
        margin = t1 - t2
        ax2.fill_between(i_range, t2, t1, alpha=0.2, color="green",
                         label="CTI margin")
        ax2.set_xlabel("Current (A)")
        ax2.set_ylabel("Trip time (s)")
        ax2.set_title("Coordination Time Interval (CTI)", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.4)
        ax2.set_ylim(0, 5)

        self._prot_canvas.draw()

    # ==================================================================
    # TAB 7 – SPEED CONTROLLER  (PID + Fuzzy)
    # ==================================================================
    def _tab_speed_ctrl(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="🎯 Speed Controller")

        ctrl = ttk.LabelFrame(frm, text="Controller Settings")
        ctrl.pack(fill=tk.X, padx=6, pady=4)
        ctrl.columnconfigure((1, 4), weight=1)

        self._slider_row(ctrl, "Reference N_ref (rpm)", self._nref, 100, 3000, 0, ".0f")
        self._slider_row(ctrl, "Load TL (N·m)",         self._tl,   0, 40,   1)
        self._slider_row(ctrl, "PID Kp",                self._kp,   0.1, 10,  2)
        self._slider_row(ctrl, "PID Ki",                self._ki,   0.0, 5,   3)
        self._slider_row(ctrl, "PID Kd",                self._kd,   0.0, 1,   4)

        ttk.Button(ctrl, text="▶ Run Both Controllers",
                   command=self._run_speed_ctrl).grid(
                       row=5, column=0, columnspan=3, pady=6)

        self._sc_fig = Figure(figsize=(12, 6), dpi=90, tight_layout=True)
        self._sc_canvas = self._embed_figure(self._sc_fig, frm)
        self._run_speed_ctrl()

    def _run_speed_ctrl(self):
        calc = self._make_calc()
        vs = calc.vs
        ra = calc.ra
        la = calc.la
        j = calc.j
        b = calc.b
        km = calc.km

        nref = self._nref.get()
        wref = rpm2rad(nref)
        tl_base = self._tl.get()
        dt = 2e-3
        t_total = 5.0
        t_step = 2.5   # load step time

        pid = PIDController(kp=self._kp.get(), ki=self._ki.get(),
                            kd=self._kd.get(), dt=dt,
                            out_min=0.0, out_max=1.0)
        fuz = FuzzyController(e_range=wref if wref > 0 else 1.0,
                              de_range=wref / 2 if wref > 0 else 1.0,
                              out_scale=0.4)

        def simulate(controller_type):
            state = np.array([0.0, 0.0])
            alpha = 0.5
            t_arr, n_arr, ia_arr, alpha_arr = [], [], [], []
            de_prev = 0.0
            for step_i in range(int(t_total / dt)):
                t_now = step_i * dt
                tl = tl_base if t_now < t_step else tl_base * 1.5
                omega = state[1]
                error = wref - omega
                if controller_type == "pid":
                    alpha = pid.step(error / max(wref, 1))
                else:
                    de = (error - de_prev) / dt
                    alpha = 0.5 + fuz.step(error, de, dt)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    de_prev = error
                va = alpha * vs
                state = rk4(state, va, km, ra, la, j, b, tl, dt)
                state[0] = np.clip(state[0], -200, 200)
                state[1] = np.clip(state[1], -100, 1500)
                t_arr.append(t_now)
                n_arr.append(rad2rpm(state[1]))
                ia_arr.append(state[0])
                alpha_arr.append(alpha)
            return np.array(t_arr), np.array(n_arr), np.array(ia_arr), np.array(alpha_arr)

        pid.reset()
        t_p, n_p, ia_p, al_p = simulate("pid")
        t_f, n_f, ia_f, al_f = simulate("fuzzy")

        fig = self._sc_fig
        fig.clear()
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(t_p, n_p, "b-", lw=1.2, label="PID")
        ax1.plot(t_f, n_f, "r--", lw=1.2, label="Fuzzy")
        ax1.axhline(nref, color="k", lw=0.8, linestyle=":")
        ax1.axvline(t_step, color="gray", lw=0.8, linestyle="--",
                    label="Load step")
        ax1.set_title("Speed Response", fontsize=9)
        ax1.set_ylabel("Speed (rpm)")
        ax1.set_xlabel("Time (s)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_p, ia_p, "b-", lw=1.2, label="PID")
        ax2.plot(t_f, ia_f, "r--", lw=1.2, label="Fuzzy")
        ax2.axvline(t_step, color="gray", lw=0.8, linestyle="--")
        ax2.set_title("Armature Current (A)", fontsize=9)
        ax2.set_ylabel("Ia (A)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.4)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t_p, al_p * 100, "b-", lw=1.2, label="PID α")
        ax3.plot(t_f, al_f * 100, "r--", lw=1.2, label="Fuzzy α")
        ax3.axvline(t_step, color="gray", lw=0.8, linestyle="--")
        ax3.set_title("Duty Cycle α (%)", fontsize=9)
        ax3.set_ylabel("α (%)")
        ax3.set_xlabel("Time (s)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.4)

        ax4 = fig.add_subplot(2, 2, 4)
        err_p = np.abs(nref - n_p)
        err_f = np.abs(nref - n_f)
        ax4.plot(t_p, err_p, "b-", lw=1.2, label="PID error")
        ax4.plot(t_f, err_f, "r--", lw=1.2, label="Fuzzy error")
        ax4.axvline(t_step, color="gray", lw=0.8, linestyle="--",
                    label="Load step")
        ax4.set_title("Speed Error |N_ref − N| (rpm)", fontsize=9)
        ax4.set_ylabel("Error (rpm)")
        ax4.set_xlabel("Time (s)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.4)

        self._sc_canvas.draw()

    # ==================================================================
    # TAB 8 – THERMAL & ECONOMIC
    # ==================================================================
    def _tab_thermal(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="🌡 Thermal & Economic")

        ctrl = ttk.LabelFrame(frm, text="Analysis Parameters")
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        self._rth = tk.DoubleVar(value=0.8)    # thermal resistance °C/W
        self._cth = tk.DoubleVar(value=200.0)  # thermal capacitance J/°C
        self._tamb = tk.DoubleVar(value=25.0)  # ambient temperature °C
        self._hours = tk.DoubleVar(value=8.0)  # daily operating hours
        self._tariff = tk.DoubleVar(value=0.15)  # $/kWh

        self._slider_row(ctrl, "Thermal resistance R_th (°C/W)",
                         self._rth, 0.1, 5.0, 0)
        self._slider_row(ctrl, "Thermal capacitance C_th (J/°C)",
                         self._cth, 10, 1000, 1, ".0f")
        self._slider_row(ctrl, "Ambient temperature T_amb (°C)",
                         self._tamb, 0, 50, 2, ".0f")
        self._slider_row(ctrl, "Daily operating hours",
                         self._hours, 1, 24, 3, ".0f")
        self._slider_row(ctrl, "Electricity tariff ($/kWh)",
                         self._tariff, 0.05, 0.5, 4)

        ttk.Button(ctrl, text="Compute",
                   command=self._draw_thermal).grid(
                       row=5, column=0, columnspan=3, pady=6)

        self._therm_res = tk.Text(ctrl, height=5, width=80,
                                  font=("Courier", 9), bg="#fff3e0")
        self._therm_res.grid(row=6, column=0, columnspan=3,
                             padx=4, pady=4, sticky="ew")

        self._therm_fig = Figure(figsize=(11, 5), dpi=90, tight_layout=True)
        self._therm_canvas = self._embed_figure(self._therm_fig, frm)
        self._draw_thermal()

    def _draw_thermal(self):
        calc = self._make_calc()
        rth = self._rth.get()
        cth = self._cth.get()
        tamb = self._tamb.get()
        hours = self._hours.get()
        tariff = self._tariff.get()

        ia_m = self._ia_m.get()
        ia_r = self._ia_r.get()
        t = np.linspace(0, 3600 * 2, 5000)  # 2 hours

        dt_m = calc.thermal_rise(ia_m, t, rth, cth)  # temp rise motoring
        dt_r = calc.thermal_rise(ia_r, t, rth, cth)  # temp rise regen

        t_m_abs = tamb + dt_m
        t_r_abs = tamb + dt_r

        rm = calc.motoring(self._n_m.get(), ia_m)
        rr = calc.regeneration(self._n_r.get(), ia_r)

        # Economic
        e_mot_kWh = rm["p_source"] / 1000 * hours     # kWh/day motoring
        e_ret_kWh = rr["p_returned"] / 1000 * hours   # kWh/day returned
        cost_no_regen = e_mot_kWh * tariff
        cost_with_regen = (e_mot_kWh - e_ret_kWh) * tariff
        saving_daily = cost_no_regen - cost_with_regen
        saving_yearly = saving_daily * 260  # working days

        therm_txt = (
            f"  Motoring: P_loss = {rm['p_ra']:.1f} W, "
            f"ΔT_ss = {dt_m[-1]:.1f} °C, "
            f"T_max = {t_m_abs[-1]:.1f} °C\n"
            f"  Regen:    P_loss = {rr['p_ra']:.1f} W, "
            f"ΔT_ss = {dt_r[-1]:.1f} °C, "
            f"T_max = {t_r_abs[-1]:.1f} °C\n"
            f"  Daily energy consumed (motoring) : {e_mot_kWh:.3f} kWh/day\n"
            f"  Daily energy returned  (regen)   : {e_ret_kWh:.3f} kWh/day\n"
            f"  Daily cost saving                : ${saving_daily:.4f}\n"
            f"  Annual cost saving (260 days)    : ${saving_yearly:.2f}"
        )
        self._therm_res.config(state="normal")
        self._therm_res.delete("1.0", tk.END)
        self._therm_res.insert("1.0", therm_txt)
        self._therm_res.config(state="disabled")

        fig = self._therm_fig
        fig.clear()
        t_min = t / 60.0

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(t_min, t_m_abs, "r-", lw=1.5, label="Motoring")
        ax1.plot(t_min, t_r_abs, "b--", lw=1.5, label="Regen")
        ax1.axhline(tamb + 40, color="orange", linestyle=":",
                    label="Warning +40°C")
        ax1.axhline(tamb + 80, color="red", linestyle=":",
                    label="Limit +80°C")
        ax1.set_title("Winding Temperature", fontsize=9)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)

        ax2 = fig.add_subplot(1, 3, 2)
        modes = ["Motoring\nPs", "Regen\nPmech", "Regen\nPreturn",
                 "Motoring\nPRa", "Regen\nPRa"]
        vals = [rm["p_source"], rr["p_mech"], rr["p_returned"],
                rm["p_ra"], rr["p_ra"]]
        colors = ["steelblue", "orange", "green", "tomato", "tomato"]
        ax2.bar(modes, vals, color=colors, edgecolor="k", linewidth=0.5)
        ax2.set_title("Power Summary (W)", fontsize=9)
        ax2.set_ylabel("Power (W)")
        ax2.grid(True, axis="y", alpha=0.4)
        for xi, vi in enumerate(vals):
            ax2.text(xi, vi + 10, f"{vi:.0f}", ha="center", fontsize=8)

        ax3 = fig.add_subplot(1, 3, 3)
        daily_hours = np.arange(0, 13)
        costs_std = daily_hours * rm["p_source"] / 1000 * tariff
        costs_reg = daily_hours * (rm["p_source"] - rr["p_returned"]) / 1000 * tariff
        ax3.plot(daily_hours, costs_std, "r-", lw=1.5, label="Without regen")
        ax3.plot(daily_hours, costs_reg, "g-", lw=1.5, label="With regen")
        ax3.fill_between(daily_hours, costs_reg, costs_std,
                         alpha=0.3, color="green", label="Savings")
        ax3.set_title("Daily Energy Cost ($)", fontsize=9)
        ax3.set_xlabel("Operating hours")
        ax3.set_ylabel("Cost ($)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.4)

        self._therm_canvas.draw()

    # ==================================================================
    # TAB 9 – HARMONIC ANALYSIS
    # ==================================================================
    def _tab_harmonic(self):
        frm = ttk.Frame(self._nb)
        self._nb.add(frm, text="〜 Harmonics & Power Quality")

        ctrl = ttk.LabelFrame(frm, text="Harmonic Analysis Settings")
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        self._harm_mode = tk.StringVar(value="motoring")
        ttk.Label(ctrl, text="Mode:").grid(row=0, column=0, padx=4)
        ttk.Radiobutton(ctrl, text="Motoring (α=0.674)",
                        variable=self._harm_mode,
                        value="motoring").grid(row=0, column=1)
        ttk.Radiobutton(ctrl, text="Regeneration (α₂=0.413)",
                        variable=self._harm_mode,
                        value="regen").grid(row=0, column=2)
        ttk.Button(ctrl, text="Compute & Plot",
                   command=self._draw_harmonics).grid(row=0, column=3, padx=8)

        self._harm_res = tk.Text(ctrl, height=5, width=80,
                                 font=("Courier", 9), bg="#e8f5e9")
        self._harm_res.grid(row=1, column=0, columnspan=4,
                            padx=4, pady=4, sticky="ew")

        self._harm_fig = Figure(figsize=(11, 5), dpi=90, tight_layout=True)
        self._harm_canvas = self._embed_figure(self._harm_fig, frm)
        self._draw_harmonics()

    def _draw_harmonics(self):
        calc = self._make_calc()
        mode = self._harm_mode.get()

        if mode == "motoring":
            rm = calc.motoring(self._n_m.get(), self._ia_m.get())
            alpha = rm["alpha"]
            ia_dc = self._ia_m.get()
        else:
            rr = calc.regeneration(self._n_r.get(), self._ia_r.get())
            alpha = 1.0 - rr["alpha2"]  # effective duty for voltage
            ia_dc = -self._ia_r.get()

        v_dc, harmon, v_n = calc.harmonic_spectrum(alpha, n_harmonics=20)
        thd_v = calc.thd(v_n)

        # Current harmonics (assuming La large → only fundamental passes)
        # Ih = Vh / (n × 2π × f × La)
        la = calc.la
        f0 = calc.f
        z_n = harmon * 2 * PI * f0 * la
        i_n = v_n / np.sqrt(calc.ra ** 2 + z_n ** 2)
        thd_i = calc.thd(i_n) if len(i_n) > 1 else 0.0

        # Power factor
        v1 = v_n[0] if len(v_n) > 0 else 1.0
        i1 = i_n[0] if len(i_n) > 0 else 1.0
        phi1 = math.atan2(2 * PI * f0 * la * i1, calc.ra * i1)
        dpf = math.cos(phi1)
        pf = dpf / math.sqrt(1 + (thd_i / 100) ** 2) if thd_i < 9999 else 0

        res_txt = (
            f"  Mode: {mode}   α = {alpha:.4f}\n"
            f"  DC component      : {v_dc:.2f} V\n"
            f"  Fundamental (V1)  : {v1:.2f} V  at {f0:.0f} Hz\n"
            f"  THD (voltage)     : {thd_v:.2f} %\n"
            f"  THD (current)     : {thd_i:.2f} %\n"
            f"  Displacement PF   : {dpf:.4f}\n"
            f"  True Power Factor : {pf:.4f}"
        )
        self._harm_res.config(state="normal")
        self._harm_res.delete("1.0", tk.END)
        self._harm_res.insert("1.0", res_txt)
        self._harm_res.config(state="disabled")

        fig = self._harm_fig
        fig.clear()

        ax1 = fig.add_subplot(1, 3, 1)
        f_ax = harmon * f0
        ax1.bar(f_ax, v_n, width=f0 * 0.6, color="steelblue",
                edgecolor="k", linewidth=0.5)
        ax1.set_title("Voltage Harmonic Spectrum", fontsize=9)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude (V)")
        ax1.grid(True, axis="y", alpha=0.4)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.bar(f_ax, i_n * 1000, width=f0 * 0.6, color="tomato",
                edgecolor="k", linewidth=0.5)
        ax2.set_title("Current Harmonic Spectrum (mA)", fontsize=9)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude (mA)")
        ax2.grid(True, axis="y", alpha=0.4)

        # Time-domain waveform with harmonics
        ax3 = fig.add_subplot(1, 3, 3)
        t_td = np.linspace(0, 4.0 / f0, 2000)
        v_time = v_dc * np.ones_like(t_td)
        for hn, vn_val in zip(harmon, v_n):
            v_time += vn_val * np.sin(2 * PI * hn * f0 * t_td)
        # Ideal chopper waveform
        v_ideal = np.where((t_td % (1.0 / f0)) < alpha / f0,
                           calc.vs, 0.0)
        ax3.plot(t_td * 1000, v_ideal, "gray", lw=0.8, label="Ideal PWM")
        ax3.plot(t_td * 1000, v_time, "b-", lw=1.2, label="Fourier (20 harm)")
        ax3.set_title("Voltage Waveform Reconstruction", fontsize=9)
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Voltage (V)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.4)

        self._harm_canvas.draw()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    root = tk.Tk()
    TwoQuadrantApp(root)   # holds reference; garbage-collection prevention
    root.mainloop()


if __name__ == "__main__":
    main()
