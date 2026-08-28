"""
Textual-based Real-Time Interactive Terminal UI for Piper Arm VLA Teleoperation.
--------------------------------------------------------------------------------
Provides an interactive dashboard with clickable buttons and live telemetry:
  - Live TCP Cartesian Position (X, Y, Z) and Orientation (Roll, Pitch, Yaw)
  - 6 Arm Joint States (q1..q6) with dynamic range meters
  - Analog Gripper Millimeter Gauge (0..40 mm)
  - SpaceMouse 6-DOF Hardware Input Telemetry
  - LeRobot VLA Multi-Modal Data Collection Status (Frames, FPS, Episodes, Task)
  - Clickable Terminal Action Buttons (Record, Save, Discard, Home, Speed, SpaceMouse)
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Grid, Horizontal, Vertical
    from textual.widgets import Header, Footer, Static, Label, Button, Rule
    from textual.reactive import reactive
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


if TEXTUAL_AVAILABLE:
    class TeleopTUIApp(App):
    Screen {
        background: #0b0f19;
        color: #e2e8f0;
    }

    Header {
        background: #111827;
        color: #38bdf8;
        text-style: bold;
    }

    Footer {
        background: #111827;
        color: #94a3b8;
    }

    #main-container {
        height: 1fr;
        padding: 1;
    }

    #top-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        height: 1fr;
    }

    .card {
        background: #161f30;
        border: solid #26354a;
        padding: 1;
        height: 100%;
    }

    .card-title {
        color: #38bdf8;
        text-style: bold;
        padding-bottom: 1;
        border-bottom: solid #26354a;
    }

    #button-bar {
        height: auto;
        background: #111827;
        border: solid #26354a;
        padding: 1;
        margin-top: 1;
        align: center middle;
    }

    Button {
        margin: 0 1;
        min-width: 16;
        height: 3;
        text-style: bold;
    }

    .btn-record-idle {
        background: #dc2626;
        color: #ffffff;
    }

    .btn-record-active {
        background: #16a34a;
        color: #ffffff;
    }

    #rec-status-box {
        background: #1e293b;
        border: solid #334155;
        padding: 1;
        margin-top: 1;
        height: auto;
        content-align: center middle;
    }
    """

    BINDINGS = [
        ("space", "toggle_recording", "Record / Stop"),
        ("c", "toggle_recording", "Record / Stop"),
        ("n", "discard_recording", "Discard Ep"),
        ("h", "home_robot", "Home Pose"),
        ("p", "toggle_spacemouse", "SpaceMouse"),
        ("1", "speed_fine", "Fine Speed"),
        ("2", "speed_normal", "Normal Speed"),
        ("3", "speed_fast", "Fast Speed"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, teleop_runner=None, **kwargs):
        super().__init__(**kwargs)
        self.teleop = teleop_runner
        self.title = "Agilex Piper 6-DOF Pure TCP Teleoperation & LeRobot VLA Pipeline"
        self._last_rec_state = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main-container"):
            with Container(id="top-grid"):
                # 1. Left Card: TCP Cartesian Pose & SpaceMouse
                with Vertical(classes="card"):
                    yield Label("TCP CARTESIAN & SPATEMOUSE", classes="card-title")
                    yield Static(id="tcp-pos-display")
                    yield Static(id="tcp-rot-display")
                    yield Rule()
                    yield Static(id="spacemouse-display")

                # 2. Center Card: 6 Joint States & Gripper
                with Vertical(classes="card"):
                    yield Label("ROBOT JOINTS & GRIPPER", classes="card-title")
                    yield Static(id="joints-display")
                    yield Rule()
                    yield Static(id="gripper-display")

                # 3. Right Card: LeRobot VLA Data Collection Pipeline
                with Vertical(classes="card"):
                    yield Label("LEROBOT VLA DATA COLLECTION", classes="card-title")
                    yield Static(id="task-display")
                    yield Static(id="rec-status-box")
                    yield Static(id="dataset-stats-display")

            # Bottom Interactive Action Button Bar
            with Horizontal(id="button-bar"):
                yield Button("🔴 Start Recording [Space]", id="btn-record", variant="error")
                yield Button("🗑️ Discard Ep [N]", id="btn-discard", variant="warning")
                yield Button("🏠 Home Pose [H]", id="btn-home", variant="primary")
                yield Button("🎮 SpaceMouse [P]", id="btn-sm", variant="default")
                yield Button("⚡ Fine [1]", id="btn-sp-1", variant="default")
                yield Button("⚡ Normal [2]", id="btn-sp-2", variant="success")
                yield Button("⚡ Fast [3]", id="btn-sp-3", variant="default")
                yield Button("❌ Quit [Q]", id="btn-quit", variant="error")

        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.033, self.refresh_telemetry)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle mouse clicks on all terminal action buttons."""
        b_id = event.button.id
        if b_id == "btn-record":
            self.action_toggle_recording()
        elif b_id == "btn-discard":
            self.action_discard_recording()
        elif b_id == "btn-home":
            self.action_home_robot()
        elif b_id == "btn-sm":
            self.action_toggle_spacemouse()
        elif b_id == "btn-sp-1":
            self.action_speed_fine()
        elif b_id == "btn-sp-2":
            self.action_speed_normal()
        elif b_id == "btn-sp-3":
            self.action_speed_fast()
        elif b_id == "btn-quit":
            self.action_quit()

    def refresh_telemetry(self) -> None:
        """Fetch latest telemetry from runner and update UI elements."""
        if self.teleop is None:
            return

        snap = self.teleop.get_telemetry_snapshot()

        # Update TCP Cartesian Pose
        ee_pos = snap.get("ee_pos", [0, 0, 0])
        ee_rpy = snap.get("ee_rpy", [0, 0, 0])
        pos_text = (
            f"[dim]Position (m):[/dim]\n"
            f"  X: [bold cyan]{ee_pos[0]:+7.4f}[/bold cyan]  "
            f"Y: [bold cyan]{ee_pos[1]:+7.4f}[/bold cyan]  "
            f"Z: [bold cyan]{ee_pos[2]:+7.4f}[/bold cyan]\n"
        )
        rot_text = (
            f"[dim]Orientation (°):[/dim]\n"
            f"  Roll: [bold yellow]{ee_rpy[0]:+6.1f}°[/bold yellow]  "
            f"Pitch: [bold yellow]{ee_rpy[1]:+6.1f}°[/bold yellow]  "
            f"Yaw: [bold yellow]{ee_rpy[2]:+6.1f}°[/bold yellow]"
        )
        self.query_one("#tcp-pos-display", Static).update(pos_text)
        self.query_one("#tcp-rot-display", Static).update(rot_text)

        # Update SpaceMouse
        sm_conn = snap.get("sm_connected", False)
        sm_en = snap.get("sm_enabled", False)
        sm_status = "[bold green]CONNECTED (Active)[/bold green]" if (sm_conn and sm_en) else (
            "[bold yellow]CONNECTED (Paused)[/bold yellow]" if sm_conn else "[dim red]NOT DETECTED[/dim red]"
        )
        sm_text = (
            f"[dim]SpaceMouse Status:[/dim] {sm_status}\n"
            f"[dim]Speed Mode:[/dim] [bold green]{snap.get('speed_mode', 'Normal')}[/bold green]\n"
            f"[dim]Control Scheme:[/dim] [bold cyan]Decoupled TCP + Gripper[/bold cyan]"
        )
        self.query_one("#spacemouse-display", Static).update(sm_text)

        # Update Joints q1..q6
        qpos_deg = snap.get("qpos_deg", [0] * 6)
        joint_names = ["q1 (Base)", "q2 (Shoulder)", "q3 (Elbow)", "q4 (Wrist1)", "q5 (Wrist2)", "q6 (Wrist3)"]
        j_lines = ["[dim]Arm Joint Positions:[/dim]"]
        for i, name in enumerate(joint_names):
            val = qpos_deg[i] if i < len(qpos_deg) else 0.0
            pct = int(np.clip((val + 180) / 360 * 20, 0, 20))
            bar = "━" * pct + "╸" + "─" * (20 - pct)
            j_lines.append(f"  {name:13s} [bold cyan]{val:+6.1f}°[/bold cyan] [dim]{bar}[/dim]")
        self.query_one("#joints-display", Static).update("\n".join(j_lines))

        # Update Gripper
        grip_mm = snap.get("gripper_mm", 0.0)
        grip_pct = int(np.clip(grip_mm / 40.0 * 20, 0, 20))
        grip_bar = "█" * grip_pct + "░" * (20 - grip_pct)
        grip_text = (
            f"[dim]Gripper Aperture:[/dim] [bold green]{grip_mm:4.1f} mm[/bold green] / 40.0 mm\n"
            f"[{'bold green' if grip_mm > 5 else 'bold yellow'}]{grip_bar}[/]"
        )
        self.query_one("#gripper-display", Static).update(grip_text)

        # Update LeRobot Recorder Status
        task = snap.get("task_description", "pick up object")
        is_rec = snap.get("is_recording", False)
        cur_ep = snap.get("current_ep", 0)
        n_frames = snap.get("ep_frames", 0)
        ep_sec = snap.get("ep_seconds", 0.0)

        task_text = f"[dim]Task Instruction:[/dim]\n[bold white]\"{task}\"[/bold white]"
        self.query_one("#task-display", Static).update(task_text)

        if is_rec:
            status_box_text = (
                f"[bold red]● RECORDING EPISODE {cur_ep:04d}[/bold red]\n"
                f"[bold white]{n_frames} frames[/bold white] | [bold cyan]{ep_sec:.1f}s[/bold cyan] @ 30 FPS"
            )
        else:
            status_box_text = (
                f"[dim]○ IDLE (Next: Episode {cur_ep:04d})[/dim]\n"
                f"[dim]Click button or press [Space] to Record[/dim]"
            )
        self.query_one("#rec-status-box", Static).update(status_box_text)

        # Dynamically update Record Button Label & Variant
        if is_rec != self._last_rec_state:
            self._last_rec_state = is_rec
            rec_btn = self.query_one("#btn-record", Button)
            if is_rec:
                rec_btn.label = "💾 Save Ep [Space]"
                rec_btn.variant = "success"
            else:
                rec_btn.label = "🔴 Record [Space]"
                rec_btn.variant = "error"

        total_eps = snap.get("total_episodes", 0)
        total_frames = snap.get("total_frames", 0)
        stats_text = (
            f"[dim]Dataset Directory:[/dim] [cyan]{snap.get('data_dir', 'data')}[/cyan]\n"
            f"[dim]Total Dataset Size:[/dim] [bold green]{total_eps} episodes[/bold green] ({total_frames} total frames)\n"
            f"[dim]Modalities:[/dim] [bold magenta]Wrist 2D + Wrist 3D + Side + Top-Down[/bold magenta]"
        )
        self.query_one("#dataset-stats-display", Static).update(stats_text)

    # Action Handlers mapped to Textual hotkeys and Buttons
    def action_toggle_recording(self) -> None:
        if self.teleop:
            self.teleop.toggle_recording()

    def action_discard_recording(self) -> None:
        if self.teleop:
            self.teleop.discard_recording()

    def action_home_robot(self) -> None:
        if self.teleop:
            self.teleop.home_robot()

    def action_toggle_spacemouse(self) -> None:
        if self.teleop:
            self.teleop.toggle_spacemouse()

    def action_speed_fine(self) -> None:
        if self.teleop:
            self.teleop.set_speed("1")
            self._highlight_speed_btn("btn-sp-1")

    def action_speed_normal(self) -> None:
        if self.teleop:
            self.teleop.set_speed("2")
            self._highlight_speed_btn("btn-sp-2")

    def action_speed_fast(self) -> None:
        if self.teleop:
            self.teleop.set_speed("3")
            self._highlight_speed_btn("btn-sp-3")

    def _highlight_speed_btn(self, active_id: str):
        for s_id in ["btn-sp-1", "btn-sp-2", "btn-sp-3"]:
            btn = self.query_one(f"#{s_id}", Button)
            btn.variant = "success" if s_id == active_id else "default"


def run_tui(teleop_runner):
    """Launch the Textual TUI Application."""
    if not TEXTUAL_AVAILABLE:
        print("\033[1;33m[TUI] Textual package not installed. Running standard CLI mode.\033[0m")
        return None
    app = TeleopTUIApp(teleop_runner=teleop_runner)
    app.run()
