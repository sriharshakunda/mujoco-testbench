"""
Multi-Camera & Live Telemetry Graph Real-Time Visualizer for Piper Arm Teleoperation.
------------------------------------------------------------------------------------
Uses OpenCV (cv2.imshow) / GLFW for hardware-accelerated 60+ FPS multi-camera & graph display.

Stitches 4 camera streams + live telemetry chart into an annotated 640x720 dashboard:
  - Row 1 (y=0..240)   : [1. Wrist 2D RGB (320x240)]      | [2. Wrist 3D Metric Depth (320x240)]
  - Row 2 (y=240..480) : [3. Side View 2D RGB (320x240)]   | [4. Top-Down Overview 2D RGB (320x240)]
  - Row 3 (y=480..720) : [5. Live 7-DOF Real-Time Telemetry Graph (q1..q6 + Gripper mm) (640x240)]
"""

import sys
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def depth_to_colormap(depth: np.ndarray, min_d: float = 0.05, max_d: float = 1.0) -> np.ndarray:
    """Convert float32 metric depth array to RGB plasma false color."""
    d_norm = np.clip((depth - min_d) / (max_d - min_d + 1e-6), 0.0, 1.0)
    r = np.clip(np.sin(d_norm * np.pi * 1.5) * 255, 0, 255).astype(np.uint8)
    g = np.clip(np.sin(d_norm * np.pi) * 255, 0, 255).astype(np.uint8)
    b = np.clip(np.cos(d_norm * np.pi * 0.5) * 255, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


class MultiCameraVisualizer:
    """
    Real-time multi-camera & live telemetry graph dashboard visualizer.
    Primary Backend: OpenCV (cv2.imshow) for non-conflicting multi-window display alongside MuJoCo.
    """

    def __init__(
        self,
        title: str = "Piper Arm Multi-Camera & Live Telemetry [LeRobot VLA]",
        frame_h: int = 240,
        frame_w: int = 320,
        include_graph: bool = True,
    ):
        self.title = title
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.include_graph = include_graph
        self.grid_w = frame_w * 2  # 640
        self.grid_h = frame_h * 3 if include_graph else frame_h * 2  # 720 or 480
        self.is_open = False
        self.backend = None

        # 1. Primary Backend: OpenCV cv2
        if CV2_AVAILABLE:
            try:
                cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
                self.backend = "cv2"
                self.is_open = True
                print("\033[1;32m[Visualizer] OpenCV Multi-Camera & Telemetry Dashboard Window Opened Successfully.\033[0m")
                return
            except Exception as e_cv2:
                print(f"\033[1;33m[Visualizer] OpenCV window creation failed ({e_cv2}), attempting GLFW fallback...\033[0m")

        # 2. Secondary Backend: GLFW + OpenGL
        try:
            import glfw
            from OpenGL import GL as gl

            self.glfw = glfw
            self.gl = gl

            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW")

            glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

            self.window = glfw.create_window(self.grid_w, self.grid_h, title, None, None)
            if not self.window:
                raise RuntimeError("Failed to create GLFW window")

            glfw.make_context_current(self.window)
            glfw.swap_interval(0)

            self.tex_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, self.grid_w, self.grid_h, 0, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            gl.glEnable(gl.GL_TEXTURE_2D)

            self.backend = "glfw"
            self.is_open = True
            print("\033[1;32m[Visualizer] GLFW OpenGL Multi-Camera & Telemetry Window Opened Successfully.\033[0m")
            return
        except Exception as e_glfw:
            print(f"\033[1;33m[Visualizer] GLFW backend unavailable ({e_glfw}). Running headless visualizer.\033[0m")
            self.is_open = False

    def update(
        self,
        wrist_rgb: np.ndarray,
        wrist_depth: Optional[np.ndarray],
        scene_rgb: np.ndarray,
        topdown_rgb: np.ndarray,
        graph_rgb: Optional[np.ndarray] = None,
    ) -> None:
        """Render new camera frames + telemetry graph onto dashboard window."""
        if not self.is_open:
            return

        try:
            # 1. Convert depth to RGB false color if 2D float array
            if wrist_depth is not None and hasattr(wrist_depth, "shape") and len(wrist_depth.shape) == 2:
                depth_rgb = depth_to_colormap(wrist_depth)
            elif wrist_depth is not None and hasattr(wrist_depth, "shape"):
                depth_rgb = wrist_depth
            else:
                depth_rgb = wrist_rgb

            # 2. Resize inputs to frame grid dimensions if needed
            w_rgb = cv2.resize(wrist_rgb, (self.frame_w, self.frame_h)) if wrist_rgb.shape[:2] != (self.frame_h, self.frame_w) else wrist_rgb
            d_rgb = cv2.resize(depth_rgb, (self.frame_w, self.frame_h)) if depth_rgb.shape[:2] != (self.frame_h, self.frame_w) else depth_rgb
            s_rgb = cv2.resize(scene_rgb, (self.frame_w, self.frame_h)) if scene_rgb.shape[:2] != (self.frame_h, self.frame_w) else scene_rgb
            t_rgb = cv2.resize(topdown_rgb, (self.frame_w, self.frame_h)) if topdown_rgb.shape[:2] != (self.frame_h, self.frame_w) else topdown_rgb

            # 3. Assemble dashboard grid (640x720)
            grid = np.empty((self.grid_h, self.grid_w, 3), dtype=np.uint8)
            # Row 1: Wrist RGB & Wrist Depth
            grid[0:self.frame_h, 0:self.frame_w] = w_rgb
            grid[0:self.frame_h, self.frame_w:self.grid_w] = d_rgb
            # Row 2: Scene Extrinsic RGB & Top-Down RGB
            grid[self.frame_h:self.frame_h*2, 0:self.frame_w] = s_rgb
            grid[self.frame_h:self.frame_h*2, self.frame_w:self.grid_w] = t_rgb

            # Row 3: Live Telemetry Graph (if enabled)
            if self.include_graph:
                if graph_rgb is not None:
                    g_rgb = cv2.resize(graph_rgb, (self.grid_w, self.frame_h)) if graph_rgb.shape[:2] != (self.frame_h, self.grid_w) else graph_rgb
                    grid[self.frame_h*2:self.grid_h, 0:self.grid_w] = g_rgb
                else:
                    grid[self.frame_h*2:self.grid_h, 0:self.grid_w] = 20

            # 4. Overlay labels with PIL
            img = Image.fromarray(grid)
            draw = ImageDraw.Draw(img)
            labels = [
                (10, 8, "1. Gripper View 2D RGB"),
                (self.frame_w + 10, 8, "2. Gripper View 3D Depth"),
                (10, self.frame_h + 8, "3. Side View 2D RGB (Extrinsic)"),
                (self.frame_w + 10, self.frame_h + 8, "4. Front Overview 2D RGB"),
            ]
            for x, y, text in labels:
                draw.rectangle([(x - 4, y - 2), (x + len(text) * 7 + 4, y + 14)], fill=(0, 0, 0, 180))
                draw.text((x, y), text, fill=(255, 255, 255))

            annotated_grid = np.array(img)

            # 5. Display via selected backend
            if self.backend == "cv2" and CV2_AVAILABLE:
                bgr_grid = cv2.cvtColor(annotated_grid, cv2.COLOR_RGB2BGR)
                cv2.imshow(self.title, bgr_grid)
                cv2.waitKey(1)
                return

            if self.backend == "glfw" and hasattr(self, "window") and self.window is not None:
                if self.glfw.window_should_close(self.window):
                    self.close()
                    return

                rendered_bytes = annotated_grid.tobytes()
                self.glfw.make_context_current(self.window)
                gl = self.gl
                gl.glViewport(0, 0, self.grid_w, self.grid_h)
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                gl.glOrtho(0, self.grid_w, self.grid_h, 0, -1, 1)
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
                gl.glEnable(gl.GL_TEXTURE_2D)

                gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_id)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.grid_w, self.grid_h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rendered_bytes
                )
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glBegin(gl.GL_QUADS)
                gl.glTexCoord2f(0, 0); gl.glVertex2f(0, 0)
                gl.glTexCoord2f(1, 0); gl.glVertex2f(self.grid_w, 0)
                gl.glTexCoord2f(1, 1); gl.glVertex2f(self.grid_w, self.grid_h)
                gl.glTexCoord2f(0, 1); gl.glVertex2f(0, self.grid_h)
                gl.glEnd()

                self.glfw.swap_buffers(self.window)
                self.glfw.poll_events()
                return
        except Exception as e:
            pass

    def close(self) -> None:
        """Close visualizer window cleanly."""
        self.is_open = False
        if self.backend == "cv2" and CV2_AVAILABLE:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
        elif self.backend == "glfw" and hasattr(self, "glfw") and hasattr(self, "window") and self.window is not None:
            try:
                self.glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None
