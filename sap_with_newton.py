from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from sim.sap_runtime import sap_control_from_newton, sap_model_from_newton, sap_state_from_newton
from sim.solver_sap import SolverSAP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Newton cartpole model and viewer, stepped by this repo's SAP solver by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Warp device, e.g. cuda:0 or cpu. Defaults to CUDA if present.")
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl", help="Newton viewer backend.")
    parser.add_argument("--headless", action="store_true", help="Create the GL viewer without a visible window.")
    parser.add_argument("--display", default=":2", help="DISPLAY used by the Newton GL viewer when DISPLAY is unset.")
    parser.add_argument("--width", type=int, default=1280, help="GL viewer width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="GL viewer height in pixels.")
    parser.add_argument("--num-frames", type=int, default=0, help="Frame limit; 0 means run until the viewer closes.")
    parser.add_argument("--fps", type=float, default=60.0, help="Render frame rate used for simulation time.")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps per rendered frame.")
    parser.add_argument("--world-count", type=int, default=1, help="Number of replicated Newton cartpole worlds.")
    parser.add_argument("--solver", choices=("sap", "newton"), default="sap", help="Simulation solver backend.")
    parser.add_argument(
        "--collision",
        choices=("none", "newton"),
        default="none",
        help="Collision source. Cartpole does not need contacts, but Newton contacts can be enabled for experiments.",
    )
    parser.add_argument("--contact-cap", type=int, default=128, help="SAP rigid-contact capacity per world.")
    parser.add_argument("--solver-iterations", type=int, default=30, help="SAP contact solve iterations.")
    parser.add_argument("--contact-tau-d", type=float, default=0.01, help="SAP fallback dissipation time scale.")
    parser.add_argument(
        "--contact-preset",
        choices=("approx32", "approx64", "drake"),
        default="drake",
        help="SAP contact/Jacobian precision preset.",
    )
    parser.add_argument(
        "--line-search",
        choices=("monotone_decay", "armijo_decay", "exact_root"),
        default="armijo_decay",
        help="SAP line-search variant.",
    )
    parser.add_argument("--cart-q", type=float, default=0.0, help="Initial cart prismatic coordinate.")
    parser.add_argument("--pole1-q", type=float, default=0.3, help="Initial first pole revolute coordinate.")
    parser.add_argument("--pole2-q", type=float, default=0.0, help="Initial second pole revolute coordinate.")
    parser.add_argument("--body-armature", type=float, default=0.1, help="Extra diagonal inertia added to each body.")
    parser.add_argument("--joint-armature", type=float, default=0.1, help="Default joint armature used by Newton builder.")
    parser.add_argument("--shape-density", type=float, default=100.0, help="Default Newton shape density.")
    parser.add_argument("--print-every", type=int, default=60, help="Print joint state every N frames; 0 disables.")
    parser.add_argument("--record-mp4", default=None, help="Optional MP4 path recorded from the Newton GL framebuffer.")
    parser.add_argument("--record-fps", type=float, default=None, help="MP4 FPS. Defaults to --fps.")
    parser.add_argument("--quiet", action="store_true", help="Suppress Warp compile messages where possible.")
    return parser


def select_device(device_name: str | None):
    if device_name is None:
        device_name = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.set_device(device_name)
    return wp.get_device(device_name)


def create_newton_cartpole_model(args: argparse.Namespace, device):
    cartpole = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
    cartpole.default_shape_cfg.density = float(args.shape_density)
    cartpole.default_joint_cfg.armature = float(args.joint_armature)

    cartpole.add_usd(
        newton.examples.get_asset("cartpole.usda"),
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )

    body_armature = float(args.body_armature)
    if body_armature:
        for body in range(cartpole.body_count):
            inertia_np = np.asarray(cartpole.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * body_armature
            cartpole.body_inertia[body] = wp.mat33(inertia_np)

    if len(cartpole.joint_q) < 3:
        raise ValueError("Expected cartpole.usda to load three joint coordinates.")
    cartpole.joint_q[-3:] = [float(args.cart_q), float(args.pole1_q), float(args.pole2_q)]

    builder = newton.ModelBuilder()
    builder.replicate(cartpole, max(int(args.world_count), 1), spacing=(1.0, 2.0, 0.0))
    return builder.finalize(device=device)


def create_viewer(args: argparse.Namespace):
    if args.viewer == "gl":
        if args.display and "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = str(args.display)
        if args.headless:
            import pyglet

            pyglet.options["headless"] = True
        return newton.viewer.ViewerGL(
            width=int(args.width),
            height=int(args.height),
            vsync=True,
            headless=bool(args.headless),
        )

    frame_count = args.num_frames if args.num_frames > 0 else 240
    return newton.viewer.ViewerNull(num_frames=frame_count)


class Mp4Recorder:
    def __init__(self, path: str, fps: float):
        self.path = Path(path)
        self.fps = float(fps)
        self.writer = None

    def write(self, frame: wp.array) -> None:
        frame_np = np.ascontiguousarray(frame.numpy())
        height, width = frame_np.shape[:2]

        if self.writer is None:
            import imageio_ffmpeg

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio_ffmpeg.write_frames(
                str(self.path),
                size=(width, height),
                fps=self.fps,
                codec="libx264",
                quality=8,
                output_params=["-pix_fmt", "yuv420p"],
                ffmpeg_log_level="error",
            )
            self.writer.send(None)

        self.writer.send(frame_np)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class NewtonCartpoleDemo:
    def __init__(self, viewer, args: argparse.Namespace, device):
        self.viewer = viewer
        self.frame_dt = 1.0 / float(args.fps)
        self.sim_substeps = max(int(args.substeps), 1)
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0
        self.frame_id = 0
        self.frame_limit = max(int(args.num_frames), 0)
        self.print_every = max(int(args.print_every), 0)
        self.solver_name = str(args.solver)
        self.collision = str(args.collision)

        self.model = create_newton_cartpole_model(args, device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model) if self.collision == "newton" else None
        self.contacts = self.collision_pipeline.contacts() if self.collision_pipeline is not None else None

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)

        if self.solver_name == "sap":
            self.sap_model = sap_model_from_newton(self.model)
            self.solver = SolverSAP(
                self.sap_model,
                max_rigid_contact=int(args.contact_cap),
                max_iterations=int(args.solver_iterations),
                contact_tau_d=float(args.contact_tau_d),
                contact_preset_variant=str(args.contact_preset),
                line_search_variant=str(args.line_search),
            )
            self.sap_state_0 = sap_state_from_newton(self.state_0)
            self.sap_state_1 = sap_state_from_newton(self.state_1)
            self.sap_control = sap_control_from_newton(self.control)
        else:
            self.solver = newton.solvers.SolverMuJoCo(self.model)
            self.sap_state_0 = None
            self.sap_state_1 = None
            self.sap_control = None

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(7.3, -14.0, 2.3), pitch=-5.0, yaw=-225.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 90.0

    def step(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self.contacts is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.solver_name == "sap":
                self.solver.step(self.sap_state_0, self.sap_state_1, self.sap_control, self.contacts, self.sim_dt)
                self.sap_state_0, self.sap_state_1 = self.sap_state_1, self.sap_state_0
            else:
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self.frame_id += 1
        self._maybe_print_state()

    def _maybe_print_state(self) -> None:
        if not self.print_every or self.frame_id % self.print_every != 0:
            return
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        q = joint_q[: min(3, joint_q.shape[0])]
        qd = joint_qd[: min(3, joint_qd.shape[0])]
        print(
            f"frame={self.frame_id} time={self.sim_time:.3f}s "
            f"solver={self.solver_name} q={np.array2string(q, precision=4)} "
            f"qd={np.array2string(qd, precision=4)}"
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def should_stop(self) -> bool:
        return self.frame_limit > 0 and self.frame_id >= self.frame_limit


def main() -> None:
    args = build_parser().parse_args()
    if args.quiet:
        wp.config.log_level = wp.LOG_WARNING
    if args.record_mp4 is not None and args.viewer != "gl":
        raise ValueError("--record-mp4 requires --viewer gl")

    device = select_device(args.device)
    viewer = create_viewer(args)
    demo = NewtonCartpoleDemo(viewer, args, device)
    recorder = Mp4Recorder(args.record_mp4, args.record_fps or args.fps) if args.record_mp4 is not None else None

    try:
        while viewer.is_running() and not demo.should_stop():
            if viewer.should_step():
                demo.step()
            demo.render()
            if recorder is not None:
                recorder.write(viewer.get_frame(render_ui=False))
    finally:
        if recorder is not None:
            recorder.close()
            print(f"Saved MP4 recording to {recorder.path}")
        viewer.close()


if __name__ == "__main__":
    main()
