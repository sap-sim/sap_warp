from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
import numpy as np
import warp as wp

from sim.blocked_cholesky import BlockCholeskySolverBatched
from sim.contact_jacobian import SapContactJacobianResult
from sim.sap_helpers import (
    _clamp_antiderivative_f32,
    _clamp_derivative_f32,
    _clamp_scalar_f32,
    _clamp_antiderivative_f64,
    _clamp_derivative_f64,
    _clamp_scalar_f64,
    _compute_effective_pd_gains_sap,
    _compute_effective_pd_gains_sap_f32,
    _contact_projection_cost_from_vc_sap,
    _contact_projection_cost_from_vc_sap_f32,
    _contact_projection_cost_from_velocity_sap,
    _contact_projection_cost_from_velocity_sap_f32,
    _sap_armijo_ok,
    _sap_armijo_ok_f32,
    _m33d,
    _m33f,
    _v3d,
    _v3f,
    _zero_m33d,
    _zero_m33f,
)
from sim.sap_runtime import (
    Control,
    Model,
    SAP_JOINT_D6,
    SAP_JOINT_DISTANCE,
    SAP_JOINT_FREE,
    SAP_JOINT_PRISMATIC,
    SAP_JOINT_REVOLUTE,
    SAP_JOINT_TARGET_NONE,
    State,
)

wp.config.enable_backward = False

_PI = 3.141592653589793
_CONTACT_SOFT_NORM_TOL = 1.0e-7
_SAP_PD_BETA = 0.1
_SAP_LIMIT_BETA = 0.1
_SAP_LIMIT_STIFFNESS = 1.0e12
_SAP_LIMIT_WINDOW_FACTOR = 2.0
_SAP_EXACT_LINE_SEARCH_ALPHA_MAX = 1.5
_SAP_EXACT_LINE_SEARCH_MAX_ITERATIONS = 100
_SAP_EXACT_LINE_SEARCH_F_TOLERANCE = 1.0e-8

_CONTACT_MODE_NONE = 0
_CONTACT_MODE_STICTION = 1
_CONTACT_MODE_SLIDING = 2
_CONTACT_MODE_FRICTIONLESS = 3
_CONTACT_HESSIAN_GEMM_TILE_M = 8
_CONTACT_HESSIAN_GEMM_TILE_N = 8
_CONTACT_HESSIAN_GEMM_TILE_K = 32


def normalize_sap_line_search_mode(value: str) -> str:
    """Normalize a line-search variant string to the canonical value accepted by the SAP contact solve."""
    mode = str(value).strip().lower().replace("-", "_")
    if mode == "monotone_decay":
        return "monotone_decay"
    if mode == "armijo_decay":
        return "armijo_decay"
    if mode == "exact_root":
        return "exact_root"
    raise ValueError(
        "line_search_variant must be 'monotone_decay', 'armijo_decay', or 'exact_root', "
        f"got {value!r}."
    )


@wp.kernel
def _copy_f32_to_f64(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float64)):
    i = wp.tid()
    dst[i] = wp.float64(src[i])


@wp.kernel
def _copy_f64(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _copy_f32(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _copy_f64_to_f32(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = wp.float32(src[i])


@wp.kernel
def _copy_flat_f64_to_env_f32_batched(
    src: wp.array(dtype=wp.float64),
    dof_per_env: int,
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    env, i = wp.tid()
    if i < dof_per_env:
        dst[env, i] = wp.float32(src[env * dof_per_env + i])


@wp.kernel
def _copy_flat_f32_to_env_f64_batched(
    src: wp.array(dtype=wp.float32),
    dof_per_env: int,
    dst: wp.array(dtype=wp.float64, ndim=2),
):
    env, i = wp.tid()
    if i < dof_per_env:
        dst[env, i] = wp.float64(src[env * dof_per_env + i])


@wp.kernel
def _copy_env_f64_to_env_f32_batched(
    src: wp.array(dtype=wp.float64, ndim=2),
    dof_per_env: int,
    dst: wp.array(dtype=wp.float32, ndim=2),
):
    env, i = wp.tid()
    if i < dof_per_env:
        dst[env, i] = wp.float32(src[env, i])


@wp.kernel
def _copy_env_f32_to_env_f64_batched(
    src: wp.array(dtype=wp.float32, ndim=2),
    dof_per_env: int,
    dst: wp.array(dtype=wp.float64, ndim=2),
):
    env, i = wp.tid()
    if i < dof_per_env:
        dst[env, i] = wp.float64(src[env, i])


@wp.kernel
def _copy_2d_f64_to_f32(src: wp.array(dtype=wp.float64, ndim=2), dst: wp.array(dtype=wp.float32, ndim=2)):
    i, j = wp.tid()
    dst[i, j] = wp.float32(src[i, j])


@wp.kernel
def _copy_3d_f64_to_f32(src: wp.array(dtype=wp.float64, ndim=3), dst: wp.array(dtype=wp.float32, ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = wp.float32(src[i, j, k])


@wp.kernel
def _copy_4d_f64_to_f32(src: wp.array(dtype=wp.float64, ndim=4), dst: wp.array(dtype=wp.float32, ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = wp.float32(src[i, j, k, l])


@wp.kernel
def _copy_vec3d_2d_to_vec3(src: wp.array(dtype=wp.vec3d, ndim=2), dst: wp.array(dtype=wp.vec3, ndim=2)):
    i, j = wp.tid()
    v = src[i, j]
    dst[i, j] = wp.vec3(wp.float32(v.x), wp.float32(v.y), wp.float32(v.z))


@wp.kernel
def _copy_mat33d_2d_to_mat33(src: wp.array(dtype=wp.mat33d, ndim=2), dst: wp.array(dtype=wp.mat33, ndim=2)):
    i, j = wp.tid()
    m = src[i, j]
    dst[i, j] = wp.mat33(
        wp.float32(m[0, 0]), wp.float32(m[0, 1]), wp.float32(m[0, 2]),
        wp.float32(m[1, 0]), wp.float32(m[1, 1]), wp.float32(m[1, 2]),
        wp.float32(m[2, 0]), wp.float32(m[2, 1]), wp.float32(m[2, 2]),
    )


@dataclass(frozen=True)
class SapContactSolveResult:
    """Views into buffers owned by `SapContactSolve`."""

    v_env: wp.array
    v_flat: wp.array
    cost: wp.array
    previous_cost: wp.array
    grad: wp.array
    hessian: wp.array
    constraint_impulse: wp.array
    dynamics_impulse: wp.array
    contact_gamma: wp.array
    contact_g: wp.array
    contact_vc: wp.array
    contact_y: wp.array
    contact_rt: wp.array
    contact_rn: wp.array
    contact_cost: wp.array
    contact_mode: wp.array
    pd_active: wp.array
    pd_y: wp.array
    pd_gamma: wp.array
    pd_hdiag: wp.array
    pd_cost: wp.array
    pd_kp_eff: wp.array
    pd_kd_eff: wp.array
    limit_lower_active: wp.array
    limit_upper_active: wp.array
    limit_lower_gamma: wp.array
    limit_upper_gamma: wp.array
    limit_grad: wp.array
    limit_hdiag: wp.array
    limit_cost: wp.array
    first_dv: wp.array
    alpha: wp.array
    newton_iterations_env: wp.array
    line_search_iterations_env: wp.array
    newton_active: wp.array
    converged_env: wp.array
    optimality_reached_env: wp.array
    cost_reached_env: wp.array
    iterations: int
    line_search_iterations: int
    converged: bool

@cache
def _make_contact_solve_kernel_table(scalar):
    if scalar == wp.float32:
        vec3 = wp.vec3
        mat33 = wp.mat33
        v3 = _v3f
        m33 = _m33f
        zero_m33 = _zero_m33f
        clamp_scalar = _clamp_scalar_f32
        clamp_derivative = _clamp_derivative_f32
        clamp_antiderivative = _clamp_antiderivative_f32
        compute_effective_pd_gains = _compute_effective_pd_gains_sap_f32
        contact_projection_cost_from_vc = _contact_projection_cost_from_vc_sap_f32
        contact_projection_cost_from_velocity = _contact_projection_cost_from_velocity_sap_f32
        sap_armijo_ok = _sap_armijo_ok_f32
    elif scalar == wp.float64:
        vec3 = wp.vec3d
        mat33 = wp.mat33d
        v3 = _v3d
        m33 = _m33d
        zero_m33 = _zero_m33d
        clamp_scalar = _clamp_scalar_f64
        clamp_derivative = _clamp_derivative_f64
        clamp_antiderivative = _clamp_antiderivative_f64
        compute_effective_pd_gains = _compute_effective_pd_gains_sap
        contact_projection_cost_from_vc = _contact_projection_cost_from_vc_sap
        contact_projection_cost_from_velocity = _contact_projection_cost_from_velocity_sap
        sap_armijo_ok = _sap_armijo_ok
    else:
        raise ValueError(f"Unsupported contact solve dtype {scalar!r}.")

    @wp.kernel(module="unique")
    def _copy_flat_to_env_batched(
        src: wp.array(dtype=scalar),
        dof_per_env: int,
        dst: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i < dof_per_env:
            dst[env, i] = src[env * dof_per_env + i]


    @wp.kernel(module="unique")
    def _copy_env_to_flat_batched(
        src: wp.array(dtype=scalar, ndim=2),
        dof_per_env: int,
        dst: wp.array(dtype=scalar),
    ):
        env, i = wp.tid()
        if i < dof_per_env:
            dst[env * dof_per_env + i] = src[env, i]


    @wp.kernel(module="unique")
    def _copy_env_to_env_batched(
        src: wp.array(dtype=scalar, ndim=2),
        dof_per_env: int,
        dst: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i < dof_per_env:
            dst[env, i] = src[env, i]


    @wp.kernel(module="unique")
    def _copy_solve_velocity_inputs_flat_batched(
        v_star_src: wp.array(dtype=scalar),
        v0_src: wp.array(dtype=scalar),
        dof_per_env: int,
        v_star_dst: wp.array(dtype=scalar, ndim=2),
        v0_dst: wp.array(dtype=scalar, ndim=2),
        v_dst: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i < dof_per_env:
            flat = env * dof_per_env + i
            v_star_dst[env, i] = v_star_src[flat]
            v0 = v0_src[flat]
            v0_dst[env, i] = v0
            v_dst[env, i] = v0


    @wp.kernel(module="unique")
    def _copy_solve_velocity_inputs_flat_batched_with_guess_flag(
        v_star_src: wp.array(dtype=scalar),
        v0_src: wp.array(dtype=scalar),
        v_guess_src: wp.array(dtype=scalar),
        use_v_guess: wp.array(dtype=int),
        dof_per_env: int,
        v_star_dst: wp.array(dtype=scalar, ndim=2),
        v0_dst: wp.array(dtype=scalar, ndim=2),
        v_dst: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i < dof_per_env:
            flat = env * dof_per_env + i
            v_star_dst[env, i] = v_star_src[flat]
            v0 = v0_src[flat]
            v0_dst[env, i] = v0
            v = v0
            if use_v_guess[0] != 0:
                v = v_guess_src[flat]
            v_dst[env, i] = v
            v_guess_src[flat] = v


    @wp.kernel(module="unique")
    def _initialize_and_mark_unconstrained_free_envs_batched(
        dof_per_env: int,
        contact_count: wp.array(dtype=int),
        pd_active: wp.array(dtype=int, ndim=2),
        limit_lower_active: wp.array(dtype=int, ndim=2),
        limit_upper_active: wp.array(dtype=int, ndim=2),
        participating_dof: wp.array(dtype=int, ndim=2),
        v_star: wp.array(dtype=scalar, ndim=2),
        v: wp.array(dtype=scalar, ndim=2),
        v_flat: wp.array(dtype=scalar),
        first_dv: wp.array(dtype=scalar, ndim=2),
        newton_iterations_env: wp.array(dtype=int),
        ls_iterations_total: wp.array(dtype=int),
        alpha: wp.array(dtype=scalar),
        previous_cost: wp.array(dtype=scalar),
        converged_env: wp.array(dtype=int),
        optimality_reached_env: wp.array(dtype=int),
        cost_reached_env: wp.array(dtype=int),
        stage2_active_env: wp.array(dtype=int),
        newton_active: wp.array(dtype=int),
        stage2_active_count: wp.array(dtype=int),
    ):
        env = wp.tid()

        newton_iterations_env[env] = 0
        ls_iterations_total[env] = 0
        alpha[env] = scalar(1.0)
        previous_cost[env] = scalar(0.0)
        optimality_reached_env[env] = 0
        cost_reached_env[env] = 0

        unconstrained = contact_count[env] == 0
        for i in range(dof_per_env):
            if pd_active[env, i] == 1 or limit_lower_active[env, i] == 1 or limit_upper_active[env, i] == 1:
                unconstrained = False
        if unconstrained:
            converged_env[env] = 1
            stage2_active_env[env] = 0
            newton_active[env] = 0
            for i in range(dof_per_env):
                first_dv[env, i] = v_star[env, i] - v[env, i]
                v[env, i] = v_star[env, i]
                v_flat[env * dof_per_env + i] = v_star[env, i]
        else:
            converged_env[env] = 0
            stage2_active_env[env] = 1
            newton_active[env] = 1
            for i in range(dof_per_env):
                if participating_dof[env, i] == 0:
                    first_dv[env, i] = v_star[env, i] - v[env, i]
                    v[env, i] = v_star[env, i]
                    v_flat[env * dof_per_env + i] = v_star[env, i]
            wp.atomic_add(stage2_active_count, 0, 1)


    @wp.kernel(module="unique")
    def _initialize_newton_loop_state(
        newton_loop_iteration: wp.array(dtype=int),
        newton_max_reached: wp.array(dtype=int),
    ):
        newton_loop_iteration[0] = 0
        newton_max_reached[0] = 0


    @wp.kernel(module="unique")
    def _extract_a_diag_data_batched(
        dof_per_env: int,
        a_mat: wp.array(dtype=scalar, ndim=3),
        a_inv_diag: wp.array(dtype=scalar, ndim=2),
        d_scale: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return
        diag = a_mat[env, i, i]
        if diag < scalar(1.0e-12) or not wp.isfinite(diag):
            diag = scalar(1.0e-12)
        a_inv_diag[env, i] = scalar(1.0) / diag
        d_scale[env, i] = scalar(1.0) / wp.sqrt(diag)


    @wp.kernel(module="unique")
    def _clear_participating_dofs_batched(
        participating_dof: wp.array(dtype=int, ndim=2),
    ):
        env, i = wp.tid()
        participating_dof[env, i] = 0


    @wp.kernel(module="unique")
    def _mark_contact_participating_dofs_batched(
        dof_per_env: int,
        max_contacts: int,
        contact_env_count: wp.array(dtype=int),
        contact_env_body0: wp.array(dtype=int, ndim=2),
        contact_env_body1: wp.array(dtype=int, ndim=2),
        contact_env_jacobian: wp.array(dtype=scalar, ndim=4),
        body_dof_start: wp.array(dtype=int),
        body_dof_count: wp.array(dtype=int),
        participating_dof: wp.array(dtype=int, ndim=2),
    ):
        env, c, i = wp.tid()
        if i >= dof_per_env:
            return
        count = contact_env_count[env]
        if count > max_contacts:
            count = max_contacts
        if c >= count:
            return

        b0 = contact_env_body0[env, c]
        if b0 >= 0:
            start0 = body_dof_start[b0]
            count0 = body_dof_count[b0]
            if start0 >= 0 and count0 > 0:
                if i >= start0 and i < start0 + count0:
                    participating_dof[env, i] = 1

        b1 = contact_env_body1[env, c]
        if b1 >= 0:
            start1 = body_dof_start[b1]
            count1 = body_dof_count[b1]
            if start1 >= 0 and count1 > 0:
                if i >= start1 and i < start1 + count1:
                    participating_dof[env, i] = 1

        if (
            contact_env_jacobian[env, c, 0, i] != scalar(0.0)
            or contact_env_jacobian[env, c, 1, i] != scalar(0.0)
            or contact_env_jacobian[env, c, 2, i] != scalar(0.0)
        ):
            participating_dof[env, i] = 1


    @wp.kernel(module="unique")
    def _mark_model_participating_dofs_batched(
        dof_per_env: int,
        pd_active: wp.array(dtype=int, ndim=2),
        limit_lower_active: wp.array(dtype=int, ndim=2),
        limit_upper_active: wp.array(dtype=int, ndim=2),
        participating_dof: wp.array(dtype=int, ndim=2),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return
        if pd_active[env, i] == 1 or limit_lower_active[env, i] == 1 or limit_upper_active[env, i] == 1:
            participating_dof[env, i] = 1


    @wp.kernel(module="unique")
    def _build_pd_terms_sap_batched(
        enabled: int,
        dof_per_env: int,
        dof_coord_index: wp.array(dtype=int),
        dof_target_index: wp.array(dtype=int),
        joint_target_mode: wp.array(dtype=int),
        joint_target_ke: wp.array(dtype=float),
        joint_target_kd: wp.array(dtype=float),
        joint_effort_limit: wp.array(dtype=float),
        joint_q: wp.array(dtype=scalar),
        joint_target_pos: wp.array(dtype=float),
        joint_target_vel: wp.array(dtype=float),
        joint_act: wp.array(dtype=float),
        a_inv_diag: wp.array(dtype=scalar, ndim=2),
        dt: scalar,
        mode_none: int,
        pd_active: wp.array(dtype=int, ndim=2),
        pd_a: wp.array(dtype=scalar, ndim=2),
        pd_gain: wp.array(dtype=scalar, ndim=2),
        pd_limit: wp.array(dtype=scalar, ndim=2),
        pd_kp_eff: wp.array(dtype=scalar, ndim=2),
        pd_kd_eff: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return

        dof = env * dof_per_env + i
        coord = dof_coord_index[dof]
        target_dof = dof_target_index[dof]
        pd_active[env, i] = 0
        pd_a[env, i] = scalar(0.0)
        pd_gain[env, i] = scalar(0.0)
        pd_limit[env, i] = scalar(0.0)
        pd_kp_eff[env, i] = scalar(0.0)
        pd_kd_eff[env, i] = scalar(0.0)

        if target_dof < 0:
            target_dof = dof

        if enabled == 0 or joint_target_mode[target_dof] == mode_none:
            return

        kp = scalar(joint_target_ke[target_dof])
        kd = scalar(joint_target_kd[target_dof])
        if kp <= scalar(0.0) and kd <= scalar(0.0):
            return
        if coord < 0 and kp > scalar(0.0):
            return

        eff = compute_effective_pd_gains(kp, kd, scalar(dt), a_inv_diag[env, i])
        kp_eff = eff[0]
        kd_eff = eff[1]
        gain = scalar(dt) * kp_eff + kd_eff
        if gain <= scalar(0.0):
            return

        q0 = scalar(0.0)
        qd = scalar(0.0)
        if coord >= 0:
            q0 = scalar(joint_q[coord])
            qd = scalar(joint_target_pos[target_dof])
        vd = scalar(joint_target_vel[target_dof])
        u0 = scalar(joint_act[target_dof])

        pd_active[env, i] = 1
        pd_a[env, i] = kp_eff * (qd - q0) + kd_eff * vd + u0
        pd_gain[env, i] = gain
        pd_limit[env, i] = scalar(joint_effort_limit[target_dof])
        pd_kp_eff[env, i] = kp_eff
        pd_kd_eff[env, i] = kd_eff


    @wp.kernel(module="unique")
    def _eval_pd_terms_sap_batched(
        add_pd: int,
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        pd_active: wp.array(dtype=int, ndim=2),
        pd_a: wp.array(dtype=scalar, ndim=2),
        pd_gain: wp.array(dtype=scalar, ndim=2),
        pd_limit: wp.array(dtype=scalar, ndim=2),
        v: wp.array(dtype=scalar, ndim=2),
        dt: scalar,
        pd_y: wp.array(dtype=scalar, ndim=2),
        pd_gamma: wp.array(dtype=scalar, ndim=2),
        pd_hdiag: wp.array(dtype=scalar, ndim=2),
        pd_cost: wp.array(dtype=scalar, ndim=2),
        total_cost: wp.array(dtype=scalar),
    ):
        env, i = wp.tid()
        if i >= dof_per_env or add_pd == 0 or active_env[env] == 0 or pd_active[env, i] == 0:
            if i < dof_per_env:
                pd_y[env, i] = scalar(0.0)
                pd_gamma[env, i] = scalar(0.0)
                pd_hdiag[env, i] = scalar(0.0)
                pd_cost[env, i] = scalar(0.0)
            return

        gain = pd_gain[env, i]
        y = pd_a[env, i] - gain * v[env, i]
        gamma = scalar(dt) * clamp_scalar(y, pd_limit[env, i])
        hdiag = scalar(dt) * gain * clamp_derivative(y, pd_limit[env, i])
        cost = scalar(0.0)
        if gain > scalar(0.0):
            cost = (scalar(dt) / gain) * clamp_antiderivative(y, pd_limit[env, i])

        pd_y[env, i] = y
        pd_gamma[env, i] = gamma
        pd_hdiag[env, i] = hdiag
        pd_cost[env, i] = cost
        wp.atomic_add(total_cost, env, cost)


    @wp.kernel(module="unique")
    def _build_limit_terms_sap_batched(
        enabled: int,
        dof_per_env: int,
        dof_coord_index: wp.array(dtype=int),
        limit_supported: wp.array(dtype=int),
        joint_limit_lower: wp.array(dtype=scalar),
        joint_limit_upper: wp.array(dtype=scalar),
        joint_limit_ke: wp.array(dtype=scalar),
        joint_limit_kd: wp.array(dtype=scalar),
        joint_q: wp.array(dtype=scalar),
        v0: wp.array(dtype=scalar, ndim=2),
        v_star: wp.array(dtype=scalar, ndim=2),
        a_inv_diag: wp.array(dtype=scalar, ndim=2),
        dt: scalar,
        lower_active: wp.array(dtype=int, ndim=2),
        upper_active: wp.array(dtype=int, ndim=2),
        lower_vhat: wp.array(dtype=scalar, ndim=2),
        upper_vhat: wp.array(dtype=scalar, ndim=2),
        lower_r: wp.array(dtype=scalar, ndim=2),
        upper_r: wp.array(dtype=scalar, ndim=2),
        lower_rinv: wp.array(dtype=scalar, ndim=2),
        upper_rinv: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return

        dof = env * dof_per_env + i
        coord = dof_coord_index[dof]
        lower_active[env, i] = 0
        upper_active[env, i] = 0
        lower_vhat[env, i] = scalar(0.0)
        upper_vhat[env, i] = scalar(0.0)
        lower_r[env, i] = scalar(1.0)
        upper_r[env, i] = scalar(1.0)
        lower_rinv[env, i] = scalar(0.0)
        upper_rinv[env, i] = scalar(0.0)
        if enabled == 0 or coord < 0 or limit_supported[dof] == 0:
            return

        ke = scalar(joint_limit_ke[dof])
        if ke <= scalar(0.0):
            return

        q0 = scalar(joint_q[coord])
        lower = scalar(joint_limit_lower[dof])
        upper = scalar(joint_limit_upper[dof])
        speed_scale = wp.max(wp.abs(v0[env, i]), wp.abs(v_star[env, i]))
        window = scalar(_SAP_LIMIT_WINDOW_FACTOR) * scalar(dt) * speed_scale

        kd = scalar(joint_limit_kd[dof])
        tau_d = scalar(0.0)
        if kd > scalar(0.0):
            tau_d = kd / ke
        beta = scalar(_SAP_LIMIT_BETA)
        beta_factor = beta * beta / (scalar(4.0) * scalar(_PI) * scalar(_PI))
        r_nr = beta_factor * wp.max(a_inv_diag[env, i], scalar(1.0e-12))
        r_soft = scalar(1.0) / (
            scalar(dt) * ke * (scalar(dt) + tau_d)
        )
        r = wp.max(r_nr, r_soft)
        rinv = scalar(1.0) / r

        if wp.isfinite(lower):
            g = q0 - lower
            if g <= window:
                lower_active[env, i] = 1
                lower_vhat[env, i] = -g / (scalar(dt) + tau_d)
                lower_r[env, i] = r
                lower_rinv[env, i] = rinv

        if wp.isfinite(upper):
            g = upper - q0
            if g <= window:
                upper_active[env, i] = 1
                upper_vhat[env, i] = -g / (scalar(dt) + tau_d)
                upper_r[env, i] = r
                upper_rinv[env, i] = rinv


    @wp.kernel(module="unique")
    def _eval_limit_terms_sap_batched(
        add_limits: int,
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        lower_active: wp.array(dtype=int, ndim=2),
        upper_active: wp.array(dtype=int, ndim=2),
        lower_vhat: wp.array(dtype=scalar, ndim=2),
        upper_vhat: wp.array(dtype=scalar, ndim=2),
        lower_r: wp.array(dtype=scalar, ndim=2),
        upper_r: wp.array(dtype=scalar, ndim=2),
        lower_rinv: wp.array(dtype=scalar, ndim=2),
        upper_rinv: wp.array(dtype=scalar, ndim=2),
        v: wp.array(dtype=scalar, ndim=2),
        lower_gamma_out: wp.array(dtype=scalar, ndim=2),
        upper_gamma_out: wp.array(dtype=scalar, ndim=2),
        limit_grad: wp.array(dtype=scalar, ndim=2),
        limit_hdiag: wp.array(dtype=scalar, ndim=2),
        limit_cost: wp.array(dtype=scalar, ndim=2),
        total_cost: wp.array(dtype=scalar),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return

        grad = scalar(0.0)
        hdiag = scalar(0.0)
        cost = scalar(0.0)
        lower_gamma = scalar(0.0)
        upper_gamma = scalar(0.0)

        if add_limits == 1 and active_env[env] == 1:
            if lower_active[env, i] == 1:
                gamma = lower_rinv[env, i] * (lower_vhat[env, i] - v[env, i])
                if gamma > scalar(0.0):
                    lower_gamma = gamma
                    grad = grad + gamma
                    hdiag = hdiag + lower_rinv[env, i]
                    cost = cost + scalar(0.5) * lower_r[env, i] * gamma * gamma

            if upper_active[env, i] == 1:
                gamma = upper_rinv[env, i] * (upper_vhat[env, i] + v[env, i])
                if gamma > scalar(0.0):
                    upper_gamma = gamma
                    grad = grad - gamma
                    hdiag = hdiag + upper_rinv[env, i]
                    cost = cost + scalar(0.5) * upper_r[env, i] * gamma * gamma

        lower_gamma_out[env, i] = lower_gamma
        upper_gamma_out[env, i] = upper_gamma
        limit_grad[env, i] = grad
        limit_hdiag[env, i] = hdiag
        limit_cost[env, i] = cost
        if active_env[env] == 1:
            wp.atomic_add(total_cost, env, cost)


    @wp.kernel(module="unique")
    def _projection_eval_contact_sap_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        max_contacts: int,
        contact_count: wp.array(dtype=int),
        contact_jac: wp.array(dtype=scalar, ndim=4),
        contact_phi0: wp.array(dtype=scalar, ndim=2),
        contact_w_eff: wp.array(dtype=scalar, ndim=2),
        contact_mu: wp.array(dtype=scalar, ndim=2),
        contact_k: wp.array(dtype=scalar, ndim=2),
        contact_tau_d: wp.array(dtype=scalar, ndim=2),
        v: wp.array(dtype=scalar, ndim=2),
        beta: scalar,
        sigma: scalar,
        dt: scalar,
        contact_gamma: wp.array(dtype=vec3, ndim=2),
        contact_g: wp.array(dtype=mat33, ndim=2),
        contact_vc: wp.array(dtype=vec3, ndim=2),
        contact_y: wp.array(dtype=vec3, ndim=2),
        contact_rt: wp.array(dtype=scalar, ndim=2),
        contact_rn: wp.array(dtype=scalar, ndim=2),
        contact_cost: wp.array(dtype=scalar, ndim=2),
        contact_mode: wp.array(dtype=int, ndim=2),
        total_cost: wp.array(dtype=scalar),
    ):
        env, c = wp.tid()
        zero = v3(scalar(0.0), scalar(0.0), scalar(0.0))
        if c >= max_contacts or c >= contact_count[env]:
            return
        if active_env[env] == 0:
            contact_gamma[env, c] = zero
            contact_g[env, c] = zero_m33()
            contact_vc[env, c] = zero
            contact_y[env, c] = zero
            contact_rt[env, c] = scalar(0.0)
            contact_rn[env, c] = scalar(0.0)
            contact_cost[env, c] = scalar(0.0)
            contact_mode[env, c] = _CONTACT_MODE_NONE
            return

        vc = zero
        for j in range(dof_per_env):
            vj = v[env, j]
            vc = vec3(
                vc.x + contact_jac[env, c, 0, j] * vj,
                vc.y + contact_jac[env, c, 1, j] * vj,
                vc.z + contact_jac[env, c, 2, j] * vj,
            )

        wi = contact_w_eff[env, c]
        if wi < scalar(1.0e-12) or not wp.isfinite(wi):
            wi = scalar(1.0e-12)

        beta64 = scalar(beta)
        beta_factor = beta64 * beta64 / (scalar(4.0) * scalar(_PI) * scalar(_PI))
        rn_hard = beta_factor * wi
        k_c = contact_k[env, c]
        if k_c <= scalar(0.0) or not wp.isfinite(k_c):
            k_c = scalar(1.0)
        rn_soft = scalar(1.0) / (
            scalar(dt) * k_c * (scalar(dt) + wp.max(contact_tau_d[env, c], scalar(0.0)))
        )
        rn = wp.max(rn_hard, rn_soft)
        rt = scalar(sigma) * wi
        if rt < scalar(1.0e-30):
            rt = scalar(1.0e-30)
        if rn < scalar(1.0e-30):
            rn = scalar(1.0e-30)

        rt_inv = scalar(1.0) / rt
        rn_inv = scalar(1.0) / rn
        tau_c = wp.max(contact_tau_d[env, c], scalar(0.0))
        vhat_n = -contact_phi0[env, c] / (scalar(dt) + tau_c)
        y = vec3(-rt_inv * vc.x, -rt_inv * vc.y, rn_inv * (vhat_n - vc.z))

        mu = contact_mu[env, c]
        if mu < scalar(0.0) or not wp.isfinite(mu):
            mu = scalar(0.0)

        yr = wp.sqrt(
            y.x * y.x
            + y.y * y.y
            + scalar(_CONTACT_SOFT_NORM_TOL) * scalar(_CONTACT_SOFT_NORM_TOL)
        )
        t_hat = v3(scalar(0.0), scalar(0.0), scalar(0.0))
        if yr > scalar(0.0):
            t_hat = v3(y.x / yr, y.y / yr, scalar(0.0))

        gamma = zero
        g_mat = zero_m33()
        mode = _CONTACT_MODE_NONE

        if mu <= scalar(1.0e-12):
            if y.z > scalar(0.0):
                gamma = v3(scalar(0.0), scalar(0.0), y.z)
                g_mat = m33(
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    rn_inv,
                )
                mode = _CONTACT_MODE_FRICTIONLESS
        else:
            mu_tilde = mu * wp.sqrt(rt / rn)
            mu_hat = mu * rt / rn
            factor = scalar(1.0) / (scalar(1.0) + mu_tilde * mu_tilde)

            if yr <= mu * y.z:
                gamma = y
                g_mat = m33(
                    rt_inv,
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    rt_inv,
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    rn_inv,
                )
                mode = _CONTACT_MODE_STICTION
            elif (-mu_hat * yr < y.z) and (y.z < yr / mu):
                gamma_n = (y.z + mu_hat * yr) * factor
                gamma = v3(mu * gamma_n * t_hat.x, mu * gamma_n * t_hat.y, gamma_n)

                p00 = t_hat.x * t_hat.x
                p01 = t_hat.x * t_hat.y
                p10 = t_hat.y * t_hat.x
                p11 = t_hat.y * t_hat.y
                pp00 = scalar(1.0) - p00
                pp01 = -p01
                pp10 = -p10
                pp11 = scalar(1.0) - p11

                gn_over_yr = scalar(0.0)
                if yr > scalar(0.0):
                    gn_over_yr = gamma_n / yr

                dgt_dyt00 = mu * (gn_over_yr * pp00 + mu_hat * factor * p00)
                dgt_dyt01 = mu * (gn_over_yr * pp01 + mu_hat * factor * p01)
                dgt_dyt10 = mu * (gn_over_yr * pp10 + mu_hat * factor * p10)
                dgt_dyt11 = mu * (gn_over_yr * pp11 + mu_hat * factor * p11)
                dgt_dyn0 = mu * factor * t_hat.x
                dgt_dyn1 = mu * factor * t_hat.y
                dgn_dyt0 = mu_hat * factor * t_hat.x
                dgn_dyt1 = mu_hat * factor * t_hat.y

                g_mat = m33(
                    dgt_dyt00 * rt_inv,
                    dgt_dyt01 * rt_inv,
                    dgt_dyn0 * rn_inv,
                    dgt_dyt10 * rt_inv,
                    dgt_dyt11 * rt_inv,
                    dgt_dyn1 * rn_inv,
                    dgn_dyt0 * rt_inv,
                    dgn_dyt1 * rt_inv,
                    factor * rn_inv,
                )
                mode = _CONTACT_MODE_SLIDING

        cost = scalar(0.5) * (
            rt * (gamma.x * gamma.x + gamma.y * gamma.y) + rn * gamma.z * gamma.z
        )

        contact_gamma[env, c] = gamma
        contact_g[env, c] = g_mat
        contact_vc[env, c] = vc
        contact_y[env, c] = y
        contact_rt[env, c] = rt
        contact_rn[env, c] = rn
        contact_cost[env, c] = cost
        contact_mode[env, c] = mode
        wp.atomic_add(total_cost, env, cost)


    @wp.kernel(module="unique")
    def _projection_cost_only_contact_sap_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        max_contacts: int,
        contact_count: wp.array(dtype=int),
        contact_jac: wp.array(dtype=scalar, ndim=4),
        contact_phi0: wp.array(dtype=scalar, ndim=2),
        contact_w_eff: wp.array(dtype=scalar, ndim=2),
        contact_mu: wp.array(dtype=scalar, ndim=2),
        contact_k: wp.array(dtype=scalar, ndim=2),
        contact_tau_d: wp.array(dtype=scalar, ndim=2),
        v: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        beta: scalar,
        sigma: scalar,
        dt: scalar,
        contact_cost: wp.array(dtype=scalar, ndim=2),
        total_cost: wp.array(dtype=scalar),
    ):
        env, c = wp.tid()
        if c >= max_contacts or c >= contact_count[env]:
            return
        if active_env[env] == 0:
            contact_cost[env, c] = scalar(0.0)
            return

        cost = contact_projection_cost_from_velocity(
            env,
            c,
            dof_per_env,
            contact_jac,
            contact_phi0,
            contact_w_eff,
            contact_mu,
            contact_k,
            contact_tau_d,
            v,
            dv,
            scalar(0.0),
            beta,
            sigma,
            dt,
        )
        contact_cost[env, c] = cost
        wp.atomic_add(total_cost, env, cost)


    @wp.kernel(module="unique")
    def _projection_eval_contact_gamma_sap_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        max_contacts: int,
        contact_count: wp.array(dtype=int),
        contact_jac: wp.array(dtype=scalar, ndim=4),
        contact_phi0: wp.array(dtype=scalar, ndim=2),
        contact_w_eff: wp.array(dtype=scalar, ndim=2),
        contact_mu: wp.array(dtype=scalar, ndim=2),
        contact_k: wp.array(dtype=scalar, ndim=2),
        contact_tau_d: wp.array(dtype=scalar, ndim=2),
        v: wp.array(dtype=scalar, ndim=2),
        beta: scalar,
        sigma: scalar,
        dt: scalar,
        contact_gamma: wp.array(dtype=vec3, ndim=2),
        contact_vc: wp.array(dtype=vec3, ndim=2),
        contact_cost: wp.array(dtype=scalar, ndim=2),
        total_cost: wp.array(dtype=scalar),
    ):
        env, c = wp.tid()
        zero = v3(scalar(0.0), scalar(0.0), scalar(0.0))
        if c >= max_contacts or c >= contact_count[env]:
            return
        if active_env[env] == 0:
            contact_gamma[env, c] = zero
            contact_vc[env, c] = zero
            contact_cost[env, c] = scalar(0.0)
            return

        vc = zero
        for j in range(dof_per_env):
            vj = v[env, j]
            vc = vec3(
                vc.x + contact_jac[env, c, 0, j] * vj,
                vc.y + contact_jac[env, c, 1, j] * vj,
                vc.z + contact_jac[env, c, 2, j] * vj,
            )

        wi = contact_w_eff[env, c]
        if wi < scalar(1.0e-12) or not wp.isfinite(wi):
            wi = scalar(1.0e-12)

        beta_factor = beta * beta / (scalar(4.0) * scalar(_PI) * scalar(_PI))
        rn_hard = beta_factor * wi
        k_c = contact_k[env, c]
        if k_c <= scalar(0.0) or not wp.isfinite(k_c):
            k_c = scalar(1.0)
        tau_c = wp.max(contact_tau_d[env, c], scalar(0.0))
        rn_soft = scalar(1.0) / (dt * k_c * (dt + tau_c))
        rn = wp.max(rn_hard, rn_soft)
        rt = sigma * wi
        if rt < scalar(1.0e-30):
            rt = scalar(1.0e-30)
        if rn < scalar(1.0e-30):
            rn = scalar(1.0e-30)

        rt_inv = scalar(1.0) / rt
        rn_inv = scalar(1.0) / rn
        vhat_n = -contact_phi0[env, c] / (dt + tau_c)
        y = vec3(-rt_inv * vc.x, -rt_inv * vc.y, rn_inv * (vhat_n - vc.z))

        mu = contact_mu[env, c]
        if mu < scalar(0.0) or not wp.isfinite(mu):
            mu = scalar(0.0)

        yr = wp.sqrt(
            y.x * y.x
            + y.y * y.y
            + scalar(_CONTACT_SOFT_NORM_TOL) * scalar(_CONTACT_SOFT_NORM_TOL)
        )
        t_hat = zero
        if yr > scalar(0.0):
            t_hat = v3(y.x / yr, y.y / yr, scalar(0.0))

        gamma = zero
        if mu <= scalar(1.0e-12):
            if y.z > scalar(0.0):
                gamma = v3(scalar(0.0), scalar(0.0), y.z)
        else:
            mu_tilde = mu * wp.sqrt(rt / rn)
            mu_hat = mu * rt / rn
            factor = scalar(1.0) / (scalar(1.0) + mu_tilde * mu_tilde)
            if yr <= mu * y.z:
                gamma = y
            elif (-mu_hat * yr < y.z) and (y.z < yr / mu):
                gamma_n = (y.z + mu_hat * yr) * factor
                gamma = v3(mu * gamma_n * t_hat.x, mu * gamma_n * t_hat.y, gamma_n)

        cost = scalar(0.5) * (
            rt * (gamma.x * gamma.x + gamma.y * gamma.y) + rn * gamma.z * gamma.z
        )
        contact_gamma[env, c] = gamma
        contact_vc[env, c] = vc
        contact_cost[env, c] = cost
        wp.atomic_add(total_cost, env, cost)


    @wp.kernel(module="unique")
    def _projection_eval_contact_hessian_sap_batched(
        active_env: wp.array(dtype=int),
        max_contacts: int,
        contact_count: wp.array(dtype=int),
        contact_phi0: wp.array(dtype=scalar, ndim=2),
        contact_w_eff: wp.array(dtype=scalar, ndim=2),
        contact_mu: wp.array(dtype=scalar, ndim=2),
        contact_k: wp.array(dtype=scalar, ndim=2),
        contact_tau_d: wp.array(dtype=scalar, ndim=2),
        contact_vc: wp.array(dtype=vec3, ndim=2),
        beta: scalar,
        sigma: scalar,
        dt: scalar,
        contact_g: wp.array(dtype=mat33, ndim=2),
        contact_y: wp.array(dtype=vec3, ndim=2),
        contact_rt: wp.array(dtype=scalar, ndim=2),
        contact_rn: wp.array(dtype=scalar, ndim=2),
        contact_mode: wp.array(dtype=int, ndim=2),
    ):
        env, c = wp.tid()
        zero = v3(scalar(0.0), scalar(0.0), scalar(0.0))
        if c >= max_contacts or c >= contact_count[env]:
            return
        if active_env[env] == 0:
            contact_g[env, c] = zero_m33()
            contact_y[env, c] = zero
            contact_rt[env, c] = scalar(0.0)
            contact_rn[env, c] = scalar(0.0)
            contact_mode[env, c] = _CONTACT_MODE_NONE
            return

        wi = contact_w_eff[env, c]
        if wi < scalar(1.0e-12) or not wp.isfinite(wi):
            wi = scalar(1.0e-12)

        beta64 = scalar(beta)
        beta_factor = beta64 * beta64 / (scalar(4.0) * scalar(_PI) * scalar(_PI))
        rn_hard = beta_factor * wi
        k_c = contact_k[env, c]
        if k_c <= scalar(0.0) or not wp.isfinite(k_c):
            k_c = scalar(1.0)
        tau_c = wp.max(contact_tau_d[env, c], scalar(0.0))
        rn_soft = scalar(1.0) / (scalar(dt) * k_c * (scalar(dt) + tau_c))
        rn = wp.max(rn_hard, rn_soft)
        rt = scalar(sigma) * wi
        if rt < scalar(1.0e-30):
            rt = scalar(1.0e-30)
        if rn < scalar(1.0e-30):
            rn = scalar(1.0e-30)

        rt_inv = scalar(1.0) / rt
        rn_inv = scalar(1.0) / rn
        vhat_n = -contact_phi0[env, c] / (scalar(dt) + tau_c)
        vc = contact_vc[env, c]
        y = vec3(-rt_inv * vc.x, -rt_inv * vc.y, rn_inv * (vhat_n - vc.z))

        mu = contact_mu[env, c]
        if mu < scalar(0.0) or not wp.isfinite(mu):
            mu = scalar(0.0)

        yr = wp.sqrt(
            y.x * y.x
            + y.y * y.y
            + scalar(_CONTACT_SOFT_NORM_TOL) * scalar(_CONTACT_SOFT_NORM_TOL)
        )
        t_hat = zero
        if yr > scalar(0.0):
            t_hat = v3(y.x / yr, y.y / yr, scalar(0.0))

        g_mat = zero_m33()
        mode = _CONTACT_MODE_NONE

        if mu <= scalar(1.0e-12):
            if y.z > scalar(0.0):
                g_mat = m33(
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    rn_inv,
                )
                mode = _CONTACT_MODE_FRICTIONLESS
        else:
            mu_tilde = mu * wp.sqrt(rt / rn)
            mu_hat = mu * rt / rn
            factor = scalar(1.0) / (scalar(1.0) + mu_tilde * mu_tilde)

            if yr <= mu * y.z:
                g_mat = m33(
                    rt_inv,
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    rt_inv,
                    scalar(0.0),
                    scalar(0.0),
                    scalar(0.0),
                    rn_inv,
                )
                mode = _CONTACT_MODE_STICTION
            elif (-mu_hat * yr < y.z) and (y.z < yr / mu):
                gamma_n = (y.z + mu_hat * yr) * factor
                p00 = t_hat.x * t_hat.x
                p01 = t_hat.x * t_hat.y
                p10 = t_hat.y * t_hat.x
                p11 = t_hat.y * t_hat.y
                pp00 = scalar(1.0) - p00
                pp01 = -p01
                pp10 = -p10
                pp11 = scalar(1.0) - p11

                gn_over_yr = scalar(0.0)
                if yr > scalar(0.0):
                    gn_over_yr = gamma_n / yr

                dgt_dyt00 = mu * (gn_over_yr * pp00 + mu_hat * factor * p00)
                dgt_dyt01 = mu * (gn_over_yr * pp01 + mu_hat * factor * p01)
                dgt_dyt10 = mu * (gn_over_yr * pp10 + mu_hat * factor * p10)
                dgt_dyt11 = mu * (gn_over_yr * pp11 + mu_hat * factor * p11)
                dgt_dyn0 = mu * factor * t_hat.x
                dgt_dyn1 = mu * factor * t_hat.y
                dgn_dyt0 = mu_hat * factor * t_hat.x
                dgn_dyt1 = mu_hat * factor * t_hat.y

                g_mat = m33(
                    dgt_dyt00 * rt_inv,
                    dgt_dyt01 * rt_inv,
                    dgt_dyn0 * rn_inv,
                    dgt_dyt10 * rt_inv,
                    dgt_dyt11 * rt_inv,
                    dgt_dyn1 * rn_inv,
                    dgn_dyt0 * rt_inv,
                    dgn_dyt1 * rt_inv,
                    factor * rn_inv,
                )
                mode = _CONTACT_MODE_SLIDING

        contact_g[env, c] = g_mat
        contact_y[env, c] = y
        contact_rt[env, c] = rt
        contact_rn[env, c] = rn
        contact_mode[env, c] = mode


    @wp.kernel(module="unique")
    def _accumulate_pd_impulse_batched(
        add_pd: int,
        dof_per_env: int,
        pd_active: wp.array(dtype=int, ndim=2),
        pd_gamma: wp.array(dtype=scalar, ndim=2),
        constraint_impulse: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if add_pd == 1 and i < dof_per_env and pd_active[env, i] == 1:
            constraint_impulse[env, i] = constraint_impulse[env, i] + pd_gamma[env, i]


    @wp.kernel(module="unique")
    def _accumulate_limit_impulse_batched(
        add_limits: int,
        dof_per_env: int,
        limit_grad: wp.array(dtype=scalar, ndim=2),
        constraint_impulse: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if add_limits == 1 and i < dof_per_env:
            constraint_impulse[env, i] = constraint_impulse[env, i] + limit_grad[env, i]


    @cache
    def _make_contact_impulse_single_tile_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _contact_impulse_single_tile(
            dof_per_env: int,
            max_contacts: int,
            contact_count: wp.array(dtype=int),
            contact_jac: wp.array(dtype=scalar, ndim=4),
            contact_gamma: wp.array(dtype=vec3, ndim=2),
            constraint_impulse: wp.array(dtype=scalar, ndim=2),
        ):
            env, i, tid = wp.tid()
            if i >= dof_per_env:
                return

            count = contact_count[env]
            if count > max_contacts:
                count = max_contacts

            value = scalar(0.0)
            stride = wp.block_dim()
            c = tid
            while c < count:
                gamma = contact_gamma[env, c]
                value = value + contact_jac[env, c, 0, i] * gamma.x
                value = value + contact_jac[env, c, 1, i] * gamma.y
                value = value + contact_jac[env, c, 2, i] * gamma.z
                c = c + stride

            values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            values[tid] = value
            total = wp.tile_sum(values)[0]
            if tid == 0:
                constraint_impulse[env, i] = total

        return _contact_impulse_single_tile


    @cache
    def _make_contact_hessian_single_tile_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _contact_hessian_single_tile(
            dof_per_env: int,
            upper_count: int,
            max_contacts: int,
            contact_count: wp.array(dtype=int),
            contact_jac: wp.array(dtype=scalar, ndim=4),
            contact_g: wp.array(dtype=mat33, ndim=2),
            hess_contact: wp.array(dtype=scalar, ndim=3),
        ):
            env, upper, tid = wp.tid()
            if upper >= upper_count:
                return

            i = wp.int32(0)
            rem = wp.int32(upper)
            row_count = wp.int32(dof_per_env)
            while rem >= row_count:
                rem = rem - row_count
                i = i + 1
                row_count = row_count - 1
            j = i + rem

            count = contact_count[env]
            if count > max_contacts:
                count = max_contacts

            value = scalar(0.0)
            stride = wp.block_dim()
            c = tid
            while c < count:
                g = contact_g[env, c]
                for r in range(3):
                    ji = contact_jac[env, c, r, i]
                    for s in range(3):
                        value = value + ji * g[r, s] * contact_jac[env, c, s, j]
                c = c + stride

            values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            values[tid] = value
            total = wp.tile_sum(values)[0]
            if tid == 0:
                hess_contact[env, i, j] = total
                if j != i:
                    hess_contact[env, j, i] = total

        return _contact_hessian_single_tile


    @cache
    def _make_pack_contact_hessian_gemm_inputs_kernel(tile_k: int, tile_dof: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _pack_contact_hessian_gemm_inputs(
            dof_per_env: int,
            contact_capacity: int,
            padded_contact_rows: int,
            padded_dof: int,
            contact_count: wp.array(dtype=int),
            contact_jac: wp.array(dtype=scalar, ndim=4),
            contact_g: wp.array(dtype=mat33, ndim=2),
            contact_j_flat: wp.array(dtype=scalar, ndim=3),
            contact_gj_flat: wp.array(dtype=scalar, ndim=3),
        ):
            env, k_tile, dof_tile, tid = wp.tid()
            k0 = k_tile * tile_k
            d0 = dof_tile * tile_dof

            linear = tid
            while linear < tile_k * tile_dof:
                kk = linear // tile_dof
                dd = linear - kk * tile_dof
                k = k0 + kk
                d = d0 + dd

                j_value = scalar(0.0)
                gj_value = scalar(0.0)
                if k < padded_contact_rows and d < padded_dof:
                    c = k // 3
                    r = k - c * 3
                    count = contact_count[env]
                    if count > contact_capacity:
                        count = contact_capacity
                    if c < count and d < dof_per_env:
                        j_value = contact_jac[env, c, r, d]
                        g = contact_g[env, c]
                        gj_value = (
                            g[r, 0] * contact_jac[env, c, 0, d]
                            + g[r, 1] * contact_jac[env, c, 1, d]
                            + g[r, 2] * contact_jac[env, c, 2, d]
                        )
                    contact_j_flat[env, k, d] = j_value
                    contact_gj_flat[env, k, d] = gj_value
                linear = linear + wp.block_dim()

        return _pack_contact_hessian_gemm_inputs


    @cache
    def _make_contact_hessian_gemm_tile_kernel(tile_m: int, tile_n: int, tile_k: int):

        @wp.kernel(enable_backward=False, module="unique")
        def _contact_hessian_gemm_tile(
            dof_per_env: int,
            padded_contact_rows: int,
            contact_j_flat: wp.array(dtype=scalar, ndim=3),
            contact_gj_flat: wp.array(dtype=scalar, ndim=3),
            hess_contact: wp.array(dtype=scalar, ndim=3),
        ):
            env, row_tile, col_tile, tid = wp.tid()
            row0 = row_tile * tile_m
            col0 = col_tile * tile_n
            if row0 >= dof_per_env or col0 >= dof_per_env or col0 > row0 + tile_m - 1:
                return

            j_env = contact_j_flat[env]
            gj_env = contact_gj_flat[env]
            acc = wp.tile_zeros((tile_m, tile_n), dtype=scalar)
            k0 = wp.int32(0)
            while k0 < padded_contact_rows:
                j_tile = wp.tile_load(
                    j_env,
                    shape=(tile_k, tile_m),
                    offset=(k0, row0),
                    storage="shared",
                )
                gj_tile = wp.tile_load(
                    gj_env,
                    shape=(tile_k, tile_n),
                    offset=(k0, col0),
                    storage="shared",
                )
                wp.tile_matmul(wp.tile_transpose(j_tile), gj_tile, acc)
                k0 = k0 + tile_k

            linear = tid
            while linear < tile_m * tile_n:
                rr = linear // tile_n
                cc = linear - rr * tile_n
                i = row0 + rr
                j = col0 + cc
                if i < dof_per_env and j < dof_per_env and j <= i:
                    value = acc[rr, cc]
                    hess_contact[env, i, j] = value
                    if j != i:
                        hess_contact[env, j, i] = value
                linear = linear + wp.block_dim()

        return _contact_hessian_gemm_tile


    @cache
    def _make_assemble_grad_and_dynamics_impulse_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _assemble_grad_and_dynamics_impulse_tiled(
            active_env: wp.array(dtype=int),
            dof_per_env: int,
            v: wp.array(dtype=scalar, ndim=2),
            v_star: wp.array(dtype=scalar, ndim=2),
            a_mat: wp.array(dtype=scalar, ndim=3),
            add_pd: int,
            pd_active: wp.array(dtype=int, ndim=2),
            pd_gamma: wp.array(dtype=scalar, ndim=2),
            add_limits: int,
            limit_grad: wp.array(dtype=scalar, ndim=2),
            constraint_impulse: wp.array(dtype=scalar, ndim=2),
            grad: wp.array(dtype=scalar, ndim=2),
            dynamics_impulse: wp.array(dtype=scalar, ndim=2),
        ):
            env, i, tid = wp.tid()
            if i >= dof_per_env:
                return
            if active_env[env] == 0:
                if tid == 0:
                    constraint_impulse[env, i] = scalar(0.0)
                    grad[env, i] = scalar(0.0)
                    dynamics_impulse[env, i] = scalar(0.0)
                return

            local_a_res = scalar(0.0)
            local_a_v = scalar(0.0)
            stride = wp.block_dim()
            j = tid
            while j < dof_per_env:
                aij = a_mat[env, i, j]
                local_a_res = local_a_res + aij * (v[env, j] - v_star[env, j])
                local_a_v = local_a_v + aij * v[env, j]
                j = j + stride

            res_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            res_values[tid] = local_a_res
            v_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            v_values[tid] = local_a_v
            a_res = wp.tile_sum(res_values)[0]
            a_v = wp.tile_sum(v_values)[0]
            if tid == 0:
                impulse = constraint_impulse[env, i]
                if add_pd == 1 and pd_active[env, i] == 1:
                    impulse = impulse + pd_gamma[env, i]
                if add_limits == 1:
                    impulse = impulse + limit_grad[env, i]

                constraint_impulse[env, i] = impulse
                grad[env, i] = a_res - impulse
                dynamics_impulse[env, i] = a_v

        return _assemble_grad_and_dynamics_impulse_tiled


    @cache
    def _make_assemble_model_terms_and_grad_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _assemble_model_terms_and_grad_tiled(
            active_env: wp.array(dtype=int),
            dof_per_env: int,
            v: wp.array(dtype=scalar, ndim=2),
            v_star: wp.array(dtype=scalar, ndim=2),
            a_mat: wp.array(dtype=scalar, ndim=3),
            add_pd: int,
            pd_active: wp.array(dtype=int, ndim=2),
            pd_a: wp.array(dtype=scalar, ndim=2),
            pd_gain: wp.array(dtype=scalar, ndim=2),
            pd_limit: wp.array(dtype=scalar, ndim=2),
            dt: scalar,
            pd_y: wp.array(dtype=scalar, ndim=2),
            pd_gamma: wp.array(dtype=scalar, ndim=2),
            pd_hdiag: wp.array(dtype=scalar, ndim=2),
            pd_cost: wp.array(dtype=scalar, ndim=2),
            add_limits: int,
            lower_active: wp.array(dtype=int, ndim=2),
            upper_active: wp.array(dtype=int, ndim=2),
            lower_vhat: wp.array(dtype=scalar, ndim=2),
            upper_vhat: wp.array(dtype=scalar, ndim=2),
            lower_r: wp.array(dtype=scalar, ndim=2),
            upper_r: wp.array(dtype=scalar, ndim=2),
            lower_rinv: wp.array(dtype=scalar, ndim=2),
            upper_rinv: wp.array(dtype=scalar, ndim=2),
            lower_gamma_out: wp.array(dtype=scalar, ndim=2),
            upper_gamma_out: wp.array(dtype=scalar, ndim=2),
            limit_grad: wp.array(dtype=scalar, ndim=2),
            limit_hdiag: wp.array(dtype=scalar, ndim=2),
            limit_cost: wp.array(dtype=scalar, ndim=2),
            constraint_impulse: wp.array(dtype=scalar, ndim=2),
            grad: wp.array(dtype=scalar, ndim=2),
            dynamics_impulse: wp.array(dtype=scalar, ndim=2),
            total_cost: wp.array(dtype=scalar),
        ):
            env, i, tid = wp.tid()
            if i >= dof_per_env:
                return
            if active_env[env] == 0:
                if tid == 0:
                    pd_y[env, i] = scalar(0.0)
                    pd_gamma[env, i] = scalar(0.0)
                    pd_hdiag[env, i] = scalar(0.0)
                    pd_cost[env, i] = scalar(0.0)
                    lower_gamma_out[env, i] = scalar(0.0)
                    upper_gamma_out[env, i] = scalar(0.0)
                    limit_grad[env, i] = scalar(0.0)
                    limit_hdiag[env, i] = scalar(0.0)
                    limit_cost[env, i] = scalar(0.0)
                    constraint_impulse[env, i] = scalar(0.0)
                    grad[env, i] = scalar(0.0)
                    dynamics_impulse[env, i] = scalar(0.0)
                return

            local_a_res = scalar(0.0)
            local_a_v = scalar(0.0)
            stride = wp.block_dim()
            j = tid
            while j < dof_per_env:
                aij = a_mat[env, i, j]
                local_a_res = local_a_res + aij * (v[env, j] - v_star[env, j])
                local_a_v = local_a_v + aij * v[env, j]
                j = j + stride

            res_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            res_values[tid] = local_a_res
            a_res = wp.tile_sum(res_values)[0]
            v_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            v_values[tid] = local_a_v
            a_v = wp.tile_sum(v_values)[0]

            if tid == 0:
                vi = v[env, i]
                model_cost = scalar(0.5) * (vi - v_star[env, i]) * a_res

                pd_y_value = scalar(0.0)
                pd_gamma_value = scalar(0.0)
                pd_hdiag_value = scalar(0.0)
                pd_cost_value = scalar(0.0)
                if add_pd == 1 and pd_active[env, i] == 1:
                    gain = pd_gain[env, i]
                    pd_y_value = pd_a[env, i] - gain * vi
                    pd_gamma_value = scalar(dt) * clamp_scalar(pd_y_value, pd_limit[env, i])
                    pd_hdiag_value = scalar(dt) * gain * clamp_derivative(pd_y_value, pd_limit[env, i])
                    if gain > scalar(0.0):
                        pd_cost_value = (scalar(dt) / gain) * clamp_antiderivative(
                            pd_y_value, pd_limit[env, i]
                        )
                    model_cost = model_cost + pd_cost_value

                limit_grad_value = scalar(0.0)
                limit_hdiag_value = scalar(0.0)
                limit_cost_value = scalar(0.0)
                lower_gamma = scalar(0.0)
                upper_gamma = scalar(0.0)
                if add_limits == 1:
                    if lower_active[env, i] == 1:
                        limit_gamma = lower_rinv[env, i] * (lower_vhat[env, i] - vi)
                        if limit_gamma > scalar(0.0):
                            lower_gamma = limit_gamma
                            limit_grad_value = limit_grad_value + limit_gamma
                            limit_hdiag_value = limit_hdiag_value + lower_rinv[env, i]
                            limit_cost_value = (
                                limit_cost_value + scalar(0.5) * lower_r[env, i] * limit_gamma * limit_gamma
                            )
                    if upper_active[env, i] == 1:
                        limit_gamma = upper_rinv[env, i] * (upper_vhat[env, i] + vi)
                        if limit_gamma > scalar(0.0):
                            upper_gamma = limit_gamma
                            limit_grad_value = limit_grad_value - limit_gamma
                            limit_hdiag_value = limit_hdiag_value + upper_rinv[env, i]
                            limit_cost_value = (
                                limit_cost_value + scalar(0.5) * upper_r[env, i] * limit_gamma * limit_gamma
                            )
                    model_cost = model_cost + limit_cost_value

                impulse = constraint_impulse[env, i] + pd_gamma_value + limit_grad_value
                constraint_impulse[env, i] = impulse
                grad[env, i] = a_res - impulse
                dynamics_impulse[env, i] = a_v
                pd_y[env, i] = pd_y_value
                pd_gamma[env, i] = pd_gamma_value
                pd_hdiag[env, i] = pd_hdiag_value
                pd_cost[env, i] = pd_cost_value
                lower_gamma_out[env, i] = lower_gamma
                upper_gamma_out[env, i] = upper_gamma
                limit_grad[env, i] = limit_grad_value
                limit_hdiag[env, i] = limit_hdiag_value
                limit_cost[env, i] = limit_cost_value
                wp.atomic_add(total_cost, env, model_cost)

        return _assemble_model_terms_and_grad_tiled


    @wp.kernel(module="unique")
    def _assemble_hessian_total_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        a_mat: wp.array(dtype=scalar, ndim=3),
        hess_contact: wp.array(dtype=scalar, ndim=3),
        has_pd: int,
        has_limits: int,
        pd_hdiag: wp.array(dtype=scalar, ndim=2),
        limit_hdiag: wp.array(dtype=scalar, ndim=2),
        hess: wp.array(dtype=scalar, ndim=3),
    ):
        env, i, j = wp.tid()
        if i >= dof_per_env or j >= dof_per_env:
            return
        if active_env[env] == 0:
            hess[env, i, j] = scalar(0.0)
            return
        value = a_mat[env, i, j] + hess_contact[env, i, j]
        if i == j:
            if has_pd == 1:
                value = value + pd_hdiag[env, i]
            if has_limits == 1:
                value = value + limit_hdiag[env, i]
        hess[env, i, j] = value


    @cache
    def _make_compute_base_cost_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _compute_base_cost_tiled(
            active_env: wp.array(dtype=int),
            dof_per_env: int,
            v: wp.array(dtype=scalar, ndim=2),
            v_star: wp.array(dtype=scalar, ndim=2),
            a_mat: wp.array(dtype=scalar, ndim=3),
            out_cost: wp.array(dtype=scalar),
        ):
            env, tid = wp.tid()
            if active_env[env] == 0:
                if tid == 0:
                    out_cost[env] = scalar(0.0)
                return

            local_cost = scalar(0.0)
            stride = wp.block_dim()
            entry = tid
            entry_count = dof_per_env * dof_per_env
            while entry < entry_count:
                i = entry // dof_per_env
                j = entry - i * dof_per_env
                residual_i = v[env, i] - v_star[env, i]
                residual_j = v[env, j] - v_star[env, j]
                local_cost = local_cost + scalar(0.5) * residual_i * a_mat[env, i, j] * residual_j
                entry = entry + stride

            values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            values[tid] = local_cost
            total = wp.tile_sum(values)[0]
            if tid == 0:
                out_cost[env] = total

        return _compute_base_cost_tiled


    @cache
    def _make_compute_line_search_base_coeffs_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _compute_line_search_base_coeffs_tiled(
            active_env: wp.array(dtype=int),
            dof_per_env: int,
            v: wp.array(dtype=scalar, ndim=2),
            v_star: wp.array(dtype=scalar, ndim=2),
            dv: wp.array(dtype=scalar, ndim=2),
            a_mat: wp.array(dtype=scalar, ndim=3),
            out_base0: wp.array(dtype=scalar),
            out_base_linear: wp.array(dtype=scalar),
            out_base_quadratic: wp.array(dtype=scalar),
        ):
            env, tid = wp.tid()
            if active_env[env] == 0:
                if tid == 0:
                    out_base0[env] = scalar(0.0)
                    out_base_linear[env] = scalar(0.0)
                    out_base_quadratic[env] = scalar(0.0)
                return

            local_base0 = scalar(0.0)
            local_base_linear = scalar(0.0)
            local_base_quadratic = scalar(0.0)
            stride = wp.block_dim()
            entry = tid
            entry_count = dof_per_env * dof_per_env
            while entry < entry_count:
                i = entry // dof_per_env
                j = entry - i * dof_per_env
                residual_i = v[env, i] - v_star[env, i]
                residual_j = v[env, j] - v_star[env, j]
                dv_i = dv[env, i]
                dv_j = dv[env, j]
                aij = a_mat[env, i, j]
                local_base0 = local_base0 + scalar(0.5) * residual_i * aij * residual_j
                local_base_linear = local_base_linear + dv_i * aij * residual_j
                local_base_quadratic = local_base_quadratic + dv_i * aij * dv_j
                entry = entry + stride

            base_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            base_values[tid] = local_base0
            base0 = wp.tile_sum(base_values)[0]

            linear_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            linear_values[tid] = local_base_linear
            base_linear = wp.tile_sum(linear_values)[0]

            quadratic_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            quadratic_values[tid] = local_base_quadratic
            base_quadratic = wp.tile_sum(quadratic_values)[0]

            if tid == 0:
                out_base0[env] = base0
                out_base_linear[env] = base_linear
                out_base_quadratic[env] = base_quadratic

        return _compute_line_search_base_coeffs_tiled


    @cache
    def _make_compute_line_search_contact_delta_velocity_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _compute_line_search_contact_delta_velocity_tiled(
            active_env: wp.array(dtype=int),
            dof_per_env: int,
            max_contacts: int,
            contact_count: wp.array(dtype=int),
            contact_jac: wp.array(dtype=scalar, ndim=4),
            dv: wp.array(dtype=scalar, ndim=2),
            out_dvc: wp.array(dtype=vec3, ndim=2),
        ):
            env, c, tid = wp.tid()
            zero = v3(scalar(0.0), scalar(0.0), scalar(0.0))
            if c >= max_contacts or c >= contact_count[env]:
                return
            if active_env[env] == 0:
                if tid == 0:
                    out_dvc[env, c] = zero
                return

            local_dvcx = scalar(0.0)
            local_dvcy = scalar(0.0)
            local_dvcz = scalar(0.0)
            stride = wp.block_dim()
            j = tid
            while j < dof_per_env:
                dvj = dv[env, j]
                local_dvcx = local_dvcx + contact_jac[env, c, 0, j] * dvj
                local_dvcy = local_dvcy + contact_jac[env, c, 1, j] * dvj
                local_dvcz = local_dvcz + contact_jac[env, c, 2, j] * dvj
                j = j + stride

            dvcx_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            dvcx_values[tid] = local_dvcx
            dvcy_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            dvcy_values[tid] = local_dvcy
            dvcz_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            dvcz_values[tid] = local_dvcz

            dvcx = wp.tile_sum(dvcx_values)[0]
            dvcy = wp.tile_sum(dvcy_values)[0]
            dvcz = wp.tile_sum(dvcz_values)[0]

            if tid == 0:
                out_dvc[env, c] = v3(dvcx, dvcy, dvcz)

        return _compute_line_search_contact_delta_velocity_tiled


    @cache
    def _make_compute_norm_terms_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _compute_norm_terms_tiled(
            active_env: wp.array(dtype=int),
            dof_per_env: int,
            participating_dof: wp.array(dtype=int, ndim=2),
            d_scale: wp.array(dtype=scalar, ndim=2),
            grad: wp.array(dtype=scalar, ndim=2),
            dynamics_impulse: wp.array(dtype=scalar, ndim=2),
            constraint_impulse: wp.array(dtype=scalar, ndim=2),
            grad_norm2: wp.array(dtype=scalar),
            p_norm2: wp.array(dtype=scalar),
            jc_norm2: wp.array(dtype=scalar),
        ):
            env, tid = wp.tid()
            if active_env[env] == 0:
                if tid == 0:
                    grad_norm2[env] = scalar(0.0)
                    p_norm2[env] = scalar(0.0)
                    jc_norm2[env] = scalar(0.0)
                return

            grad_acc = scalar(0.0)
            p_acc = scalar(0.0)
            jc_acc = scalar(0.0)
            stride = wp.block_dim()
            i = tid
            while i < dof_per_env:
                if participating_dof[env, i] == 1:
                    s = d_scale[env, i]
                    g = s * grad[env, i]
                    p = s * dynamics_impulse[env, i]
                    jc = s * constraint_impulse[env, i]
                    grad_acc = grad_acc + g * g
                    p_acc = p_acc + p * p
                    jc_acc = jc_acc + jc * jc
                i = i + stride

            grad_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            grad_values[tid] = grad_acc
            p_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            p_values[tid] = p_acc
            jc_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            jc_values[tid] = jc_acc

            grad_total = wp.tile_sum(grad_values)[0]
            p_total = wp.tile_sum(p_values)[0]
            jc_total = wp.tile_sum(jc_values)[0]
            if tid == 0:
                grad_norm2[env] = grad_total
                p_norm2[env] = p_total
                jc_norm2[env] = jc_total

        return _compute_norm_terms_tiled


    @cache
    def _make_compute_norm_terms_and_update_active_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _compute_norm_terms_and_update_active_tiled(
            active_env_input: wp.array(dtype=int),
            dof_per_env: int,
            participating_dof: wp.array(dtype=int, ndim=2),
            d_scale: wp.array(dtype=scalar, ndim=2),
            grad: wp.array(dtype=scalar, ndim=2),
            dynamics_impulse: wp.array(dtype=scalar, ndim=2),
            constraint_impulse: wp.array(dtype=scalar, ndim=2),
            cost: wp.array(dtype=scalar),
            previous_cost: wp.array(dtype=scalar),
            alpha: wp.array(dtype=scalar),
            iteration: int,
            optimality_abs_tol: scalar,
            optimality_rel_tol: scalar,
            cost_abs_tol: scalar,
            cost_rel_tol: scalar,
            cost_min_alpha: scalar,
            single_env_count: int,
            active_env: wp.array(dtype=int),
            converged_env: wp.array(dtype=int),
            optimality_reached_env: wp.array(dtype=int),
            cost_reached_env: wp.array(dtype=int),
            newton_iterations_env: wp.array(dtype=int),
            active_count: wp.array(dtype=int),
            grad_norm2: wp.array(dtype=scalar),
            p_norm2: wp.array(dtype=scalar),
            jc_norm2: wp.array(dtype=scalar),
        ):
            env, tid = wp.tid()
            if active_env_input[env] == 0 or converged_env[env] == 1:
                if tid == 0:
                    grad_norm2[env] = scalar(0.0)
                    p_norm2[env] = scalar(0.0)
                    jc_norm2[env] = scalar(0.0)
                    active_env[env] = 0
                    if single_env_count == 1:
                        active_count[0] = 0
                return

            grad_acc = scalar(0.0)
            p_acc = scalar(0.0)
            jc_acc = scalar(0.0)
            stride = wp.block_dim()
            i = tid
            while i < dof_per_env:
                if participating_dof[env, i] == 1:
                    s = d_scale[env, i]
                    g = s * grad[env, i]
                    p = s * dynamics_impulse[env, i]
                    jc = s * constraint_impulse[env, i]
                    grad_acc = grad_acc + g * g
                    p_acc = p_acc + p * p
                    jc_acc = jc_acc + jc * jc
                i = i + stride

            grad_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            grad_values[tid] = grad_acc
            p_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            p_values[tid] = p_acc
            jc_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            jc_values[tid] = jc_acc

            grad_total = wp.tile_sum(grad_values)[0]
            p_total = wp.tile_sum(p_values)[0]
            jc_total = wp.tile_sum(jc_values)[0]

            if tid == 0:
                grad_norm2[env] = grad_total
                p_norm2[env] = p_total
                jc_norm2[env] = jc_total

                grad_norm = wp.sqrt(wp.max(grad_total, scalar(0.0)))
                p_norm = wp.sqrt(wp.max(p_total, scalar(0.0)))
                jc_norm = wp.sqrt(wp.max(jc_total, scalar(0.0)))
                opt_tol = scalar(optimality_abs_tol) + scalar(optimality_rel_tol) * wp.max(p_norm, jc_norm)

                opt_reached = 0
                cost_reached = 0
                if grad_norm <= opt_tol:
                    opt_reached = 1
                if iteration > 0 and alpha[env] > cost_min_alpha:
                    scale = scalar(0.5) * (wp.abs(cost[env]) + wp.abs(previous_cost[env]))
                    tol = scalar(cost_abs_tol) + scalar(cost_rel_tol) * scale
                    if wp.abs(cost[env] - previous_cost[env]) < tol:
                        cost_reached = 1

                active = 1
                if opt_reached == 1 or cost_reached == 1:
                    active = 0

                active_env[env] = active
                optimality_reached_env[env] = opt_reached
                cost_reached_env[env] = cost_reached
                converged_env[env] = 1 - active
                if active == 0:
                    newton_iterations_env[env] = iteration
                if single_env_count == 1:
                    active_count[0] = active
                elif active == 1:
                    wp.atomic_add(active_count, 0, 1)

        return _compute_norm_terms_and_update_active_tiled


    @cache
    def _make_compute_norm_terms_and_update_active_conditional_tiled_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _compute_norm_terms_and_update_active_conditional_tiled(
            active_env_input: wp.array(dtype=int),
            dof_per_env: int,
            participating_dof: wp.array(dtype=int, ndim=2),
            d_scale: wp.array(dtype=scalar, ndim=2),
            grad: wp.array(dtype=scalar, ndim=2),
            dynamics_impulse: wp.array(dtype=scalar, ndim=2),
            constraint_impulse: wp.array(dtype=scalar, ndim=2),
            cost: wp.array(dtype=scalar),
            previous_cost: wp.array(dtype=scalar),
            alpha: wp.array(dtype=scalar),
            iteration_count: wp.array(dtype=int),
            max_iterations: int,
            optimality_abs_tol: scalar,
            optimality_rel_tol: scalar,
            cost_abs_tol: scalar,
            cost_rel_tol: scalar,
            cost_min_alpha: scalar,
            single_env_count: int,
            active_env: wp.array(dtype=int),
            converged_env: wp.array(dtype=int),
            optimality_reached_env: wp.array(dtype=int),
            cost_reached_env: wp.array(dtype=int),
            newton_iterations_env: wp.array(dtype=int),
            active_count: wp.array(dtype=int),
            max_reached: wp.array(dtype=int),
            grad_norm2: wp.array(dtype=scalar),
            p_norm2: wp.array(dtype=scalar),
            jc_norm2: wp.array(dtype=scalar),
        ):
            env, tid = wp.tid()
            iteration = iteration_count[0]
            if active_env_input[env] == 0 or converged_env[env] == 1:
                if tid == 0:
                    grad_norm2[env] = scalar(0.0)
                    p_norm2[env] = scalar(0.0)
                    jc_norm2[env] = scalar(0.0)
                    active_env[env] = 0
                    if single_env_count == 1:
                        active_count[0] = 0
                return

            grad_acc = scalar(0.0)
            p_acc = scalar(0.0)
            jc_acc = scalar(0.0)
            stride = wp.block_dim()
            i = tid
            while i < dof_per_env:
                if participating_dof[env, i] == 1:
                    s = d_scale[env, i]
                    g = s * grad[env, i]
                    p = s * dynamics_impulse[env, i]
                    jc = s * constraint_impulse[env, i]
                    grad_acc = grad_acc + g * g
                    p_acc = p_acc + p * p
                    jc_acc = jc_acc + jc * jc
                i = i + stride

            grad_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            grad_values[tid] = grad_acc
            p_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            p_values[tid] = p_acc
            jc_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
            jc_values[tid] = jc_acc

            grad_total = wp.tile_sum(grad_values)[0]
            p_total = wp.tile_sum(p_values)[0]
            jc_total = wp.tile_sum(jc_values)[0]

            if tid == 0:
                grad_norm2[env] = grad_total
                p_norm2[env] = p_total
                jc_norm2[env] = jc_total

                grad_norm = wp.sqrt(wp.max(grad_total, scalar(0.0)))
                p_norm = wp.sqrt(wp.max(p_total, scalar(0.0)))
                jc_norm = wp.sqrt(wp.max(jc_total, scalar(0.0)))
                opt_tol = scalar(optimality_abs_tol) + scalar(optimality_rel_tol) * wp.max(p_norm, jc_norm)

                opt_reached = 0
                cost_reached = 0
                if grad_norm <= opt_tol:
                    opt_reached = 1
                if iteration > 0 and alpha[env] > cost_min_alpha:
                    scale = scalar(0.5) * (wp.abs(cost[env]) + wp.abs(previous_cost[env]))
                    tol = scalar(cost_abs_tol) + scalar(cost_rel_tol) * scale
                    if wp.abs(cost[env] - previous_cost[env]) < tol:
                        cost_reached = 1

                optimality_reached_env[env] = opt_reached
                cost_reached_env[env] = cost_reached
                if opt_reached == 1 or cost_reached == 1:
                    active_env[env] = 0
                    converged_env[env] = 1
                    newton_iterations_env[env] = iteration
                    if single_env_count == 1:
                        active_count[0] = 0
                    return

                newton_iterations_env[env] = iteration
                if iteration >= max_iterations:
                    active_env[env] = 0
                    max_reached[0] = 1
                    if single_env_count == 1:
                        active_count[0] = 0
                    return

                active_env[env] = 1
                converged_env[env] = 0
                if single_env_count == 1:
                    active_count[0] = 1
                else:
                    wp.atomic_add(active_count, 0, 1)

        return _compute_norm_terms_and_update_active_conditional_tiled


    @wp.kernel(module="unique")
    def _increment_scalar_i32(value: wp.array(dtype=int)):
        value[0] = value[0] + 1


    @wp.kernel(module="unique")
    def _set_scalar_i32(value: wp.array(dtype=int), new_value: int):
        value[0] = new_value


    @cache
    def _create_pack_dense_to_padded_batched_kernel(dtype):
        @wp.kernel(module="unique")
        def _pack_dense_to_padded_batched(
            src: wp.array(dtype=scalar, ndim=3),
            dst: wp.array(dtype=dtype, ndim=3),
            active_size: int,
            diag_shift: scalar,
        ):
            env, i, j = wp.tid()
            if i < active_size and j < active_size:
                out = dtype(src[env, i, j])
                if i == j:
                    out = out + dtype(diag_shift)
                dst[env, i, j] = out
            elif i == j:
                dst[env, i, j] = dtype(1.0)
            else:
                dst[env, i, j] = dtype(0.0)

        return _pack_dense_to_padded_batched


    @cache
    def _create_pack_grad_to_padded_rhs_batched_kernel(dtype):
        @wp.kernel(module="unique")
        def _pack_grad_to_padded_rhs_batched(
            grad: wp.array(dtype=scalar, ndim=2),
            rhs: wp.array(dtype=dtype, ndim=3),
            active_size: int,
        ):
            env, i = wp.tid()
            if i < active_size:
                rhs[env, i, 0] = -dtype(grad[env, i])
            else:
                rhs[env, i, 0] = dtype(0.0)

        return _pack_grad_to_padded_rhs_batched


    @cache
    def _create_pack_dense_and_grad_to_padded_batched_kernel(dtype):
        @wp.kernel(module="unique")
        def _pack_dense_and_grad_to_padded_batched(
            hessian: wp.array(dtype=scalar, ndim=3),
            grad: wp.array(dtype=scalar, ndim=2),
            chol_a: wp.array(dtype=dtype, ndim=3),
            rhs: wp.array(dtype=dtype, ndim=3),
            active_size: int,
            diag_shift: scalar,
        ):
            env, i, j = wp.tid()
            if i < active_size and j < active_size:
                out = dtype(hessian[env, i, j])
                if i == j:
                    out = out + dtype(diag_shift)
                chol_a[env, i, j] = out
            elif i == j:
                chol_a[env, i, j] = dtype(1.0)
            else:
                chol_a[env, i, j] = dtype(0.0)

            if j == 0:
                if i < active_size:
                    rhs[env, i, 0] = -dtype(grad[env, i])
                else:
                    rhs[env, i, 0] = dtype(0.0)

        return _pack_dense_and_grad_to_padded_batched


    @cache
    def _create_unpack_solution_batched_kernel(dtype):
        @wp.kernel(module="unique")
        def _unpack_solution_batched(
            src: wp.array(dtype=dtype, ndim=3),
            dst: wp.array(dtype=scalar, ndim=2),
            active_size: int,
        ):
            env, i = wp.tid()
            if i < active_size:
                dst[env, i] = scalar(src[env, i, 0])

        return _unpack_solution_batched


    @cache
    def _create_unpack_solution_and_first_batched_kernel(dtype):
        @wp.kernel(module="unique")
        def _unpack_solution_and_first_batched(
            active_env: wp.array(dtype=int),
            loop_iteration: wp.array(dtype=int),
            src: wp.array(dtype=dtype, ndim=3),
            dst: wp.array(dtype=scalar, ndim=2),
            first_dv: wp.array(dtype=scalar, ndim=2),
            active_size: int,
        ):
            env, i = wp.tid()
            if i < active_size:
                value = scalar(src[env, i, 0])
                dst[env, i] = value
                if active_env[env] == 1 and loop_iteration[0] == 0:
                    first_dv[env, i] = value

        return _unpack_solution_and_first_batched


    @wp.kernel(module="unique")
    def _compute_search_direction_data_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        a_mat: wp.array(dtype=scalar, ndim=3),
        v: wp.array(dtype=scalar, ndim=2),
        v_star: wp.array(dtype=scalar, ndim=2),
        grad: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        dp: wp.array(dtype=scalar, ndim=2),
        dell0: wp.array(dtype=scalar),
        dell_a0: wp.array(dtype=scalar),
        d2ell_a: wp.array(dtype=scalar),
    ):
        env, i = wp.tid()
        if active_env[env] == 0 or i >= dof_per_env:
            return

        acc = scalar(0.0)
        for j in range(dof_per_env):
            acc = acc + a_mat[env, i, j] * dv[env, j]
        dp[env, i] = acc
        wp.atomic_add(dell0, env, grad[env, i] * dv[env, i])
        wp.atomic_add(dell_a0, env, acc * (v[env, i] - v_star[env, i]))
        wp.atomic_add(d2ell_a, env, dv[env, i] * acc)


    @wp.kernel(module="unique")
    def _compute_search_direction_data_serial_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        a_mat: wp.array(dtype=scalar, ndim=3),
        v: wp.array(dtype=scalar, ndim=2),
        v_star: wp.array(dtype=scalar, ndim=2),
        grad: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        dp: wp.array(dtype=scalar, ndim=2),
        dell0: wp.array(dtype=scalar),
        dell_a0: wp.array(dtype=scalar),
        d2ell_a: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        d0 = scalar(0.0)
        da0 = scalar(0.0)
        d2 = scalar(0.0)
        if active_env[env] == 1:
            for i in range(dof_per_env):
                acc = scalar(0.0)
                for j in range(dof_per_env):
                    acc = acc + a_mat[env, i, j] * dv[env, j]
                dp[env, i] = acc
                d0 = d0 + grad[env, i] * dv[env, i]
                da0 = da0 + acc * (v[env, i] - v_star[env, i])
                d2 = d2 + dv[env, i] * acc
        dell0[env] = d0
        dell_a0[env] = da0
        d2ell_a[env] = d2


    @wp.kernel(module="unique")
    def _axpy_to_trial_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        x: wp.array(dtype=scalar, ndim=2),
        direction: wp.array(dtype=scalar, ndim=2),
        alpha: wp.array(dtype=scalar),
        out: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return
        if active_env[env] == 1:
            out[env, i] = x[env, i] + alpha[env] * direction[env, i]
        else:
            out[env, i] = x[env, i]


    @wp.kernel(module="unique")
    def _compute_line_derivative_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        v_trial: wp.array(dtype=scalar, ndim=2),
        v_star: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        dp: wp.array(dtype=scalar, ndim=2),
        constraint_impulse: wp.array(dtype=scalar, ndim=2),
        derivative: wp.array(dtype=scalar),
    ):
        env, i = wp.tid()
        if active_env[env] == 0 or i >= dof_per_env:
            return
        value = dp[env, i] * (v_trial[env, i] - v_star[env, i])
        value = value - dv[env, i] * constraint_impulse[env, i]
        wp.atomic_add(derivative, env, value)


    @wp.kernel(module="unique")
    def _compute_line_derivative_serial_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        v_trial: wp.array(dtype=scalar, ndim=2),
        v_star: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        dp: wp.array(dtype=scalar, ndim=2),
        constraint_impulse: wp.array(dtype=scalar, ndim=2),
        derivative: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        out = scalar(0.0)
        if active_env[env] == 1:
            for i in range(dof_per_env):
                value = dp[env, i] * (v_trial[env, i] - v_star[env, i])
                value = value - dv[env, i] * constraint_impulse[env, i]
                out = out + value
        derivative[env] = out


    @wp.kernel(module="unique")
    def _compute_line_second_derivative_serial_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        max_contacts: int,
        contact_count: wp.array(dtype=int),
        contact_jac: wp.array(dtype=scalar, ndim=4),
        contact_g: wp.array(dtype=mat33, ndim=2),
        has_pd: int,
        has_limits: int,
        pd_hdiag: wp.array(dtype=scalar, ndim=2),
        limit_hdiag: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        d2ell_a: wp.array(dtype=scalar),
        derivative2: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        if active_env[env] == 0:
            derivative2[env] = scalar(0.0)
            return

        out = d2ell_a[env]
        count = contact_count[env]
        if count > max_contacts:
            count = max_contacts
        c = int(0)
        while c < count:
            dvc = v3(scalar(0.0), scalar(0.0), scalar(0.0))
            for j in range(dof_per_env):
                dvj = dv[env, j]
                dvc = vec3(
                    dvc.x + contact_jac[env, c, 0, j] * dvj,
                    dvc.y + contact_jac[env, c, 1, j] * dvj,
                    dvc.z + contact_jac[env, c, 2, j] * dvj,
                )
            g = contact_g[env, c]
            gdvc = v3(
                g[0, 0] * dvc.x + g[0, 1] * dvc.y + g[0, 2] * dvc.z,
                g[1, 0] * dvc.x + g[1, 1] * dvc.y + g[1, 2] * dvc.z,
                g[2, 0] * dvc.x + g[2, 1] * dvc.y + g[2, 2] * dvc.z,
            )
            out = out + dvc.x * gdvc.x + dvc.y * gdvc.y + dvc.z * gdvc.z
            c = c + 1

        for i in range(dof_per_env):
            hdiag = scalar(0.0)
            if has_pd == 1:
                hdiag = hdiag + pd_hdiag[env, i]
            if has_limits == 1:
                hdiag = hdiag + limit_hdiag[env, i]
            out = out + hdiag * dv[env, i] * dv[env, i]
        derivative2[env] = out


    @wp.kernel(module="unique")
    def _replace_trial_cost_with_sap_line_search_cost_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        max_contacts: int,
        contact_count: wp.array(dtype=int),
        contact_cost: wp.array(dtype=scalar, ndim=2),
        has_pd: int,
        has_limits: int,
        pd_cost: wp.array(dtype=scalar, ndim=2),
        limit_cost: wp.array(dtype=scalar, ndim=2),
        momentum_cost0: wp.array(dtype=scalar),
        dell_a0: wp.array(dtype=scalar),
        d2ell_a: wp.array(dtype=scalar),
        alpha: wp.array(dtype=scalar),
        trial_cost: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        if active_env[env] == 0:
            return

        regularizer_cost = scalar(0.0)
        count = contact_count[env]
        if count > max_contacts:
            count = max_contacts
        c = int(0)
        while c < count:
            regularizer_cost = regularizer_cost + contact_cost[env, c]
            c = c + 1
        for i in range(dof_per_env):
            if has_pd == 1:
                regularizer_cost = regularizer_cost + pd_cost[env, i]
            if has_limits == 1:
                regularizer_cost = regularizer_cost + limit_cost[env, i]

        a = alpha[env]
        momentum_cost = (
            momentum_cost0[env]
            + a * dell_a0[env]
            + scalar(0.5) * a * a * d2ell_a[env]
        )
        trial_cost[env] = momentum_cost + regularizer_cost


    @wp.kernel(module="unique")
    def _init_unit_decay_line_search_state(
        newton_active: wp.array(dtype=int),
        current_cost: wp.array(dtype=scalar),
        alpha: wp.array(dtype=scalar),
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        alpha[env] = scalar(1.0)
        ls_active[env] = 0
        ls_accepted[env] = 0
        ls_status[env] = 0
        ls_iterations[env] = 0
        accepted_cost[env] = current_cost[env]
        if newton_active[env] == 1:
            ls_active[env] = 1


    @wp.kernel(module="unique")
    def _update_unit_decay_line_search_state(
        trial_cost: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        alpha: wp.array(dtype=scalar),
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
        next_active_count: wp.array(dtype=int),
        min_alpha: scalar,
        cost_relax_r: scalar,
        cost_relax_a: scalar,
        decay: scalar,
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return

        ls_iterations[env] = ls_iterations[env] + 1
        current = current_cost[env]
        tol = cost_relax_a + cost_relax_r * wp.abs(current)
        if trial_cost[env] <= current + tol:
            ls_active[env] = 0
            ls_accepted[env] = 1
            accepted_cost[env] = trial_cost[env]
            return

        a = alpha[env] * decay
        alpha[env] = a
        if a < min_alpha:
            ls_active[env] = 0
            ls_status[env] = -2
            return

        wp.atomic_add(next_active_count, 0, 1)


    @cache
    def _make_unit_decay_line_search_fused_parallel_kernel(tile_size: int):
        @wp.kernel(enable_backward=False, module="unique")
        def _run_unit_decay_line_search_fused_parallel_batched(
            newton_active: wp.array(dtype=int),
            dof_per_env: int,
            max_contacts: int,
            contact_count: wp.array(dtype=int),
            contact_vc0: wp.array(dtype=vec3, ndim=2),
            contact_dvc: wp.array(dtype=vec3, ndim=2),
            contact_phi0: wp.array(dtype=scalar, ndim=2),
            contact_w_eff: wp.array(dtype=scalar, ndim=2),
            contact_mu: wp.array(dtype=scalar, ndim=2),
            contact_k: wp.array(dtype=scalar, ndim=2),
            contact_tau_d: wp.array(dtype=scalar, ndim=2),
            pd_active: wp.array(dtype=int, ndim=2),
            pd_a: wp.array(dtype=scalar, ndim=2),
            pd_gain: wp.array(dtype=scalar, ndim=2),
            pd_limit: wp.array(dtype=scalar, ndim=2),
            limit_lower_active: wp.array(dtype=int, ndim=2),
            limit_upper_active: wp.array(dtype=int, ndim=2),
            limit_lower_vhat: wp.array(dtype=scalar, ndim=2),
            limit_upper_vhat: wp.array(dtype=scalar, ndim=2),
            limit_lower_r: wp.array(dtype=scalar, ndim=2),
            limit_upper_r: wp.array(dtype=scalar, ndim=2),
            limit_lower_rinv: wp.array(dtype=scalar, ndim=2),
            limit_upper_rinv: wp.array(dtype=scalar, ndim=2),
            v: wp.array(dtype=scalar, ndim=2),
            dv: wp.array(dtype=scalar, ndim=2),
            v_flat: wp.array(dtype=scalar),
            line_base0: wp.array(dtype=scalar),
            line_base_linear: wp.array(dtype=scalar),
            line_base_quadratic: wp.array(dtype=scalar),
            beta: scalar,
            sigma: scalar,
            dt: scalar,
            has_contact_terms: int,
            has_pd_terms: int,
            has_limit_terms: int,
            max_iterations: int,
            decay: scalar,
            min_alpha: scalar,
            cost_relax_r: scalar,
            cost_relax_a: scalar,
            alpha_out: wp.array(dtype=scalar),
            current_cost: wp.array(dtype=scalar),
            previous_cost: wp.array(dtype=scalar),
            accepted_cost: wp.array(dtype=scalar),
            ls_active: wp.array(dtype=int),
            ls_accepted: wp.array(dtype=int),
            ls_status: wp.array(dtype=int),
            ls_iterations: wp.array(dtype=int),
            ls_iterations_total: wp.array(dtype=int),
        ):
            env, tid = wp.tid()
            current = current_cost[env]
            stride = wp.block_dim()

            if newton_active[env] == 0:
                i = tid
                while i < dof_per_env:
                    v_flat[env * dof_per_env + i] = v[env, i]
                    i = i + stride
                if tid == 0:
                    alpha_out[env] = scalar(1.0)
                    ls_active[env] = 0
                    ls_accepted[env] = 0
                    ls_status[env] = 0
                    ls_iterations[env] = 0
                    accepted_cost[env] = current
                return

            base0 = line_base0[env]
            base_linear = line_base_linear[env]
            base_quadratic = line_base_quadratic[env]

            alpha = scalar(1.0)
            trial_cost = current
            accepted_flag = int(0)
            status_value = int(0)
            iterations_value = int(0)

            for _ in range(max_iterations):
                momentum_cost = base0 + alpha * base_linear + scalar(0.5) * alpha * alpha * base_quadratic

                local_regularizer = scalar(0.0)
                if has_contact_terms != 0:
                    count = contact_count[env]
                    if count > max_contacts:
                        count = max_contacts
                    c = tid
                    while c < count:
                        vc_base = contact_vc0[env, c]
                        dvc = contact_dvc[env, c]
                        vc = v3(
                            vc_base.x + alpha * dvc.x,
                            vc_base.y + alpha * dvc.y,
                            vc_base.z + alpha * dvc.z,
                        )
                        local_regularizer = local_regularizer + contact_projection_cost_from_vc(
                            env,
                            c,
                            vc,
                            contact_phi0,
                            contact_w_eff,
                            contact_mu,
                            contact_k,
                            contact_tau_d,
                            beta,
                            sigma,
                            dt,
                        )
                        c = c + stride

                i = tid
                while i < dof_per_env:
                    vi = v[env, i] + alpha * dv[env, i]
                    if has_pd_terms != 0 and pd_active[env, i] == 1:
                        gain = pd_gain[env, i]
                        if gain > scalar(0.0):
                            y = pd_a[env, i] - gain * vi
                            local_regularizer = (
                                local_regularizer
                                + (dt / gain) * clamp_antiderivative(y, pd_limit[env, i])
                            )

                    if has_limit_terms != 0:
                        if limit_lower_active[env, i] == 1:
                            lower_gamma = limit_lower_rinv[env, i] * (limit_lower_vhat[env, i] - vi)
                            if lower_gamma > scalar(0.0):
                                local_regularizer = (
                                    local_regularizer
                                    + scalar(0.5) * limit_lower_r[env, i] * lower_gamma * lower_gamma
                                )
                        if limit_upper_active[env, i] == 1:
                            upper_gamma = limit_upper_rinv[env, i] * (limit_upper_vhat[env, i] + vi)
                            if upper_gamma > scalar(0.0):
                                local_regularizer = (
                                    local_regularizer
                                    + scalar(0.5) * limit_upper_r[env, i] * upper_gamma * upper_gamma
                                )
                    i = i + stride

                regularizer_values = wp.tile_zeros((tile_size,), dtype=scalar, storage="shared")
                regularizer_values[tid] = local_regularizer
                regularizer_cost = wp.tile_sum(regularizer_values)[0]
                trial_cost = momentum_cost + regularizer_cost

                iterations_value = iterations_value + 1
                tol = cost_relax_a + cost_relax_r * wp.abs(current)
                if trial_cost <= current + tol:
                    accepted_flag = 1
                    break
                alpha = alpha * decay
                if alpha < min_alpha:
                    status_value = -2
                    break

            i = tid
            while i < dof_per_env:
                vi = v[env, i]
                if accepted_flag == 1:
                    vi = vi + alpha * dv[env, i]
                    v[env, i] = vi
                v_flat[env * dof_per_env + i] = vi
                i = i + stride

            if tid == 0:
                alpha_out[env] = alpha
                ls_active[env] = 0
                ls_accepted[env] = accepted_flag
                ls_status[env] = status_value
                ls_iterations[env] = iterations_value
                if accepted_flag == 1:
                    accepted_cost[env] = trial_cost
                    previous_cost[env] = current
                    current_cost[env] = trial_cost
                    ls_iterations_total[env] = ls_iterations_total[env] + iterations_value
                else:
                    accepted_cost[env] = current
                    previous_cost[env] = current
                    current_cost[env] = current
                    ls_iterations_total[env] = ls_iterations_total[env] + iterations_value

        return _run_unit_decay_line_search_fused_parallel_batched


    @wp.kernel(module="unique")
    def _init_sap_backtracking_state(
        newton_active: wp.array(dtype=int),
        current_cost: wp.array(dtype=scalar),
        dell0: wp.array(dtype=scalar),
        alpha_max: scalar,
        relative_slop: scalar,
        alpha: wp.array(dtype=scalar),
        alpha_prev: wp.array(dtype=scalar),
        ell_prev: wp.array(dtype=scalar),
        ell_slop: wp.array(dtype=scalar),
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        ls_active[env] = 0
        ls_accepted[env] = 0
        ls_status[env] = 0
        ls_iterations[env] = 0
        accepted_cost[env] = current_cost[env]

        if newton_active[env] == 0:
            return

        amax = scalar(alpha_max)
        alpha[env] = amax
        alpha_prev[env] = amax
        ell_prev[env] = current_cost[env]
        scale = wp.max(scalar(1.0), wp.abs(current_cost[env]))
        ell_slop[env] = (scalar(relative_slop) / scalar(10.0)) * scale
        if dell0[env] >= scalar(0.0) or not wp.isfinite(dell0[env]):
            ls_status[env] = -1
            return
        ls_active[env] = 1


    @wp.kernel(module="unique")
    def _accept_sap_alpha_max(
        trial_cost: wp.array(dtype=scalar),
        trial_derivative: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        relative_slop: scalar,
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        alpha: wp.array(dtype=scalar),
        alpha_prev: wp.array(dtype=scalar),
        ell_prev: wp.array(dtype=scalar),
        ell_slop: wp.array(dtype=scalar),
        accepted_cost: wp.array(dtype=scalar),
        next_active_count: wp.array(dtype=int),
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return

        ell = trial_cost[env]
        scale = wp.max(scalar(1.0), scalar(0.5) * (wp.abs(ell) + wp.abs(current_cost[env])))
        slop = (scalar(relative_slop) / scalar(10.0)) * scale
        ell_slop[env] = slop
        if trial_derivative[env] < scalar(0.0) or trial_derivative[env] < slop:
            ls_active[env] = 0
            ls_accepted[env] = 1
            accepted_cost[env] = ell
            return

        alpha_prev[env] = alpha[env]
        ell_prev[env] = ell
        wp.atomic_add(next_active_count, 0, 1)


    @wp.kernel(module="unique")
    def _scale_sap_backtracking_alpha(
        ls_active: wp.array(dtype=int),
        alpha: wp.array(dtype=scalar),
        rho: scalar,
    ):
        env = wp.tid()
        if ls_active[env] == 1:
            alpha[env] = alpha[env] * scalar(rho)


    @wp.kernel(module="unique")
    def _update_sap_backtracking_iteration(
        trial_cost: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        dell0: wp.array(dtype=scalar),
        alpha: wp.array(dtype=scalar),
        alpha_prev: wp.array(dtype=scalar),
        ell_prev: wp.array(dtype=scalar),
        ell_slop: wp.array(dtype=scalar),
        armijo_c: scalar,
        iteration: int,
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
        next_active_count: wp.array(dtype=int),
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return

        ell = trial_cost[env]
        a = alpha[env]
        a_prev = alpha_prev[env]
        e_prev = ell_prev[env]
        denom = a - a_prev
        dell_approx = scalar(0.0)
        if wp.abs(denom) > scalar(0.0):
            dell_approx = (ell - e_prev) / denom

        accept = 0
        accept_prev = 0
        if wp.abs(dell_approx) < ell_slop[env]:
            accept = 1
        elif ell > e_prev and sap_armijo_ok(a, ell, current_cost[env], dell0[env], scalar(armijo_c)):
            if sap_armijo_ok(a_prev, e_prev, current_cost[env], dell0[env], scalar(armijo_c)):
                accept = 1
                accept_prev = 1
            else:
                accept = 1

        if accept == 1:
            ls_active[env] = 0
            ls_accepted[env] = 1
            ls_iterations[env] = iteration
            if accept_prev == 1:
                alpha[env] = a_prev
                accepted_cost[env] = e_prev
            else:
                accepted_cost[env] = ell
            return

        alpha_prev[env] = a
        ell_prev[env] = ell
        wp.atomic_add(next_active_count, 0, 1)


    @wp.kernel(module="unique")
    def _update_sap_backtracking_iteration_conditional(
        trial_cost: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        dell0: wp.array(dtype=scalar),
        alpha: wp.array(dtype=scalar),
        alpha_prev: wp.array(dtype=scalar),
        ell_prev: wp.array(dtype=scalar),
        ell_slop: wp.array(dtype=scalar),
        armijo_c: scalar,
        iteration_count: wp.array(dtype=int),
        max_iterations: int,
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
        next_active_count: wp.array(dtype=int),
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return

        iteration = iteration_count[0]
        ell = trial_cost[env]
        a = alpha[env]
        a_prev = alpha_prev[env]
        e_prev = ell_prev[env]
        denom = a - a_prev
        dell_approx = scalar(0.0)
        if wp.abs(denom) > scalar(0.0):
            dell_approx = (ell - e_prev) / denom

        accept = 0
        accept_prev = 0
        if wp.abs(dell_approx) < ell_slop[env]:
            accept = 1
        elif ell > e_prev and sap_armijo_ok(a, ell, current_cost[env], dell0[env], scalar(armijo_c)):
            if sap_armijo_ok(a_prev, e_prev, current_cost[env], dell0[env], scalar(armijo_c)):
                accept = 1
                accept_prev = 1
            else:
                accept = 1

        if accept == 1:
            ls_active[env] = 0
            ls_accepted[env] = 1
            ls_iterations[env] = iteration
            if accept_prev == 1:
                alpha[env] = a_prev
                accepted_cost[env] = e_prev
            else:
                accepted_cost[env] = ell
            return

        if iteration >= max_iterations - 1:
            ls_active[env] = 0
            if sap_armijo_ok(alpha[env], trial_cost[env], current_cost[env], dell0[env], scalar(armijo_c)):
                ls_accepted[env] = 1
                ls_iterations[env] = max_iterations
                accepted_cost[env] = trial_cost[env]
            else:
                ls_status[env] = -2
            return

        alpha_prev[env] = a
        ell_prev[env] = ell
        wp.atomic_add(next_active_count, 0, 1)


    @wp.kernel(module="unique")
    def _finalize_sap_backtracking(
        trial_cost: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        dell0: wp.array(dtype=scalar),
        alpha: wp.array(dtype=scalar),
        armijo_c: scalar,
        max_iterations: int,
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return
        if sap_armijo_ok(alpha[env], trial_cost[env], current_cost[env], dell0[env], scalar(armijo_c)):
            ls_active[env] = 0
            ls_accepted[env] = 1
            ls_iterations[env] = max_iterations
            accepted_cost[env] = trial_cost[env]
        else:
            ls_status[env] = -2


    @wp.kernel(module="unique")
    def _copy_i32_batched(src: wp.array(dtype=int), dst: wp.array(dtype=int)):
        env = wp.tid()
        dst[env] = src[env]


    @wp.kernel(module="unique")
    def _init_sap_exact_alpha_max_state(
        newton_active: wp.array(dtype=int),
        current_cost: wp.array(dtype=scalar),
        dell0: wp.array(dtype=scalar),
        alpha_max: scalar,
        alpha: wp.array(dtype=scalar),
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        alpha[env] = scalar(alpha_max)
        ls_active[env] = 0
        ls_accepted[env] = 0
        ls_status[env] = 0
        ls_iterations[env] = 0
        accepted_cost[env] = current_cost[env]
        if newton_active[env] == 0:
            return
        if dell0[env] >= scalar(0.0) or not wp.isfinite(dell0[env]):
            ls_status[env] = -1
            return
        ls_active[env] = 1


    @wp.kernel(module="unique")
    def _init_sap_exact_root_state(
        trial_cost: wp.array(dtype=scalar),
        trial_derivative: wp.array(dtype=scalar),
        trial_second_derivative: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        dell0: wp.array(dtype=scalar),
        alpha_max: scalar,
        cost_abs_tol: scalar,
        cost_rel_tol: scalar,
        root_tolerance: scalar,
        alpha: wp.array(dtype=scalar),
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        accepted_cost: wp.array(dtype=scalar),
        exact_scale: wp.array(dtype=scalar),
        exact_x_lower: wp.array(dtype=scalar),
        exact_x_upper: wp.array(dtype=scalar),
        exact_f_lower: wp.array(dtype=scalar),
        exact_f_upper: wp.array(dtype=scalar),
        exact_root: wp.array(dtype=scalar),
        exact_minus_dx: wp.array(dtype=scalar),
        exact_minus_dx_previous: wp.array(dtype=scalar),
        exact_x_tolerance: wp.array(dtype=scalar),
        next_active_count: wp.array(dtype=int),
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return

        if trial_derivative[env] <= scalar(0.0):
            ls_active[env] = 0
            ls_accepted[env] = 1
            alpha[env] = scalar(alpha_max)
            accepted_cost[env] = trial_cost[env]
            return

        if -dell0[env] < scalar(cost_abs_tol) + scalar(cost_rel_tol) * trial_cost[env]:
            ls_active[env] = 0
            ls_accepted[env] = 1
            alpha[env] = scalar(1.0)
            return

        scale = -dell0[env]
        f_upper = trial_derivative[env] / scale
        if not wp.isfinite(f_upper) or f_upper <= scalar(0.0):
            ls_active[env] = 0
            ls_status[env] = -3
            return

        alpha_guess = -dell0[env] / trial_second_derivative[env]
        if alpha_guess > scalar(alpha_max):
            alpha_guess = scalar(alpha_max)
        if (
            not wp.isfinite(alpha_guess)
            or alpha_guess < scalar(0.0)
            or alpha_guess > scalar(alpha_max)
            or scalar(root_tolerance) * alpha_guess <= scalar(0.0)
        ):
            ls_active[env] = 0
            ls_status[env] = -4
            return

        exact_scale[env] = scale
        exact_x_lower[env] = scalar(0.0)
        exact_x_upper[env] = scalar(alpha_max)
        exact_f_lower[env] = scalar(-1.0)
        exact_f_upper[env] = f_upper
        exact_root[env] = alpha_guess
        exact_minus_dx[env] = -scalar(alpha_max)
        exact_minus_dx_previous[env] = -scalar(alpha_max)
        exact_x_tolerance[env] = scalar(root_tolerance) * alpha_guess
        alpha[env] = alpha_guess
        accepted_cost[env] = current_cost[env]
        wp.atomic_add(next_active_count, 0, 1)


    @wp.kernel(module="unique")
    def _update_sap_exact_root_state(
        trial_derivative: wp.array(dtype=scalar),
        trial_second_derivative: wp.array(dtype=scalar),
        root_tolerance: scalar,
        iteration_count: wp.array(dtype=int),
        max_iterations: int,
        alpha: wp.array(dtype=scalar),
        ls_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        ls_status: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        exact_scale: wp.array(dtype=scalar),
        exact_x_lower: wp.array(dtype=scalar),
        exact_x_upper: wp.array(dtype=scalar),
        exact_f_lower: wp.array(dtype=scalar),
        exact_f_upper: wp.array(dtype=scalar),
        exact_root: wp.array(dtype=scalar),
        exact_minus_dx: wp.array(dtype=scalar),
        exact_minus_dx_previous: wp.array(dtype=scalar),
        exact_x_tolerance: wp.array(dtype=scalar),
        next_active_count: wp.array(dtype=int),
    ):
        env = wp.tid()
        if ls_active[env] == 0:
            return

        iteration = iteration_count[0]
        scale = exact_scale[env]
        f = trial_derivative[env] / scale
        df = trial_second_derivative[env] / scale
        root = exact_root[env]
        f_upper = exact_f_upper[env]

        lower_update = 0
        if (f < scalar(0.0) and f_upper >= scalar(0.0)) or (
            f >= scalar(0.0) and f_upper < scalar(0.0)
        ):
            lower_update = 1

        if lower_update == 1:
            exact_x_lower[env] = root
            exact_f_lower[env] = f
        else:
            exact_x_upper[env] = root
            exact_f_upper[env] = f

        if wp.abs(f) < scalar(root_tolerance):
            ls_active[env] = 0
            ls_accepted[env] = 1
            ls_iterations[env] = iteration
            alpha[env] = root
            return

        if iteration >= max_iterations:
            ls_active[env] = 0
            ls_status[env] = -5
            return

        newton_is_slow = scalar(0.0)
        if scalar(2.0) * wp.abs(f) > wp.abs(exact_minus_dx_previous[env] * df):
            newton_is_slow = scalar(1.0)
        exact_minus_dx_previous[env] = exact_minus_dx[env]

        bisect_minus_dx = scalar(0.5) * (exact_x_lower[env] - exact_x_upper[env])
        bisect_root = exact_x_lower[env] - bisect_minus_dx

        newton_minus_dx = scalar(0.0)
        newton_root = root
        if wp.abs(df) > scalar(0.0):
            newton_minus_dx = f / df
            newton_root = root - newton_minus_dx

        use_bisect = 0
        if newton_is_slow > scalar(0.0):
            use_bisect = 1
        if newton_root < exact_x_lower[env] or newton_root > exact_x_upper[env]:
            use_bisect = 1
        if not wp.isfinite(newton_root) or not wp.isfinite(newton_minus_dx):
            use_bisect = 1

        next_root = newton_root
        next_minus_dx = newton_minus_dx
        if use_bisect == 1:
            next_root = bisect_root
            next_minus_dx = bisect_minus_dx

        if wp.abs(next_minus_dx) < exact_x_tolerance[env]:
            ls_active[env] = 0
            ls_accepted[env] = 1
            ls_iterations[env] = iteration
            alpha[env] = next_root
            return

        exact_root[env] = next_root
        exact_minus_dx[env] = next_minus_dx
        alpha[env] = next_root
        wp.atomic_add(next_active_count, 0, 1)


    @wp.kernel(module="unique")
    def _store_exact_accepted_cost(
        ls_accepted: wp.array(dtype=int),
        trial_cost: wp.array(dtype=scalar),
        accepted_cost: wp.array(dtype=scalar),
    ):
        env = wp.tid()
        if ls_accepted[env] == 1:
            accepted_cost[env] = trial_cost[env]


    @wp.kernel(module="unique")
    def _commit_line_search_step_batched(
        newton_active: wp.array(dtype=int),
        ls_accepted: wp.array(dtype=int),
        dof_per_env: int,
        alpha: wp.array(dtype=scalar),
        v: wp.array(dtype=scalar, ndim=2),
        dv: wp.array(dtype=scalar, ndim=2),
        v_flat: wp.array(dtype=scalar),
        current_cost: wp.array(dtype=scalar),
        previous_cost: wp.array(dtype=scalar),
        accepted_cost: wp.array(dtype=scalar),
    ):
        env, i = wp.tid()
        if i >= dof_per_env:
            return
        if newton_active[env] == 1 and ls_accepted[env] == 1:
            if i == 0:
                previous_cost[env] = current_cost[env]
                current_cost[env] = accepted_cost[env]
            v[env, i] = v[env, i] + alpha[env] * dv[env, i]
        v_flat[env * dof_per_env + i] = v[env, i]


    @wp.kernel(module="unique")
    def _accumulate_line_search_iterations_batched(
        ls_accepted: wp.array(dtype=int),
        ls_iterations: wp.array(dtype=int),
        total_iterations: wp.array(dtype=int),
    ):
        env = wp.tid()
        if ls_accepted[env] == 1:
            total_iterations[env] = total_iterations[env] + ls_iterations[env]


    @wp.kernel(module="unique")
    def _copy_first_dv_active_batched(
        active_env: wp.array(dtype=int),
        dof_per_env: int,
        dv: wp.array(dtype=scalar, ndim=2),
        first_dv: wp.array(dtype=scalar, ndim=2),
    ):
        env, i = wp.tid()
        if i < dof_per_env and active_env[env] == 1:
            first_dv[env, i] = dv[env, i]


    @wp.kernel(module="unique")
    def _count_active_int(
        active: wp.array(dtype=int),
        count: wp.array(dtype=int),
    ):
        env = wp.tid()
        if active[env] == 1:
            wp.atomic_add(count, 0, 1)


    @wp.kernel(module="unique")
    def _activate_unconverged_envs_batched(
        converged_env: wp.array(dtype=int),
        active_env: wp.array(dtype=int),
    ):
        env = wp.tid()
        active_env[env] = 1 - converged_env[env]



    return SimpleNamespace(
        _copy_flat_to_env_batched=_copy_flat_to_env_batched,
        _copy_env_to_flat_batched=_copy_env_to_flat_batched,
        _copy_env_to_env_batched=_copy_env_to_env_batched,
        _copy_solve_velocity_inputs_flat_batched=_copy_solve_velocity_inputs_flat_batched,
        _copy_solve_velocity_inputs_flat_batched_with_guess_flag=_copy_solve_velocity_inputs_flat_batched_with_guess_flag,
        _initialize_and_mark_unconstrained_free_envs_batched=_initialize_and_mark_unconstrained_free_envs_batched,
        _initialize_newton_loop_state=_initialize_newton_loop_state,
        _extract_a_diag_data_batched=_extract_a_diag_data_batched,
        _clear_participating_dofs_batched=_clear_participating_dofs_batched,
        _mark_contact_participating_dofs_batched=_mark_contact_participating_dofs_batched,
        _mark_model_participating_dofs_batched=_mark_model_participating_dofs_batched,
        _build_pd_terms_sap_batched=_build_pd_terms_sap_batched,
        _eval_pd_terms_sap_batched=_eval_pd_terms_sap_batched,
        _build_limit_terms_sap_batched=_build_limit_terms_sap_batched,
        _eval_limit_terms_sap_batched=_eval_limit_terms_sap_batched,
        _projection_eval_contact_sap_batched=_projection_eval_contact_sap_batched,
        _projection_cost_only_contact_sap_batched=_projection_cost_only_contact_sap_batched,
        _projection_eval_contact_gamma_sap_batched=_projection_eval_contact_gamma_sap_batched,
        _projection_eval_contact_hessian_sap_batched=_projection_eval_contact_hessian_sap_batched,
        _accumulate_pd_impulse_batched=_accumulate_pd_impulse_batched,
        _accumulate_limit_impulse_batched=_accumulate_limit_impulse_batched,
        _make_contact_impulse_single_tile_kernel=_make_contact_impulse_single_tile_kernel,
        _make_contact_hessian_single_tile_kernel=_make_contact_hessian_single_tile_kernel,
        _make_pack_contact_hessian_gemm_inputs_kernel=_make_pack_contact_hessian_gemm_inputs_kernel,
        _make_contact_hessian_gemm_tile_kernel=_make_contact_hessian_gemm_tile_kernel,
        _make_assemble_grad_and_dynamics_impulse_tiled_kernel=_make_assemble_grad_and_dynamics_impulse_tiled_kernel,
        _make_assemble_model_terms_and_grad_tiled_kernel=_make_assemble_model_terms_and_grad_tiled_kernel,
        _assemble_hessian_total_batched=_assemble_hessian_total_batched,
        _make_compute_base_cost_tiled_kernel=_make_compute_base_cost_tiled_kernel,
        _make_compute_line_search_base_coeffs_tiled_kernel=_make_compute_line_search_base_coeffs_tiled_kernel,
        _make_compute_line_search_contact_delta_velocity_tiled_kernel=_make_compute_line_search_contact_delta_velocity_tiled_kernel,
        _make_compute_norm_terms_tiled_kernel=_make_compute_norm_terms_tiled_kernel,
        _make_compute_norm_terms_and_update_active_tiled_kernel=_make_compute_norm_terms_and_update_active_tiled_kernel,
        _make_compute_norm_terms_and_update_active_conditional_tiled_kernel=_make_compute_norm_terms_and_update_active_conditional_tiled_kernel,
        _increment_scalar_i32=_increment_scalar_i32,
        _set_scalar_i32=_set_scalar_i32,
        _create_pack_dense_to_padded_batched_kernel=_create_pack_dense_to_padded_batched_kernel,
        _create_pack_grad_to_padded_rhs_batched_kernel=_create_pack_grad_to_padded_rhs_batched_kernel,
        _create_pack_dense_and_grad_to_padded_batched_kernel=_create_pack_dense_and_grad_to_padded_batched_kernel,
        _create_unpack_solution_batched_kernel=_create_unpack_solution_batched_kernel,
        _create_unpack_solution_and_first_batched_kernel=_create_unpack_solution_and_first_batched_kernel,
        _compute_search_direction_data_batched=_compute_search_direction_data_batched,
        _compute_search_direction_data_serial_batched=_compute_search_direction_data_serial_batched,
        _axpy_to_trial_batched=_axpy_to_trial_batched,
        _compute_line_derivative_batched=_compute_line_derivative_batched,
        _compute_line_derivative_serial_batched=_compute_line_derivative_serial_batched,
        _compute_line_second_derivative_serial_batched=_compute_line_second_derivative_serial_batched,
        _replace_trial_cost_with_sap_line_search_cost_batched=_replace_trial_cost_with_sap_line_search_cost_batched,
        _init_unit_decay_line_search_state=_init_unit_decay_line_search_state,
        _update_unit_decay_line_search_state=_update_unit_decay_line_search_state,
        _make_unit_decay_line_search_fused_parallel_kernel=_make_unit_decay_line_search_fused_parallel_kernel,
        _init_sap_backtracking_state=_init_sap_backtracking_state,
        _accept_sap_alpha_max=_accept_sap_alpha_max,
        _scale_sap_backtracking_alpha=_scale_sap_backtracking_alpha,
        _update_sap_backtracking_iteration=_update_sap_backtracking_iteration,
        _update_sap_backtracking_iteration_conditional=_update_sap_backtracking_iteration_conditional,
        _finalize_sap_backtracking=_finalize_sap_backtracking,
        _copy_i32_batched=_copy_i32_batched,
        _init_sap_exact_alpha_max_state=_init_sap_exact_alpha_max_state,
        _init_sap_exact_root_state=_init_sap_exact_root_state,
        _update_sap_exact_root_state=_update_sap_exact_root_state,
        _store_exact_accepted_cost=_store_exact_accepted_cost,
        _commit_line_search_step_batched=_commit_line_search_step_batched,
        _accumulate_line_search_iterations_batched=_accumulate_line_search_iterations_batched,
        _copy_first_dv_active_batched=_copy_first_dv_active_batched,
        _count_active_int=_count_active_int,
        _activate_unconverged_envs_batched=_activate_unconverged_envs_batched,
    )

class SapContactSolve:
    """SAP SAP stage2 contact solve in SAP-order generalized velocities.

    This class consumes `SapContactJacobianResult` buffers. Runtime collision
    detection stays outside this module and is adapted to `SapContacts` before
    entering the solver components.
    """

    def __init__(
        self,
        model: Model,
        *,
        max_rigid_contact: int = 128,
        contact_beta: float = 1.0,
        contact_sigma: float = 1.0e-3,
        contact_tau_d: float = 0.1,
        block_size: int | None = None,
        diag_shift: float = 0.0,
        contact_assembly_tile_size: int = 256,
        solve_precision: str = "fp64",
        linear_solve_precision: str = "fp64",
    ):
        if not isinstance(model, Model):
            raise TypeError("SapContactSolve requires SapModel; convert in the frontend adapter before construction.")
        if int(model.joint_dof_count) <= 0:
            raise ValueError("SapContactSolve requires a model with positive joint_dof_count.")

        self.model = model
        self.device = model.device
        self.dof_count = int(model.joint_dof_count)
        self.body_count = int(model.body_count)
        self.num_envs = int(getattr(model, "world_count", 1))
        if (
            self.num_envs <= 0
            or self.dof_count % self.num_envs != 0
            or self.body_count % self.num_envs != 0
        ):
            raise ValueError(
                "SapContactSolve requires contiguous equal-sized env dof/body blocks "
                f"(num_envs={self.num_envs}, dof_count={self.dof_count}, body_count={self.body_count})."
            )
        self.dof_per_env = self.dof_count // self.num_envs
        self.bodies_per_env = self.body_count // self.num_envs
        self.max_rigid_contact = int(max_rigid_contact)
        if self.max_rigid_contact <= 0:
            self.max_rigid_contact = 1

        self.contact_beta = float(contact_beta)
        self.contact_sigma = float(contact_sigma)
        self.contact_tau_d = float(contact_tau_d)
        self.diag_shift = float(diag_shift)
        self.solve_precision = self._normalize_solve_precision(solve_precision)
        self.solve_dtype = wp.float32 if self.solve_precision == "fp32" else wp.float64
        self.numpy_dtype = np.float32 if self.solve_precision == "fp32" else np.float64
        self.vec3_dtype = wp.vec3 if self.solve_precision == "fp32" else wp.vec3d
        self.mat33_dtype = wp.mat33 if self.solve_precision == "fp32" else wp.mat33d
        self.k = _make_contact_solve_kernel_table(self.solve_dtype)
        self.linear_solve_precision = self._normalize_linear_solve_precision(linear_solve_precision)
        self.contact_assembly_tile_size = int(contact_assembly_tile_size)
        if self.contact_assembly_tile_size <= 0:
            self.contact_assembly_tile_size = 256
        self.unit_line_search_tile_size = 128
        self.unit_line_search_contact_vc_tile_size = 128

        if block_size is None:
            block_size = 32 if wp.get_device(self.device).is_cuda and self.dof_per_env > 32 else 16
        self.block_size = int(block_size)
        self.linear_solve_dtype = wp.float32 if self.linear_solve_precision == "fp32" else wp.float64
        self._pack_dense_to_padded_batched = self.k._create_pack_dense_to_padded_batched_kernel(
            self.linear_solve_dtype
        )
        self._pack_grad_to_padded_rhs_batched = self.k._create_pack_grad_to_padded_rhs_batched_kernel(
            self.linear_solve_dtype
        )
        self._pack_dense_and_grad_to_padded_batched = self.k._create_pack_dense_and_grad_to_padded_batched_kernel(
            self.linear_solve_dtype
        )
        self._unpack_solution_batched = self.k._create_unpack_solution_batched_kernel(self.linear_solve_dtype)
        self._unpack_solution_and_first_batched = self.k._create_unpack_solution_and_first_batched_kernel(
            self.linear_solve_dtype
        )
        self._base_cost_tiled = self.k._make_compute_base_cost_tiled_kernel(self.unit_line_search_tile_size)
        self._unit_line_search_base_coeffs = self.k._make_compute_line_search_base_coeffs_tiled_kernel(
            self.unit_line_search_tile_size
        )
        self._unit_line_search_contact_delta_velocity = self.k._make_compute_line_search_contact_delta_velocity_tiled_kernel(
            self.unit_line_search_contact_vc_tile_size
        )
        self._grad_dynamics_impulse_tiled = self.k._make_assemble_grad_and_dynamics_impulse_tiled_kernel(
            self.unit_line_search_tile_size
        )
        self._contact_hessian_gemm_tile = self.k._make_contact_hessian_gemm_tile_kernel(
            _CONTACT_HESSIAN_GEMM_TILE_M,
            _CONTACT_HESSIAN_GEMM_TILE_N,
            _CONTACT_HESSIAN_GEMM_TILE_K,
        )
        self._pack_contact_hessian_gemm_inputs = self.k._make_pack_contact_hessian_gemm_inputs_kernel(
            _CONTACT_HESSIAN_GEMM_TILE_K,
            max(_CONTACT_HESSIAN_GEMM_TILE_M, _CONTACT_HESSIAN_GEMM_TILE_N),
        )
        self._model_terms_grad_tiled = self.k._make_assemble_model_terms_and_grad_tiled_kernel(
            self.unit_line_search_tile_size
        )
        self._norm_terms_tiled = self.k._make_compute_norm_terms_tiled_kernel(self.unit_line_search_tile_size)
        self._norm_terms_update_active_tiled = self.k._make_compute_norm_terms_and_update_active_tiled_kernel(
            self.unit_line_search_tile_size
        )
        self._norm_terms_update_active_conditional_tiled = (
            self.k._make_compute_norm_terms_and_update_active_conditional_tiled_kernel(
                self.unit_line_search_tile_size
            )
        )
        self._unit_line_search_fused_parallel = self.k._make_unit_decay_line_search_fused_parallel_kernel(
            self.unit_line_search_tile_size
        )
        self.block_solver = BlockCholeskySolverBatched(
            max_num_equations=self.dof_per_env,
            batch_size=self.num_envs,
            block_size=self.block_size,
            device=self.device,
            dtype=self.linear_solve_dtype,
        )
        self.padded_dof = self.block_solver.max_num_equations

        self._mode_none = int(SAP_JOINT_TARGET_NONE)
        self.dof_coord_index, self.dof_target_index, self.limit_supported = self._build_dof_maps(model)
        self.body_dof_start, self.body_dof_count = self._build_body_dof_maps(model)
        self._model_can_have_pd_terms = self._detect_model_pd_terms()
        self._model_can_have_limit_terms = self._detect_model_limit_terms()
        self._has_pd_terms = False
        self._has_limit_terms = False

        self.zero_control = wp.zeros((self.dof_count,), dtype=self.solve_dtype, device=self.device)
        self.joint_q_input = wp.zeros((model.joint_coord_count,), dtype=self.solve_dtype, device=self.device)
        self.joint_limit_lower_solve = wp.array(
            np.asarray(model.joint_limit_lower.numpy(), dtype=self.numpy_dtype).reshape(-1),
            dtype=self.solve_dtype,
            device=self.device,
        )
        self.joint_limit_upper_solve = wp.array(
            np.asarray(model.joint_limit_upper.numpy(), dtype=self.numpy_dtype).reshape(-1),
            dtype=self.solve_dtype,
            device=self.device,
        )
        self.joint_limit_ke_solve = wp.array(
            np.asarray(model.joint_limit_ke.numpy(), dtype=self.numpy_dtype).reshape(-1),
            dtype=self.solve_dtype,
            device=self.device,
        )
        self.joint_limit_kd_solve = wp.array(
            np.asarray(model.joint_limit_kd.numpy(), dtype=self.numpy_dtype).reshape(-1),
            dtype=self.solve_dtype,
            device=self.device,
        )

        shape_dof = (self.num_envs, self.dof_per_env)
        shape_mat = (self.num_envs, self.dof_per_env, self.dof_per_env)
        shape_contact = (self.num_envs, self.max_rigid_contact)
        self.contact_hessian_gemm_padded_contact_rows = (
            (self.max_rigid_contact * 3 + _CONTACT_HESSIAN_GEMM_TILE_K - 1)
            // _CONTACT_HESSIAN_GEMM_TILE_K
            * _CONTACT_HESSIAN_GEMM_TILE_K
        )
        contact_hessian_tile_dof = max(_CONTACT_HESSIAN_GEMM_TILE_M, _CONTACT_HESSIAN_GEMM_TILE_N)
        self.contact_hessian_gemm_padded_dof = (
            (self.dof_per_env + contact_hessian_tile_dof - 1)
            // contact_hessian_tile_dof
            * contact_hessian_tile_dof
        )
        shape_contact_gemm = (
            self.num_envs,
            self.contact_hessian_gemm_padded_contact_rows,
            self.contact_hessian_gemm_padded_dof,
        )

        self.v_env = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.v_flat = wp.zeros((self.dof_count,), dtype=self.solve_dtype, device=self.device)
        self.v_star_env = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.v0_env = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.v_trial = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)

        self.a_inv_diag = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.d_scale = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.participating_dof = wp.zeros(shape_dof, dtype=int, device=self.device)

        self.cost = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.previous_cost = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.accepted_cost = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)

        self.grad = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.constraint_impulse = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.dynamics_impulse = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.hess_contact = wp.zeros(shape_mat, dtype=self.solve_dtype, device=self.device)
        self.hessian = wp.zeros(shape_mat, dtype=self.solve_dtype, device=self.device)
        self.contact_hessian_j_flat = wp.zeros(shape_contact_gemm, dtype=self.solve_dtype, device=self.device)
        self.contact_hessian_gj_flat = wp.zeros(shape_contact_gemm, dtype=self.solve_dtype, device=self.device)
        self.contact_assembly_upper_count = self.dof_per_env * (self.dof_per_env + 1) // 2

        self.trial_constraint_impulse = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_pd_y = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_pd_gamma = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_pd_hdiag = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_pd_cost = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_limit_lower_gamma = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_limit_upper_gamma = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_limit_grad = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_limit_hdiag = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_limit_cost = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.trial_contact_gamma = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)
        self.trial_contact_g = wp.zeros(shape_contact, dtype=self.mat33_dtype, device=self.device)
        self.trial_contact_vc = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)
        self.trial_contact_y = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)
        self.trial_contact_rt = wp.zeros(shape_contact, dtype=self.solve_dtype, device=self.device)
        self.trial_contact_rn = wp.zeros(shape_contact, dtype=self.solve_dtype, device=self.device)
        self.trial_contact_cost = wp.zeros(shape_contact, dtype=self.solve_dtype, device=self.device)
        self.trial_contact_mode = wp.zeros(shape_contact, dtype=int, device=self.device)
        self.trial_cost = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.trial_derivative = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.trial_second_derivative = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)

        self.contact_gamma = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)
        self.contact_g = wp.zeros(shape_contact, dtype=self.mat33_dtype, device=self.device)
        self.contact_vc = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)
        self.contact_y = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)
        self.contact_rt = wp.zeros(shape_contact, dtype=self.solve_dtype, device=self.device)
        self.contact_rn = wp.zeros(shape_contact, dtype=self.solve_dtype, device=self.device)
        self.contact_cost = wp.zeros(shape_contact, dtype=self.solve_dtype, device=self.device)
        self.contact_mode = wp.zeros(shape_contact, dtype=int, device=self.device)
        self.contact_tau_d_fallback = wp.full(shape_contact, self.contact_tau_d, dtype=self.solve_dtype, device=self.device)

        self.pd_active = wp.zeros(shape_dof, dtype=int, device=self.device)
        self.pd_a = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_gain = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_limit = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_kp_eff = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_kd_eff = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_y = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_gamma = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_hdiag = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.pd_cost = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)

        self.limit_lower_active = wp.zeros(shape_dof, dtype=int, device=self.device)
        self.limit_upper_active = wp.zeros(shape_dof, dtype=int, device=self.device)
        self.limit_lower_vhat = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_upper_vhat = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_lower_r = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_upper_r = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_lower_rinv = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_upper_rinv = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_lower_gamma = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_upper_gamma = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_grad = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_hdiag = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.limit_cost = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)

        self.newton_active = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.converged_env = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.optimality_reached_env = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.cost_reached_env = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.stage2_active_env = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.stage2_active_count = wp.zeros((1,), dtype=int, device=self.device)
        self.all_env_active = wp.ones((self.num_envs,), dtype=int, device=self.device)
        self.active_count = wp.zeros((1,), dtype=int, device=self.device)
        self.newton_loop_iteration = wp.zeros((1,), dtype=int, device=self.device)
        self.newton_max_reached = wp.zeros((1,), dtype=int, device=self.device)

        self.grad_norm2 = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.p_norm2 = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.jc_norm2 = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)

        self.chol_a = wp.zeros(
            (self.num_envs, self.padded_dof, self.padded_dof),
            dtype=self.linear_solve_dtype,
            device=self.device,
        )
        self.chol_rhs = wp.zeros(
            (self.num_envs, self.padded_dof, 1),
            dtype=self.linear_solve_dtype,
            device=self.device,
        )
        self.chol_x = wp.zeros(
            (self.num_envs, self.padded_dof, 1),
            dtype=self.linear_solve_dtype,
            device=self.device,
        )
        self.dv = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.first_dv = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.dp = wp.zeros(shape_dof, dtype=self.solve_dtype, device=self.device)
        self.dell0 = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.dell_a0 = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.d2ell_a = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.line_momentum_cost = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.line_search_base0 = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.line_search_base_linear = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.line_search_base_quadratic = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.line_search_contact_dvc = wp.zeros(shape_contact, dtype=self.vec3_dtype, device=self.device)

        self.alpha = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.newton_iterations_env = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.alpha_prev = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.ell_prev = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.ell_slop = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.ls_active = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.ls_accepted = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.ls_status = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.ls_iterations = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.ls_iterations_total = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.ls_active_count = wp.zeros((1,), dtype=int, device=self.device)
        self.ls_loop_iteration = wp.zeros((1,), dtype=int, device=self.device)
        self.exact_scale = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_x_lower = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_x_upper = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_f_lower = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_f_upper = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_root = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_minus_dx = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_minus_dx_previous = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)
        self.exact_x_tolerance = wp.zeros((self.num_envs,), dtype=self.solve_dtype, device=self.device)

        self._contact_result_f32 = None
        if self.solve_dtype == wp.float32:
            self._contact_env_phi0_f32 = wp.zeros(shape_contact, dtype=wp.float32, device=self.device)
            self._contact_env_jacobian_f32 = wp.zeros(
                (self.num_envs, self.max_rigid_contact, 3, self.dof_per_env),
                dtype=wp.float32,
                device=self.device,
            )
            self._contact_env_w_eff_f32 = wp.zeros(shape_contact, dtype=wp.float32, device=self.device)
            self._contact_env_mu_f32 = wp.zeros(shape_contact, dtype=wp.float32, device=self.device)
            self._contact_env_stiffness_f32 = wp.zeros(shape_contact, dtype=wp.float32, device=self.device)
            self._contact_env_tau_d_f32 = wp.zeros(shape_contact, dtype=wp.float32, device=self.device)
            self._dynamics_matrix_env_f32 = wp.zeros(shape_mat, dtype=wp.float32, device=self.device)
            self._body_jacobian_local_f32 = wp.zeros(
                (int(model.body_count), 6, self.dof_per_env),
                dtype=wp.float32,
                device=self.device,
            )
            self._contact_env_R_WC_f32 = wp.zeros(
                (self.num_envs, self.max_rigid_contact, 3, 3),
                dtype=wp.float32,
                device=self.device,
            )
            self._contact_env_point_f32 = wp.zeros(shape_contact, dtype=wp.vec3, device=self.device)
            self._contact_env_witness0_f32 = wp.zeros_like(self._contact_env_point_f32)
            self._contact_env_witness1_f32 = wp.zeros_like(self._contact_env_point_f32)

        self.last_iterations = 0
        self.last_line_search_iterations = 0

    @staticmethod
    def _normalize_solve_precision(value: str) -> str:
        precision = str(value).strip().lower()
        if precision == "f32":
            precision = "fp32"
        elif precision == "f64":
            precision = "fp64"
        if precision not in {"fp32", "fp64"}:
            raise ValueError(
                "solve_precision must be 'fp32'/'f32' or 'fp64'/'f64', "
                f"got {value!r}."
            )
        return precision

    @staticmethod
    def _normalize_linear_solve_precision(value: str) -> str:
        precision = str(value).strip().lower()
        if precision == "f32":
            precision = "fp32"
        elif precision == "f64":
            precision = "fp64"
        if precision not in {"fp32", "fp64"}:
            raise ValueError(
                "linear_solve_precision must be 'fp32'/'f32' or 'fp64'/'f64', "
                f"got {value!r}."
            )
        return precision

    def _build_dof_maps(self, model: Model) -> tuple[wp.array, wp.array, wp.array]:
        dof_coord_index = np.full(self.dof_count, -1, dtype=np.int32)
        dof_target_index = np.arange(self.dof_count, dtype=np.int32)
        limit_supported = np.zeros(self.dof_count, dtype=np.int32)

        joint_type = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        free_types = {
            int(SAP_JOINT_FREE),
            int(SAP_JOINT_DISTANCE),
        }
        supported_types = {
            int(SAP_JOINT_PRISMATIC),
            int(SAP_JOINT_REVOLUTE),
            int(SAP_JOINT_D6),
        }

        for joint_idx, jtype in enumerate(joint_type):
            dof_start = int(joint_qd_start[joint_idx])
            if int(jtype) in free_types:
                for axis in range(3):
                    dof_target_index[dof_start + axis] = dof_start + axis + 3
                    dof_target_index[dof_start + axis + 3] = dof_start + axis

            if int(jtype) not in supported_types:
                continue
            coord_start = int(joint_q_start[joint_idx])
            axis_count = int(joint_dof_dim[joint_idx, 0] + joint_dof_dim[joint_idx, 1])
            for axis in range(axis_count):
                dof = dof_start + axis
                if 0 <= dof < self.dof_count:
                    dof_coord_index[dof] = coord_start + axis
                    if axis_count == 1:
                        limit_supported[dof] = 1

        return (
            wp.array(dof_coord_index, dtype=int, device=self.device),
            wp.array(dof_target_index, dtype=int, device=self.device),
            wp.array(limit_supported, dtype=int, device=self.device),
        )

    def _build_body_dof_maps(self, model: Model) -> tuple[wp.array, wp.array]:
        body_dof_start = np.full(self.body_count, -1, dtype=np.int32)
        body_dof_count = np.zeros(self.body_count, dtype=np.int32)

        articulation_start = np.asarray(model.articulation_start.numpy(), dtype=np.int32).reshape(-1)
        joint_child = np.asarray(model.joint_child.numpy(), dtype=np.int32).reshape(-1)
        joint_qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int32).reshape(-1)
        joint_dof_dim = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int32)

        for art_idx in range(int(model.articulation_count)):
            first_joint = int(articulation_start[art_idx])
            last_joint = int(articulation_start[art_idx + 1])
            starts: list[int] = []
            dof_count = 0
            for joint_idx in range(first_joint, last_joint):
                axis_count = int(joint_dof_dim[joint_idx, 0] + joint_dof_dim[joint_idx, 1])
                if axis_count <= 0:
                    continue
                starts.append(int(joint_qd_start[joint_idx]))
                dof_count += axis_count
            if not starts or dof_count <= 0:
                continue
            dof_start = min(starts)
            for joint_idx in range(first_joint, last_joint):
                body = int(joint_child[joint_idx])
                if body < 0 or body >= self.body_count:
                    continue
                env = body // self.bodies_per_env
                body_dof_start[body] = dof_start - env * self.dof_per_env
                body_dof_count[body] = dof_count

        return (
            wp.array(body_dof_start, dtype=int, device=self.device),
            wp.array(body_dof_count, dtype=int, device=self.device),
        )

    def _detect_model_pd_terms(self) -> bool:
        target_mode = np.asarray(self.model.joint_target_mode.numpy(), dtype=np.int64).reshape(-1)
        target_ke = np.asarray(self.model.joint_target_ke.numpy(), dtype=np.float64).reshape(-1)
        target_kd = np.asarray(self.model.joint_target_kd.numpy(), dtype=np.float64).reshape(-1)
        count = min(target_mode.size, target_ke.size, target_kd.size)
        if count <= 0:
            return False
        active = (target_mode[:count] != self._mode_none) & (
            (target_ke[:count] > 0.0) | (target_kd[:count] > 0.0)
        )
        return bool(np.any(active))

    def _detect_model_limit_terms(self) -> bool:
        limit_supported = np.asarray(self.limit_supported.numpy(), dtype=np.int64).reshape(-1)
        lower = np.asarray(self.model.joint_limit_lower.numpy(), dtype=np.float64).reshape(-1)
        upper = np.asarray(self.model.joint_limit_upper.numpy(), dtype=np.float64).reshape(-1)
        ke = np.asarray(self.model.joint_limit_ke.numpy(), dtype=np.float64).reshape(-1)
        count = min(limit_supported.size, lower.size, upper.size, ke.size)
        if count <= 0:
            return False
        active = (
            (limit_supported[:count] != 0)
            & (ke[:count] > 0.0)
            & (np.isfinite(lower[:count]) | np.isfinite(upper[:count]))
        )
        return bool(np.any(active))

    def _as_control_array(self, arr) -> wp.array:
        if arr is None:
            return self.zero_control
        return arr

    def _prepare_joint_q_input(self, state: State) -> None:
        if self.solve_dtype == wp.float64:
            kernel = _copy_f64 if state.joint_q.dtype == wp.float64 else _copy_f32_to_f64
        else:
            kernel = _copy_f32 if state.joint_q.dtype == wp.float32 else _copy_f64_to_f32
        wp.launch(
            kernel,
            dim=self.model.joint_coord_count,
            inputs=[state.joint_q, self.joint_q_input],
            device=self.device,
        )

    def _build_participating_dof_mask(self, contact_result: SapContactJacobianResult) -> None:
        contact_capacity = self._contact_capacity(contact_result)
        wp.launch(
            self.k._clear_participating_dofs_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.participating_dof],
            device=self.device,
        )
        wp.launch(
            self.k._mark_contact_participating_dofs_batched,
            dim=(self.num_envs, contact_capacity, self.dof_per_env),
            inputs=[
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_body0,
                contact_result.contact_env_body1,
                contact_result.contact_env_jacobian,
                self.body_dof_start,
                self.body_dof_count,
                self.participating_dof,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._mark_model_participating_dofs_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                self.dof_per_env,
                self.pd_active,
                self.limit_lower_active,
                self.limit_upper_active,
                self.participating_dof,
            ],
            device=self.device,
        )

    def _contact_capacity(self, contact_result: SapContactJacobianResult) -> int:
        result_slots = int(contact_result.contact_env_jacobian.shape[1])
        return max(1, min(self.max_rigid_contact, result_slots))

    def _compute_base_cost(
        self,
        active_env: wp.array,
        v: wp.array,
        a_mat: wp.array,
        out_cost: wp.array,
    ) -> None:
        wp.launch_tiled(
            self._base_cost_tiled,
            dim=self.num_envs,
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                active_env,
                self.dof_per_env,
                v,
                self.v_star_env,
                a_mat,
                out_cost,
            ],
            device=self.device,
        )

    def _assemble_grad_and_dynamics_impulse(
        self,
        active_env: wp.array,
        v: wp.array,
        a_mat: wp.array,
        constraint_impulse: wp.array,
        add_constraint_terms: bool = False,
    ) -> None:
        add_pd = int(bool(add_constraint_terms) and bool(self._has_pd_terms))
        add_limits = int(bool(add_constraint_terms) and bool(self._has_limit_terms))
        wp.launch_tiled(
            self._grad_dynamics_impulse_tiled,
            dim=(self.num_envs, self.dof_per_env),
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                active_env,
                self.dof_per_env,
                v,
                self.v_star_env,
                a_mat,
                add_pd,
                self.pd_active,
                self.pd_gamma,
                add_limits,
                self.limit_grad,
                constraint_impulse,
                self.grad,
                self.dynamics_impulse,
            ],
            device=self.device,
        )

    def _assemble_model_terms_and_grad(
        self,
        active_env: wp.array,
        v: wp.array,
        a_mat: wp.array,
        dt: float,
        *,
        pd_y: wp.array,
        pd_gamma: wp.array,
        pd_hdiag: wp.array,
        pd_cost: wp.array,
        lower_gamma: wp.array,
        upper_gamma: wp.array,
        limit_grad: wp.array,
        limit_hdiag: wp.array,
        limit_cost: wp.array,
        constraint_impulse: wp.array,
        cost: wp.array,
    ) -> None:
        wp.launch_tiled(
            self._model_terms_grad_tiled,
            dim=(self.num_envs, self.dof_per_env),
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                active_env,
                self.dof_per_env,
                v,
                self.v_star_env,
                a_mat,
                int(self._has_pd_terms),
                self.pd_active,
                self.pd_a,
                self.pd_gain,
                self.pd_limit,
                float(dt),
                pd_y,
                pd_gamma,
                pd_hdiag,
                pd_cost,
                int(self._has_limit_terms),
                self.limit_lower_active,
                self.limit_upper_active,
                self.limit_lower_vhat,
                self.limit_upper_vhat,
                self.limit_lower_r,
                self.limit_upper_r,
                self.limit_lower_rinv,
                self.limit_upper_rinv,
                lower_gamma,
                upper_gamma,
                limit_grad,
                limit_hdiag,
                limit_cost,
                constraint_impulse,
                self.grad,
                self.dynamics_impulse,
                cost,
            ],
            device=self.device,
        )

    def _compute_norm_terms(self, active_env: wp.array) -> None:
        wp.launch_tiled(
            self._norm_terms_tiled,
            dim=self.num_envs,
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                active_env,
                self.dof_per_env,
                self.participating_dof,
                self.d_scale,
                self.grad,
                self.dynamics_impulse,
                self.constraint_impulse,
                self.grad_norm2,
                self.p_norm2,
                self.jc_norm2,
            ],
            device=self.device,
        )

    def _compute_norm_terms_and_update_active(
        self,
        active_env: wp.array,
        iteration: int,
        *,
        optimality_abs_tol: float,
        optimality_rel_tol: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        cost_min_alpha: float,
    ) -> None:
        single_env_count = int(self.num_envs == 1)
        if single_env_count == 0:
            self.active_count.zero_()
        wp.launch_tiled(
            self._norm_terms_update_active_tiled,
            dim=self.num_envs,
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                active_env,
                self.dof_per_env,
                self.participating_dof,
                self.d_scale,
                self.grad,
                self.dynamics_impulse,
                self.constraint_impulse,
                self.cost,
                self.previous_cost,
                self.alpha,
                int(iteration),
                float(optimality_abs_tol),
                float(optimality_rel_tol),
                float(cost_abs_tol),
                float(cost_rel_tol),
                float(cost_min_alpha),
                single_env_count,
                self.newton_active,
                self.converged_env,
                self.optimality_reached_env,
                self.cost_reached_env,
                self.newton_iterations_env,
                self.active_count,
                self.grad_norm2,
                self.p_norm2,
                self.jc_norm2,
            ],
            device=self.device,
        )

    def _compute_norm_terms_and_update_active_conditional(
        self,
        active_env: wp.array,
        *,
        max_iterations: int,
        optimality_abs_tol: float,
        optimality_rel_tol: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        cost_min_alpha: float,
    ) -> None:
        single_env_count = int(self.num_envs == 1)
        if single_env_count == 0:
            self.active_count.zero_()
        wp.launch_tiled(
            self._norm_terms_update_active_conditional_tiled,
            dim=self.num_envs,
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                active_env,
                self.dof_per_env,
                self.participating_dof,
                self.d_scale,
                self.grad,
                self.dynamics_impulse,
                self.constraint_impulse,
                self.cost,
                self.previous_cost,
                self.alpha,
                self.newton_loop_iteration,
                int(max_iterations),
                float(optimality_abs_tol),
                float(optimality_rel_tol),
                float(cost_abs_tol),
                float(cost_rel_tol),
                float(cost_min_alpha),
                single_env_count,
                self.newton_active,
                self.converged_env,
                self.optimality_reached_env,
                self.cost_reached_env,
                self.newton_iterations_env,
                self.active_count,
                self.newton_max_reached,
                self.grad_norm2,
                self.p_norm2,
                self.jc_norm2,
            ],
            device=self.device,
        )

    def _solver_update_active(
        self,
        contact_result: SapContactJacobianResult,
        active_env: wp.array,
        dt: float,
        *,
        max_iterations: int,
        optimality_abs_tol: float,
        optimality_rel_tol: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        cost_min_alpha: float,
    ) -> None:
        grad_ready = self._evaluate_problem(
            contact_result,
            self.v_env,
            active_env,
            dt,
            include_hessian=False,
            cost=self.cost,
            constraint_impulse=self.constraint_impulse,
            contact_gamma=self.contact_gamma,
            contact_g=self.contact_g,
            contact_vc=self.contact_vc,
            contact_y=self.contact_y,
            contact_rt=self.contact_rt,
            contact_rn=self.contact_rn,
            contact_cost=self.contact_cost,
            contact_mode=self.contact_mode,
            pd_y=self.pd_y,
            pd_gamma=self.pd_gamma,
            pd_hdiag=self.pd_hdiag,
            pd_cost=self.pd_cost,
            lower_gamma=self.limit_lower_gamma,
            upper_gamma=self.limit_upper_gamma,
            limit_grad=self.limit_grad,
            limit_hdiag=self.limit_hdiag,
            limit_cost=self.limit_cost,
            defer_constraint_terms=True,
        )
        if not grad_ready:
            self._assemble_grad_and_dynamics_impulse(
                active_env,
                self.v_env,
                contact_result.dynamics_matrix_env,
                self.constraint_impulse,
                add_constraint_terms=True,
            )
        self._compute_norm_terms_and_update_active_conditional(
            active_env,
            max_iterations=int(max_iterations),
            optimality_abs_tol=float(optimality_abs_tol),
            optimality_rel_tol=float(optimality_rel_tol),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            cost_min_alpha=float(cost_min_alpha),
        )

    def _cast_contact_result(self, contact_result: SapContactJacobianResult) -> SapContactJacobianResult:
        if self.solve_dtype == wp.float64 or contact_result.dynamics_matrix_env.dtype == wp.float32:
            return contact_result

        wp.launch(
            _copy_3d_f64_to_f32,
            dim=self._dynamics_matrix_env_f32.shape,
            inputs=[contact_result.dynamics_matrix_env, self._dynamics_matrix_env_f32],
            device=self.device,
        )
        wp.launch(
            _copy_2d_f64_to_f32,
            dim=self._contact_env_phi0_f32.shape,
            inputs=[contact_result.contact_env_phi0, self._contact_env_phi0_f32],
            device=self.device,
        )
        wp.launch(
            _copy_4d_f64_to_f32,
            dim=self._contact_env_jacobian_f32.shape,
            inputs=[contact_result.contact_env_jacobian, self._contact_env_jacobian_f32],
            device=self.device,
        )
        wp.launch(
            _copy_2d_f64_to_f32,
            dim=self._contact_env_w_eff_f32.shape,
            inputs=[contact_result.contact_env_w_eff, self._contact_env_w_eff_f32],
            device=self.device,
        )
        wp.launch(
            _copy_2d_f64_to_f32,
            dim=self._contact_env_mu_f32.shape,
            inputs=[contact_result.contact_env_mu, self._contact_env_mu_f32],
            device=self.device,
        )
        wp.launch(
            _copy_2d_f64_to_f32,
            dim=self._contact_env_stiffness_f32.shape,
            inputs=[contact_result.contact_env_stiffness, self._contact_env_stiffness_f32],
            device=self.device,
        )
        tau_src = contact_result.contact_env_tau_d
        if tau_src is not None:
            wp.launch(
                _copy_2d_f64_to_f32,
                dim=self._contact_env_tau_d_f32.shape,
                inputs=[tau_src, self._contact_env_tau_d_f32],
                device=self.device,
            )
        wp.launch(
            _copy_3d_f64_to_f32,
            dim=self._body_jacobian_local_f32.shape,
            inputs=[contact_result.body_jacobian_local, self._body_jacobian_local_f32],
            device=self.device,
        )
        wp.launch(
            _copy_4d_f64_to_f32,
            dim=self._contact_env_R_WC_f32.shape,
            inputs=[contact_result.contact_env_R_WC, self._contact_env_R_WC_f32],
            device=self.device,
        )
        wp.launch(
            _copy_vec3d_2d_to_vec3,
            dim=self._contact_env_point_f32.shape,
            inputs=[contact_result.contact_env_point, self._contact_env_point_f32],
            device=self.device,
        )
        wp.launch(
            _copy_vec3d_2d_to_vec3,
            dim=self._contact_env_witness0_f32.shape,
            inputs=[contact_result.contact_env_witness0, self._contact_env_witness0_f32],
            device=self.device,
        )
        wp.launch(
            _copy_vec3d_2d_to_vec3,
            dim=self._contact_env_witness1_f32.shape,
            inputs=[contact_result.contact_env_witness1, self._contact_env_witness1_f32],
            device=self.device,
        )

        self._contact_result_f32 = SapContactJacobianResult(
            contact_count=contact_result.contact_count,
            truncated_contact_count=contact_result.truncated_contact_count,
            contact_env_count=contact_result.contact_env_count,
            contact_env_phi0=self._contact_env_phi0_f32,
            contact_env_jacobian=self._contact_env_jacobian_f32,
            contact_env_w_eff=self._contact_env_w_eff_f32,
            contact_env_mu=self._contact_env_mu_f32,
            contact_env_stiffness=self._contact_env_stiffness_f32,
            contact_env_tau_d=self._contact_env_tau_d_f32,
            contact_env_R_WC=self._contact_env_R_WC_f32,
            contact_env_point=self._contact_env_point_f32,
            contact_env_witness0=self._contact_env_witness0_f32,
            contact_env_witness1=self._contact_env_witness1_f32,
            contact_env_body0=contact_result.contact_env_body0,
            contact_env_body1=contact_result.contact_env_body1,
            body_jacobian_local=self._body_jacobian_local_f32,
            dynamics_matrix_env=self._dynamics_matrix_env_f32,
        )
        return self._contact_result_f32

    def _load_velocity(self, src: wp.array, dst: wp.array) -> None:
        if src.shape == (self.dof_count,):
            if src.dtype == self.solve_dtype:
                wp.launch(
                    self.k._copy_flat_to_env_batched,
                    dim=(self.num_envs, self.dof_per_env),
                    inputs=[src, self.dof_per_env, dst],
                    device=self.device,
                )
            elif self.solve_dtype == wp.float32:
                wp.launch(
                    _copy_flat_f64_to_env_f32_batched,
                    dim=(self.num_envs, self.dof_per_env),
                    inputs=[src, self.dof_per_env, dst],
                    device=self.device,
                )
            else:
                wp.launch(
                    _copy_flat_f32_to_env_f64_batched,
                    dim=(self.num_envs, self.dof_per_env),
                    inputs=[src, self.dof_per_env, dst],
                    device=self.device,
                )
        elif src.shape == (self.num_envs, self.dof_per_env):
            if src.dtype == self.solve_dtype:
                wp.copy(dst, src)
            elif self.solve_dtype == wp.float32:
                wp.launch(
                    _copy_env_f64_to_env_f32_batched,
                    dim=(self.num_envs, self.dof_per_env),
                    inputs=[src, self.dof_per_env, dst],
                    device=self.device,
                )
            else:
                wp.launch(
                    _copy_env_f32_to_env_f64_batched,
                    dim=(self.num_envs, self.dof_per_env),
                    inputs=[src, self.dof_per_env, dst],
                    device=self.device,
                )
        else:
            raise ValueError(
                "velocity input must be flat SAP-order `(joint_dof_count,)` "
                "or env-local `(num_envs, dof_per_env)`, got "
                f"{src.shape!r}"
            )

    def prepare(
        self,
        contact_result: SapContactJacobianResult,
        state: State,
        control: Control | None,
        dt: float,
        v_star: wp.array,
        *,
        v0: wp.array | None = None,
        v_guess: wp.array | None = None,
        v_guess_active: wp.array | None = None,
    ) -> None:
        """Prepare contact-solve buffers for the current active contact set before iterative solve
        evaluation.
        """
        contact_result = self._cast_contact_result(contact_result)
        if v0 is None:
            v0 = v_guess if v_guess is not None else v_star
        if not isinstance(state, State):
            raise TypeError("SapContactSolve.prepare requires SapState.")
        if control is None or not isinstance(control, Control):
            raise TypeError("SapContactSolve.prepare requires SapControl.")
        self._has_pd_terms = self._model_can_have_pd_terms
        self._has_limit_terms = self._model_can_have_limit_terms

        if (
            v_guess_active is not None
            and v_guess is not None
            and v_star.shape == (self.dof_count,)
            and v0.shape == (self.dof_count,)
            and v_guess.shape == (self.dof_count,)
            and v_star.dtype == self.solve_dtype
            and v0.dtype == self.solve_dtype
            and v_guess.dtype == self.solve_dtype
        ):
            wp.launch(
                self.k._copy_solve_velocity_inputs_flat_batched_with_guess_flag,
                dim=(self.num_envs, self.dof_per_env),
                inputs=[
                    v_star,
                    v0,
                    v_guess,
                    v_guess_active,
                    self.dof_per_env,
                    self.v_star_env,
                    self.v0_env,
                    self.v_env,
                ],
                device=self.device,
            )
        elif (
            v_guess is None
            and v_star.shape == (self.dof_count,)
            and v0.shape == (self.dof_count,)
            and v_star.dtype == self.solve_dtype
            and v0.dtype == self.solve_dtype
        ):
            wp.launch(
                self.k._copy_solve_velocity_inputs_flat_batched,
                dim=(self.num_envs, self.dof_per_env),
                inputs=[
                    v_star,
                    v0,
                    self.dof_per_env,
                    self.v_star_env,
                    self.v0_env,
                    self.v_env,
                ],
                device=self.device,
            )
        else:
            self._load_velocity(v_star, self.v_star_env)
            self._load_velocity(v0, self.v0_env)
            self._load_velocity(v_guess if v_guess is not None else v0, self.v_env)
        self._prepare_joint_q_input(state)

        wp.launch(
            self.k._extract_a_diag_data_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                self.dof_per_env,
                contact_result.dynamics_matrix_env,
                self.a_inv_diag,
                self.d_scale,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._build_pd_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_pd_terms),
                self.dof_per_env,
                self.dof_coord_index,
                self.dof_target_index,
                self.model.joint_target_mode,
                self.model.joint_target_ke,
                self.model.joint_target_kd,
                self.model.joint_effort_limit,
                self.joint_q_input,
                self._as_control_array(getattr(control, "joint_target_pos", None)),
                self._as_control_array(getattr(control, "joint_target_vel", None)),
                self._as_control_array(getattr(control, "joint_act", None)),
                self.a_inv_diag,
                float(dt),
                self._mode_none,
                self.pd_active,
                self.pd_a,
                self.pd_gain,
                self.pd_limit,
                self.pd_kp_eff,
                self.pd_kd_eff,
            ],
            device=self.device,
        )

        wp.launch(
            self.k._build_limit_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_limit_terms),
                self.dof_per_env,
                self.dof_coord_index,
                self.limit_supported,
                self.joint_limit_lower_solve,
                self.joint_limit_upper_solve,
                self.joint_limit_ke_solve,
                self.joint_limit_kd_solve,
                self.joint_q_input,
                self.v0_env,
                self.v_star_env,
                self.a_inv_diag,
                float(dt),
                self.limit_lower_active,
                self.limit_upper_active,
                self.limit_lower_vhat,
                self.limit_upper_vhat,
                self.limit_lower_r,
                self.limit_upper_r,
                self.limit_lower_rinv,
                self.limit_upper_rinv,
            ],
            device=self.device,
        )
        self._build_participating_dof_mask(contact_result)

    def _assemble_contact_impulse_from_terms(
        self,
        contact_result: SapContactJacobianResult,
        *,
        contact_capacity: int,
        contact_gamma: wp.array,
        constraint_impulse: wp.array,
    ) -> None:
        tile_size = int(self.contact_assembly_tile_size)
        wp.launch_tiled(
            self.k._make_contact_impulse_single_tile_kernel(tile_size),
            dim=(self.num_envs, self.dof_per_env),
            block_dim=tile_size,
            inputs=[
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                contact_gamma,
                constraint_impulse,
            ],
            device=self.device,
        )

    def _assemble_contact_hessian_from_terms(
        self,
        contact_result: SapContactJacobianResult,
        *,
        contact_capacity: int,
        contact_g: wp.array,
    ) -> None:
        contact_hessian_tile_dof = max(_CONTACT_HESSIAN_GEMM_TILE_M, _CONTACT_HESSIAN_GEMM_TILE_N)
        wp.launch_tiled(
            self._pack_contact_hessian_gemm_inputs,
            dim=(
                self.num_envs,
                self.contact_hessian_gemm_padded_contact_rows // _CONTACT_HESSIAN_GEMM_TILE_K,
                self.contact_hessian_gemm_padded_dof // contact_hessian_tile_dof,
            ),
            block_dim=128,
            inputs=[
                self.dof_per_env,
                contact_capacity,
                self.contact_hessian_gemm_padded_contact_rows,
                self.contact_hessian_gemm_padded_dof,
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                contact_g,
                self.contact_hessian_j_flat,
                self.contact_hessian_gj_flat,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self._contact_hessian_gemm_tile,
            dim=(
                self.num_envs,
                (self.dof_per_env + _CONTACT_HESSIAN_GEMM_TILE_M - 1) // _CONTACT_HESSIAN_GEMM_TILE_M,
                (self.dof_per_env + _CONTACT_HESSIAN_GEMM_TILE_N - 1) // _CONTACT_HESSIAN_GEMM_TILE_N,
            ),
            block_dim=128,
            inputs=[
                self.dof_per_env,
                self.contact_hessian_gemm_padded_contact_rows,
                self.contact_hessian_j_flat,
                self.contact_hessian_gj_flat,
                self.hess_contact,
            ],
            device=self.device,
        )

    def _ensure_hessian_terms_for_active_envs(
        self,
        contact_result: SapContactJacobianResult,
        active_env: wp.array,
        dt: float,
    ) -> int:
        contact_capacity = self._contact_capacity(contact_result)
        contact_tau_d = getattr(contact_result, "contact_env_tau_d", None)
        if contact_tau_d is None:
            self.contact_tau_d_fallback.fill_(self.contact_tau_d)
            contact_tau_d = self.contact_tau_d_fallback
        wp.launch(
            self.k._projection_eval_contact_hessian_sap_batched,
            dim=(self.num_envs, contact_capacity),
            inputs=[
                active_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_phi0,
                contact_result.contact_env_w_eff,
                contact_result.contact_env_mu,
                contact_result.contact_env_stiffness,
                contact_tau_d,
                self.contact_vc,
                self.contact_beta,
                self.contact_sigma,
                float(dt),
                self.contact_g,
                self.contact_y,
                self.contact_rt,
                self.contact_rn,
                self.contact_mode,
            ],
            device=self.device,
        )
        return contact_capacity

    def _evaluate_cost_terms(
        self,
        contact_result: SapContactJacobianResult,
        v: wp.array,
        active_env: wp.array,
        dt: float,
        *,
        cost: wp.array,
        contact_gamma: wp.array,
        contact_g: wp.array,
        contact_vc: wp.array,
        contact_y: wp.array,
        contact_rt: wp.array,
        contact_rn: wp.array,
        contact_cost: wp.array,
        contact_mode: wp.array,
        pd_y: wp.array,
        pd_gamma: wp.array,
        pd_hdiag: wp.array,
        pd_cost: wp.array,
        lower_gamma: wp.array,
        upper_gamma: wp.array,
        limit_grad: wp.array,
        limit_hdiag: wp.array,
        limit_cost: wp.array,
    ) -> int:
        contact_capacity = self._contact_capacity(contact_result)
        contact_tau_d = getattr(contact_result, "contact_env_tau_d", None)
        if contact_tau_d is None:
            self.contact_tau_d_fallback.fill_(self.contact_tau_d)
            contact_tau_d = self.contact_tau_d_fallback

        self._compute_base_cost(active_env, v, contact_result.dynamics_matrix_env, cost)
        wp.launch(
            self.k._projection_eval_contact_sap_batched,
            dim=(self.num_envs, contact_capacity),
            inputs=[
                active_env,
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                contact_result.contact_env_phi0,
                contact_result.contact_env_w_eff,
                contact_result.contact_env_mu,
                contact_result.contact_env_stiffness,
                contact_tau_d,
                v,
                self.contact_beta,
                self.contact_sigma,
                float(dt),
                contact_gamma,
                contact_g,
                contact_vc,
                contact_y,
                contact_rt,
                contact_rn,
                contact_cost,
                contact_mode,
                cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._eval_pd_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_pd_terms),
                active_env,
                self.dof_per_env,
                self.pd_active,
                self.pd_a,
                self.pd_gain,
                self.pd_limit,
                v,
                float(dt),
                pd_y,
                pd_gamma,
                pd_hdiag,
                pd_cost,
                cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._eval_limit_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_limit_terms),
                active_env,
                self.dof_per_env,
                self.limit_lower_active,
                self.limit_upper_active,
                self.limit_lower_vhat,
                self.limit_upper_vhat,
                self.limit_lower_r,
                self.limit_upper_r,
                self.limit_lower_rinv,
                self.limit_upper_rinv,
                v,
                lower_gamma,
                upper_gamma,
                limit_grad,
                limit_hdiag,
                limit_cost,
                cost,
            ],
            device=self.device,
        )
        return contact_capacity

    def _evaluate_cost_terms_no_contact_hessian(
        self,
        contact_result: SapContactJacobianResult,
        v: wp.array,
        active_env: wp.array,
        dt: float,
        *,
        cost: wp.array,
        contact_gamma: wp.array,
        contact_vc: wp.array,
        contact_cost: wp.array,
        pd_y: wp.array,
        pd_gamma: wp.array,
        pd_hdiag: wp.array,
        pd_cost: wp.array,
        lower_gamma: wp.array,
        upper_gamma: wp.array,
        limit_grad: wp.array,
        limit_hdiag: wp.array,
        limit_cost: wp.array,
    ) -> int:
        contact_capacity = self._contact_capacity(contact_result)
        contact_tau_d = getattr(contact_result, "contact_env_tau_d", None)
        if contact_tau_d is None:
            self.contact_tau_d_fallback.fill_(self.contact_tau_d)
            contact_tau_d = self.contact_tau_d_fallback

        self._compute_base_cost(active_env, v, contact_result.dynamics_matrix_env, cost)
        wp.launch(
            self.k._projection_eval_contact_gamma_sap_batched,
            dim=(self.num_envs, contact_capacity),
            inputs=[
                active_env,
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                contact_result.contact_env_phi0,
                contact_result.contact_env_w_eff,
                contact_result.contact_env_mu,
                contact_result.contact_env_stiffness,
                contact_tau_d,
                v,
                self.contact_beta,
                self.contact_sigma,
                float(dt),
                contact_gamma,
                contact_vc,
                contact_cost,
                cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._eval_pd_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_pd_terms),
                active_env,
                self.dof_per_env,
                self.pd_active,
                self.pd_a,
                self.pd_gain,
                self.pd_limit,
                v,
                float(dt),
                pd_y,
                pd_gamma,
                pd_hdiag,
                pd_cost,
                cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._eval_limit_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_limit_terms),
                active_env,
                self.dof_per_env,
                self.limit_lower_active,
                self.limit_upper_active,
                self.limit_lower_vhat,
                self.limit_upper_vhat,
                self.limit_lower_r,
                self.limit_upper_r,
                self.limit_lower_rinv,
                self.limit_upper_rinv,
                v,
                lower_gamma,
                upper_gamma,
                limit_grad,
                limit_hdiag,
                limit_cost,
                cost,
            ],
            device=self.device,
        )
        return contact_capacity

    def _evaluate_problem(
        self,
        contact_result: SapContactJacobianResult,
        v: wp.array,
        active_env: wp.array,
        dt: float,
        *,
        include_hessian: bool,
        cost: wp.array,
        constraint_impulse: wp.array,
        contact_gamma: wp.array,
        contact_g: wp.array,
        contact_vc: wp.array,
        contact_y: wp.array,
        contact_rt: wp.array,
        contact_rn: wp.array,
        contact_cost: wp.array,
        contact_mode: wp.array,
        pd_y: wp.array,
        pd_gamma: wp.array,
        pd_hdiag: wp.array,
        pd_cost: wp.array,
        lower_gamma: wp.array,
        upper_gamma: wp.array,
        limit_grad: wp.array,
        limit_hdiag: wp.array,
        limit_cost: wp.array,
        defer_constraint_terms: bool = False,
    ) -> bool:
        if bool(defer_constraint_terms):
            contact_capacity = self._contact_capacity(contact_result)
            contact_tau_d = getattr(contact_result, "contact_env_tau_d", None)
            if contact_tau_d is None:
                self.contact_tau_d_fallback.fill_(self.contact_tau_d)
                contact_tau_d = self.contact_tau_d_fallback

            cost.zero_()
            if include_hessian:
                wp.launch(
                    self.k._projection_eval_contact_sap_batched,
                    dim=(self.num_envs, contact_capacity),
                    inputs=[
                        active_env,
                        self.dof_per_env,
                        contact_capacity,
                        contact_result.contact_env_count,
                        contact_result.contact_env_jacobian,
                        contact_result.contact_env_phi0,
                        contact_result.contact_env_w_eff,
                        contact_result.contact_env_mu,
                        contact_result.contact_env_stiffness,
                        contact_tau_d,
                        v,
                        self.contact_beta,
                        self.contact_sigma,
                        float(dt),
                        contact_gamma,
                        contact_g,
                        contact_vc,
                        contact_y,
                        contact_rt,
                        contact_rn,
                        contact_cost,
                        contact_mode,
                        cost,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    self.k._projection_eval_contact_gamma_sap_batched,
                    dim=(self.num_envs, contact_capacity),
                    inputs=[
                        active_env,
                        self.dof_per_env,
                        contact_capacity,
                        contact_result.contact_env_count,
                        contact_result.contact_env_jacobian,
                        contact_result.contact_env_phi0,
                        contact_result.contact_env_w_eff,
                        contact_result.contact_env_mu,
                        contact_result.contact_env_stiffness,
                        contact_tau_d,
                        v,
                        self.contact_beta,
                        self.contact_sigma,
                        float(dt),
                        contact_gamma,
                        contact_vc,
                        contact_cost,
                        cost,
                    ],
                    device=self.device,
                )

            self._assemble_contact_impulse_from_terms(
                contact_result,
                contact_capacity=contact_capacity,
                contact_gamma=contact_gamma,
                constraint_impulse=constraint_impulse,
            )
            self._assemble_model_terms_and_grad(
                active_env,
                v,
                contact_result.dynamics_matrix_env,
                dt,
                pd_y=pd_y,
                pd_gamma=pd_gamma,
                pd_hdiag=pd_hdiag,
                pd_cost=pd_cost,
                lower_gamma=lower_gamma,
                upper_gamma=upper_gamma,
                limit_grad=limit_grad,
                limit_hdiag=limit_hdiag,
                limit_cost=limit_cost,
                constraint_impulse=constraint_impulse,
                cost=cost,
            )

            if include_hessian:
                self._assemble_hessian_from_terms(
                    contact_result,
                    active_env=active_env,
                    contact_capacity=contact_capacity,
                    contact_g=contact_g,
                    pd_hdiag=pd_hdiag,
                    limit_hdiag=limit_hdiag,
                )
            return True

        if include_hessian:
            contact_capacity = self._evaluate_cost_terms(
                contact_result,
                v,
                active_env,
                dt,
                cost=cost,
                contact_gamma=contact_gamma,
                contact_g=contact_g,
                contact_vc=contact_vc,
                contact_y=contact_y,
                contact_rt=contact_rt,
                contact_rn=contact_rn,
                contact_cost=contact_cost,
                contact_mode=contact_mode,
                pd_y=pd_y,
                pd_gamma=pd_gamma,
                pd_hdiag=pd_hdiag,
                pd_cost=pd_cost,
                lower_gamma=lower_gamma,
                upper_gamma=upper_gamma,
                limit_grad=limit_grad,
                limit_hdiag=limit_hdiag,
                limit_cost=limit_cost,
            )
        else:
            contact_capacity = self._evaluate_cost_terms_no_contact_hessian(
                contact_result,
                v,
                active_env,
                dt,
                cost=cost,
                contact_gamma=contact_gamma,
                contact_vc=contact_vc,
                contact_cost=contact_cost,
                pd_y=pd_y,
                pd_gamma=pd_gamma,
                pd_hdiag=pd_hdiag,
                pd_cost=pd_cost,
                lower_gamma=lower_gamma,
                upper_gamma=upper_gamma,
                limit_grad=limit_grad,
                limit_hdiag=limit_hdiag,
                limit_cost=limit_cost,
            )

        self._assemble_contact_impulse_from_terms(
            contact_result,
            contact_capacity=contact_capacity,
            contact_gamma=contact_gamma,
            constraint_impulse=constraint_impulse,
        )
        wp.launch(
            self.k._accumulate_pd_impulse_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_pd_terms and not bool(defer_constraint_terms)),
                self.dof_per_env,
                self.pd_active,
                pd_gamma,
                constraint_impulse,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._accumulate_limit_impulse_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_limit_terms and not bool(defer_constraint_terms)),
                self.dof_per_env,
                limit_grad,
                constraint_impulse,
            ],
            device=self.device,
        )

        if include_hessian:
            self._assemble_hessian_from_terms(
                contact_result,
                active_env=active_env,
                contact_capacity=contact_capacity,
                contact_g=contact_g,
                pd_hdiag=pd_hdiag,
                limit_hdiag=limit_hdiag,
            )
        return False

    def _assemble_hessian_from_terms(
        self,
        contact_result: SapContactJacobianResult,
        *,
        active_env: wp.array | None = None,
        contact_capacity: int | None = None,
        contact_g: wp.array | None = None,
        pd_hdiag: wp.array | None = None,
        limit_hdiag: wp.array | None = None,
    ) -> None:
        if active_env is None:
            active_env = self.all_env_active
        if contact_capacity is None:
            contact_capacity = self._contact_capacity(contact_result)
        if contact_g is None:
            contact_g = self.contact_g
        if pd_hdiag is None:
            pd_hdiag = self.pd_hdiag
        if limit_hdiag is None:
            limit_hdiag = self.limit_hdiag

        self._assemble_contact_hessian_from_terms(
            contact_result,
            contact_capacity=contact_capacity,
            contact_g=contact_g,
        )
        wp.launch(
            self.k._assemble_hessian_total_batched,
            dim=(self.num_envs, self.dof_per_env, self.dof_per_env),
            inputs=[
                active_env,
                self.dof_per_env,
                contact_result.dynamics_matrix_env,
                self.hess_contact,
                int(self._has_pd_terms),
                int(self._has_limit_terms),
                pd_hdiag,
                limit_hdiag,
                self.hessian,
            ],
            device=self.device,
        )

    def evaluate_itemwise(
        self,
        contact_result: SapContactJacobianResult,
        state: State,
        control: Control | None,
        dt: float,
        v_star: wp.array,
        *,
        v: wp.array | None = None,
        v0: wp.array | None = None,
    ) -> SapContactSolveResult:
        """Evaluate itemwise SAP objective, gradient, and line-search quantities for diagnostics or solver
        internals.
        """
        self.prepare(contact_result, state, control, dt, v_star, v0=v0, v_guess=v)
        grad_ready = self._evaluate_problem(
            contact_result,
            self.v_env,
            self.all_env_active,
            dt,
            include_hessian=True,
            cost=self.cost,
            constraint_impulse=self.constraint_impulse,
            contact_gamma=self.contact_gamma,
            contact_g=self.contact_g,
            contact_vc=self.contact_vc,
            contact_y=self.contact_y,
            contact_rt=self.contact_rt,
            contact_rn=self.contact_rn,
            contact_cost=self.contact_cost,
            contact_mode=self.contact_mode,
            pd_y=self.pd_y,
            pd_gamma=self.pd_gamma,
            pd_hdiag=self.pd_hdiag,
            pd_cost=self.pd_cost,
            lower_gamma=self.limit_lower_gamma,
            upper_gamma=self.limit_upper_gamma,
            limit_grad=self.limit_grad,
            limit_hdiag=self.limit_hdiag,
            limit_cost=self.limit_cost,
            defer_constraint_terms=True,
        )
        if not grad_ready:
            self._assemble_grad_and_dynamics_impulse(
                self.all_env_active,
                self.v_env,
                contact_result.dynamics_matrix_env,
                self.constraint_impulse,
                add_constraint_terms=True,
            )
        wp.launch(
            self.k._copy_env_to_flat_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.v_env, self.dof_per_env, self.v_flat],
            device=self.device,
        )
        return self._make_result(0, 0, True)

    def _solve_newton_direction(self, *, store_first_dv: bool = False) -> None:
        wp.launch(
            self._pack_dense_and_grad_to_padded_batched,
            dim=(self.num_envs, self.padded_dof, self.padded_dof),
            inputs=[
                self.hessian,
                self.grad,
                self.chol_a,
                self.chol_rhs,
                self.dof_per_env,
                self.diag_shift,
            ],
            device=self.device,
        )
        self.block_solver.factorize_masked(self.chol_a, self.dof_per_env, self.newton_active)
        self.block_solver.solve_masked(self.chol_rhs, self.chol_x, self.newton_active)
        if store_first_dv:
            wp.launch(
                self._unpack_solution_and_first_batched,
                dim=(self.num_envs, self.padded_dof),
                inputs=[
                    self.newton_active,
                    self.newton_loop_iteration,
                    self.chol_x,
                    self.dv,
                    self.first_dv,
                    self.dof_per_env,
                ],
                device=self.device,
            )
            return
        wp.launch(
            self._unpack_solution_batched,
            dim=(self.num_envs, self.padded_dof),
            inputs=[self.chol_x, self.dv, self.dof_per_env],
            device=self.device,
        )

    def _run_sap_backtracking(
        self,
        contact_result: SapContactJacobianResult,
        dt: float,
        *,
        armijo_c: float,
        rho: float,
        alpha_max: float,
        max_iterations: int,
        relative_slop: float,
        check_errors: bool = True,
    ) -> None:
        if int(max_iterations) <= 0:
            return
        self.dell0.zero_()
        self.dell_a0.zero_()
        self.d2ell_a.zero_()
        wp.launch(
            self.k._compute_search_direction_data_serial_batched,
            dim=self.num_envs,
            inputs=[
                self.newton_active,
                self.dof_per_env,
                contact_result.dynamics_matrix_env,
                self.v_env,
                self.v_star_env,
                self.grad,
                self.dv,
                self.dp,
                self.dell0,
                self.dell_a0,
                self.d2ell_a,
            ],
            device=self.device,
        )
        self._compute_base_cost(
            self.newton_active,
            self.v_env,
            contact_result.dynamics_matrix_env,
            self.line_momentum_cost,
        )

        wp.launch(
            self.k._init_sap_backtracking_state,
            dim=self.num_envs,
            inputs=[
                self.newton_active,
                self.cost,
                self.dell0,
                float(alpha_max),
                float(relative_slop),
                self.alpha,
                self.alpha_prev,
                self.ell_prev,
                self.ell_slop,
                self.ls_active,
                self.ls_accepted,
                self.ls_status,
                self.ls_iterations,
                self.accepted_cost,
            ],
                device=self.device,
            )

        wp.launch(
            self.k._axpy_to_trial_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.ls_active, self.dof_per_env, self.v_env, self.dv, self.alpha, self.v_trial],
            device=self.device,
        )
        self._evaluate_trial(contact_result, dt)
        self._replace_trial_cost_with_sap_line_search_cost(contact_result)
        self._compute_trial_derivative()

        self.ls_active_count.zero_()
        wp.launch(
            self.k._accept_sap_alpha_max,
            dim=self.num_envs,
            inputs=[
                self.trial_cost,
                self.trial_derivative,
                self.cost,
                float(relative_slop),
                self.ls_active,
                self.ls_accepted,
                self.alpha,
                self.alpha_prev,
                self.ell_prev,
                self.ell_slop,
                self.accepted_cost,
                self.ls_active_count,
            ],
                device=self.device,
            )

        if int(max_iterations) > 1:
            self.ls_loop_iteration.fill_(1)
            wp.capture_while(
                self.ls_active_count,
                while_body=self._run_sap_backtracking_body,
                contact_result=contact_result,
                dt=float(dt),
                armijo_c=float(armijo_c),
                rho=float(rho),
                max_iterations=int(max_iterations),
            )

        if check_errors:
            self._raise_line_search_errors_if_any(stage="backtracking")
        wp.launch(
            self.k._accumulate_line_search_iterations_batched,
            dim=self.num_envs,
            inputs=[self.ls_accepted, self.ls_iterations, self.ls_iterations_total],
                device=self.device,
            )

        wp.launch(
            self.k._commit_line_search_step_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                self.newton_active,
                self.ls_accepted,
                self.dof_per_env,
                self.alpha,
                self.v_env,
                self.dv,
                self.v_flat,
                self.cost,
                self.previous_cost,
                self.accepted_cost,
            ],
            device=self.device,
        )

    def _run_sap_backtracking_body(
        self,
        *,
        contact_result: SapContactJacobianResult,
        dt: float,
        armijo_c: float,
        rho: float,
        max_iterations: int,
    ) -> None:
        wp.launch(
            self.k._scale_sap_backtracking_alpha,
            dim=self.num_envs,
            inputs=[self.ls_active, self.alpha, float(rho)],
            device=self.device,
        )
        wp.launch(
            self.k._axpy_to_trial_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.ls_active, self.dof_per_env, self.v_env, self.dv, self.alpha, self.v_trial],
            device=self.device,
        )
        self._evaluate_trial(contact_result, dt)
        self._replace_trial_cost_with_sap_line_search_cost(contact_result)
        self.ls_active_count.zero_()
        wp.launch(
            self.k._update_sap_backtracking_iteration_conditional,
            dim=self.num_envs,
            inputs=[
                self.trial_cost,
                self.cost,
                self.dell0,
                self.alpha,
                self.alpha_prev,
                self.ell_prev,
                self.ell_slop,
                float(armijo_c),
                self.ls_loop_iteration,
                int(max_iterations),
                self.ls_active,
                self.ls_accepted,
                self.ls_status,
                self.ls_iterations,
                self.accepted_cost,
                self.ls_active_count,
            ],
            device=self.device,
        )
        wp.launch(self.k._increment_scalar_i32, dim=1, inputs=[self.ls_loop_iteration], device=self.device)

    def _run_unit_decay_line_search(
        self,
        contact_result: SapContactJacobianResult,
        dt: float,
        *,
        max_iterations: int,
        decay: float,
        min_alpha: float,
        cost_relax_r: float,
        cost_relax_a: float,
        check_errors: bool = True,
    ) -> None:
        contact_tau_d = getattr(contact_result, "contact_env_tau_d", None)
        if contact_tau_d is None:
            self.contact_tau_d_fallback.fill_(self.contact_tau_d)
            contact_tau_d = self.contact_tau_d_fallback
        contact_capacity = self._contact_capacity(contact_result)
        wp.launch_tiled(
            self._unit_line_search_base_coeffs,
            dim=self.num_envs,
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                self.newton_active,
                self.dof_per_env,
                self.v_env,
                self.v_star_env,
                self.dv,
                contact_result.dynamics_matrix_env,
                self.line_search_base0,
                self.line_search_base_linear,
                self.line_search_base_quadratic,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self._unit_line_search_contact_delta_velocity,
            dim=(self.num_envs, contact_capacity),
            block_dim=self.unit_line_search_contact_vc_tile_size,
            inputs=[
                self.newton_active,
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                self.dv,
                self.line_search_contact_dvc,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self._unit_line_search_fused_parallel,
            dim=self.num_envs,
            block_dim=self.unit_line_search_tile_size,
            inputs=[
                self.newton_active,
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                self.contact_vc,
                self.line_search_contact_dvc,
                contact_result.contact_env_phi0,
                contact_result.contact_env_w_eff,
                contact_result.contact_env_mu,
                contact_result.contact_env_stiffness,
                contact_tau_d,
                self.pd_active,
                self.pd_a,
                self.pd_gain,
                self.pd_limit,
                self.limit_lower_active,
                self.limit_upper_active,
                self.limit_lower_vhat,
                self.limit_upper_vhat,
                self.limit_lower_r,
                self.limit_upper_r,
                self.limit_lower_rinv,
                self.limit_upper_rinv,
                self.v_env,
                self.dv,
                self.v_flat,
                self.line_search_base0,
                self.line_search_base_linear,
                self.line_search_base_quadratic,
                self.solve_dtype(self.contact_beta),
                self.solve_dtype(self.contact_sigma),
                self.solve_dtype(dt),
                1,
                int(self._has_pd_terms),
                int(self._has_limit_terms),
                int(max_iterations),
                self.solve_dtype(decay),
                self.solve_dtype(min_alpha),
                self.solve_dtype(cost_relax_r),
                self.solve_dtype(cost_relax_a),
                self.alpha,
                self.cost,
                self.previous_cost,
                self.accepted_cost,
                self.ls_active,
                self.ls_accepted,
                self.ls_status,
                self.ls_iterations,
                self.ls_iterations_total,
            ],
            device=self.device,
        )
        if check_errors:
            self._raise_line_search_errors_if_any(stage="unit_device")

    def _run_conditional_line_search(
        self,
        contact_result: SapContactJacobianResult,
        dt: float,
        *,
        line_search_variant: str,
        line_search_max_iterations: int,
        armijo_c: float,
        rho: float,
        line_search_relative_slop: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        check_errors: bool,
    ) -> None:
        if line_search_variant == "monotone_decay":
            self._run_unit_decay_line_search(
                contact_result,
                dt,
                max_iterations=int(line_search_max_iterations),
                decay=0.5,
                min_alpha=1.0e-8,
                cost_relax_r=1.0e-12,
                cost_relax_a=1.0e-14,
                check_errors=bool(check_errors),
            )
            return

        if line_search_variant == "armijo_decay":
            self._run_sap_backtracking(
                contact_result,
                dt,
                armijo_c=float(armijo_c),
                rho=float(rho),
                alpha_max=1.0 / float(rho),
                max_iterations=int(line_search_max_iterations),
                relative_slop=float(line_search_relative_slop),
                check_errors=bool(check_errors),
            )
            return

        if line_search_variant == "exact_root":
            self._run_sap_exact(
                contact_result,
                dt,
                max_iterations=int(line_search_max_iterations),
                cost_abs_tol=float(cost_abs_tol),
                cost_rel_tol=float(cost_rel_tol),
                check_errors=bool(check_errors),
            )
            return

        raise ValueError(f"Unsupported SAP line search variant {line_search_variant!r}.")

    def _run_unit_conditional_newton_body(
        self,
        *,
        contact_result: SapContactJacobianResult,
        dt: float,
        max_iterations: int,
        optimality_abs_tol: float,
        optimality_rel_tol: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        cost_min_alpha: float,
        line_search_max_iterations: int,
        line_search_variant: str,
        armijo_c: float,
        rho: float,
        line_search_relative_slop: float,
        check_line_search_errors: bool,
    ) -> None:
        self._run_unit_conditional_newton_step(
            contact_result=contact_result,
            dt=float(dt),
            line_search_max_iterations=int(line_search_max_iterations),
            line_search_variant=line_search_variant,
            armijo_c=float(armijo_c),
            rho=float(rho),
            line_search_relative_slop=float(line_search_relative_slop),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            check_line_search_errors=bool(check_line_search_errors),
        )
        self._solver_update_active(
            contact_result,
            self.stage2_active_env,
            float(dt),
            max_iterations=int(max_iterations),
            optimality_abs_tol=float(optimality_abs_tol),
            optimality_rel_tol=float(optimality_rel_tol),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            cost_min_alpha=float(cost_min_alpha),
        )

    def _run_unit_conditional_newton_step(
        self,
        *,
        contact_result: SapContactJacobianResult,
        dt: float,
        line_search_max_iterations: int,
        line_search_variant: str,
        armijo_c: float,
        rho: float,
        line_search_relative_slop: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        check_line_search_errors: bool,
    ) -> None:
        contact_capacity = self._ensure_hessian_terms_for_active_envs(
            contact_result,
            self.newton_active,
            dt,
        )
        self._assemble_hessian_from_terms(
            contact_result,
            active_env=self.newton_active,
            contact_capacity=contact_capacity,
        )
        self._solve_newton_direction(store_first_dv=True)
        self._run_conditional_line_search(
            contact_result,
            dt,
            line_search_variant=line_search_variant,
            line_search_max_iterations=int(line_search_max_iterations),
            armijo_c=float(armijo_c),
            rho=float(rho),
            line_search_relative_slop=float(line_search_relative_slop),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            check_errors=bool(check_line_search_errors),
        )
        wp.launch(self.k._increment_scalar_i32, dim=1, inputs=[self.newton_loop_iteration], device=self.device)

    def _run_unit_conditional_newton_loop(
        self,
        contact_result: SapContactJacobianResult,
        dt: float,
        *,
        max_iterations: int,
        optimality_abs_tol: float,
        optimality_rel_tol: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        line_search_max_iterations: int,
        line_search_variant: str,
        armijo_c: float,
        rho: float,
        line_search_relative_slop: float,
        cost_min_alpha: float,
        collect_iteration_stats: bool,
        check_line_search_errors: bool,
        loop_counters_initialized: bool = False,
        v_flat_seeded: bool = False,
    ) -> SapContactSolveResult:
        if not bool(loop_counters_initialized):
            wp.launch(
                self.k._initialize_newton_loop_state,
                dim=1,
                inputs=[self.newton_loop_iteration, self.newton_max_reached],
                device=self.device,
            )

        self._solver_update_active(
            contact_result,
            self.stage2_active_env,
            float(dt),
            max_iterations=int(max_iterations),
            optimality_abs_tol=float(optimality_abs_tol),
            optimality_rel_tol=float(optimality_rel_tol),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            cost_min_alpha=float(cost_min_alpha),
        )
        if int(max_iterations) > 0:
            wp.capture_while(
                self.active_count,
                while_body=self._run_unit_conditional_newton_body,
                contact_result=contact_result,
                dt=float(dt),
                max_iterations=int(max_iterations),
                optimality_abs_tol=float(optimality_abs_tol),
                optimality_rel_tol=float(optimality_rel_tol),
                cost_abs_tol=float(cost_abs_tol),
                cost_rel_tol=float(cost_rel_tol),
                cost_min_alpha=float(cost_min_alpha),
                line_search_max_iterations=int(line_search_max_iterations),
                line_search_variant=line_search_variant,
                armijo_c=float(armijo_c),
                rho=float(rho),
                line_search_relative_slop=float(line_search_relative_slop),
                check_line_search_errors=False,
            )

        if not bool(v_flat_seeded):
            wp.launch(
                self.k._copy_env_to_flat_batched,
                dim=(self.num_envs, self.dof_per_env),
                inputs=[self.v_env, self.dof_per_env, self.v_flat],
                device=self.device,
            )

        self.last_iterations = -1
        self.last_line_search_iterations = -1
        converged = True
        return self._make_result(self.last_iterations, self.last_line_search_iterations, converged)

    def _run_unit_conditional_newton_loop_capture(
        self,
        *,
        contact_result: SapContactJacobianResult,
        dt: float,
        max_iterations: int,
        optimality_abs_tol: float,
        optimality_rel_tol: float,
        cost_abs_tol: float,
        cost_rel_tol: float,
        line_search_max_iterations: int,
        line_search_variant: str,
        armijo_c: float,
        rho: float,
        line_search_relative_slop: float,
        cost_min_alpha: float,
        collect_iteration_stats: bool,
        check_line_search_errors: bool,
        loop_counters_initialized: bool,
        v_flat_seeded: bool,
    ) -> None:
        self._run_unit_conditional_newton_loop(
            contact_result,
            float(dt),
            max_iterations=int(max_iterations),
            optimality_abs_tol=float(optimality_abs_tol),
            optimality_rel_tol=float(optimality_rel_tol),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            line_search_max_iterations=int(line_search_max_iterations),
            line_search_variant=line_search_variant,
            armijo_c=float(armijo_c),
            rho=float(rho),
            line_search_relative_slop=float(line_search_relative_slop),
            cost_min_alpha=float(cost_min_alpha),
            collect_iteration_stats=bool(collect_iteration_stats),
            check_line_search_errors=bool(check_line_search_errors),
            loop_counters_initialized=bool(loop_counters_initialized),
            v_flat_seeded=bool(v_flat_seeded),
        )

    def _run_sap_exact(
        self,
        contact_result: SapContactJacobianResult,
        dt: float,
        *,
        max_iterations: int,
        cost_abs_tol: float,
        cost_rel_tol: float,
        check_errors: bool = True,
    ) -> None:
        if int(max_iterations) <= 0:
            return
        self.dell0.zero_()
        self.dell_a0.zero_()
        self.d2ell_a.zero_()
        wp.launch(
            self.k._compute_search_direction_data_serial_batched,
            dim=self.num_envs,
            inputs=[
                self.newton_active,
                self.dof_per_env,
                contact_result.dynamics_matrix_env,
                self.v_env,
                self.v_star_env,
                self.grad,
                self.dv,
                self.dp,
                self.dell0,
                self.dell_a0,
                self.d2ell_a,
            ],
            device=self.device,
        )
        self._compute_base_cost(
            self.newton_active,
            self.v_env,
            contact_result.dynamics_matrix_env,
            self.line_momentum_cost,
        )

        alpha_max = float(_SAP_EXACT_LINE_SEARCH_ALPHA_MAX)
        wp.launch(
            self.k._init_sap_exact_alpha_max_state,
            dim=self.num_envs,
            inputs=[
                self.newton_active,
                self.cost,
                self.dell0,
                float(alpha_max),
                self.alpha,
                self.ls_active,
                self.ls_accepted,
                self.ls_status,
                self.ls_iterations,
                self.accepted_cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._axpy_to_trial_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.ls_active, self.dof_per_env, self.v_env, self.dv, self.alpha, self.v_trial],
            device=self.device,
        )
        self._evaluate_trial(contact_result, dt, include_contact_hessian=True)
        self._replace_trial_cost_with_sap_line_search_cost(contact_result)
        self._compute_trial_derivative()
        self._compute_trial_second_derivative(contact_result)
        self.ls_active_count.zero_()
        wp.launch(
            self.k._init_sap_exact_root_state,
            dim=self.num_envs,
            inputs=[
                self.trial_cost,
                self.trial_derivative,
                self.trial_second_derivative,
                self.cost,
                self.dell0,
                float(alpha_max),
                float(cost_abs_tol),
                float(cost_rel_tol),
                float(_SAP_EXACT_LINE_SEARCH_F_TOLERANCE),
                self.alpha,
                self.ls_active,
                self.ls_accepted,
                self.ls_status,
                self.accepted_cost,
                self.exact_scale,
                self.exact_x_lower,
                self.exact_x_upper,
                self.exact_f_lower,
                self.exact_f_upper,
                self.exact_root,
                self.exact_minus_dx,
                self.exact_minus_dx_previous,
                self.exact_x_tolerance,
                self.ls_active_count,
            ],
            device=self.device,
        )

        self.ls_loop_iteration.fill_(1)
        wp.capture_while(
            self.ls_active_count,
            while_body=self._run_sap_exact_root_body,
            contact_result=contact_result,
            dt=float(dt),
            max_iterations=int(max_iterations),
        )

        wp.launch(self.k._copy_i32_batched, dim=self.num_envs, inputs=[self.ls_accepted, self.ls_active], device=self.device)
        wp.launch(
            self.k._axpy_to_trial_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.ls_active, self.dof_per_env, self.v_env, self.dv, self.alpha, self.v_trial],
            device=self.device,
        )
        self._evaluate_trial_cost_only(contact_result, dt)
        self._replace_trial_cost_with_sap_line_search_cost(contact_result)
        wp.launch(
            self.k._store_exact_accepted_cost,
            dim=self.num_envs,
            inputs=[self.ls_accepted, self.trial_cost, self.accepted_cost],
            device=self.device,
        )

        wp.launch(
            self.k._accumulate_line_search_iterations_batched,
            dim=self.num_envs,
            inputs=[self.ls_accepted, self.ls_iterations, self.ls_iterations_total],
            device=self.device,
        )
        wp.launch(
            self.k._commit_line_search_step_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                self.newton_active,
                self.ls_accepted,
                self.dof_per_env,
                self.alpha,
                self.v_env,
                self.dv,
                self.v_flat,
                self.cost,
                self.previous_cost,
                self.accepted_cost,
            ],
            device=self.device,
        )
        if check_errors:
            self._raise_line_search_errors_if_any(stage="exact")

    def _run_sap_exact_root_body(
        self,
        *,
        contact_result: SapContactJacobianResult,
        dt: float,
        max_iterations: int,
    ) -> None:
        wp.launch(
            self.k._axpy_to_trial_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[self.ls_active, self.dof_per_env, self.v_env, self.dv, self.alpha, self.v_trial],
            device=self.device,
        )
        self._evaluate_trial(contact_result, dt, include_contact_hessian=True)
        self._replace_trial_cost_with_sap_line_search_cost(contact_result)
        self._compute_trial_derivative()
        self._compute_trial_second_derivative(contact_result)
        self.ls_active_count.zero_()
        wp.launch(
            self.k._update_sap_exact_root_state,
            dim=self.num_envs,
            inputs=[
                self.trial_derivative,
                self.trial_second_derivative,
                float(_SAP_EXACT_LINE_SEARCH_F_TOLERANCE),
                self.ls_loop_iteration,
                int(max_iterations),
                self.alpha,
                self.ls_active,
                self.ls_accepted,
                self.ls_status,
                self.ls_iterations,
                self.exact_scale,
                self.exact_x_lower,
                self.exact_x_upper,
                self.exact_f_lower,
                self.exact_f_upper,
                self.exact_root,
                self.exact_minus_dx,
                self.exact_minus_dx_previous,
                self.exact_x_tolerance,
                self.ls_active_count,
            ],
            device=self.device,
        )
        wp.launch(self.k._increment_scalar_i32, dim=1, inputs=[self.ls_loop_iteration], device=self.device)

    def _evaluate_trial(
        self,
        contact_result: SapContactJacobianResult,
        dt: float,
        *,
        include_contact_hessian: bool = False,
    ) -> None:
        self._evaluate_problem(
            contact_result,
            self.v_trial,
            self.ls_active,
            dt,
            include_hessian=bool(include_contact_hessian),
            cost=self.trial_cost,
            constraint_impulse=self.trial_constraint_impulse,
            contact_gamma=self.trial_contact_gamma,
            contact_g=self.trial_contact_g,
            contact_vc=self.trial_contact_vc,
            contact_y=self.trial_contact_y,
            contact_rt=self.trial_contact_rt,
            contact_rn=self.trial_contact_rn,
            contact_cost=self.trial_contact_cost,
            contact_mode=self.trial_contact_mode,
            pd_y=self.trial_pd_y,
            pd_gamma=self.trial_pd_gamma,
            pd_hdiag=self.trial_pd_hdiag,
            pd_cost=self.trial_pd_cost,
            lower_gamma=self.trial_limit_lower_gamma,
            upper_gamma=self.trial_limit_upper_gamma,
            limit_grad=self.trial_limit_grad,
            limit_hdiag=self.trial_limit_hdiag,
            limit_cost=self.trial_limit_cost,
        )

    def _evaluate_trial_cost_only(self, contact_result: SapContactJacobianResult, dt: float) -> None:
        self.trial_cost.zero_()
        contact_capacity = self._contact_capacity(contact_result)
        contact_tau_d = getattr(contact_result, "contact_env_tau_d", None)
        if contact_tau_d is None:
            self.contact_tau_d_fallback.fill_(self.contact_tau_d)
            contact_tau_d = self.contact_tau_d_fallback

        self._compute_base_cost(
            self.ls_active,
            self.v_trial,
            contact_result.dynamics_matrix_env,
            self.trial_cost,
        )
        wp.launch(
            self.k._projection_cost_only_contact_sap_batched,
            dim=(self.num_envs, contact_capacity),
            inputs=[
                self.ls_active,
                self.dof_per_env,
                contact_capacity,
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                contact_result.contact_env_phi0,
                contact_result.contact_env_w_eff,
                contact_result.contact_env_mu,
                contact_result.contact_env_stiffness,
                contact_tau_d,
                self.v_trial,
                self.dv,
                self.contact_beta,
                self.contact_sigma,
                float(dt),
                self.trial_contact_cost,
                self.trial_cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._eval_pd_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_pd_terms),
                self.ls_active,
                self.dof_per_env,
                self.pd_active,
                self.pd_a,
                self.pd_gain,
                self.pd_limit,
                self.v_trial,
                float(dt),
                self.trial_pd_y,
                self.trial_pd_gamma,
                self.trial_pd_hdiag,
                self.trial_pd_cost,
                self.trial_cost,
            ],
            device=self.device,
        )
        wp.launch(
            self.k._eval_limit_terms_sap_batched,
            dim=(self.num_envs, self.dof_per_env),
            inputs=[
                int(self._has_limit_terms),
                self.ls_active,
                self.dof_per_env,
                self.limit_lower_active,
                self.limit_upper_active,
                self.limit_lower_vhat,
                self.limit_upper_vhat,
                self.limit_lower_r,
                self.limit_upper_r,
                self.limit_lower_rinv,
                self.limit_upper_rinv,
                self.v_trial,
                self.trial_limit_lower_gamma,
                self.trial_limit_upper_gamma,
                self.trial_limit_grad,
                self.trial_limit_hdiag,
                self.trial_limit_cost,
                self.trial_cost,
            ],
            device=self.device,
        )

    def _replace_trial_cost_with_sap_line_search_cost(self, contact_result: SapContactJacobianResult) -> None:
        wp.launch(
            self.k._replace_trial_cost_with_sap_line_search_cost_batched,
            dim=self.num_envs,
            inputs=[
                self.ls_active,
                self.dof_per_env,
                self._contact_capacity(contact_result),
                contact_result.contact_env_count,
                self.trial_contact_cost,
                int(self._has_pd_terms),
                int(self._has_limit_terms),
                self.trial_pd_cost,
                self.trial_limit_cost,
                self.line_momentum_cost,
                self.dell_a0,
                self.d2ell_a,
                self.alpha,
                self.trial_cost,
            ],
            device=self.device,
        )

    def _compute_trial_derivative(self) -> None:
        self.trial_derivative.zero_()
        wp.launch(
            self.k._compute_line_derivative_serial_batched,
            dim=self.num_envs,
            inputs=[
                self.ls_active,
                self.dof_per_env,
                self.v_trial,
                self.v_star_env,
                self.dv,
                self.dp,
                self.trial_constraint_impulse,
                self.trial_derivative,
            ],
            device=self.device,
        )

    def _compute_trial_second_derivative(self, contact_result: SapContactJacobianResult) -> None:
        self.trial_second_derivative.zero_()
        wp.launch(
            self.k._compute_line_second_derivative_serial_batched,
            dim=self.num_envs,
            inputs=[
                self.ls_active,
                self.dof_per_env,
                self._contact_capacity(contact_result),
                contact_result.contact_env_count,
                contact_result.contact_env_jacobian,
                self.trial_contact_g,
                int(self._has_pd_terms),
                int(self._has_limit_terms),
                self.trial_pd_hdiag,
                self.trial_limit_hdiag,
                self.dv,
                self.d2ell_a,
                self.trial_second_derivative,
            ],
            device=self.device,
        )

    def _raise_line_search_errors_if_any(self, *, stage: str) -> None:
        status = self.ls_status.numpy()
        bad = np.nonzero(status < 0)[0]
        if bad.size == 0:
            return
        env = int(bad[0])
        code = int(status[env])
        dell0 = float(self.dell0.numpy()[env])
        cost = float(self.cost.numpy()[env])
        alpha = float(self.alpha.numpy()[env])
        raise RuntimeError(
            "SapContactSolve line search failed "
            f"(stage={stage}, env={env}, status={code}, dell0={dell0:.6e}, "
            f"cost={cost:.6e}, alpha={alpha:.6e})."
        )

    def solve(
        self,
        contact_result: SapContactJacobianResult,
        state: State,
        control: Control | None,
        dt: float,
        v_star: wp.array,
        *,
        v0: wp.array | None = None,
        v_guess: wp.array | None = None,
        v_guess_active: wp.array | None = None,
        max_iterations: int = 100,
        optimality_abs_tol: float = 1.0e-14,
        optimality_rel_tol: float = 1.0e-6,
        cost_abs_tol: float | None = None,
        cost_rel_tol: float | None = None,
        line_search_max_iterations: int = 40,
        armijo_c: float = 1.0e-4,
        rho: float = 0.8,
        line_search_relative_slop: float | None = None,
        line_search_variant: str = "monotone_decay",
        collect_iteration_stats: bool = True,
        check_line_search_errors: bool = True,
        graph_conditional: bool = True,
    ) -> SapContactSolveResult:
        """Solve the SAP velocity objective for the active contacts and write the next generalized velocity
        in SAP order.
        """
        if line_search_relative_slop is None:
            line_search_relative_slop = 1000.0 * np.finfo(self.numpy_dtype).eps
        contact_result = self._cast_contact_result(contact_result)
        line_search_variant = normalize_sap_line_search_mode(line_search_variant)
        if not bool(graph_conditional):
            raise ValueError(
                "SapContactSolve only supports graph_conditional=True; "
                "the Python Newton-loop fallback has been removed."
            )
        if line_search_variant == "armijo_decay" and not (0.0 < float(rho) < 1.0):
            raise ValueError(f"rho must lie in (0, 1), got {rho!r}.")
        if line_search_variant == "exact_root" and int(line_search_max_iterations) == 40:
            line_search_max_iterations = int(_SAP_EXACT_LINE_SEARCH_MAX_ITERATIONS)
        if cost_abs_tol is None:
            cost_abs_tol = 0.0 if line_search_variant == "monotone_decay" else 1.0e-30
        if cost_rel_tol is None:
            cost_rel_tol = 5.0e-3 if line_search_variant == "monotone_decay" else 1.0e-15
        cost_min_alpha = 0.0 if line_search_variant == "monotone_decay" else 0.5

        self.prepare(
            contact_result,
            state,
            control,
            dt,
            v_star,
            v0=v0,
            v_guess=v_guess,
            v_guess_active=v_guess_active,
        )
        self.last_iterations = 0
        self.last_line_search_iterations = 0
        self.stage2_active_count.zero_()
        wp.launch(
            self.k._initialize_and_mark_unconstrained_free_envs_batched,
            dim=self.num_envs,
            inputs=[
                self.dof_per_env,
                contact_result.contact_env_count,
                self.pd_active,
                self.limit_lower_active,
                self.limit_upper_active,
                self.participating_dof,
                self.v_star_env,
                self.v_env,
                self.v_flat,
                self.first_dv,
                self.newton_iterations_env,
                self.ls_iterations_total,
                self.alpha,
                self.previous_cost,
                self.converged_env,
                self.optimality_reached_env,
                self.cost_reached_env,
                self.stage2_active_env,
                self.newton_active,
                self.stage2_active_count,
            ],
            device=self.device,
        )
        wp.capture_if(
            self.stage2_active_count,
            on_true=self._run_unit_conditional_newton_loop_capture,
            contact_result=contact_result,
            dt=float(dt),
            max_iterations=int(max_iterations),
            optimality_abs_tol=float(optimality_abs_tol),
            optimality_rel_tol=float(optimality_rel_tol),
            cost_abs_tol=float(cost_abs_tol),
            cost_rel_tol=float(cost_rel_tol),
            line_search_max_iterations=int(line_search_max_iterations),
            line_search_variant=line_search_variant,
            armijo_c=float(armijo_c),
            rho=float(rho),
            line_search_relative_slop=float(line_search_relative_slop),
            cost_min_alpha=float(cost_min_alpha),
            collect_iteration_stats=bool(collect_iteration_stats),
            check_line_search_errors=bool(check_line_search_errors),
            loop_counters_initialized=False,
            v_flat_seeded=(v_guess is self.v_flat),
        )
        if v_guess_active is not None:
            wp.launch(
                self.k._set_scalar_i32,
                dim=1,
                inputs=[v_guess_active, 1],
                device=self.device,
            )
        iterations_env = np.asarray(self.newton_iterations_env.numpy(), dtype=np.int64).reshape(-1)
        line_search_iterations_env = np.asarray(self.ls_iterations_total.numpy(), dtype=np.int64).reshape(-1)
        converged_env = np.asarray(self.converged_env.numpy(), dtype=np.int64).reshape(-1)
        self.last_iterations = int(iterations_env.max(initial=0))
        self.last_line_search_iterations = int(line_search_iterations_env.max(initial=0))
        converged = bool(np.all(converged_env != 0))
        return self._make_result(self.last_iterations, self.last_line_search_iterations, converged)

    def _make_result(self, iterations: int, line_search_iterations: int, converged: bool) -> SapContactSolveResult:
        return SapContactSolveResult(
            v_env=self.v_env,
            v_flat=self.v_flat,
            cost=self.cost,
            previous_cost=self.previous_cost,
            grad=self.grad,
            hessian=self.hessian,
            constraint_impulse=self.constraint_impulse,
            dynamics_impulse=self.dynamics_impulse,
            contact_gamma=self.contact_gamma,
            contact_g=self.contact_g,
            contact_vc=self.contact_vc,
            contact_y=self.contact_y,
            contact_rt=self.contact_rt,
            contact_rn=self.contact_rn,
            contact_cost=self.contact_cost,
            contact_mode=self.contact_mode,
            pd_active=self.pd_active,
            pd_y=self.pd_y,
            pd_gamma=self.pd_gamma,
            pd_hdiag=self.pd_hdiag,
            pd_cost=self.pd_cost,
            pd_kp_eff=self.pd_kp_eff,
            pd_kd_eff=self.pd_kd_eff,
            limit_lower_active=self.limit_lower_active,
            limit_upper_active=self.limit_upper_active,
            limit_lower_gamma=self.limit_lower_gamma,
            limit_upper_gamma=self.limit_upper_gamma,
            limit_grad=self.limit_grad,
            limit_hdiag=self.limit_hdiag,
            limit_cost=self.limit_cost,
            first_dv=self.first_dv,
            alpha=self.alpha,
            newton_iterations_env=self.newton_iterations_env,
            line_search_iterations_env=self.ls_iterations_total,
            newton_active=self.newton_active,
            converged_env=self.converged_env,
            optimality_reached_env=self.optimality_reached_env,
            cost_reached_env=self.cost_reached_env,
            iterations=int(iterations),
            line_search_iterations=int(line_search_iterations),
            converged=bool(converged),
        )


__all__ = [
    "SapContactSolve",
    "SapContactSolveResult",
    "normalize_sap_line_search_mode",
    "_CONTACT_MODE_NONE",
    "_CONTACT_MODE_STICTION",
    "_CONTACT_MODE_SLIDING",
    "_CONTACT_MODE_FRICTIONLESS",
]
