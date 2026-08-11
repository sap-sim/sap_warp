"""Contact Jacobian assembly for the SAP velocity solve. The public entry point is
SapContactJacobian, which reads free-motion state, active contacts, body inertia, and material
arrays, then produces per-contact Jacobians and per-environment dynamics blocks consumed by
SapContactSolve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from sim.sap_helpers import (
    _body_com_world_d,
    _body_com_world_f32,
    _body_com_world_f32_pose,
    _contact_env_from_shapes,
    _contact_mu_pair,
    _contact_tau_pair,
    _sap_combine_stiffness,
    _sap_make_from_one_unit_vector_z,
    _sap_make_from_one_unit_vector_z_f32,
    _infer_contact_mu_fallback,
    _resolve_contact_shape_mu,
    _resolve_contact_shape_stiffness,
    _safe_normalize,
    _safe_normalize_f32,
    _sap_body_direction_weight_d,
    _sap_body_direction_weight_f32,
    _sap_body_direction_weight_f32_pose,
    _shape_stiffness_or_fallback,
    _transform_point_d,
    _transform_point_f32_pose,
    _transform_point_f32_pose_d,
    _transform_translation_f32_pose_d,
    _v3_zero_f32,
    _v3d_zero,
    _vec3_to_vec3d,
)
from sim.free_motion import SapFreeMotion
from sim.sap_runtime import (
    Control,
    Model,
    SapContacts,
    State,
)

wp.config.enable_backward = False


@wp.func
def _newton_contact_material_override(
    tid: int,
    base_mu: wp.float64,
    base_k: wp.float64,
    base_tau: wp.float64,
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction: wp.array(dtype=wp.float32),
):
    """Map Newton's optional per-contact hydroelastic material into kSAP data.

    A positive per-contact stiffness is the version-pinned discriminator used
    by Newton 1.5 hydroelastic contacts.  Its damping is converted from a
    Kelvin-Voigt coefficient to SAP's time scale using tau=d/k, and its
    friction entry scales (rather than replaces) the shape-pair coefficient.
    """
    mu = base_mu
    stiffness = base_k
    tau = base_tau
    contact_stiffness = wp.float64(rigid_contact_stiffness[tid])
    if contact_stiffness > wp.float64(0.0):
        stiffness = contact_stiffness
        contact_damping = wp.float64(rigid_contact_damping[tid])
        if contact_damping > wp.float64(0.0):
            tau = contact_damping / contact_stiffness
        friction_scale = wp.float64(rigid_contact_friction[tid])
        if friction_scale > wp.float64(0.0):
            mu = mu * friction_scale
    return wp.vec3d(mu, stiffness, tau)


@dataclass(frozen=True)
class SapContactJacobianResult:
    """Views into buffers owned by `SapContactJacobian`."""

    contact_count: int
    truncated_contact_count: int
    contact_env_count: wp.array
    contact_env_phi0: wp.array
    contact_env_jacobian: wp.array
    contact_env_w_eff: wp.array
    contact_env_mu: wp.array
    contact_env_stiffness: wp.array
    contact_env_R_WC: wp.array
    contact_env_point: wp.array
    contact_env_witness0: wp.array
    contact_env_witness1: wp.array
    contact_env_body0: wp.array
    contact_env_body1: wp.array
    body_jacobian_local: wp.array
    dynamics_matrix_env: wp.array
    contact_env_tau_d: wp.array | None = None
    local_joint_qd_sap_input: wp.array | None = None
    local_free_motion_joint_qd_sap: wp.array | None = None
    local_dynamics_matrix_env: wp.array | None = None
    local_contact_env_phi0: wp.array | None = None
    local_contact_env_jacobian: wp.array | None = None
    local_contact_env_w_eff: wp.array | None = None
    local_contact_env_mu: wp.array | None = None
    local_contact_env_stiffness: wp.array | None = None
    local_contact_env_tau_d: wp.array | None = None
    local_contact_env_R_WC: wp.array | None = None
    local_contact_env_point: wp.array | None = None
    local_contact_env_body0: wp.array | None = None
    local_contact_env_body1: wp.array | None = None

@wp.kernel
def _assemble_body_origin_jacobians_local_sap(
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    j_flat: wp.array(dtype=wp.float64),
    body_src_row_start: wp.array(dtype=int),
    body_local_dof_start: wp.array(dtype=int),
    body_cols: wp.array(dtype=int),
):
    # Rows are SAP spatial velocity order at body origin: angular, then translational.
    body, row, col = wp.tid()
    src_start = body_src_row_start[body]
    if src_start < 0:
        body_jac_local[body, row, col] = wp.float64(0.0)
        return

    dof_start = body_local_dof_start[body]
    cols = body_cols[body]
    if col >= dof_start and col < dof_start + cols:
        local_col = col - dof_start
        body_jac_local[body, row, col] = j_flat[src_start + row * cols + local_col]
    else:
        body_jac_local[body, row, col] = wp.float64(0.0)


@wp.kernel
def _assemble_dynamics_matrix_multi_env_sap(
    a_env: wp.array(dtype=wp.float64, ndim=3),
    h_flat: wp.array(dtype=wp.float64),
    joint_armature: wp.array(dtype=wp.float64),
    articulation_h_start: wp.array(dtype=int),
    articulation_h_rows: wp.array(dtype=int),
    env_articulation_ids: wp.array(dtype=int, ndim=2),
    env_articulation_count: wp.array(dtype=int),
    articulation_local_dof_start: wp.array(dtype=int),
    max_articulations_per_env: int,
    dof_per_env: int,
):
    env, i, j = wp.tid()
    out = wp.float64(0.0)

    count = env_articulation_count[env]
    if count > max_articulations_per_env:
        count = max_articulations_per_env

    for slot in range(count):
        art = env_articulation_ids[env, slot]
        if art < 0:
            continue
        rows = articulation_h_rows[art]
        local_dof_start = articulation_local_dof_start[art]
        if (
            i >= local_dof_start
            and i < local_dof_start + rows
            and j >= local_dof_start
            and j < local_dof_start + rows
        ):
            li = i - local_dof_start
            lj = j - local_dof_start
            src = articulation_h_start[art] + li * rows + lj
            out = h_flat[src]

    if i == j:
        out = out + wp.float64(joint_armature[env * dof_per_env + i])

    a_env[env, i, j] = out


@wp.kernel
def _compute_contact_weights_diag_delassus_batched(
    max_local_contacts: int,
    dof_per_env: int,
    contact_count_env: wp.array(dtype=int),
    contact_jac_env: wp.array(dtype=wp.float64, ndim=4),
    a_mat_env: wp.array(dtype=wp.float64, ndim=3),
    contact_w_eff_env: wp.array(dtype=wp.float64, ndim=2),
):
    # Diagonal Delassus estimate: ||J * diag(A)^-1 * J^T||_F / 3.
    env, c = wp.tid()
    if c >= max_local_contacts or c >= contact_count_env[env]:
        contact_w_eff_env[env, c] = wp.float64(1.0e-12)
        return

    w00 = wp.float64(0.0)
    w01 = wp.float64(0.0)
    w02 = wp.float64(0.0)
    w11 = wp.float64(0.0)
    w12 = wp.float64(0.0)
    w22 = wp.float64(0.0)
    for j in range(dof_per_env):
        aii = a_mat_env[env, j, j]
        if aii < wp.float64(1.0e-12):
            aii = wp.float64(1.0e-12)
        inv_aii = wp.float64(1.0) / aii
        j0 = contact_jac_env[env, c, 0, j]
        j1 = contact_jac_env[env, c, 1, j]
        j2 = contact_jac_env[env, c, 2, j]
        w00 = w00 + inv_aii * j0 * j0
        w01 = w01 + inv_aii * j0 * j1
        w02 = w02 + inv_aii * j0 * j2
        w11 = w11 + inv_aii * j1 * j1
        w12 = w12 + inv_aii * j1 * j2
        w22 = w22 + inv_aii * j2 * j2

    frob = wp.sqrt(
        w00 * w00
        + w11 * w11
        + w22 * w22
        + wp.float64(2.0) * (w01 * w01 + w02 * w02 + w12 * w12)
    )
    w = frob / wp.float64(3.0)
    if w < wp.float64(1.0e-12):
        w = wp.float64(1.0e-12)
    contact_w_eff_env[env, c] = w


@wp.kernel
def _compute_contact_weights_sap_body_batched(
    max_local_contacts: int,
    contact_count_env: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformd),
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_com: wp.array(dtype=wp.vec3d),
    contact_body0: wp.array(dtype=int, ndim=2),
    contact_body1: wp.array(dtype=int, ndim=2),
    contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    contact_w_eff: wp.array(dtype=wp.float64, ndim=2),
):
    env, c = wp.tid()
    if c >= max_local_contacts or c >= contact_count_env[env]:
        return

    cx = _safe_normalize(
        wp.vec3d(contact_R_WC[env, c, 0, 0], contact_R_WC[env, c, 1, 0], contact_R_WC[env, c, 2, 0])
    )
    cy = _safe_normalize(
        wp.vec3d(contact_R_WC[env, c, 0, 1], contact_R_WC[env, c, 1, 1], contact_R_WC[env, c, 2, 1])
    )
    cz = _safe_normalize(
        wp.vec3d(contact_R_WC[env, c, 0, 2], contact_R_WC[env, c, 1, 2], contact_R_WC[env, c, 2, 2])
    )

    wt1 = wp.float64(0.0)
    wt2 = wp.float64(0.0)
    wn = wp.float64(0.0)

    b0 = contact_body0[env, c]
    if b0 >= 0:
        r0 = contact_witness0[env, c] - _body_com_world_d(b0, body_q, body_com)
        wt1 = wt1 + _sap_body_direction_weight_d(b0, r0, cx, body_q, body_mass, body_inertia)
        wt2 = wt2 + _sap_body_direction_weight_d(b0, r0, cy, body_q, body_mass, body_inertia)
        wn = wn + _sap_body_direction_weight_d(b0, r0, cz, body_q, body_mass, body_inertia)

    b1 = contact_body1[env, c]
    if b1 >= 0:
        r1 = contact_witness1[env, c] - _body_com_world_d(b1, body_q, body_com)
        wt1 = wt1 + _sap_body_direction_weight_d(b1, r1, cx, body_q, body_mass, body_inertia)
        wt2 = wt2 + _sap_body_direction_weight_d(b1, r1, cy, body_q, body_mass, body_inertia)
        wn = wn + _sap_body_direction_weight_d(b1, r1, cz, body_q, body_mass, body_inertia)

    w = (wt1 + wt2 + wn) / wp.float64(3.0)
    if w < wp.float64(1.0e-12):
        w = wp.float64(1.0e-12)
    contact_w_eff[env, c] = w


@wp.kernel
def _compute_contact_weights_sap_body_batched_f32(
    max_local_contacts: int,
    contact_count_env: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformd),
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_com: wp.array(dtype=wp.vec3d),
    contact_body0: wp.array(dtype=int, ndim=2),
    contact_body1: wp.array(dtype=int, ndim=2),
    contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    contact_w_eff: wp.array(dtype=wp.float64, ndim=2),
):
    env, c = wp.tid()
    if c >= max_local_contacts or c >= contact_count_env[env]:
        return

    cx = _safe_normalize_f32(
        wp.vec3(
            wp.float32(contact_R_WC[env, c, 0, 0]),
            wp.float32(contact_R_WC[env, c, 1, 0]),
            wp.float32(contact_R_WC[env, c, 2, 0]),
        )
    )
    cy = _safe_normalize_f32(
        wp.vec3(
            wp.float32(contact_R_WC[env, c, 0, 1]),
            wp.float32(contact_R_WC[env, c, 1, 1]),
            wp.float32(contact_R_WC[env, c, 2, 1]),
        )
    )
    cz = _safe_normalize_f32(
        wp.vec3(
            wp.float32(contact_R_WC[env, c, 0, 2]),
            wp.float32(contact_R_WC[env, c, 1, 2]),
            wp.float32(contact_R_WC[env, c, 2, 2]),
        )
    )

    wt1 = wp.float32(0.0)
    wt2 = wp.float32(0.0)
    wn = wp.float32(0.0)

    b0 = contact_body0[env, c]
    if b0 >= 0:
        w0 = contact_witness0[env, c]
        r0 = wp.vec3(wp.float32(w0.x), wp.float32(w0.y), wp.float32(w0.z)) - _body_com_world_f32(
            b0,
            body_q,
            body_com,
        )
        wt1 = wt1 + _sap_body_direction_weight_f32(b0, r0, cx, body_q, body_mass, body_inertia)
        wt2 = wt2 + _sap_body_direction_weight_f32(b0, r0, cy, body_q, body_mass, body_inertia)
        wn = wn + _sap_body_direction_weight_f32(b0, r0, cz, body_q, body_mass, body_inertia)

    b1 = contact_body1[env, c]
    if b1 >= 0:
        w1 = contact_witness1[env, c]
        r1 = wp.vec3(wp.float32(w1.x), wp.float32(w1.y), wp.float32(w1.z)) - _body_com_world_f32(
            b1,
            body_q,
            body_com,
        )
        wt1 = wt1 + _sap_body_direction_weight_f32(b1, r1, cx, body_q, body_mass, body_inertia)
        wt2 = wt2 + _sap_body_direction_weight_f32(b1, r1, cy, body_q, body_mass, body_inertia)
        wn = wn + _sap_body_direction_weight_f32(b1, r1, cz, body_q, body_mass, body_inertia)

    w = (wt1 + wt2 + wn) / wp.float32(3.0)
    if w < wp.float32(1.0e-12):
        w = wp.float32(1.0e-12)
    contact_w_eff[env, c] = wp.float64(w)


@wp.kernel
def _compute_contact_weights_sap_body_batched_f32_pose(
    max_local_contacts: int,
    contact_count_env: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_com: wp.array(dtype=wp.vec3d),
    contact_body0: wp.array(dtype=int, ndim=2),
    contact_body1: wp.array(dtype=int, ndim=2),
    contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    contact_w_eff: wp.array(dtype=wp.float64, ndim=2),
):
    env, c = wp.tid()
    if c >= max_local_contacts or c >= contact_count_env[env]:
        return

    cx = _safe_normalize_f32(
        wp.vec3(
            wp.float32(contact_R_WC[env, c, 0, 0]),
            wp.float32(contact_R_WC[env, c, 1, 0]),
            wp.float32(contact_R_WC[env, c, 2, 0]),
        )
    )
    cy = _safe_normalize_f32(
        wp.vec3(
            wp.float32(contact_R_WC[env, c, 0, 1]),
            wp.float32(contact_R_WC[env, c, 1, 1]),
            wp.float32(contact_R_WC[env, c, 2, 1]),
        )
    )
    cz = _safe_normalize_f32(
        wp.vec3(
            wp.float32(contact_R_WC[env, c, 0, 2]),
            wp.float32(contact_R_WC[env, c, 1, 2]),
            wp.float32(contact_R_WC[env, c, 2, 2]),
        )
    )

    wt1 = wp.float32(0.0)
    wt2 = wp.float32(0.0)
    wn = wp.float32(0.0)

    b0 = contact_body0[env, c]
    if b0 >= 0:
        w0 = contact_witness0[env, c]
        r0 = wp.vec3(wp.float32(w0.x), wp.float32(w0.y), wp.float32(w0.z)) - _body_com_world_f32_pose(
            b0,
            body_q,
            body_com,
        )
        wt1 = wt1 + _sap_body_direction_weight_f32_pose(b0, r0, cx, body_q, body_mass, body_inertia)
        wt2 = wt2 + _sap_body_direction_weight_f32_pose(b0, r0, cy, body_q, body_mass, body_inertia)
        wn = wn + _sap_body_direction_weight_f32_pose(b0, r0, cz, body_q, body_mass, body_inertia)

    b1 = contact_body1[env, c]
    if b1 >= 0:
        w1 = contact_witness1[env, c]
        r1 = wp.vec3(wp.float32(w1.x), wp.float32(w1.y), wp.float32(w1.z)) - _body_com_world_f32_pose(
            b1,
            body_q,
            body_com,
        )
        wt1 = wt1 + _sap_body_direction_weight_f32_pose(b1, r1, cx, body_q, body_mass, body_inertia)
        wt2 = wt2 + _sap_body_direction_weight_f32_pose(b1, r1, cy, body_q, body_mass, body_inertia)
        wn = wn + _sap_body_direction_weight_f32_pose(b1, r1, cz, body_q, body_mass, body_inertia)

    w = (wt1 + wt2 + wn) / wp.float32(3.0)
    if w < wp.float32(1.0e-12):
        w = wp.float32(1.0e-12)
    contact_w_eff[env, c] = wp.float64(w)


@wp.kernel
def _scatter_sap_contacts_to_env_direct(
    active_count: int,
    rigid_contact_count: wp.array(dtype=int),
    bodies_per_env: int,
    dof_per_env: int,
    max_local_contacts: int,
    use_witness_jacobian: int,
    shape_body: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=wp.float32),
    rigid_contact_margin1: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transformd),
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    env_contact_count: wp.array(dtype=int),
    env_contact_phi0: wp.array(dtype=wp.float64, ndim=2),
    env_contact_jac: wp.array(dtype=wp.float64, ndim=4),
    env_contact_mu: wp.array(dtype=wp.float64, ndim=2),
    env_contact_k: wp.array(dtype=wp.float64, ndim=2),
    env_contact_tau: wp.array(dtype=wp.float64, ndim=2),
    env_contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    env_contact_point: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_body0: wp.array(dtype=int, ndim=2),
    env_contact_body1: wp.array(dtype=int, ndim=2),
    contact_input_env: wp.array(dtype=int),
    contact_input_slot: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=wp.float64),
    fallback_mu: float,
    shape_material_ke: wp.array(dtype=wp.float64),
    fallback_k: float,
    shape_material_tau: wp.array(dtype=wp.float64),
    fallback_tau: float,
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= active_count:
        return

    live_count = rigid_contact_count[0]
    if live_count > active_count:
        live_count = active_count
    if tid >= live_count:
        return

    contact_input_env[tid] = -1
    contact_input_slot[tid] = -1

    sh0 = rigid_contact_shape0[tid]
    sh1 = rigid_contact_shape1[tid]

    b0 = -1
    b1 = -1
    if sh0 >= 0:
        b0 = shape_body[sh0]
    if sh1 >= 0:
        b1 = shape_body[sh1]

    env = _contact_env_from_shapes(bodies_per_env, shape_body, sh0, sh1)
    if env < 0:
        return

    slot = wp.atomic_add(env_contact_count, env, 1)
    if slot >= max_local_contacts:
        return
    contact_input_env[tid] = env
    contact_input_slot[tid] = slot

    p0 = _v3d_zero()
    p1 = _v3d_zero()
    x0 = _vec3_to_vec3d(rigid_contact_point0[tid])
    x1 = _vec3_to_vec3d(rigid_contact_point1[tid])
    if b0 >= 0:
        p0 = wp.transform_get_translation(body_q[b0])
        x0 = _transform_point_d(body_q[b0], x0)
    if b1 >= 0:
        p1 = wp.transform_get_translation(body_q[b1])
        x1 = _transform_point_d(body_q[b1], x1)

    n = _safe_normalize(_vec3_to_vec3d(rigid_contact_normal[tid]))
    cx, cy, cz = _sap_make_from_one_unit_vector_z(n)

    k0 = _shape_stiffness_or_fallback(shape_material_ke, sh0, fallback_k)
    k1 = _shape_stiffness_or_fallback(shape_material_ke, sh1, fallback_k)
    denom = k0 + k1
    w0 = wp.float64(0.5)
    w1 = wp.float64(0.5)
    is_hydro = rigid_contact_stiffness[tid] > wp.float32(0.0)
    if denom != wp.float64(0.0) and not is_hydro:
        w0 = k0 / denom
        w1 = k1 / denom
    pC = w0 * x0 + w1 * x1

    env_contact_body0[env, slot] = b0
    env_contact_body1[env, slot] = b1
    env_contact_witness0[env, slot] = x0
    env_contact_witness1[env, slot] = x1
    env_contact_point[env, slot] = pC
    env_contact_phi0[env, slot] = (
        wp.dot(x1 - x0, cz)
        - wp.float64(rigid_contact_margin0[tid])
        - wp.float64(rigid_contact_margin1[tid])
    )
    material = _newton_contact_material_override(
        tid,
        _contact_mu_pair(shape_material_mu, sh0, sh1, fallback_mu),
        _sap_combine_stiffness(k0, k1),
        _contact_tau_pair(shape_material_tau, sh0, sh1, fallback_tau),
        rigid_contact_stiffness,
        rigid_contact_damping,
        rigid_contact_friction,
    )
    env_contact_mu[env, slot] = material[0]
    env_contact_k[env, slot] = material[1]
    env_contact_tau[env, slot] = material[2]

    # R_WC is stored with world rows and contact-frame columns.
    env_contact_R_WC[env, slot, 0, 0] = cx.x
    env_contact_R_WC[env, slot, 1, 0] = cx.y
    env_contact_R_WC[env, slot, 2, 0] = cx.z
    env_contact_R_WC[env, slot, 0, 1] = cy.x
    env_contact_R_WC[env, slot, 1, 1] = cy.y
    env_contact_R_WC[env, slot, 2, 1] = cy.z
    env_contact_R_WC[env, slot, 0, 2] = cz.x
    env_contact_R_WC[env, slot, 1, 2] = cz.y
    env_contact_R_WC[env, slot, 2, 2] = cz.z

    for dof in range(dof_per_env):
        v0 = _v3d_zero()
        v1 = _v3d_zero()
        xj0 = pC
        xj1 = pC
        if use_witness_jacobian != 0:
            xj0 = x0
            xj1 = x1

        if b0 >= 0:
            wA = wp.vec3d(
                body_jac_local[b0, 0, dof],
                body_jac_local[b0, 1, dof],
                body_jac_local[b0, 2, dof],
            )
            vBoA = wp.vec3d(
                body_jac_local[b0, 3, dof],
                body_jac_local[b0, 4, dof],
                body_jac_local[b0, 5, dof],
            )
            v0 = vBoA + wp.cross(wA, xj0 - p0)
        if b1 >= 0:
            wB = wp.vec3d(
                body_jac_local[b1, 0, dof],
                body_jac_local[b1, 1, dof],
                body_jac_local[b1, 2, dof],
            )
            vBoB = wp.vec3d(
                body_jac_local[b1, 3, dof],
                body_jac_local[b1, 4, dof],
                body_jac_local[b1, 5, dof],
            )
            v1 = vBoB + wp.cross(wB, xj1 - p1)

        rel = v1 - v0
        env_contact_jac[env, slot, 0, dof] = wp.dot(cx, rel)
        env_contact_jac[env, slot, 1, dof] = wp.dot(cy, rel)
        env_contact_jac[env, slot, 2, dof] = wp.dot(cz, rel)


@wp.kernel
def _scatter_sap_contacts_to_env_direct_f32_pose(
    active_count: int,
    rigid_contact_count: wp.array(dtype=int),
    bodies_per_env: int,
    dof_per_env: int,
    max_local_contacts: int,
    use_witness_jacobian: int,
    shape_body: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=wp.float32),
    rigid_contact_margin1: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    env_contact_count: wp.array(dtype=int),
    env_contact_phi0: wp.array(dtype=wp.float64, ndim=2),
    env_contact_jac: wp.array(dtype=wp.float64, ndim=4),
    env_contact_mu: wp.array(dtype=wp.float64, ndim=2),
    env_contact_k: wp.array(dtype=wp.float64, ndim=2),
    env_contact_tau: wp.array(dtype=wp.float64, ndim=2),
    env_contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    env_contact_point: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_body0: wp.array(dtype=int, ndim=2),
    env_contact_body1: wp.array(dtype=int, ndim=2),
    contact_input_env: wp.array(dtype=int),
    contact_input_slot: wp.array(dtype=int),
    compute_sap_weight: int,
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_com: wp.array(dtype=wp.vec3d),
    env_contact_w_eff: wp.array(dtype=wp.float64, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float64),
    fallback_mu: float,
    shape_material_ke: wp.array(dtype=wp.float64),
    fallback_k: float,
    shape_material_tau: wp.array(dtype=wp.float64),
    fallback_tau: float,
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= active_count:
        return

    live_count = rigid_contact_count[0]
    if live_count > active_count:
        live_count = active_count
    if tid >= live_count:
        return

    contact_input_env[tid] = -1
    contact_input_slot[tid] = -1

    sh0 = rigid_contact_shape0[tid]
    sh1 = rigid_contact_shape1[tid]

    b0 = -1
    b1 = -1
    if sh0 >= 0:
        b0 = shape_body[sh0]
    if sh1 >= 0:
        b1 = shape_body[sh1]

    env = _contact_env_from_shapes(bodies_per_env, shape_body, sh0, sh1)
    if env < 0:
        return

    slot = wp.atomic_add(env_contact_count, env, 1)
    if slot >= max_local_contacts:
        return
    contact_input_env[tid] = env
    contact_input_slot[tid] = slot

    p0 = _v3_zero_f32()
    p1 = _v3_zero_f32()
    x0 = rigid_contact_point0[tid]
    x1 = rigid_contact_point1[tid]
    if b0 >= 0:
        p0 = wp.transform_get_translation(body_q[b0])
        x0 = _transform_point_f32_pose(body_q[b0], x0)
    if b1 >= 0:
        p1 = wp.transform_get_translation(body_q[b1])
        x1 = _transform_point_f32_pose(body_q[b1], x1)

    n = _safe_normalize_f32(rigid_contact_normal[tid])
    cx, cy, cz = _sap_make_from_one_unit_vector_z_f32(n)

    k0 = _shape_stiffness_or_fallback(shape_material_ke, sh0, fallback_k)
    k1 = _shape_stiffness_or_fallback(shape_material_ke, sh1, fallback_k)
    k0f = wp.float32(k0)
    k1f = wp.float32(k1)
    denom = k0f + k1f
    w0 = wp.float32(0.5)
    w1 = wp.float32(0.5)
    is_hydro = rigid_contact_stiffness[tid] > wp.float32(0.0)
    if denom != wp.float32(0.0) and not is_hydro:
        w0 = k0f / denom
        w1 = k1f / denom
    pC = w0 * x0 + w1 * x1

    env_contact_body0[env, slot] = b0
    env_contact_body1[env, slot] = b1
    env_contact_witness0[env, slot] = wp.vec3d(wp.float64(x0.x), wp.float64(x0.y), wp.float64(x0.z))
    env_contact_witness1[env, slot] = wp.vec3d(wp.float64(x1.x), wp.float64(x1.y), wp.float64(x1.z))
    env_contact_point[env, slot] = wp.vec3d(wp.float64(pC.x), wp.float64(pC.y), wp.float64(pC.z))
    env_contact_phi0[env, slot] = (
        wp.float64(wp.dot(x1 - x0, cz))
        - wp.float64(rigid_contact_margin0[tid])
        - wp.float64(rigid_contact_margin1[tid])
    )
    material = _newton_contact_material_override(
        tid,
        _contact_mu_pair(shape_material_mu, sh0, sh1, fallback_mu),
        _sap_combine_stiffness(k0, k1),
        _contact_tau_pair(shape_material_tau, sh0, sh1, fallback_tau),
        rigid_contact_stiffness,
        rigid_contact_damping,
        rigid_contact_friction,
    )
    env_contact_mu[env, slot] = material[0]
    env_contact_k[env, slot] = material[1]
    env_contact_tau[env, slot] = material[2]

    env_contact_R_WC[env, slot, 0, 0] = wp.float64(cx.x)
    env_contact_R_WC[env, slot, 1, 0] = wp.float64(cx.y)
    env_contact_R_WC[env, slot, 2, 0] = wp.float64(cx.z)
    env_contact_R_WC[env, slot, 0, 1] = wp.float64(cy.x)
    env_contact_R_WC[env, slot, 1, 1] = wp.float64(cy.y)
    env_contact_R_WC[env, slot, 2, 1] = wp.float64(cy.z)
    env_contact_R_WC[env, slot, 0, 2] = wp.float64(cz.x)
    env_contact_R_WC[env, slot, 1, 2] = wp.float64(cz.y)
    env_contact_R_WC[env, slot, 2, 2] = wp.float64(cz.z)

    if compute_sap_weight != 0:
        wt1 = wp.float32(0.0)
        wt2 = wp.float32(0.0)
        wn = wp.float32(0.0)
        if b0 >= 0:
            r0 = x0 - _body_com_world_f32_pose(b0, body_q, body_com)
            wt1 = wt1 + _sap_body_direction_weight_f32_pose(b0, r0, cx, body_q, body_mass, body_inertia)
            wt2 = wt2 + _sap_body_direction_weight_f32_pose(b0, r0, cy, body_q, body_mass, body_inertia)
            wn = wn + _sap_body_direction_weight_f32_pose(b0, r0, cz, body_q, body_mass, body_inertia)
        if b1 >= 0:
            r1 = x1 - _body_com_world_f32_pose(b1, body_q, body_com)
            wt1 = wt1 + _sap_body_direction_weight_f32_pose(b1, r1, cx, body_q, body_mass, body_inertia)
            wt2 = wt2 + _sap_body_direction_weight_f32_pose(b1, r1, cy, body_q, body_mass, body_inertia)
            wn = wn + _sap_body_direction_weight_f32_pose(b1, r1, cz, body_q, body_mass, body_inertia)
        w_eff = (wt1 + wt2 + wn) / wp.float32(3.0)
        if w_eff < wp.float32(1.0e-12):
            w_eff = wp.float32(1.0e-12)
        env_contact_w_eff[env, slot] = wp.float64(w_eff)

    for dof in range(dof_per_env):
        v0 = _v3_zero_f32()
        v1 = _v3_zero_f32()
        xj0 = pC
        xj1 = pC
        if use_witness_jacobian != 0:
            xj0 = x0
            xj1 = x1

        if b0 >= 0:
            wA = wp.vec3(
                wp.float32(body_jac_local[b0, 0, dof]),
                wp.float32(body_jac_local[b0, 1, dof]),
                wp.float32(body_jac_local[b0, 2, dof]),
            )
            vBoA = wp.vec3(
                wp.float32(body_jac_local[b0, 3, dof]),
                wp.float32(body_jac_local[b0, 4, dof]),
                wp.float32(body_jac_local[b0, 5, dof]),
            )
            v0 = vBoA + wp.cross(wA, xj0 - p0)
        if b1 >= 0:
            wB = wp.vec3(
                wp.float32(body_jac_local[b1, 0, dof]),
                wp.float32(body_jac_local[b1, 1, dof]),
                wp.float32(body_jac_local[b1, 2, dof]),
            )
            vBoB = wp.vec3(
                wp.float32(body_jac_local[b1, 3, dof]),
                wp.float32(body_jac_local[b1, 4, dof]),
                wp.float32(body_jac_local[b1, 5, dof]),
            )
            v1 = vBoB + wp.cross(wB, xj1 - p1)

        rel = v1 - v0
        env_contact_jac[env, slot, 0, dof] = wp.float64(wp.dot(cx, rel))
        env_contact_jac[env, slot, 1, dof] = wp.float64(wp.dot(cy, rel))
        env_contact_jac[env, slot, 2, dof] = wp.float64(wp.dot(cz, rel))


@wp.kernel
def _scatter_sap_contacts_to_env_direct_f64(
    active_count: int,
    rigid_contact_count: wp.array(dtype=int),
    bodies_per_env: int,
    dof_per_env: int,
    max_local_contacts: int,
    use_witness_jacobian: int,
    shape_body: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3d),
    rigid_contact_point1: wp.array(dtype=wp.vec3d),
    rigid_contact_normal: wp.array(dtype=wp.vec3d),
    rigid_contact_margin0: wp.array(dtype=wp.float64),
    rigid_contact_margin1: wp.array(dtype=wp.float64),
    body_q: wp.array(dtype=wp.transformd),
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    env_contact_count: wp.array(dtype=int),
    env_contact_phi0: wp.array(dtype=wp.float64, ndim=2),
    env_contact_jac: wp.array(dtype=wp.float64, ndim=4),
    env_contact_mu: wp.array(dtype=wp.float64, ndim=2),
    env_contact_k: wp.array(dtype=wp.float64, ndim=2),
    env_contact_tau: wp.array(dtype=wp.float64, ndim=2),
    env_contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    env_contact_point: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_body0: wp.array(dtype=int, ndim=2),
    env_contact_body1: wp.array(dtype=int, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float64),
    fallback_mu: float,
    shape_material_ke: wp.array(dtype=wp.float64),
    fallback_k: float,
    shape_material_tau: wp.array(dtype=wp.float64),
    fallback_tau: float,
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= active_count:
        return

    live_count = rigid_contact_count[0]
    if live_count > active_count:
        live_count = active_count
    if tid >= live_count:
        return

    sh0 = rigid_contact_shape0[tid]
    sh1 = rigid_contact_shape1[tid]

    b0 = -1
    b1 = -1
    if sh0 >= 0:
        b0 = shape_body[sh0]
    if sh1 >= 0:
        b1 = shape_body[sh1]

    env = _contact_env_from_shapes(bodies_per_env, shape_body, sh0, sh1)
    if env < 0:
        return

    slot = wp.atomic_add(env_contact_count, env, 1)
    if slot >= max_local_contacts:
        return

    p0 = _v3d_zero()
    p1 = _v3d_zero()
    x0 = rigid_contact_point0[tid]
    x1 = rigid_contact_point1[tid]
    if b0 >= 0:
        p0 = wp.transform_get_translation(body_q[b0])
        x0 = _transform_point_d(body_q[b0], x0)
    if b1 >= 0:
        p1 = wp.transform_get_translation(body_q[b1])
        x1 = _transform_point_d(body_q[b1], x1)

    n = _safe_normalize(rigid_contact_normal[tid])
    cx, cy, cz = _sap_make_from_one_unit_vector_z(n)

    k0 = _shape_stiffness_or_fallback(shape_material_ke, sh0, fallback_k)
    k1 = _shape_stiffness_or_fallback(shape_material_ke, sh1, fallback_k)
    denom = k0 + k1
    w0 = wp.float64(0.5)
    w1 = wp.float64(0.5)
    is_hydro = rigid_contact_stiffness[tid] > wp.float32(0.0)
    if denom != wp.float64(0.0) and not is_hydro:
        w0 = k0 / denom
        w1 = k1 / denom
    pC = w0 * x0 + w1 * x1

    env_contact_body0[env, slot] = b0
    env_contact_body1[env, slot] = b1
    env_contact_witness0[env, slot] = x0
    env_contact_witness1[env, slot] = x1
    env_contact_point[env, slot] = pC
    env_contact_phi0[env, slot] = wp.dot(x1 - x0, cz) - rigid_contact_margin0[tid] - rigid_contact_margin1[tid]
    material = _newton_contact_material_override(
        tid,
        _contact_mu_pair(shape_material_mu, sh0, sh1, fallback_mu),
        _sap_combine_stiffness(k0, k1),
        _contact_tau_pair(shape_material_tau, sh0, sh1, fallback_tau),
        rigid_contact_stiffness,
        rigid_contact_damping,
        rigid_contact_friction,
    )
    env_contact_mu[env, slot] = material[0]
    env_contact_k[env, slot] = material[1]
    env_contact_tau[env, slot] = material[2]

    env_contact_R_WC[env, slot, 0, 0] = cx.x
    env_contact_R_WC[env, slot, 1, 0] = cx.y
    env_contact_R_WC[env, slot, 2, 0] = cx.z
    env_contact_R_WC[env, slot, 0, 1] = cy.x
    env_contact_R_WC[env, slot, 1, 1] = cy.y
    env_contact_R_WC[env, slot, 2, 1] = cy.z
    env_contact_R_WC[env, slot, 0, 2] = cz.x
    env_contact_R_WC[env, slot, 1, 2] = cz.y
    env_contact_R_WC[env, slot, 2, 2] = cz.z

    for dof in range(dof_per_env):
        v0 = _v3d_zero()
        v1 = _v3d_zero()
        xj0 = pC
        xj1 = pC
        if use_witness_jacobian != 0:
            xj0 = x0
            xj1 = x1

        if b0 >= 0:
            wA = wp.vec3d(
                body_jac_local[b0, 0, dof],
                body_jac_local[b0, 1, dof],
                body_jac_local[b0, 2, dof],
            )
            vBoA = wp.vec3d(
                body_jac_local[b0, 3, dof],
                body_jac_local[b0, 4, dof],
                body_jac_local[b0, 5, dof],
            )
            v0 = vBoA + wp.cross(wA, xj0 - p0)
        if b1 >= 0:
            wB = wp.vec3d(
                body_jac_local[b1, 0, dof],
                body_jac_local[b1, 1, dof],
                body_jac_local[b1, 2, dof],
            )
            vBoB = wp.vec3d(
                body_jac_local[b1, 3, dof],
                body_jac_local[b1, 4, dof],
                body_jac_local[b1, 5, dof],
            )
            v1 = vBoB + wp.cross(wB, xj1 - p1)

        rel = v1 - v0
        env_contact_jac[env, slot, 0, dof] = wp.dot(cx, rel)
        env_contact_jac[env, slot, 1, dof] = wp.dot(cy, rel)
        env_contact_jac[env, slot, 2, dof] = wp.dot(cz, rel)


@wp.kernel
def _scatter_sap_contacts_to_env_direct_f64_f32_pose(
    active_count: int,
    rigid_contact_count: wp.array(dtype=int),
    bodies_per_env: int,
    dof_per_env: int,
    max_local_contacts: int,
    use_witness_jacobian: int,
    shape_body: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3d),
    rigid_contact_point1: wp.array(dtype=wp.vec3d),
    rigid_contact_normal: wp.array(dtype=wp.vec3d),
    rigid_contact_margin0: wp.array(dtype=wp.float64),
    rigid_contact_margin1: wp.array(dtype=wp.float64),
    body_q: wp.array(dtype=wp.transform),
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    env_contact_count: wp.array(dtype=int),
    env_contact_phi0: wp.array(dtype=wp.float64, ndim=2),
    env_contact_jac: wp.array(dtype=wp.float64, ndim=4),
    env_contact_mu: wp.array(dtype=wp.float64, ndim=2),
    env_contact_k: wp.array(dtype=wp.float64, ndim=2),
    env_contact_tau: wp.array(dtype=wp.float64, ndim=2),
    env_contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    env_contact_point: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_body0: wp.array(dtype=int, ndim=2),
    env_contact_body1: wp.array(dtype=int, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float64),
    fallback_mu: float,
    shape_material_ke: wp.array(dtype=wp.float64),
    fallback_k: float,
    shape_material_tau: wp.array(dtype=wp.float64),
    fallback_tau: float,
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= active_count:
        return

    live_count = rigid_contact_count[0]
    if live_count > active_count:
        live_count = active_count
    if tid >= live_count:
        return

    sh0 = rigid_contact_shape0[tid]
    sh1 = rigid_contact_shape1[tid]

    b0 = -1
    b1 = -1
    if sh0 >= 0:
        b0 = shape_body[sh0]
    if sh1 >= 0:
        b1 = shape_body[sh1]

    env = _contact_env_from_shapes(bodies_per_env, shape_body, sh0, sh1)
    if env < 0:
        return

    slot = wp.atomic_add(env_contact_count, env, 1)
    if slot >= max_local_contacts:
        return

    p0 = _v3d_zero()
    p1 = _v3d_zero()
    x0 = rigid_contact_point0[tid]
    x1 = rigid_contact_point1[tid]
    if b0 >= 0:
        p0 = _transform_translation_f32_pose_d(body_q[b0])
        x0 = _transform_point_f32_pose_d(body_q[b0], x0)
    if b1 >= 0:
        p1 = _transform_translation_f32_pose_d(body_q[b1])
        x1 = _transform_point_f32_pose_d(body_q[b1], x1)

    n = _safe_normalize(rigid_contact_normal[tid])
    cx, cy, cz = _sap_make_from_one_unit_vector_z(n)

    k0 = _shape_stiffness_or_fallback(shape_material_ke, sh0, fallback_k)
    k1 = _shape_stiffness_or_fallback(shape_material_ke, sh1, fallback_k)
    denom = k0 + k1
    w0 = wp.float64(0.5)
    w1 = wp.float64(0.5)
    is_hydro = rigid_contact_stiffness[tid] > wp.float32(0.0)
    if denom != wp.float64(0.0) and not is_hydro:
        w0 = k0 / denom
        w1 = k1 / denom
    pC = w0 * x0 + w1 * x1

    env_contact_body0[env, slot] = b0
    env_contact_body1[env, slot] = b1
    env_contact_witness0[env, slot] = x0
    env_contact_witness1[env, slot] = x1
    env_contact_point[env, slot] = pC
    env_contact_phi0[env, slot] = wp.dot(x1 - x0, cz) - rigid_contact_margin0[tid] - rigid_contact_margin1[tid]
    material = _newton_contact_material_override(
        tid,
        _contact_mu_pair(shape_material_mu, sh0, sh1, fallback_mu),
        _sap_combine_stiffness(k0, k1),
        _contact_tau_pair(shape_material_tau, sh0, sh1, fallback_tau),
        rigid_contact_stiffness,
        rigid_contact_damping,
        rigid_contact_friction,
    )
    env_contact_mu[env, slot] = material[0]
    env_contact_k[env, slot] = material[1]
    env_contact_tau[env, slot] = material[2]

    env_contact_R_WC[env, slot, 0, 0] = cx.x
    env_contact_R_WC[env, slot, 1, 0] = cx.y
    env_contact_R_WC[env, slot, 2, 0] = cx.z
    env_contact_R_WC[env, slot, 0, 1] = cy.x
    env_contact_R_WC[env, slot, 1, 1] = cy.y
    env_contact_R_WC[env, slot, 2, 1] = cy.z
    env_contact_R_WC[env, slot, 0, 2] = cz.x
    env_contact_R_WC[env, slot, 1, 2] = cz.y
    env_contact_R_WC[env, slot, 2, 2] = cz.z

    for dof in range(dof_per_env):
        v0 = _v3d_zero()
        v1 = _v3d_zero()
        xj0 = pC
        xj1 = pC
        if use_witness_jacobian != 0:
            xj0 = x0
            xj1 = x1

        if b0 >= 0:
            wA = wp.vec3d(
                body_jac_local[b0, 0, dof],
                body_jac_local[b0, 1, dof],
                body_jac_local[b0, 2, dof],
            )
            vBoA = wp.vec3d(
                body_jac_local[b0, 3, dof],
                body_jac_local[b0, 4, dof],
                body_jac_local[b0, 5, dof],
            )
            v0 = vBoA + wp.cross(wA, xj0 - p0)
        if b1 >= 0:
            wB = wp.vec3d(
                body_jac_local[b1, 0, dof],
                body_jac_local[b1, 1, dof],
                body_jac_local[b1, 2, dof],
            )
            vBoB = wp.vec3d(
                body_jac_local[b1, 3, dof],
                body_jac_local[b1, 4, dof],
                body_jac_local[b1, 5, dof],
            )
            v1 = vBoB + wp.cross(wB, xj1 - p1)

        rel = v1 - v0
        env_contact_jac[env, slot, 0, dof] = wp.dot(cx, rel)
        env_contact_jac[env, slot, 1, dof] = wp.dot(cy, rel)
        env_contact_jac[env, slot, 2, dof] = wp.dot(cz, rel)


@wp.kernel
def _scatter_sap_contact_jacobian_from_slots(
    active_count: int,
    rigid_contact_count: wp.array(dtype=int),
    dof_per_env: int,
    max_local_contacts: int,
    use_witness_jacobian: int,
    contact_input_env: wp.array(dtype=int),
    contact_input_slot: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformd),
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    env_contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    env_contact_point: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_body0: wp.array(dtype=int, ndim=2),
    env_contact_body1: wp.array(dtype=int, ndim=2),
    env_contact_jac: wp.array(dtype=wp.float64, ndim=4),
):
    contact, dof = wp.tid()
    if contact >= active_count or dof >= dof_per_env:
        return
    live_count = rigid_contact_count[0]
    if live_count > active_count:
        live_count = active_count
    if contact >= live_count:
        return

    env = contact_input_env[contact]
    slot = contact_input_slot[contact]
    if env < 0 or slot < 0 or slot >= max_local_contacts:
        return

    pC = env_contact_point[env, slot]
    xj0 = pC
    xj1 = pC
    if use_witness_jacobian != 0:
        xj0 = env_contact_witness0[env, slot]
        xj1 = env_contact_witness1[env, slot]

    b0 = env_contact_body0[env, slot]
    b1 = env_contact_body1[env, slot]
    v0 = _v3d_zero()
    v1 = _v3d_zero()
    if b0 >= 0:
        p0 = wp.transform_get_translation(body_q[b0])
        wA = wp.vec3d(
            body_jac_local[b0, 0, dof],
            body_jac_local[b0, 1, dof],
            body_jac_local[b0, 2, dof],
        )
        vBoA = wp.vec3d(
            body_jac_local[b0, 3, dof],
            body_jac_local[b0, 4, dof],
            body_jac_local[b0, 5, dof],
        )
        v0 = vBoA + wp.cross(wA, xj0 - p0)
    if b1 >= 0:
        p1 = wp.transform_get_translation(body_q[b1])
        wB = wp.vec3d(
            body_jac_local[b1, 0, dof],
            body_jac_local[b1, 1, dof],
            body_jac_local[b1, 2, dof],
        )
        vBoB = wp.vec3d(
            body_jac_local[b1, 3, dof],
            body_jac_local[b1, 4, dof],
            body_jac_local[b1, 5, dof],
        )
        v1 = vBoB + wp.cross(wB, xj1 - p1)

    cx = wp.vec3d(
        env_contact_R_WC[env, slot, 0, 0],
        env_contact_R_WC[env, slot, 1, 0],
        env_contact_R_WC[env, slot, 2, 0],
    )
    cy = wp.vec3d(
        env_contact_R_WC[env, slot, 0, 1],
        env_contact_R_WC[env, slot, 1, 1],
        env_contact_R_WC[env, slot, 2, 1],
    )
    cz = wp.vec3d(
        env_contact_R_WC[env, slot, 0, 2],
        env_contact_R_WC[env, slot, 1, 2],
        env_contact_R_WC[env, slot, 2, 2],
    )
    rel = v1 - v0
    env_contact_jac[env, slot, 0, dof] = wp.dot(cx, rel)
    env_contact_jac[env, slot, 1, dof] = wp.dot(cy, rel)
    env_contact_jac[env, slot, 2, dof] = wp.dot(cz, rel)


@wp.kernel
def _scatter_sap_contact_jacobian_from_slots_f32_pose(
    active_count: int,
    rigid_contact_count: wp.array(dtype=int),
    dof_per_env: int,
    max_local_contacts: int,
    use_witness_jacobian: int,
    contact_input_env: wp.array(dtype=int),
    contact_input_slot: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_jac_local: wp.array(dtype=wp.float64, ndim=3),
    env_contact_R_WC: wp.array(dtype=wp.float64, ndim=4),
    env_contact_point: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness0: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_witness1: wp.array(dtype=wp.vec3d, ndim=2),
    env_contact_body0: wp.array(dtype=int, ndim=2),
    env_contact_body1: wp.array(dtype=int, ndim=2),
    env_contact_jac: wp.array(dtype=wp.float64, ndim=4),
):
    contact, dof = wp.tid()
    if contact >= active_count or dof >= dof_per_env:
        return
    live_count = rigid_contact_count[0]
    if live_count > active_count:
        live_count = active_count
    if contact >= live_count:
        return

    env = contact_input_env[contact]
    slot = contact_input_slot[contact]
    if env < 0 or slot < 0 or slot >= max_local_contacts:
        return

    pC = env_contact_point[env, slot]
    xj0 = pC
    xj1 = pC
    if use_witness_jacobian != 0:
        xj0 = env_contact_witness0[env, slot]
        xj1 = env_contact_witness1[env, slot]

    b0 = env_contact_body0[env, slot]
    b1 = env_contact_body1[env, slot]
    v0 = _v3d_zero()
    v1 = _v3d_zero()
    if b0 >= 0:
        p0 = _transform_translation_f32_pose_d(body_q[b0])
        wA = wp.vec3d(
            body_jac_local[b0, 0, dof],
            body_jac_local[b0, 1, dof],
            body_jac_local[b0, 2, dof],
        )
        vBoA = wp.vec3d(
            body_jac_local[b0, 3, dof],
            body_jac_local[b0, 4, dof],
            body_jac_local[b0, 5, dof],
        )
        v0 = vBoA + wp.cross(wA, xj0 - p0)
    if b1 >= 0:
        p1 = _transform_translation_f32_pose_d(body_q[b1])
        wB = wp.vec3d(
            body_jac_local[b1, 0, dof],
            body_jac_local[b1, 1, dof],
            body_jac_local[b1, 2, dof],
        )
        vBoB = wp.vec3d(
            body_jac_local[b1, 3, dof],
            body_jac_local[b1, 4, dof],
            body_jac_local[b1, 5, dof],
        )
        v1 = vBoB + wp.cross(wB, xj1 - p1)

    cx = wp.vec3d(
        env_contact_R_WC[env, slot, 0, 0],
        env_contact_R_WC[env, slot, 1, 0],
        env_contact_R_WC[env, slot, 2, 0],
    )
    cy = wp.vec3d(
        env_contact_R_WC[env, slot, 0, 1],
        env_contact_R_WC[env, slot, 1, 1],
        env_contact_R_WC[env, slot, 2, 1],
    )
    cz = wp.vec3d(
        env_contact_R_WC[env, slot, 0, 2],
        env_contact_R_WC[env, slot, 1, 2],
        env_contact_R_WC[env, slot, 2, 2],
    )
    rel = v1 - v0
    env_contact_jac[env, slot, 0, dof] = wp.dot(cx, rel)
    env_contact_jac[env, slot, 1, dof] = wp.dot(cy, rel)
    env_contact_jac[env, slot, 2, dof] = wp.dot(cz, rel)


class SapContactJacobian:
    """Standalone Warp implementation of SAP-style contact preparation.

    Runtime input is SAP-native contact storage.  Internal quantities are
    SAP-style: body-origin spatial velocity Jacobians and FREE velocity order
    `[w, v_body_origin]`.
    """

    def __init__(
        self,
        model: Model,
        *,
        max_rigid_contact: int = 100,
        fallback_mu: float | None = None,
        fallback_stiffness: float = 1.0e10,
        fallback_tau_d: float = 0.1,
        contact_weight_mode: str = "diag_delassus",
        contact_point_mode: str = "contact_midpoint",
        capture_local_snapshots: bool = True,
        use_f64_boundary_pose: bool = True,
        free_motion_solve_precision: str = "fp64",
        sap_contact_weight_precision: str = "fp64",
    ):
        if not isinstance(model, Model):
            raise TypeError("SapContactJacobian requires SapModel; convert in the frontend adapter before construction.")
        if int(model.joint_count) <= 0 or int(model.joint_dof_count) <= 0:
            raise ValueError("SapContactJacobian requires articulated joint DOFs.")

        self.model = model
        self.device = model.device
        self.free_motion = SapFreeMotion(
            model,
            allocate_dynamics_matrix=False,
            use_f64_boundary_pose=use_f64_boundary_pose,
            linear_solve_precision=free_motion_solve_precision,
        )

        self.dof_count = int(model.joint_dof_count)
        self.body_count = int(model.body_count)
        self.num_envs = int(getattr(model, "world_count", 1))
        if (
            self.num_envs <= 0
            or self.dof_count % self.num_envs != 0
            or self.body_count % self.num_envs != 0
        ):
            raise ValueError(
                "SapContactJacobian requires equal-sized contiguous world blocks "
                f"(num_envs={self.num_envs}, dof_count={self.dof_count}, body_count={self.body_count})."
            )
        self.dof_per_env = self.dof_count // self.num_envs
        self.bodies_per_env = self.body_count // self.num_envs
        self.max_rigid_contact = int(max_rigid_contact)
        if self.max_rigid_contact <= 0:
            raise ValueError("max_rigid_contact must be positive.")
        self.max_contacts = self.max_rigid_contact * self.num_envs

        self.fallback_mu = _infer_contact_mu_fallback(model) if fallback_mu is None else float(fallback_mu)
        self.fallback_stiffness = float(fallback_stiffness)
        self.fallback_tau_d = float(fallback_tau_d)
        self.contact_weight_mode = self._normalize_contact_weight_mode(contact_weight_mode)
        self.sap_contact_weight_precision = self._normalize_precision(
            sap_contact_weight_precision,
            option_name="sap_contact_weight_precision",
        )
        self.contact_point_mode = self._normalize_contact_point_mode(contact_point_mode)
        self.capture_local_snapshots = bool(capture_local_snapshots)
        self.contact_shape_mu = self._resolve_shape_material_f64(
            model,
            exact_name="sap_debug_shape_material_mu",
            fallback_array=_resolve_contact_shape_mu(model, self.fallback_mu),
        )
        self.contact_shape_ke = self._resolve_shape_material_f64(
            model,
            exact_name="sap_debug_shape_material_ke",
            fallback_array=_resolve_contact_shape_stiffness(model, self.fallback_stiffness),
        )
        shape_count = int(np.asarray(model.shape_body.numpy()).reshape(-1).size)
        self.contact_shape_tau = self._resolve_shape_tau_material_f64(model, shape_count)

        self._setup_mappings()
        self._allocate_buffers()

    @staticmethod
    def _normalize_contact_weight_mode(contact_weight_mode: str) -> str:
        mode = str(contact_weight_mode).strip().lower().replace("-", "_")
        if mode == "diag_delassus":
            return "diag_delassus"
        if mode == "body_inertia":
            return "body_inertia"
        raise ValueError(
            "contact_weight_mode must be 'diag_delassus' or 'body_inertia', "
            f"got {contact_weight_mode!r}."
        )

    @staticmethod
    def _normalize_precision(value: str, *, option_name: str) -> str:
        precision = str(value).strip().lower()
        if precision == "f32":
            precision = "fp32"
        elif precision == "f64":
            precision = "fp64"
        if precision not in {"fp32", "fp64"}:
            raise ValueError(f"{option_name} must be 'fp32'/'f32' or 'fp64'/'f64', got {value!r}.")
        return precision

    @staticmethod
    def _normalize_contact_point_mode(contact_point_mode: str) -> str:
        mode = str(contact_point_mode).strip().lower().replace("-", "_")
        if mode == "contact_midpoint":
            return "contact_midpoint"
        if mode == "witness_point":
            return "witness_point"
        raise ValueError(
            "contact_point_mode must be 'contact_midpoint' or 'witness_point', "
            f"got {contact_point_mode!r}."
        )

    @staticmethod
    def _resolve_shape_material_f64(model: Model, *, exact_name: str, fallback_array: wp.array) -> wp.array:
        src = getattr(model, exact_name, None)
        if src is None:
            src = fallback_array
        if isinstance(src, wp.array):
            values = np.asarray(src.numpy(), dtype=np.float64)
            device = src.device
        else:
            values = np.asarray(src, dtype=np.float64)
            device = getattr(model, "device", None)
        return wp.array(values.reshape(-1), dtype=wp.float64, device=device)

    def _resolve_shape_tau_material_f64(self, model: Model, shape_count: int) -> wp.array:
        values = np.full((shape_count,), np.nan, dtype=np.float64)
        src = getattr(model, "sap_debug_shape_material_tau", None)
        if src is None:
            src = getattr(model, "shape_material_tau", None)
        if src is not None:
            if isinstance(src, wp.array):
                src_values = np.asarray(src.numpy(), dtype=np.float64).reshape(-1)
            else:
                src_values = np.asarray(src, dtype=np.float64).reshape(-1)
            count = min(shape_count, int(src_values.size))
            if count > 0:
                values[:count] = src_values[:count]

        explicit = np.isfinite(values) & (values >= 0.0)
        self._contact_shape_tau_explicit_mask_np = explicit
        self._contact_shape_tau_explicit_values_np = np.where(explicit, values, np.nan)

        material = values.copy()
        material[~explicit] = self.fallback_tau_d
        return wp.array(material, dtype=wp.float64, device=self.device)

    def set_fallback_tau_d(self, fallback_tau_d: float) -> None:
        """Set the fallback contact dissipation time scale used when contact material data does not provide
        tau.
        """
        fallback_tau_d = float(fallback_tau_d)
        if fallback_tau_d == self.fallback_tau_d:
            return

        self.fallback_tau_d = fallback_tau_d
        explicit = self._contact_shape_tau_explicit_mask_np
        if bool(np.all(explicit)):
            return

        if not bool(np.any(explicit)):
            self.contact_shape_tau.fill_(self.fallback_tau_d)
            return

        material = np.array(self._contact_shape_tau_explicit_values_np, copy=True)
        material[~explicit] = self.fallback_tau_d
        wp.copy(self.contact_shape_tau, wp.array(material, dtype=wp.float64, device=self.device))

    def _compute_contact_weights(self, contact_capacity: int, body_q: wp.array) -> None:
        if self.contact_weight_mode == "body_inertia":
            if getattr(body_q, "dtype", None) == wp.transform:
                kernel = _compute_contact_weights_sap_body_batched_f32_pose
            else:
                kernel = (
                    _compute_contact_weights_sap_body_batched_f32
                    if self.sap_contact_weight_precision == "fp32"
                    else _compute_contact_weights_sap_body_batched
                )
            wp.launch(
                kernel,
                dim=(self.num_envs, contact_capacity),
                inputs=[
                    int(contact_capacity),
                    self.contact_env_count_wp,
                    body_q,
                    self.free_motion.model_body_mass,
                    self.free_motion.model_body_inertia,
                    self.free_motion.model_body_com,
                    self.contact_env_body0_wp,
                    self.contact_env_body1_wp,
                    self.contact_env_witness0_wp,
                    self.contact_env_witness1_wp,
                    self.contact_env_R_WC_wp,
                    self.contact_env_w_eff_wp,
                ],
                device=self.device,
            )
            return

        wp.launch(
            _compute_contact_weights_diag_delassus_batched,
            dim=(self.num_envs, contact_capacity),
            inputs=[
                int(contact_capacity),
                int(self.dof_per_env),
                self.contact_env_count_wp,
                self.contact_env_jac_wp,
                self.a_env_wp,
                self.contact_env_w_eff_wp,
            ],
            device=self.device,
        )

    @staticmethod
    def _total_rigid_contact_count(contacts) -> int:
        if contacts is None or contacts.rigid_contact_count is None:
            return 0
        count_np = np.asarray(contacts.rigid_contact_count.numpy(), dtype=np.int64).reshape(-1)
        if count_np.size == 0:
            return 0
        return int(np.maximum(count_np, 0).sum())

    def _setup_mappings(self) -> None:
        body_src_row_start_np = np.full(self.body_count, -1, dtype=np.int32)
        body_dof_start_np = np.zeros(self.body_count, dtype=np.int32)
        body_cols_np = np.zeros(self.body_count, dtype=np.int32)

        articulation_start = self.model.articulation_start.numpy()
        articulation_j_start = self.free_motion.articulation_J_start.numpy()
        articulation_j_cols = self.free_motion.articulation_J_cols.numpy()
        articulation_dof_start = self.free_motion.articulation_dof_start.numpy()
        joint_child = self.model.joint_child.numpy()

        for art_idx in range(int(self.model.articulation_count)):
            first_joint = int(articulation_start[art_idx])
            last_joint = int(articulation_start[art_idx + 1])
            cols = int(articulation_j_cols[art_idx])
            dof_start = int(articulation_dof_start[art_idx])
            j_start = int(articulation_j_start[art_idx])

            for local_joint_idx in range(last_joint - first_joint):
                joint_idx = first_joint + local_joint_idx
                body = int(joint_child[joint_idx])
                body_src_row_start_np[body] = j_start + 6 * local_joint_idx * cols
                body_dof_start_np[body] = dof_start
                body_cols_np[body] = cols

        body_local_dof_start_np = body_dof_start_np.copy()
        for body in range(self.body_count):
            env = body // self.bodies_per_env
            body_local_dof_start_np[body] -= env * self.dof_per_env

        self.body_src_row_start = wp.array(body_src_row_start_np, dtype=int, device=self.device)
        self.body_local_dof_start = wp.array(body_local_dof_start_np, dtype=int, device=self.device)
        self.body_cols = wp.array(body_cols_np, dtype=int, device=self.device)

        art_count = int(self.model.articulation_count)
        art_local_dof_start_np = np.zeros(art_count, dtype=np.int32)
        env_lists: list[list[int]] = [[] for _ in range(self.num_envs)]
        for art_idx in range(art_count):
            dof_start = int(articulation_dof_start[art_idx])
            env = min(max(dof_start // self.dof_per_env, 0), self.num_envs - 1)
            art_local_dof_start_np[art_idx] = dof_start - env * self.dof_per_env
            env_lists[env].append(art_idx)

        max_arts = max((len(v) for v in env_lists), default=1)
        max_arts = max(max_arts, 1)
        env_articulation_ids_np = np.full((self.num_envs, max_arts), -1, dtype=np.int32)
        env_articulation_count_np = np.zeros((self.num_envs,), dtype=np.int32)
        for env, items in enumerate(env_lists):
            env_articulation_count_np[env] = len(items)
            for slot, art_idx in enumerate(items):
                env_articulation_ids_np[env, slot] = art_idx

        self.max_articulations_per_env = int(max_arts)
        self.env_articulation_ids = wp.array(env_articulation_ids_np, dtype=int, device=self.device)
        self.env_articulation_count = wp.array(env_articulation_count_np, dtype=int, device=self.device)
        self.articulation_local_dof_start = wp.array(art_local_dof_start_np, dtype=int, device=self.device)

    def _allocate_buffers(self) -> None:
        self.body_jac_local_wp = wp.zeros(
            (self.body_count, 6, self.dof_per_env),
            dtype=wp.float64,
            device=self.device,
        )
        self.a_env_wp = wp.zeros(
            (self.num_envs, self.dof_per_env, self.dof_per_env),
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_count_wp = wp.zeros((self.num_envs,), dtype=int, device=self.device)
        self.contact_env_phi0_wp = wp.zeros(
            (self.num_envs, self.max_rigid_contact),
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_jac_wp = wp.zeros(
            (self.num_envs, self.max_rigid_contact, 3, self.dof_per_env),
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_w_eff_wp = wp.full(
            (self.num_envs, self.max_rigid_contact),
            1.0e-12,
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_mu_wp = wp.full(
            (self.num_envs, self.max_rigid_contact),
            self.fallback_mu,
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_k_wp = wp.full(
            (self.num_envs, self.max_rigid_contact),
            self.fallback_stiffness,
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_tau_wp = wp.full(
            (self.num_envs, self.max_rigid_contact),
            self.fallback_tau_d + self.fallback_tau_d,
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_R_WC_wp = wp.zeros(
            (self.num_envs, self.max_rigid_contact, 3, 3),
            dtype=wp.float64,
            device=self.device,
        )
        self.contact_env_point_wp = wp.zeros(
            (self.num_envs, self.max_rigid_contact),
            dtype=wp.vec3d,
            device=self.device,
        )
        self.contact_env_witness0_wp = wp.zeros_like(self.contact_env_point_wp)
        self.contact_env_witness1_wp = wp.zeros_like(self.contact_env_point_wp)
        self.contact_env_body0_wp = wp.full(
            (self.num_envs, self.max_rigid_contact),
            -1,
            dtype=int,
            device=self.device,
        )
        self.contact_env_body1_wp = wp.full_like(self.contact_env_body0_wp, -1)
        self.contact_input_env_wp = wp.full((self.max_contacts,), -1, dtype=int, device=self.device)
        self.contact_input_slot_wp = wp.full_like(self.contact_input_env_wp, -1)
        self._empty_rigid_contact_count = wp.zeros((1,), dtype=int, device=self.device)
        self._empty_rigid_contact_shape = wp.full((self.max_contacts,), -1, dtype=int, device=self.device)
        self._empty_rigid_contact_point = wp.zeros((self.max_contacts,), dtype=wp.vec3, device=self.device)
        self._empty_rigid_contact_normal = wp.zeros((self.max_contacts,), dtype=wp.vec3, device=self.device)
        self._empty_rigid_contact_margin = wp.zeros((self.max_contacts,), dtype=wp.float32, device=self.device)
        self._empty_rigid_contact_material = wp.zeros(
            (self.max_contacts,), dtype=wp.float32, device=self.device
        )

        self.local_joint_qd_sap_input_wp = None
        self.local_free_motion_joint_qd_sap_wp = None
        self.local_a_env_wp = None
        self.local_contact_env_phi0_wp = None
        self.local_contact_env_jac_wp = None
        self.local_contact_env_w_eff_wp = None
        self.local_contact_env_mu_wp = None
        self.local_contact_env_k_wp = None
        self.local_contact_env_tau_wp = None
        self.local_contact_env_R_WC_wp = None
        self.local_contact_env_point_wp = None
        self.local_contact_env_body0_wp = None
        self.local_contact_env_body1_wp = None
        if self.capture_local_snapshots:
            self.local_joint_qd_sap_input_wp = wp.zeros_like(self.free_motion.joint_qd_sap_input)
            self.local_free_motion_joint_qd_sap_wp = wp.zeros_like(
                self.free_motion.free_motion_joint_qd_sap
            )
            self.local_a_env_wp = wp.zeros_like(self.a_env_wp)
            self.local_contact_env_phi0_wp = wp.zeros_like(self.contact_env_phi0_wp)
            self.local_contact_env_jac_wp = wp.zeros_like(self.contact_env_jac_wp)
            self.local_contact_env_w_eff_wp = wp.zeros_like(self.contact_env_w_eff_wp)
            self.local_contact_env_mu_wp = wp.zeros_like(self.contact_env_mu_wp)
            self.local_contact_env_k_wp = wp.zeros_like(self.contact_env_k_wp)
            self.local_contact_env_tau_wp = wp.zeros_like(self.contact_env_tau_wp)
            self.local_contact_env_R_WC_wp = wp.zeros_like(self.contact_env_R_WC_wp)
            self.local_contact_env_point_wp = wp.zeros_like(self.contact_env_point_wp)
            self.local_contact_env_body0_wp = wp.full_like(self.contact_env_body0_wp, -1)
            self.local_contact_env_body1_wp = wp.full_like(self.contact_env_body1_wp, -1)

    def _assemble_dynamics_matrix(self) -> None:
        wp.launch(
            _assemble_dynamics_matrix_multi_env_sap,
            dim=(self.num_envs, self.dof_per_env, self.dof_per_env),
            inputs=[
                self.a_env_wp,
                self.free_motion.H,
                self.free_motion.model_joint_armature,
                self.free_motion.articulation_H_start,
                self.free_motion.articulation_H_rows,
                self.env_articulation_ids,
                self.env_articulation_count,
                self.articulation_local_dof_start,
                int(self.max_articulations_per_env),
                int(self.dof_per_env),
            ],
            device=self.device,
        )
        if self.capture_local_snapshots:
            wp.copy(self.local_a_env_wp, self.a_env_wp)

    def compute(
        self,
        state_in: State,
        contacts: SapContacts,
        *,
        control: Control,
        dt: float = 0.0,
    ) -> SapContactJacobianResult:
        """Build contact Jacobians, material parameters, gap values, and per-environment dynamics blocks
        for the SAP contact solve.
        """
        if not isinstance(state_in, State):
            raise TypeError("SapContactJacobian.compute requires SapState.")
        if not isinstance(control, Control):
            raise TypeError("SapContactJacobian.compute requires SapControl.")
        if not isinstance(contacts, SapContacts):
            raise TypeError("SapContactJacobian.compute requires SapContacts.")
        # Reuse the SAP-native free-motion kernels for body-origin kinematics,
        # motion subspaces, and articulation-local H.  The global dense A is
        # not assembled here; contact preparation builds env-local A blocks.
        self.free_motion.compute(state_in, control, dt, assemble_dynamics_matrix=False)
        if self.capture_local_snapshots:
            wp.copy(self.local_joint_qd_sap_input_wp, self.free_motion.joint_qd_sap_input)
            wp.copy(self.local_free_motion_joint_qd_sap_wp, self.free_motion.free_motion_joint_qd_sap)

        total_contact_count = self._total_rigid_contact_count(contacts)
        active_count = min(total_contact_count, self.max_contacts)
        contact_capacity = max(1, min(self.max_rigid_contact, int(self.contact_env_phi0_wp.shape[1])))
        truncated = max(total_contact_count - self.max_contacts, 0)

        wp.launch(
            _assemble_body_origin_jacobians_local_sap,
            dim=(self.body_count, 6, self.dof_per_env),
            inputs=[
                self.body_jac_local_wp,
                self.free_motion.J,
                self.body_src_row_start,
                self.body_local_dof_start,
                self.body_cols,
            ],
            device=self.device,
        )
        wp.launch(
            _assemble_dynamics_matrix_multi_env_sap,
            dim=(self.num_envs, self.dof_per_env, self.dof_per_env),
            inputs=[
                self.a_env_wp,
                self.free_motion.H,
                self.free_motion.model_joint_armature,
                self.free_motion.articulation_H_start,
                self.free_motion.articulation_H_rows,
                self.env_articulation_ids,
                self.env_articulation_count,
                self.articulation_local_dof_start,
                int(self.max_articulations_per_env),
                int(self.dof_per_env),
            ],
            device=self.device,
        )
        if self.capture_local_snapshots:
            wp.copy(self.local_a_env_wp, self.a_env_wp)

        self.contact_env_count_wp.fill_(0)
        if self.capture_local_snapshots:
            self.contact_env_phi0_wp.fill_(0.0)
            self.contact_env_jac_wp.fill_(0.0)
            self.contact_env_w_eff_wp.fill_(1.0e-12)
            self.contact_env_mu_wp.fill_(self.fallback_mu)
            self.contact_env_k_wp.fill_(self.fallback_stiffness)
            self.contact_env_tau_wp.fill_(self.fallback_tau_d + self.fallback_tau_d)
            self.contact_env_R_WC_wp.fill_(0.0)
            self.contact_env_point_wp.zero_()
            self.contact_env_witness0_wp.zero_()
            self.contact_env_witness1_wp.zero_()
            self.contact_env_body0_wp.fill_(-1)
            self.contact_env_body1_wp.fill_(-1)
        if getattr(state_in.body_q, "dtype", None) == wp.transform:
            contact_body_q = state_in.body_q
        else:
            contact_body_q = self.free_motion.body_q
        contact_body_q_is_f32 = getattr(contact_body_q, "dtype", None) == wp.transform
        contact_weights_inlined = False
        rigid_contact_count = (
            contacts.rigid_contact_count if contacts.rigid_contact_count is not None else self._empty_rigid_contact_count
        )
        rigid_contact_shape0 = (
            contacts.rigid_contact_shape0 if contacts.rigid_contact_shape0 is not None else self._empty_rigid_contact_shape
        )
        rigid_contact_shape1 = (
            contacts.rigid_contact_shape1 if contacts.rigid_contact_shape1 is not None else self._empty_rigid_contact_shape
        )
        rigid_contact_point0 = (
            contacts.rigid_contact_point0 if contacts.rigid_contact_point0 is not None else self._empty_rigid_contact_point
        )
        rigid_contact_point1 = (
            contacts.rigid_contact_point1 if contacts.rigid_contact_point1 is not None else self._empty_rigid_contact_point
        )
        rigid_contact_normal = (
            contacts.rigid_contact_normal if contacts.rigid_contact_normal is not None else self._empty_rigid_contact_normal
        )
        rigid_contact_margin0 = (
            contacts.rigid_contact_margin0
            if contacts.rigid_contact_margin0 is not None
            else self._empty_rigid_contact_margin
        )
        rigid_contact_margin1 = (
            contacts.rigid_contact_margin1
            if contacts.rigid_contact_margin1 is not None
            else self._empty_rigid_contact_margin
        )
        rigid_contact_stiffness = (
            contacts.rigid_contact_stiffness
            if contacts.rigid_contact_stiffness is not None
            else self._empty_rigid_contact_material
        )
        rigid_contact_damping = (
            contacts.rigid_contact_damping
            if contacts.rigid_contact_damping is not None
            else self._empty_rigid_contact_material
        )
        rigid_contact_friction = (
            contacts.rigid_contact_friction
            if contacts.rigid_contact_friction is not None
            else self._empty_rigid_contact_material
        )
        scatter_dim = int(self.max_contacts)
        has_f64_contacts = contacts.has_f64_rigid_contacts
        if contact_body_q_is_f32 and has_f64_contacts:
            wp.launch(
                _scatter_sap_contacts_to_env_direct_f64_f32_pose,
                dim=scatter_dim,
                inputs=[
                    scatter_dim,
                    rigid_contact_count,
                    int(self.bodies_per_env),
                    int(self.dof_per_env),
                    int(self.max_rigid_contact),
                    int(self.contact_point_mode == "witness_point"),
                    self.model.shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    contacts.rigid_contact_point0d,
                    contacts.rigid_contact_point1d,
                    contacts.rigid_contact_normald,
                    contacts.rigid_contact_margin0d,
                    contacts.rigid_contact_margin1d,
                    contact_body_q,
                    self.body_jac_local_wp,
                    self.contact_env_count_wp,
                    self.contact_env_phi0_wp,
                    self.contact_env_jac_wp,
                    self.contact_env_mu_wp,
                    self.contact_env_k_wp,
                    self.contact_env_tau_wp,
                    self.contact_env_R_WC_wp,
                    self.contact_env_point_wp,
                    self.contact_env_witness0_wp,
                    self.contact_env_witness1_wp,
                    self.contact_env_body0_wp,
                    self.contact_env_body1_wp,
                    self.contact_shape_mu,
                    float(self.fallback_mu),
                    self.contact_shape_ke,
                    float(self.fallback_stiffness),
                    self.contact_shape_tau,
                    float(self.fallback_tau_d),
                    rigid_contact_stiffness,
                    rigid_contact_damping,
                    rigid_contact_friction,
                ],
                device=self.device,
            )
        elif contact_body_q_is_f32:
            wp.launch(
                _scatter_sap_contacts_to_env_direct_f32_pose,
                dim=scatter_dim,
                inputs=[
                    scatter_dim,
                    rigid_contact_count,
                    int(self.bodies_per_env),
                    int(self.dof_per_env),
                    int(self.max_rigid_contact),
                    int(self.contact_point_mode == "witness_point"),
                    self.model.shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    rigid_contact_point0,
                    rigid_contact_point1,
                    rigid_contact_normal,
                    rigid_contact_margin0,
                    rigid_contact_margin1,
                    contact_body_q,
                    self.body_jac_local_wp,
                    self.contact_env_count_wp,
                    self.contact_env_phi0_wp,
                    self.contact_env_jac_wp,
                    self.contact_env_mu_wp,
                    self.contact_env_k_wp,
                    self.contact_env_tau_wp,
                    self.contact_env_R_WC_wp,
                    self.contact_env_point_wp,
                    self.contact_env_witness0_wp,
                    self.contact_env_witness1_wp,
                    self.contact_env_body0_wp,
                    self.contact_env_body1_wp,
                    self.contact_input_env_wp,
                    self.contact_input_slot_wp,
                    int(self.contact_weight_mode == "body_inertia"),
                    self.free_motion.model_body_mass,
                    self.free_motion.model_body_inertia,
                    self.free_motion.model_body_com,
                    self.contact_env_w_eff_wp,
                    self.contact_shape_mu,
                    float(self.fallback_mu),
                    self.contact_shape_ke,
                    float(self.fallback_stiffness),
                    self.contact_shape_tau,
                    float(self.fallback_tau_d),
                    rigid_contact_stiffness,
                    rigid_contact_damping,
                    rigid_contact_friction,
                ],
                device=self.device,
            )
            contact_weights_inlined = self.contact_weight_mode == "body_inertia"
        elif has_f64_contacts:
            wp.launch(
                _scatter_sap_contacts_to_env_direct_f64,
                dim=scatter_dim,
                inputs=[
                    scatter_dim,
                    rigid_contact_count,
                    int(self.bodies_per_env),
                    int(self.dof_per_env),
                    int(self.max_rigid_contact),
                    int(self.contact_point_mode == "witness_point"),
                    self.model.shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    contacts.rigid_contact_point0d,
                    contacts.rigid_contact_point1d,
                    contacts.rigid_contact_normald,
                    contacts.rigid_contact_margin0d,
                    contacts.rigid_contact_margin1d,
                    contact_body_q,
                    self.body_jac_local_wp,
                    self.contact_env_count_wp,
                    self.contact_env_phi0_wp,
                    self.contact_env_jac_wp,
                    self.contact_env_mu_wp,
                    self.contact_env_k_wp,
                    self.contact_env_tau_wp,
                    self.contact_env_R_WC_wp,
                    self.contact_env_point_wp,
                    self.contact_env_witness0_wp,
                    self.contact_env_witness1_wp,
                    self.contact_env_body0_wp,
                    self.contact_env_body1_wp,
                    self.contact_shape_mu,
                    float(self.fallback_mu),
                    self.contact_shape_ke,
                    float(self.fallback_stiffness),
                    self.contact_shape_tau,
                    float(self.fallback_tau_d),
                    rigid_contact_stiffness,
                    rigid_contact_damping,
                    rigid_contact_friction,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _scatter_sap_contacts_to_env_direct,
                dim=scatter_dim,
                inputs=[
                    scatter_dim,
                    rigid_contact_count,
                    int(self.bodies_per_env),
                    int(self.dof_per_env),
                    int(self.max_rigid_contact),
                    int(self.contact_point_mode == "witness_point"),
                    self.model.shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    rigid_contact_point0,
                    rigid_contact_point1,
                    rigid_contact_normal,
                    rigid_contact_margin0,
                    rigid_contact_margin1,
                    contact_body_q,
                    self.body_jac_local_wp,
                    self.contact_env_count_wp,
                    self.contact_env_phi0_wp,
                    self.contact_env_jac_wp,
                    self.contact_env_mu_wp,
                    self.contact_env_k_wp,
                    self.contact_env_tau_wp,
                    self.contact_env_R_WC_wp,
                    self.contact_env_point_wp,
                    self.contact_env_witness0_wp,
                    self.contact_env_witness1_wp,
                    self.contact_env_body0_wp,
                    self.contact_env_body1_wp,
                    self.contact_input_env_wp,
                    self.contact_input_slot_wp,
                    self.contact_shape_mu,
                    float(self.fallback_mu),
                    self.contact_shape_ke,
                    float(self.fallback_stiffness),
                    self.contact_shape_tau,
                    float(self.fallback_tau_d),
                    rigid_contact_stiffness,
                    rigid_contact_damping,
                    rigid_contact_friction,
                ],
                device=self.device,
            )
        if not contact_weights_inlined:
            self._compute_contact_weights(contact_capacity, contact_body_q)
        if self.capture_local_snapshots:
            wp.copy(self.local_contact_env_phi0_wp, self.contact_env_phi0_wp)
            wp.copy(self.local_contact_env_jac_wp, self.contact_env_jac_wp)
            wp.copy(self.local_contact_env_w_eff_wp, self.contact_env_w_eff_wp)
            wp.copy(self.local_contact_env_mu_wp, self.contact_env_mu_wp)
            wp.copy(self.local_contact_env_k_wp, self.contact_env_k_wp)
            wp.copy(self.local_contact_env_tau_wp, self.contact_env_tau_wp)
            wp.copy(self.local_contact_env_R_WC_wp, self.contact_env_R_WC_wp)
            wp.copy(self.local_contact_env_point_wp, self.contact_env_point_wp)
            wp.copy(self.local_contact_env_body0_wp, self.contact_env_body0_wp)
            wp.copy(self.local_contact_env_body1_wp, self.contact_env_body1_wp)

        return SapContactJacobianResult(
            contact_count=active_count,
            truncated_contact_count=truncated,
            contact_env_count=self.contact_env_count_wp,
            contact_env_phi0=self.contact_env_phi0_wp,
            contact_env_jacobian=self.contact_env_jac_wp,
            contact_env_w_eff=self.contact_env_w_eff_wp,
            contact_env_mu=self.contact_env_mu_wp,
            contact_env_stiffness=self.contact_env_k_wp,
            contact_env_tau_d=self.contact_env_tau_wp,
            contact_env_R_WC=self.contact_env_R_WC_wp,
            contact_env_point=self.contact_env_point_wp,
            contact_env_witness0=self.contact_env_witness0_wp,
            contact_env_witness1=self.contact_env_witness1_wp,
            contact_env_body0=self.contact_env_body0_wp,
            contact_env_body1=self.contact_env_body1_wp,
            body_jacobian_local=self.body_jac_local_wp,
            dynamics_matrix_env=self.a_env_wp,
            local_joint_qd_sap_input=self.local_joint_qd_sap_input_wp,
            local_free_motion_joint_qd_sap=self.local_free_motion_joint_qd_sap_wp,
            local_dynamics_matrix_env=self.local_a_env_wp,
            local_contact_env_phi0=self.local_contact_env_phi0_wp,
            local_contact_env_jacobian=self.local_contact_env_jac_wp,
            local_contact_env_w_eff=self.local_contact_env_w_eff_wp,
            local_contact_env_mu=self.local_contact_env_mu_wp,
            local_contact_env_stiffness=self.local_contact_env_k_wp,
            local_contact_env_tau_d=self.local_contact_env_tau_wp,
            local_contact_env_R_WC=self.local_contact_env_R_WC_wp,
            local_contact_env_point=self.local_contact_env_point_wp,
            local_contact_env_body0=self.local_contact_env_body0_wp,
            local_contact_env_body1=self.local_contact_env_body1_wp,
        )
