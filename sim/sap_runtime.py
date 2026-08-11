"""SAP runtime data containers and Newton adapter helpers.

Source note: the SAP modifications in this module are based on Newton's
runtime container code and adapted so SAP Warp can wrap Newton-owned Warp
arrays directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import warp as wp

SapPrecision = Literal["f32", "f64"]
SapDofOrder = Literal["sap", "public"]
SapBodyForceOrder = Literal["sap", "public"]

SAP_JOINT_PRISMATIC = wp.constant(0)
SAP_JOINT_REVOLUTE = wp.constant(1)
SAP_JOINT_BALL = wp.constant(2)
SAP_JOINT_FIXED = wp.constant(3)
SAP_JOINT_FREE = wp.constant(4)
SAP_JOINT_DISTANCE = wp.constant(5)
SAP_JOINT_D6 = wp.constant(6)

SAP_JOINT_TARGET_NONE = wp.constant(0)

SAP_BODY_FLAG_DYNAMIC = wp.constant(1 << 0)
SAP_BODY_FLAG_KINEMATIC = wp.constant(1 << 1)
SAP_BODY_FLAG_ALL = wp.constant((1 << 0) | (1 << 1))


@dataclass(frozen=True)
class SapPrecisionPolicy:
    """Precision choices used by the SAP runtime stages. Each field selects the dtype family for state
    storage, free motion, contact Jacobians, contact solve, and linear solve buffers.
    """
    state: SapPrecision = "f32"
    free_motion: SapPrecision = "f32"
    contact_jacobian: SapPrecision = "f64"
    contact_solve: SapPrecision = "f64"
    contact_linear_solve: SapPrecision = "f32"


@dataclass(frozen=True)
class SapModel:
    """Immutable SAP model arrays describing topology, articulated bodies, shapes, materials, limits,
    drives, labels, and initial state for one or more replicated worlds.
    """
    device: Any
    joint_count: int
    joint_dof_count: int
    joint_coord_count: int
    body_count: int
    articulation_count: int
    world_count: int
    shape_count: int
    requires_grad: bool

    joint_type: wp.array
    joint_articulation: wp.array
    joint_parent: wp.array
    joint_child: wp.array
    joint_q_start: wp.array
    joint_qd_start: wp.array
    joint_dof_dim: wp.array
    joint_axis: wp.array
    joint_X_p: wp.array
    joint_X_c: wp.array
    joint_ancestor: wp.array
    articulation_start: wp.array
    joint_q: wp.array
    joint_qd: wp.array

    body_q: wp.array
    body_qd: wp.array
    body_mass: wp.array
    body_inertia: wp.array
    body_com: wp.array
    body_flags: wp.array
    body_world: wp.array
    gravity: wp.array
    joint_armature: wp.array

    shape_body: wp.array
    shape_material_mu: wp.array | None
    shape_material_ke: wp.array | None
    shape_material_tau: wp.array | None

    joint_target_mode: wp.array
    joint_target_ke: wp.array
    joint_target_kd: wp.array
    joint_target_pos: wp.array
    joint_target_vel: wp.array
    joint_f: wp.array
    joint_act: wp.array
    joint_effort_limit: wp.array
    joint_limit_lower: wp.array
    joint_limit_upper: wp.array
    joint_limit_ke: wp.array
    joint_limit_kd: wp.array

    sap_debug_body_mass: wp.array | None = None
    sap_debug_body_inertia: wp.array | None = None
    sap_debug_body_com: wp.array | None = None
    sap_debug_joint_axis: wp.array | None = None
    sap_debug_joint_X_p: wp.array | None = None
    sap_debug_joint_X_c: wp.array | None = None
    sap_debug_gravity: wp.array | None = None
    sap_debug_joint_armature: wp.array | None = None
    sap_debug_shape_material_mu: wp.array | None = None
    sap_debug_shape_material_ke: wp.array | None = None
    sap_debug_shape_material_tau: wp.array | None = None

    up_axis: int = 2
    shape_transform: wp.array | None = None
    shape_type: wp.array | None = None
    shape_scale: wp.array | None = None
    shape_flags: wp.array | None = None
    shape_margin: wp.array | None = None
    shape_gap: wp.array | None = None
    shape_is_solid: wp.array | None = None
    shape_source: list[Any] | None = None
    shape_source_ptr: wp.array | None = None
    shape_collision_radius: wp.array | None = None
    shape_world: wp.array | None = None
    shape_sdf_index: wp.array | None = None
    texture_sdf_data: wp.array | None = None
    texture_sdf_coarse_textures: list[Any] | None = None
    texture_sdf_subgrid_textures: list[Any] | None = None
    texture_sdf_subgrid_start_slots: list[Any] | None = None
    body_label: tuple[str, ...] = ()
    joint_label: tuple[str, ...] = ()
    articulation_label: tuple[str, ...] = ()
    shape_label: tuple[str, ...] = ()

    def state(self, requires_grad: bool | None = None) -> "SapState":
        """Create a mutable SapState initialized from the model position, velocity, body pose, and body
        velocity arrays.
        """
        if requires_grad is None:
            requires_grad = self.requires_grad
        return SapState(
            joint_q=wp.clone(self.joint_q, requires_grad=requires_grad),
            joint_qd=wp.clone(self.joint_qd, requires_grad=requires_grad),
            body_q=wp.clone(self.body_q, requires_grad=requires_grad),
            body_qd=wp.clone(self.body_qd, requires_grad=requires_grad),
            body_f=wp.zeros_like(self.body_qd, requires_grad=requires_grad),
            joint_qd_order="public",
            body_f_order="public",
            requires_grad=bool(requires_grad),
        )

    def control(self, requires_grad: bool | None = None, clone_variables: bool = True) -> "SapControl":
        """Create a SapControl object initialized from the model force, target, and actuation arrays."""
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            return SapControl(
                joint_f=wp.clone(self.joint_f, requires_grad=requires_grad),
                joint_target_pos=wp.clone(self.joint_target_pos, requires_grad=requires_grad),
                joint_target_vel=wp.clone(self.joint_target_vel, requires_grad=requires_grad),
                joint_act=wp.clone(self.joint_act, requires_grad=requires_grad),
                joint_f_order="public",
            )
        return SapControl(
            joint_f=self.joint_f,
            joint_target_pos=self.joint_target_pos,
            joint_target_vel=self.joint_target_vel,
            joint_act=self.joint_act,
            joint_f_order="public",
        )


@dataclass(frozen=True)
class SapState:
    """Mutable SAP simulation state. Positions, velocities, body poses, body velocities, and external
    forces live here while SapModel remains immutable.
    """
    joint_q: wp.array
    joint_qd: wp.array
    body_q: wp.array
    body_qd: wp.array | None = None
    body_f: wp.array | None = None
    joint_qd_order: SapDofOrder = "sap"
    body_f_order: SapBodyForceOrder = "sap"
    requires_grad: bool = False

    def clear_forces(self) -> None:
        """Clear external body forces in place while leaving positions and velocities unchanged."""
        if self.body_f is not None:
            self.body_f.zero_()

    def assign(self, other: "SapState") -> None:
        """Copy all state arrays from another SapState with matching optional-buffer layout."""
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            dst = getattr(self, name, None)
            src = getattr(other, name, None)
            if dst is None and src is None:
                continue
            if dst is None or src is None:
                raise ValueError(f"SapState assign mismatch for {name}")
            wp.copy(dest=dst, src=src)


@dataclass(frozen=True)
class SapControl:
    """Mutable control input for a SAP timestep, including generalized forces, drive targets, target
    velocities, and actuation values.
    """
    joint_f: wp.array
    joint_target_pos: wp.array | None = None
    joint_target_vel: wp.array | None = None
    joint_act: wp.array | None = None
    joint_f_order: SapDofOrder = "sap"

    def clear(self) -> None:
        """Clear direct forces, drive targets, target velocities, and actuation arrays in place."""
        self.joint_f.zero_()
        if self.joint_target_pos is not None:
            self.joint_target_pos.zero_()
        if self.joint_target_vel is not None:
            self.joint_target_vel.zero_()
        if self.joint_act is not None:
            self.joint_act.zero_()


@dataclass(frozen=True)
class SapContacts:
    """Compatibility contact bundle used by solver-facing SAP runtime code. It stores rigid contact
    arrays and optional f64 mirror buffers.
    """
    rigid_contact_count: wp.array | None = None
    rigid_contact_shape0: wp.array | None = None
    rigid_contact_shape1: wp.array | None = None
    rigid_contact_point0: wp.array | None = None
    rigid_contact_point1: wp.array | None = None
    rigid_contact_normal: wp.array | None = None
    rigid_contact_margin0: wp.array | None = None
    rigid_contact_margin1: wp.array | None = None
    rigid_contact_point0d: wp.array | None = None
    rigid_contact_point1d: wp.array | None = None
    rigid_contact_normald: wp.array | None = None
    rigid_contact_margin0d: wp.array | None = None
    rigid_contact_margin1d: wp.array | None = None
    rigid_contact_stiffness: wp.array | None = None
    rigid_contact_damping: wp.array | None = None
    rigid_contact_friction: wp.array | None = None

    @property
    def has_rigid_contacts(self) -> bool:
        """Return True when this contact bundle contains rigid-contact arrays."""
        return self.rigid_contact_count is not None

    @property
    def has_f64_rigid_contacts(self) -> bool:
        """Return True when all f64 rigid-contact mirror arrays are present."""
        return (
            self.rigid_contact_point0d is not None
            and self.rigid_contact_point1d is not None
            and self.rigid_contact_normald is not None
            and self.rigid_contact_margin0d is not None
            and self.rigid_contact_margin1d is not None
        )


@dataclass
class SapData:
    """Scratch data bundle reserved for SAP stage buffers that need to travel together across runtime
    calls.
    """
    body_q: wp.array | None = None
    body_qd: wp.array | None = None
    v0: wp.array | None = None
    v_star: wp.array | None = None
    qdd: wp.array | None = None
    h: wp.array | None = None
    j: wp.array | None = None
    a_env: wp.array | None = None
    contact_env_count: wp.array | None = None


def _array_size(arr: wp.array | None) -> int:
    if arr is None:
        return 0
    try:
        return int(np.asarray(arr.numpy()).reshape(-1).size)
    except Exception:
        return int(getattr(arr, "size", 0) or 0)


def sap_model_from_newton(model: SapModel | Any) -> SapModel:
    """Convert a Newton model into an immutable SapModel with SAP-owned array conventions and optional
    debug mirrors.
    """
    if isinstance(model, SapModel):
        return model

    shape_body = getattr(model, "shape_body")
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    if shape_count <= 0:
        shape_count = _array_size(shape_body)

    return SapModel(
        device=model.device,
        joint_count=int(model.joint_count),
        joint_dof_count=int(model.joint_dof_count),
        joint_coord_count=int(model.joint_coord_count),
        body_count=int(model.body_count),
        articulation_count=int(model.articulation_count),
        world_count=int(getattr(model, "world_count", 1)),
        shape_count=shape_count,
        requires_grad=bool(getattr(model, "requires_grad", False)),
        joint_type=model.joint_type,
        joint_articulation=model.joint_articulation,
        joint_parent=model.joint_parent,
        joint_child=model.joint_child,
        joint_q_start=model.joint_q_start,
        joint_qd_start=model.joint_qd_start,
        joint_dof_dim=model.joint_dof_dim,
        joint_axis=model.joint_axis,
        joint_X_p=model.joint_X_p,
        joint_X_c=model.joint_X_c,
        joint_ancestor=model.joint_ancestor,
        articulation_start=model.articulation_start,
        joint_q=model.joint_q,
        joint_qd=model.joint_qd,
        body_q=model.body_q,
        body_qd=model.body_qd,
        body_mass=model.body_mass,
        body_inertia=model.body_inertia,
        body_com=model.body_com,
        body_flags=model.body_flags,
        body_world=model.body_world,
        gravity=model.gravity,
        joint_armature=model.joint_armature,
        shape_body=shape_body,
        shape_material_mu=getattr(model, "shape_material_mu", None),
        shape_material_ke=getattr(model, "shape_material_ke", None),
        shape_material_tau=getattr(model, "shape_material_tau", None),
        joint_target_mode=model.joint_target_mode,
        joint_target_ke=model.joint_target_ke,
        joint_target_kd=model.joint_target_kd,
        # Newton 1.5 renamed the public target arrays.  Keep SAP-Warp's
        # internal names stable at this dirty adapter boundary.
        joint_target_pos=model.joint_target_q,
        joint_target_vel=model.joint_target_qd,
        joint_f=model.joint_f,
        joint_act=model.joint_act,
        joint_effort_limit=model.joint_effort_limit,
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        joint_limit_ke=model.joint_limit_ke,
        joint_limit_kd=model.joint_limit_kd,
        sap_debug_body_mass=getattr(model, "sap_debug_body_mass", None),
        sap_debug_body_inertia=getattr(model, "sap_debug_body_inertia", None),
        sap_debug_body_com=getattr(model, "sap_debug_body_com", None),
        sap_debug_joint_axis=getattr(model, "sap_debug_joint_axis", None),
        sap_debug_joint_X_p=getattr(model, "sap_debug_joint_X_p", None),
        sap_debug_joint_X_c=getattr(model, "sap_debug_joint_X_c", None),
        sap_debug_gravity=getattr(model, "sap_debug_gravity", None),
        sap_debug_joint_armature=getattr(model, "sap_debug_joint_armature", None),
        sap_debug_shape_material_mu=getattr(model, "sap_debug_shape_material_mu", None),
        sap_debug_shape_material_ke=getattr(model, "sap_debug_shape_material_ke", None),
        sap_debug_shape_material_tau=getattr(model, "sap_debug_shape_material_tau", None),
        up_axis=int(getattr(model, "up_axis", 2)),
        shape_transform=getattr(model, "shape_transform", None),
        shape_type=getattr(model, "shape_type", None),
        shape_scale=getattr(model, "shape_scale", None),
        shape_flags=getattr(model, "shape_flags", None),
        shape_margin=getattr(model, "shape_margin", None),
        shape_gap=getattr(model, "shape_gap", None),
        shape_is_solid=getattr(model, "shape_is_solid", None),
        shape_source=list(getattr(model, "shape_source", []) or []),
        shape_source_ptr=getattr(model, "shape_source_ptr", None),
        shape_collision_radius=getattr(model, "shape_collision_radius", None),
        shape_world=getattr(model, "shape_world", None),
        shape_sdf_index=getattr(model, "shape_sdf_index", None),
        texture_sdf_data=getattr(model, "texture_sdf_data", None),
        texture_sdf_coarse_textures=list(getattr(model, "texture_sdf_coarse_textures", []) or []),
        texture_sdf_subgrid_textures=list(getattr(model, "texture_sdf_subgrid_textures", []) or []),
        texture_sdf_subgrid_start_slots=list(getattr(model, "texture_sdf_subgrid_start_slots", []) or []),
        body_label=tuple(getattr(model, "body_label", ()) or ()),
        joint_label=tuple(getattr(model, "joint_label", ()) or ()),
        articulation_label=tuple(getattr(model, "articulation_label", ()) or ()),
        shape_label=tuple(getattr(model, "shape_label", ()) or ()),
    )


def sap_state_from_newton(state: SapState | Any) -> SapState:
    """Convert a Newton state into a SapState that uses the public boundary ordering expected by high-
    level callers.
    """
    if isinstance(state, SapState):
        return state
    return SapState(
        joint_q=state.joint_q,
        joint_qd=state.joint_qd,
        body_q=state.body_q,
        body_qd=getattr(state, "body_qd", None),
        body_f=getattr(state, "body_f", None),
        joint_qd_order="public",
        body_f_order="public",
        requires_grad=bool(getattr(state, "requires_grad", False)),
    )


def sap_control_from_newton(control: SapControl | Any) -> SapControl:
    """Convert a Newton control object into SapControl arrays using the public generalized-force
    ordering.
    """
    if isinstance(control, SapControl):
        return control
    return SapControl(
        joint_f=control.joint_f,
        joint_target_pos=getattr(control, "joint_target_q", None),
        joint_target_vel=getattr(control, "joint_target_qd", None),
        joint_act=getattr(control, "joint_act", None),
        joint_f_order="public",
    )


def sap_contacts_from_newton(contacts: SapContacts | Any | None) -> SapContacts:
    """Convert Newton contact storage into the SAP runtime contact bundle consumed by the solver."""
    if isinstance(contacts, SapContacts):
        return contacts
    if contacts is None:
        return SapContacts()
    return SapContacts(
        rigid_contact_count=getattr(contacts, "rigid_contact_count", None),
        rigid_contact_shape0=getattr(contacts, "rigid_contact_shape0", None),
        rigid_contact_shape1=getattr(contacts, "rigid_contact_shape1", None),
        rigid_contact_point0=getattr(contacts, "rigid_contact_point0", None),
        rigid_contact_point1=getattr(contacts, "rigid_contact_point1", None),
        rigid_contact_normal=getattr(contacts, "rigid_contact_normal", None),
        rigid_contact_margin0=getattr(contacts, "rigid_contact_margin0", None),
        rigid_contact_margin1=getattr(contacts, "rigid_contact_margin1", None),
        rigid_contact_point0d=getattr(contacts, "rigid_contact_point0d", None),
        rigid_contact_point1d=getattr(contacts, "rigid_contact_point1d", None),
        rigid_contact_normald=getattr(contacts, "rigid_contact_normald", None),
        rigid_contact_margin0d=getattr(contacts, "rigid_contact_margin0d", None),
        rigid_contact_margin1d=getattr(contacts, "rigid_contact_margin1d", None),
        rigid_contact_stiffness=getattr(contacts, "rigid_contact_stiffness", None),
        rigid_contact_damping=getattr(contacts, "rigid_contact_damping", None),
        rigid_contact_friction=getattr(contacts, "rigid_contact_friction", None),
    )


Model = SapModel
State = SapState
Control = SapControl
