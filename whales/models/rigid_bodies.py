"""
Structural models built of rigid bodies
"""

import numpy as np

from mbwind.core import System
from mbwind.elements import FreeJoint, RigidConnection, RigidBody
from mbwind import LinearisedSystem

from . import LinearSystem


def build_structure(rigid_body_configs):
    # Free joint represents the rigid-body motion of the platform
    free_joint = FreeJoint('base')

    for i, conf in enumerate(rigid_body_configs):
        name = conf.get('name', 'body{}'.format(i))
        conn = RigidConnection('conn-' + name, offset=conf['CoM'])
        body = RigidBody(name, mass=conf['mass'],
                         inertia=np.diag(conf['inertia']))
        free_joint.add_leaf(conn)
        conn.add_leaf(body)

    system = System(free_joint)
    return system


def structure(rigid_body_configs):
    system = build_structure(rigid_body_configs)
    system.update_kinematics()
    linsys = LinearisedSystem.from_system(system)
    return LinearSystem(linsys.M, linsys.C, linsys.K)
