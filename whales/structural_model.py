"""
Build a structural model of a wind turbine using mbwind
"""

import itertools
import numpy as np
from numpy import dot, pi
from mbwind.utils import rotmat_x, rotmat_y
from mbwind.core import System
from mbwind.elements import (FreeJoint, RigidBody, RigidConnection, Hinge,
                             ModalElement, DistalModalElementFromScratch)
from mbwind.modes import ModesFromScratch
from mbwind.blade import Tower
from mbwind.io import load_modes_from_Bladed
from mbwind import LinearisedSystem


class FloatingTurbineStructure(object):
    def __init__(self, structure_config):
        s = structure_config

        #### Load details of flexible elements ####
        if 'definition' in s['tower']:
            if 'mass' in s['tower']:
                raise ValueError("Both tower definition and explicit mass!")
            self.tower_definition = Tower(s['tower']['definition'])
            assert np.all(self.tower_definition.stn_pos[:, :2] == 0)  # vert.
            z_tower = self.tower_definition.stn_pos[:, 2]
            self.tower_modes = ModesFromScratch(
                z_tower - z_tower[0],
                self.tower_definition.density, 1,
                self.tower_definition.EIy, self.tower_definition.EIz)
        else:
            self.tower_definition = None
            self.tower_modes = None

        if 'blade' in s:
            self.blade_modes = load_modes_from_Bladed(s['blade']['definition'])
        else:
            self.blade_modes = None

        #### Create the elements ####

        # Free joint represents the rigid-body motion of the platform
        free_joint = FreeJoint('base')

        # This is the rigid-body mass of the platform structure
        conn_platform = RigidConnection('conn-platform',
                                        offset=s['platform']['CoM'])
        platform = RigidBody('platform',
                             mass=s['platform']['mass'],
                             inertia=np.diag(s['platform']['inertia']))
        free_joint.add_leaf(conn_platform)
        conn_platform.add_leaf(platform)

        # Make a rigid body to represent the added mass
        # (approximate to zero frequency)
        #  XXX this is skipping the coupling matrix
        #A = whales_model.A(0)
        # added_mass = RigidBody('added-mass', mass=np.diag(A[:3, :3]),
        #                           inertia=A[3:, 3:])

        # Flexible tower or equivalent rigid body
        if self.tower_modes:
            # move base of tower 10m up, and rotate so tower x-axis is vertical
            conn_tower = RigidConnection(
                'conn-tower', offset=[0, 0, z_tower[0]],
                rotation=rotmat_y(-pi/2))
            tower = DistalModalElementFromScratch(
                'tower', self.tower_modes,
                s['tower']['number of normal modes'])
        else:
            # move tower to COG
            conn_tower = RigidConnection(
                'conn-tower', offset=s['tower']['CoM'])
            tower = RigidBody('tower', s['tower']['mass'],
                              np.diag(s['tower']['inertia']))
        free_joint.add_leaf(conn_tower)
        conn_tower.add_leaf(tower)

        # The nacelle -- rigid body
        # rotate back so nacelle inertia is aligned with global coordinates
        if self.tower_modes:
            nacoff = s['nacelle']['offset from tower top']
            conn_nacelle = RigidConnection('conn-nacelle',
                                           offset=dot(rotmat_y(pi/2), nacoff),
                                           rotation=rotmat_y(pi/2))
            tower.add_leaf(conn_nacelle)
        else:
            conn_nacelle = RigidConnection(
                'conn-nacelle',
                offset=np.array([0, 0, s['nacelle']['height']]))
            free_joint.add_leaf(conn_nacelle)
        nacelle = RigidBody(
            'nacelle',
            mass=s['nacelle']['mass'],
            inertia=np.diag(s['nacelle'].get('inertia', np.zeros(3))))
        conn_nacelle.add_leaf(nacelle)

        # The rotor hub -- currently just connections (no mass)
        # rotate so rotor centre is aligned with global coordinates
        if self.tower_modes:
            rotoff = s['rotor']['offset from tower top']
            conn_rotor = RigidConnection('conn-rotor',
                                         offset=dot(rotmat_y(pi/2), rotoff),
                                         rotation=rotmat_y(pi/2))
            tower.add_leaf(conn_rotor)
        else:
            conn_rotor = RigidConnection(
                'conn-rotor',
                offset=np.array([0, 0, s['nacelle']['height']]))
            free_joint.add_leaf(conn_rotor)

        # The drive shaft rotation (rotation about x)
        shaft = Hinge('shaft', [1, 0, 0])
        conn_rotor.add_leaf(shaft)

        # The blades
        if self.blade_modes:
            rtlen = s['rotor']['root length']
            Ryx = dot(rotmat_y(-pi/2), rotmat_x(-pi/2))  # align blade modes
            for i in range(3):
                R = rotmat_x(i*2*pi/3)
                root = RigidConnection('root%d' % (i+1),
                                       offset=dot(R, [0, 0, rtlen]),
                                       rotation=dot(R, Ryx))
                blade = ModalElement('blade%d' % (i+1), self.blade_modes)
                shaft.add_leaf(root)
                root.add_leaf(blade)
        else:
            rotor = RigidBody('rotor', s['rotor']['mass'],
                              np.diag(s['rotor']['inertia']))
            shaft.add_leaf(rotor)

        # Build system
        self.system = System(free_joint)

        # Constrain missing DOFs -- tower torsion & extension not complete
        if self.tower_modes:
            self.system.prescribe(tower, vel=0, part=[0, 3])

    def set_rigid(self, what):
        if what in ('tower', 'all'):
            self.system.prescribe(self.system.elements['tower'], vel=0)
        if what in ('rotor', 'all'):
            for i in range(3):
                self.system.prescribe(self.system.elements['blade%d' % (i+1)],
                                      vel=0)

    def set_flexible(self, what):
        if what in ('tower', 'all'):
            self.system.free(self.system.elements['tower'])
            # Constrain missing DOFs -- tower torsion & extension not complete
            self.system.prescribe(self.system.elements['tower'], vel=0,
                                  parts=[0, 3])
        if what in ('rotor', 'all'):
            for i in range(3):
                self.system.free(self.system.elements['blade%d' % (i+1)])

    def set_shaft_lock(self, locked):
        if locked:
            self.system.prescribe(self.system.elements['shaft'], vel=0)
        else:
            self.system.free(self.system.elements['shaft'])

    def linearised_system(self, z0=None, zd0=None, mbc=False,
                          perturbation=None):

        # Linearise the system about the given operating point
        self.system.update_kinematics()
        linsys = LinearisedSystem.from_system(self.system, z0=z0, zd0=zd0,
                                              perturbation=perturbation)

        # Apply multi-blade coordinate transform if needed
        if mbc:
            iazimuth = self.system.qd.dofs.subset.index(
                self.system.elements['shaft']._istrain[0])
            iblades = [
                [self.system.qd.dofs.subset.index(s)
                 for s in self.system.elements['blade%d' % (i+1)]._istrain]
                for i in range(3)
            ]
            if mbc == 2:
                linsys = linsys.multiblade_transform2(iazimuth, iblades)
            else:
                linsys = linsys.multiblade_transform(iazimuth, iblades)

        return linsys

    def linearised_matrices(self, *args, **kwargs):
        linsys = self.linearised_system(*args, **kwargs)
        return linsys.M, linsys.C, linsys.K

    def describe_states(self):
        q = self.system.q
        states = [(q.owners[j], i) for i, j in enumerate(q.dofs.subset)]
        for owner, owner_states in itertools.groupby(states, lambda x: x[0]):
            numbers = [x[1] for x in owner_states]
            print '{:2}-{:2}: {}'.format(numbers[0], numbers[-1], owner)
