import numpy as np
from numpy import newaxis, dot
from scipy import integrate

from whales.stats_helpers import binorm_expectation
from whales.utils import skew, output_variance

g = 9.81

class ViscousDragModel(object):
    """Represent and linearise viscous drag on elements"""
    def __init__(self, config):
        self.Cd = config['drag coefficient']
        if len(config['members']) != 1:
            raise NotImplementedError("not generalised for multiple members yet")
        m = config['members'][0]
        length = np.linalg.norm(np.array(m['end1']) - np.array(m['end2']))
        nvec = (np.array(m['end2']) - np.array(m['end1'])) / length
        n_strips = int(np.ceil(length / m['strip width']))
        strip_width = length / n_strips
        s_centres = np.array([(i+0.5) * strip_width for i in range(n_strips)])
        self.element_lengths = strip_width * np.ones(n_strips)
        self.element_centres = np.asarray(m['end1']) + s_centres[:,np.newaxis] * nvec[np.newaxis,:]

        if np.isscalar(m['diameter']):
            self.element_diameters = m['diameter'] * np.ones(n_strips)
        else:
            diam_data = np.asarray(m['diameter'])
            self.element_diameters = np.interp(s_centres, diam_data[0], diam_data[1])

        # Element axes -- for now hard coded XXX
        # local Z axis is longitudinal; write as transformation matrix R so x=RX
        self.element_axes = np.array([np.eye(3) for i in range(n_strips)])

    def wave_velocity_transfer_function(self, w):
        """Return the transfer functions from wave elevation to fluid velocity
        Shape: (frequency, element, xyz)
        """
        k = w**2 / g
        x, _, z = self.element_centres.T
        # Transfer matrix, shape (freq, element, xyz)
        H_uf = (
            w[:,newaxis,newaxis]
            * np.exp(-1j * k[:,newaxis,newaxis] * x[newaxis,:,newaxis])
            * np.exp(k[:,newaxis,newaxis] * z[newaxis,:,newaxis])
            * np.array([1, 0, 1j])[newaxis,newaxis,:])
        return H_uf

    def structural_velocity_transfer_function(self, w, H):
        """Return the transfer functions from wave elevation to structure velocity
        H: array of transfer functions (RAOs)
        Returns: array (frequency, element, xyz)
        """
        H_us = np.zeros((len(w), len(self.element_diameters), 3), dtype=np.complex)
        for iel in range(H_us.shape[1]):
            xs = skew(self.element_centres[iel])
            for j in range(H_us.shape[0]):
                H_us[j, iel, :] = (1j*w[j] * (H[j,:3] - dot(xs, H[j,3:])))
        return H_us

    def local_relative_velocity_transfer_functions(self, w, H):
        """Return the transfer functions for relative flow velocity in the
        two locally-perpendicular directions (from wave elevation)
        Ref: Langley1984
        """
        H_rel = (self.wave_velocity_transfer_function(w) -
                 self.structural_velocity_transfer_function(w, H))

        Hi = self.resolve_perpendicular_to_elements(H_rel)
        return Hi

    def local_linearised_drag_force(self, wc, wv_sigma):
        """Calculate coefficients A & b so F_local = A w_v + b.
        w_v is the local perpendicular varying relative velocity.

         - wc is the steady perpendicular (local_ current velocity
         - wv_sigma are the standard deviations of the varying perpendicular
           velocities

        b is given by
        $$ \boldsymbol{b} = E[\boldsymbol{w} |\boldsymbol{w}|] $$

        A is given by
        $$ \boldsymbol{A} = \boldsymbol{B} \boldsymbol{D}^{-1} $$
        where
        $$ \boldsymbol{B} = E[ |\boldsymbol{w}| \boldsymbol{w}
        \boldsymbol{w}_v^{\mathrm{T}}]; \quad
        \boldsymbol{D} = E[\boldsymbol{w}_v \boldsymbol{w}_v^{\mathrm{T}}] $$
        """

        # Evaluate the expectations at each station
        A = np.empty((len(wv_sigma), 2, 2))
        b = np.empty((len(wv_sigma), 2))
        for i,wvs in enumerate(wv_sigma):
            wci = wc[i]
            # Define the functions
            def b_func(wv):
                """ w |w| """
                w = wci + wv
                return w * np.sqrt(np.sum(w**2))
            def B_func(wv):
                """ w w_v^T |w| """
                w = wci + wv
                return w[:,newaxis] * wv[newaxis,:] * np.sqrt(np.sum(w**2))
            def D_func(wv):
                """ w_v w_v^T """
                return wv[:,newaxis] * wv[newaxis,:]

            # Evaluate the expectations
            b[i] = binorm_expectation(b_func, wvs)
            B = binorm_expectation(B_func, wvs)
            D = binorm_expectation(D_func, wvs)

            # Calculate the matrix A
            if D[0,0] == 0:
                A[i] = B / D[1,1]
            elif D[1,1] == 0:
                A[i] = B / D[0,0]
            else:
                A[i] = np.dot(B, np.linalg.inv(D))

        return A, b

    def linearised_drag_force(self, w, H_wave, S_wave, vc=None):
        """Calculate coefficients A & b so F = A w_v + b.
        w_v is the local perpendicular varying relative velocity.

         - w are the frequencies used to calculate H_wave and S_wave
         - vc is the steady current velocity in global coordinates.
           It can be given for all elements or else one value for all.
        """

        # Calculate standard deviations of relative velocity
        Hi = self.local_relative_velocity_transfer_functions(w, H_wave)
        wv_sigma = np.sqrt(output_variance(w, Hi, S_wave))

        # Assume current is equal everywhere unless specified otherwise
        # Resolve into local perpendicular velocities
        if vc is None:
            vc = np.zeros(3)
        wc = self.resolve_perpendicular_to_elements(vc)

        # Calculate the local linearised drag force
        A_local, b_local = self.local_linearised_drag_force(wc, wv_sigma)

        # Transform results back to global coordinates
        P = self.element_axes[:,:,0:2] # transformation matrix [n1 n2]
        b = np.einsum('eip,ep->ei', P, b_local)
        A = np.einsum('eip,epq,ejq->eij', P, A_local, P)

        return A, b

    def resolve_perpendicular_to_elements(self, v):
        """Resolve a velocity ``v`` into 2 components normal to each element"""

        v = np.atleast_2d(v) # will broadcast across elements if necessary

        # Unit vectors in global axes -- 2 normal vectors and 1 tangent
        n1,n2,t = self.element_axes.transpose(2, 0, 1) # shape: (element, xyz)

        # Longitudinal velocity component at each element:
        #  vt = (v . t) t
        # e -> element, ij -> global xyz
        vt = np.einsum('...ei,...ei,...ej->...ej', v, t, t)

        # Project velocity in two perpendicular directions
        w1 = np.einsum('...ei,...ei->...e', (v - vt), n1)
        w2 = np.einsum('...ei,...ei->...e', (v - vt), n2)
        w = np.concatenate((w1[..., newaxis], w2[..., newaxis]), axis=-1)
        return w

    def total_drag(self, w, H_wave, S_wave, vc=None):
        r"""Calculate the linearised forces on each element then sum

        The total force and moment together are
        $$ \boldsymbol{F} = \sum \boldsymbol{L}_i \boldsymbol{F}^_i$$
        with
        $$\boldsymbol{L}_i = \begin{bmatrix}
        \boldsymbol{I} \\ \boldsymbol{\tilde{X}_i} \end{bmatrix}$$

        The drag force on each element is
        $$ \boldsymbol{F}_i = C \boldsymbol{A} (
        \boldsymbol{u}_f - \boldsymbol{u}_s ) + C \boldsymbol{b} $$
        where $C = \rho C_d D \mathrm{d}s / 2$ is a constant, and
        $\boldsymbol{A}$ is the transformed linearisation matrix.

        Using the fluid velocity transfer functions $\boldsymbol{H}_{uf \zeta}$,
        $\boldsymbol{u}_f(t) = \boldsymbol{H}_{uf \zeta}(\omega)
        \zeta_a e^{i\omega t}$, so this part goes on the RHS as a force
        $C \boldsymbol{A} \boldsymbol{H}_{u_f \zeta} \zeta_a
        e^{i\omega t} + C \boldsymbol{b}$.

        The structural response is given by
        $$ \boldsymbol{u}_s = i\omega
        \begin{bmatrix} \boldsymbol{I} & -\boldsymbol{\tilde{X}} \end{bmatrix}
        \boldsymbol{\Xi}(\omega) e^{i\omega t} $$
        so the rest of the drag force goes on the LHS as a damping matrix
        $$\boldsymbol{B}_v = C \boldsymbol{A} \boldsymbol{L}^{\mathrm{T}}$$

        This function returns Fvc = sum (C Li bi), the steady part of the force,
        and Fvv[freq] = sum (C Li Ai H_uf), which is the harmonic drag force for
        unit wave amplitude.

        NB this is very slow if w goes right to zero -- why?
        """

        # XXX
        rho = 1025

        H_uf = self.wave_velocity_transfer_function(w)
        A, b = self.linearised_drag_force(w, H_wave, S_wave, vc)

        Bv = np.zeros((6,6))                       # viscous drag damping
        Fvc = np.zeros(6)                          # constant drag force
        Fvv = np.zeros((len(w), 6), dtype=complex) # time-varying drag force
        for iel in range(len(self.element_diameters)):
            # Drag constant for each element
            C = (0.5 * rho * self.Cd *
                 self.element_diameters[iel] * self.element_lengths[iel])

            # Force acts at centre of each element
            x = self.element_centres[iel]
            L = np.r_[ np.eye(3), skew(x) ] # forces and moments about origin

            # Damping effect of viscous drag force
            Bv += C * dot(L, dot(A[iel], L.T))

            # Applied viscous drag force
            Fvc += C * dot(L, b[iel])
            for iw in range(len(w)):
                Fvv[iw] += C * dot(L, dot(A[iel], H_uf[iw, iel, :]))

        return Bv, Fvc, Fvv
