# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

def plot_6x6_transfer_function(w, H, H2=None):
    _, ax = plt.subplots(2, sharex=True, figsize=(6,8))
    ax[0].set_color_cycle(['b', 'g', 'r', 'c', 'm', 'y', 'b', 'm'])
    ax[0].plot(w, np.diagonal(20*np.log10(np.abs(H)), axis1=-2, axis2=-1))
    ax[0].plot(w, 20*np.log10(np.abs(H[:,[0,1],[4,3]])), '--')
    if H2 is not None:
        ax[0].plot(w, np.diagonal(20*np.log10(np.abs(H2)), axis1=-2, axis2=-1), alpha=0.4)
        ax[0].plot(w, 20*np.log10(np.abs(H2[:,[0,1],[4,3]])), '--', alpha=0.4)
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(-200, -30)
    ax[0].legend(range(1,7), ncol=2)
    ax[0].set_title(u'— Diagonal TFs   - - pitch/surge coupling')

    ax[1].plot(w, np.diagonal(np.angle(H), axis1=1, axis2=2))
    if H2 is not None:
        ax[1].plot(w, np.diagonal(np.angle(H2), axis1=1, axis2=2), alpha=0.2)

def plot_6x1_transfer_function(w, H, H2=None):
    _, ax = plt.subplots(2, 2, figsize=(7,7), sharex=True, sharey='row')
    ax[0,0].set_xlim(0,1)
    ax[0,1].set_ylim(-20,20)
    for a in ax.flat:
        a.grid()
        a.set_color_cycle(['b','g','r'])

    ax[0,0].plot(w, 10*np.log10(abs(H[:,0:3])))
    ax[1,0].plot(w, np.angle(H[:,0:3], True))
    if H2 is not None:
        ax[0,0].plot(w, 10*np.log10(abs(H2[:,0:3])), alpha=0.4)
        ax[1,0].plot(w, np.angle(H2[:,0:3], True), alpha=0.4)
    ax[0,0].legend(['Surge', 'Sway', 'Heave'])
    ax[0,0].set_title('Translations [m]')
    ax[0,0].set_ylabel('Magnitude [dB]')
    ax[1,0].set_ylabel('Phase [deg]')

    ax[1,0].set_yticks([-180,-90,0,90,180])
    ax[1,0].set_ylim(-200,200)

    ax[0,1].plot(w, 10*np.log10(180/np.pi*abs(H[:,3:6])))
    ax[1,1].plot(w, np.angle(H[:,3:6], True))
    if H2 is not None:
        ax[0,1].plot(w, 10*np.log10(180/np.pi*abs(H2[:,3:6])), alpha=0.4)
        ax[1,1].plot(w, np.angle(H2[:,3:6], True), alpha=0.4)
    ax[0,1].legend(['Roll', 'Pitch', 'Yaw'])
    ax[0,1].set_title('Rotations [deg]')
    ax[1,0].set_xlabel('Frequency [rad/s]')
    ax[1,1].set_xlabel('Frequency [rad/s]')
