import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Set physical constants equal to one
hbar = 1

# Set kinematics
global m
global L
m = 1   # Particle mass
L = 10  # Half-width of potential
T = 50 # Max time span


def infinite_square_well_energy(n: int) -> float:
    '''
    Compute energy of the infinite square well wavefunction for a given quantum number n.

    Arguments:
        n : int. Principal quantum number.

    Returns:
        energy : float. Quantised energy corresponding to level n. 
    '''

    energy = n**2 * np.pi**2 * hbar**2 / (2 * m * L**2)

    return energy

def infinite_square_well_wavefunction(x: float, n: int) -> float:
    '''
    Defines the wave function for the infinite square well on [-L, L].

    Arguments:
        x : float. Position.
        n : int. Principal quantum number.

    Returns: 
        psi : float. Corresponding wavefunction.
    '''

    # Normalisation constant
    N = np.sqrt(2/L)
    
    psi = N * np.sin(n * np.pi/( L) * x)

    return psi

def energy_dependent_isw_wavefunction(x: float, E: float) -> float:
    '''
    Computes the infinite square well wavefunction with a variable energy input.

    Arguments:
        x : float. Position.
        E : float. Energy.

    Returns:
        psi : float. Wavefunction. 
    '''

    psi = np.sqrt(2/L) * np.sin(np.sqrt(2 * m * E)/hbar * x)

    return psi

def interactive_energy_eigstate_plot(E: float):
    '''
    Interactive plotting feature - scroll through energies to find eigenstates.

    Arguments:
        E : float. Energy.

    Returns:
        none.
    '''
    # Input range
    x = np.linspace(0, L, 200)

    # Change line colour when an energy eigenstate is found
    n_allowed = np.arange(1,13)
    n = np.sqrt((2 * m * L**2)*E/np.pi**2)
    if (np.isclose(n,n_allowed, atol = 1e-1)).any():
        color = 'g'
    else:
        color = 'b'

    # Create the interactive plot
    fig, ax = plt.subplots()
    line, = ax.plot(x, energy_dependent_isw_wavefunction(x, E), color = color)
    ax.axhline(0, ls = '--', c = 'k')
    ax.axvline(0, c = 'k')
    ax.axvline(L, c = 'k')
    ax.set_xlim(left = - 1, right = L + 1)
    ax.set_ylim(-0.5,0.5)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\psi(x)$')
    plt.show()

def simple_time_dependence(t: float, E: float) -> complex:
    '''
    Time-dependence of wavefunction for time-independent potentials.

    Arguments:
        t : float. Time.
        E : float. Energy.

    Returns:
        phi : complex. Time-dependence of wavefunction.
    '''

    phi = np.exp(-1j * E * t / hbar)

    return phi

def main():
    x = np.linspace(-L, L, 150)
    psi = infinite_square_well_wavefunction(x, 1)

    plt.figure()
    plt.plot(x, psi)
    plt.show()

if __name__ == "__main__":
    main()