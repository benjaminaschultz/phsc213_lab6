#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt

m0 = 1.67e-27
T = 5000000
kb = 1.38e-23
c = 3e8
f0 = 1
h = 6.62e-34
h_bar = h * 0.5 / np.pi

r0 = 1.13e-10
mu = m0 * (1 * 1 / (1 + 1))
k0 = 1.85e3


def spectral_line(fs, f0, width):
    Is = np.exp(-(fs-f0)*(fs-f0) / (2.0 * width * width))
    return Is / (np.sqrt(2 * np.pi))

class Molecule():
    
    def __init__(self, mu, r0, k0):
        self.mu = mu
        self.I = mu * r0 ** 2
        self.f0 = np.sqrt(k0/self.mu) / (2 * np.pi)
    
    def interesting_evs(self, sample_width=0.1, sample_size=1000):
        delta_e0 = h * self.f0 / 1.6e-19
        return np.linspace(delta_e0 - sample_width, delta_e0 + sample_width, sample_size)

    def spectrum(self, temperature, evs):
        delta_e0 = h * self.f0 / 1.6e-19
        
        fs = evs * 1.6e-19 / h
        
        intensities = np.zeros(fs.shape)
        
        
        for j0 in np.arange(0, 30):
            for delta_j in [-1, 1]:
                j1 = j0 + delta_j
                if j1 > -1:

                    ej0 = h_bar * h_bar / (2 * self.I) * (j0 * (j0 + 1)) 
                    ej1 = h_bar * h_bar / (2 * self.I) * (j1 * (j1 + 1))
                    
                                        
                    multiplicity = max((2 * j0 + 1), (2 * j1 + 1))
                    boltzmann = max(
                        np.exp(-ej0 /(kb * temperature)),
                        np.exp(-ej1 /(kb * temperature))
                    )
                    
                    if delta_j == -1:
                        delta_e = (h * self.f0 + h_bar * h_bar / self.I * (j0 + 1)) / 1.6e-19
                        
                    elif delta_j == 1:
                        delta_e = (h * self.f0 - h_bar * h_bar / self.I * (j0)) / 1.6e-19
                    
                    f0 = delta_e * 1.6e-19 / h
                    intensities += multiplicity * boltzmann * spectral_line(fs, f0, width=3e-4*f0)

        return np.sqrt(intensities)


class Atom():
    
    def __init__(self, Z_eff=1):
        
        m0 = 1.67e-27
        self.z = Z_eff
        self.e0 = -13.6 * Z_eff * Z_eff
    
    def interesting_evs(self, sample_width=0.15, sample_size=1000):
        ranges = list()
        for n0 in range(1, 4):
            ev0 = self.e0/(n0*n0)
            for n1 in range(n0+1, 15):
                ev1 = self.e0/(n1 * n1)
                
                ranges.append(np.linspace(ev1-ev0 - sample_width, ev1-ev0 + sample_width, sample_size))
        
        return np.sort(np.concatenate(ranges))
    
    def spectrum(self, temperature, evs, width=1e-4):
        


        max_n = 15
        max_l = max_n - 1
        max_m = max_l * 2 + 1
        ns = np.arange(1, max_n + 1)
        energies = np.zeros((max_n + 1, max_l + 1, max_m))
        
        for n in ns:
            for l in np.arange(0, n):
                offset_m = l
                for m in np.arange(-l, l+1):
                    energies[n, l, m + offset_m] = self.e0 / (n * n)
                    
        fs = evs * 1.6e-19 / h
        intensities = np.zeros(fs.shape)
        # iterate possible transitions
        for n0 in ns:
            for l0 in np.arange(0, n0):
                for m0 in np.arange(-l0, l0+1):
                    offset_m0 = l0
                    for n1 in [n for n in ns if n > n0]:
                        if n1 > 10 and n0 > 10:
                            continue
                        for delta_j in [-1, 1]:
                            l1 = l0 + delta_j
                            if not (l1 < n1 and l1 >=0):
                                continue
                            for delta_m in [-1, 0, 1]:
                                m1 = m0 + delta_m
                                offset_m1 = l1
                                if m1 <= l1 and m1 >= -l:
                                    
                                    e0 = energies[n0, l0, m0 + offset_m0] 
                                    e1 = energies[n1, l1, m1 + offset_m1]
                                    deltaE = e1 - e0
                                    boltzmann = 1 - np.exp(e1 * 1.6e-19 /(kb * temperature))

                                    f0 = deltaE * 1.6e-19/ h
                                  
                                    intensities += spectral_line(fs, f0, width = width* f0) * boltzmann
        return np.power(intensities, 0.25)

atom0 = Atom(Z_eff=1.4)
atom1 = Atom(Z_eff=1.2)
mol0 = Molecule(6.8*m0, r0/5., k0)
mol1 = Molecule(6.8*m0, r0/4., k0*1.5)
evs = np.sort(
    np.concatenate([
           atom0.interesting_evs(),
           atom1.interesting_evs(),
           mol0.interesting_evs(),
           mol1.interesting_evs(),
           np.linspace(0, 1, 1000),
           np.linspace(0, 30, 10000)
    ])
)

atom0_spectrum = atom0.spectrum(20, evs)
atom1_spectrum = atom1.spectrum(20, evs)

mol0_spectrum = mol0.spectrum(300, evs)
mol1_spectrum = mol1.spectrum(300, evs)

print('done!')
noise = np.abs(np.random.normal(loc=0, scale=0.2, size=len(evs)))


def compute_mixture(c):
    return (c * mol0_spectrum + 2 * c * atom0_spectrum
        + 2 * atom1_spectrum + mol1_spectrum)


def fit_to_mixture_spectrum(observed_data):
    def compare(c):
        return np.sum(np.power(compute_mixture(c) - observed_data, 2))
    return minimize_scalar(compare).x


observed_data = compute_mixture(3.52) + noise


def plot_atom_A():
    plt.figure()
    plt.plot(evs, atom0_spectrum, '-k', label='A',  alpha=0.5)
    plt.xlabel('eV')
    plt.ylabel('Intensity')
    plt.title('A atomic spectrum')
    plt.legend()



def plot_atom_B():
    plt.figure()
    plt.plot(evs, atom1_spectrum, '-k', label='B',  alpha=0.5)
    plt.xlabel('eV')
    plt.ylabel('Intensity')
    plt.title('B Atomic spectrum')
    plt.legend()



def plot_molecule_A2():
    plt.figure()
    plt.plot(evs, mol0_spectrum, '-k', label='A',  alpha=0.5)
    plt.xlabel('eV')
    plt.ylabel('Intensity')
    plt.title('$A_2$ molecular spectrum')
    plt.legend()


def plot_molecule_B2():
    plt.figure()
    plt.plot(evs, mol1_spectrum, '-k', label='$B_2$',  alpha=0.5)
    plt.xlabel('eV')
    plt.ylabel('Intensity')
    plt.title('$B_2$ molecular spectrum')
    plt.legend()
