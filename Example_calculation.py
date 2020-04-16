import numpy as np
import matplotlib.pyplot as plt
import sys, os
rootDir = 'C:\\Users\\rsawa\\Documents\\Work_Related\\polarisability-3' 
sys.path.insert(0,rootDir)
import AtomFieldInt_V3 as AF
import AtomFieldInt_Example_functions as AFF
# global constants:
c    = 2.99792458e8  # speed of light in m/s
eps0 = 8.85419e-12   # permittivity of free space in m^-3 kg^-1 s^4 A^2
h    = 6.6260700e-34 # Planck's constant in m^2 kg / s
hbar = 1.0545718e-34 # reduced Planck's constant in m^2 kg / s
a0 = 5.29177e-11     # Bohr radius in m
e = 1.6021766208e-19 # magnitude of the charge on an electron in C
me = 9.10938356e-31  # mass of an electron in kg
kB = 1.38064852e-23  # Boltzmann's constant in m^2 kg s^-2 K^-1
amu = 1.6605390e-27  # atomic mass unit in kg
Eh = me * e**4 /(4. *np.pi *eps0 *hbar)**2  # the Hartree energy
au = e**2 * a0**2 / Eh # atomic unit for polarisability
# note that atomic unit au = 4 pi eps0 a0^3


ATOM  = AF.atom(atm = 'Rb87')
bprop = [1064e-9, 1e-3, 1e-6] # wavelength (in metres),  total power (in Watts),beam_waist (in metres)
bprop1 = [1064e-9, 0e-3, 1e-6] # wavelength (in metres),  total power (in Watts),beam_waist (in metres)
L = 0
J = 1/2.
F = 1
MF = 0
S = AF.dipole(ATOM, (L,J,F,MF), bprop)
S1 = AF.dipole(ATOM, (L,J,F,MF), bprop1)

# wavel1 = np.linspace(780, 800, 4)*1e-9
# print((S.acStarkShift(wavel1))/h*1e-6)
#AFF.compareKien()
ev, evec = S.diagHV(Bfield = 00e-4,u = [0,1,0])
ev1, evec1 = S1.diagHV(Bfield = 00e-4)
print((ev/h*1e-6)-(ev1/h*1e-6))
#print((ev1/h*1e-6))
#print(S.diagH())
# Bfield_min = 0 # In Gauss
# Bfield_max = 500 # In Gauss
# Bfield_int = 1# In Gauss
#S.zeeman_map(Bfield_min,Bfield_max,Bfield_int)
      # AF.getMFStarkShifts(wavelength = 1064e-9, # laser wavelength in m
#                     power = 1e-3,    # laser power in W
#                     beamwaist = 1e-6,      # beam waist in m
#                     ATOM = Rb)

