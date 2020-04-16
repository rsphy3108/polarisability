"""Stefan Spence 13.11.18
17.01.18 -- Run to launch UI to calculate stark shifts in Rb/Cs

Version 3: the calculations are in agreement with Arora 2007 when the hyperfine
splitting can be ignored. However, this version assumes that when hyperfine 
splitting is relevant, it is much larger than the stark shift, so that the
stark shift hamiltonian is diagonal in the hyperfine basis (no state mixing).
This assumption doesn't hold for the excited states.

Simulation of atoms in an optical tweezer.
1) Formulate the equations for Gaussian beam propagation.
2) Look at the dipole interaction as a function of laser wavelength and 
spatial position
3) calculate the polarisability for a given state at a given wavelength

14.11.18 add in dipole potential
calculate the polarisability and compare the 2-level model to including other transitions

19.11.18 extend polarisability function
now it allows several laser wavelengths and several resonant transitions 
added Boltzmann's constant to global variables to convert Joules to Kelvin

20.11.18
make the polarisability function work for multiple or individual wavelengths
correct the denominator in the polarisability function from Delta^2 - Gamma^2
to Delta^2 + Gamma^2

13.12.18
Allow the dipole.polarisability() to take wavelength as an argument

18.12.18
The previous polarisability results did not agree with literature. Since the
stark shift is the priority, load the polarisability from provided data.
Also must include vector and tensor polarizabilities
see F. L. Kien et al, Eur. Phys. J. D, 67, 92 (2013)

21.12.18
Some papers include the Stark shift for hyperfine states (Kien 2013), whereas
others use just the fine structure (see B. Arora et al, Phys. Rev. A 76, 052509 
(2007))
So we will incorporate functions for both of them.

02.01.19
Use arc (see https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/ ) 
to get the data for dipole matrix elements and transition properties
(note that arc gets all its Rb, Cs literature values from Safronova papers:
Safronova et al, PRA 60, 6 (1999)
Safronova et al, PRA 69, 022509 (2004)

07.01.19
Add in functions to calculate the polarisability
 - when the hyperfine transitions are important (not finished - needs dipole
 matrix elements for hyperfine transitions): polarisability()
 - when hyperfine transitions can be ignored: polarisabilityJ()
Arora 2007 does include hyperfine splittings in a separate equations, so make
one acStarkShift() function where hyperfine interactions can be toggled

08.01.19
Remove the duplicate starkshift/polarisability functions

14.01.19
Give state labels (n,l,j) to the transition data

15.01.19
Correct the polarisability formula (had w - w0 instead of w0 - w)
Note: the arc data doesn't have linewidths for transitions
Since they're usually quite small this usually doesn't make too much of a 
difference [Archived this version]

16.01.19
Remove functions for loading polarisability data from other papers
Store transition data in dat files so that importing arc is unnecessary

17.01.19
explicitly state that the denominator in fractions are floats, otherwise there
is integer division 

23.01.19 
write a function to match the figures for polarisability from Arora 2007
Correct a factor of 1/2 in the polarisability formula to match Arora 2007

29.01.19
When the hyperfine boolean is True, use the formula from Kien 2013

04.02.19
use Arora 2007 for hyperfine 

20.03.19
Also print the polarisability components in getStarkShift()

27.03.19
When looking at excited states with several possible mj values, average
over the possible mj values.

26.04.19
Add in a function to calculate the scattering rate at a given wavelength

20.05.19
Function to get Stark shift of MF states for Rb or Cs on cooling/repump transition

08.07.19
include the vector polarisability in stark shift calculations

23.11.19
Introduce Potassium 41

16.03.20
Replace wigner functions with ones from sympy
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from math import factorial 
from matplotlib.ticker import AutoLocator
from sympy.physics.wigner import wigner_6j, wigner_3j, clebsch_gordan
import AtomFieldInt_Example_functions as AFF

# see https://docs.sympy.org/latest/modules/physics/wigner.html for documentation
# Memoized wigner functions

# Generic memoization class.
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]
    
@Memoize
def wigner3j(*args):
    return float(wigner_3j(*args))

@Memoize
def wigner6j(*args):
    return float(wigner_6j(*args))

# Memoized Clebsch-Gordon function
@Memoize
def clebschgordan(*args):
    return  float(clebsch_gordan(*args))
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
bohr_magneton = 9.274014e-24
# note that atomic unit au = 4 pi eps0 a0^3

#####################
    
    
class atom:
    """Properties of an atom: 
    
    The transitions follow the order:
    S1/2 -> nP1/2, nP3/2
    P1/2 -> nS1/2. nD3/2
    P3/2 -> nS1/2, nD3/2, nD5/2
    
    D0:  Dipole matrix elements (C m)
    nlj: quantum numbers of the states (n, l, j)
    rw:  resonant wavelength (m) of transitions 
    w0:  resonant frequency (rad/s) of transitions 
    lw:  natural linewidth (rad/s) of transitions 
    """
    def __init__(self, atm = 'None'):
        self.atm = atm
        
        if self.atm == 'None':
            raise SyntaxError('Please enter the atom to be used. Available options are Cs133, Rb87, K41.')
            sys.exit(1)

        
        if atm == 'Cs133':
            ######### atomic properties for Cs-133:  ##########
            # file contains columns: n, l, j, dipole matrix element, wavelength, linewidth
            # for the 6S1/2 state:
            S1_2 = np.loadtxt(r'.\TransitionData\CsS1_2.dat', delimiter=',', skiprows=1)
                    
            # for the 6P1/2 state:
            P1_2 = np.loadtxt(r'.\TransitionData\CsP1_2.dat', delimiter=',', skiprows=1)
            
            # for the 6P3/2 state:
            P3_2 = np.loadtxt(r'.\TransitionData\CsP3_2.dat', delimiter=',', skiprows=1)
            self.D0S = S1_2[:,3]  # dipole matrix elements from S1/2 state
            self.D0P1 = P1_2[:,3] # dipole matrix elements from P1/2 state
            self.D0P3 = P3_2[:,3] # dipole matrix elements from P3/2 state
            self.nljS = S1_2[:,:3] # (n,l,j) quantum numbers for transitions
            self.nljP1 = P1_2[:,:3]
            self.nljP3 = P3_2[:,:3]
            self.rwS = S1_2[:,4]   # resonant wavelengths from S1/2 state (m)
            self.rwP1 = P1_2[:,4] # resonant wavelengths from P1/2 state (m)
            self.rwP3 = P3_2[:,4]  # resonant wavelengths from P3/2 state (m)
            self.w0S = 2*np.pi*c / S1_2[:,4]# resonant frequency (rad/s)
            self.w0P1 = 2*np.pi*c / P1_2[:,4]
            self.w0P3 = 2*np.pi*c / P3_2[:,4]
            self.lwS = S1_2[:,5]  # natural linewidth from S1/2 (rad/s)
            self.lwP1 = P1_2[:,5]  # natural linewidth from P1/2 (rad/s)
            self.lwP3 = P3_2[:,5] # natural linewidth from P3/2 (rad/s)
            self.m  =  133*amu
            self.I  = 7/2
            self.X  = self.atm
            # Values from steck https://steck.us/alkalidata/
            self.Ahfs_S = 2.2981579425*1e9         # Magnetic Dipole Constant for S1/2 state (Hz)
            self.Ahfs_P1 = 291.920*1e6         # Magnetic Dipole Constant for P1/2 state (Hz)
            self.Ahfs_P3 = 50.275*1e6         # Magnetic Dipole Constant for P3/2 state (Hz)
            self.Bhfs_P3 = -0.53*1e6        #Electric Quadrupole Constant for for P3/2 state (Hz)
            self.gS = 2.0023193043737       #Electron spin g-factor
            self.gL = 0.99999587          #Electron orbital g-factor
            self.gJS = 2.00254032         #Fine structure Lande g-factor for S1/2
            self.gI = -0.00039885395        #Nuclear spin g-factor
            self.gJP1 = 0.66590         #Fine structure Lande g-factor for P1/2
            self.gJP3 = 1.3340         #Fine structure Lande g-factor for P3/2
            
    
        if atm == 'Rb87':
            ######### atomic properties for Rb-87:  ###########
            # file contains columns: n, l, j, dipole matrix element, wavelength, linewidth
            # for the 6S1/2 state:
            S1_2 = np.loadtxt(r'.\TransitionData\RbS1_2.dat', delimiter=',', skiprows=1)
                    
            # for the 6P1/2 state:
            P1_2 = np.loadtxt(r'.\TransitionData\RbP1_2.dat', delimiter=',', skiprows=1)
            
            # for the 6P3/2 state:
            P3_2 = np.loadtxt(r'.\TransitionData\RbP3_2.dat', delimiter=',', skiprows=1)
            
            self.D0S = S1_2[:,3]  # dipole matrix elements from S1/2 state
            self.D0P1 = P1_2[:,3] # dipole matrix elements from P1/2 state
            self.D0P3 = P3_2[:,3] # dipole matrix elements from P3/2 state
            self.nljS = S1_2[:,:3] # (n,l,j) quantum numbers for transitions
            self.nljP1 = P1_2[:,:3]
            self.nljP3 = P3_2[:,:3]
            self.rwS = S1_2[:,4]   # resonant wavelengths from S1/2 state (m)
            self.rwP1 = P1_2[:,4] # resonant wavelengths from P1/2 state (m)
            self.rwP3 = P3_2[:,4]  # resonant wavelengths from P3/2 state (m)
            self.w0S = 2*np.pi*c / S1_2[:,4]# resonant frequency (rad/s)
            self.w0P1 = 2*np.pi*c / P1_2[:,4]
            self.w0P3 = 2*np.pi*c / P3_2[:,4]
            self.lwS = S1_2[:,5]  # natural linewidth from S1/2 (rad/s)
            self.lwP1 = P1_2[:,5]  # natural linewidth from P1/2 (rad/s)
            self.lwP3 = P3_2[:,5] # natural linewidth from P3/2 (rad/s)
            self.m  =  87*amu
            self.I  = 3/2
            self.X  = self.atm
            # Values from steck https://steck.us/alkalidata/
            self.Ahfs_S = 3.4173430545215*1e9         # Magnetic Dipole Constant for S1/2 state (Hz)
            self.Ahfs_P1 = 408.328*1e6         # Magnetic Dipole Constant for P1/2 state (Hz)
            self.Ahfs_P3 = 84.7185*1e6         # Magnetic Dipole Constant for P3/2 state (Hz)
            self.Bhfs_P3 = 12.4965*1e6        #Electric Quadrupole Constant for for P3/2 state (Hz) 
            self.gS = 2.0023193043737       #Electron spin g-factor
            self.gL = 0.99999369          #Electron orbital g-factor
            self.gI = -0.0009951414        #Nuclear spin g-factor
            self.gJS = 2.00233113         #Fine structure Lande g-factor for S1/2
            self.gJP1 = 0.666         #Fine structure Lande g-factor for P1/2
            self.gJP3 = 1.3362         #Fine structure Lande g-factor for P3/2
            
        if atm == 'K41':
        ######### atomic properties for K-41:  ###########
            # file contains columns: n, l, j, dipole matrix element, wavelength, linewidth
            # for the 4S1/2 state:
            S1_2 = np.loadtxt(r'.\TransitionData\KS1_2.dat', delimiter=',', skiprows=1)
                
            # for the 4P1/2 state:
            P1_2 = np.loadtxt(r'.\TransitionData\KP1_2.dat', delimiter=',', skiprows=1)
        
            # for the 4P3/2 state:
            P3_2 = np.loadtxt(r'.\TransitionData\KP3_2.dat', delimiter=',', skiprows=1)
            # for the 4S1/2 state:
            self.D0S = S1_2[:,3]  # dipole matrix elements from S1/2 state
            self.D0P1 = P1_2[:,3] # dipole matrix elements from P1/2 state
            self.D0P3 = P3_2[:,3] # dipole matrix elements from P3/2 state
            self.nljS = S1_2[:,:3] # (n,l,j) quantum numbers for transitions
            self.nljP1 = P1_2[:,:3]
            self.nljP3 = P3_2[:,:3]
            self.rwS = S1_2[:,4]   # resonant wavelengths from S1/2 state (m)
            self.rwP1 = P1_2[:,4] # resonant wavelengths from P1/2 state (m)
            self.rwP3 = P3_2[:,4]  # resonant wavelengths from P3/2 state (m)
            self.w0S = 2*np.pi*c / S1_2[:,4]# resonant frequency (rad/s)
            self.w0P1 = 2*np.pi*c / P1_2[:,4]
            self.w0P3 = 2*np.pi*c / P3_2[:,4]
            self.lwS = S1_2[:,5]  # natural linewidth from S1/2 (rad/s)
            self.lwP1 = P1_2[:,5]  # natural linewidth from P1/2 (rad/s)
            self.lwP3 = P3_2[:,5] # natural linewidth from P3/2 (rad/s)
            self.m  =  41*amu
            self.I  = 3/2
            self.X  = self.atm
            # Values from http://www.tobiastiecke.nl/archive/PotassiumProperties.pdf
            self.Ahfs_S = 127.0069352*1e6         # Magnetic Dipole Constant for S1/2 state (Hz)
            self.Ahfs_P1 = 15.245*1e6         # Magnetic Dipole Constant for P1/2 state (Hz)
            self.Ahfs_P3 = 3.363*1e6         # Magnetic Dipole Constant for P3/2 state (Hz)
            self.Bhfs_P3 = 3.351*1e6        #Electric Quadrupole Constant for for P3/2 state (Hz)
            self.gS = 2.0023193043622       #Electron spin g-factor
            self.gL = 1-(9.10938356*1e-31)/self.m          #Electron orbital g-factor
            self.gI = -0.00007790600        #Nuclear spin g-factor
            self.gJS = 2.00229421         #Fine structure Lande g-factor for S1/2
            self.gJP1 = 2/3         #Fine structure Lande g-factor for P1/2
            self.gJP3 = 4/3         #Fine structure Lande g-factor for P3/2
            
        if atm == 'Na23':
            ######### atomic properties for Na-23:  ###########
            # file contains columns: n, l, j, dipole matrix element, wavelength, linewidth
            # for the 3S1/2 state:
            S1_2 = np.loadtxt(r'.\TransitionData\NaS1_2.dat', delimiter=',', skiprows=1)
            # for the 3P1/2 state:
            P1_2 = np.loadtxt(r'.\TransitionData\NaP1_2.dat', delimiter=',', skiprows=1)
            # for the 3P3/2 state:
            P3_2 = np.loadtxt(r'.\TransitionData\NaP3_2.dat', delimiter=',', skiprows=1)
            # for the 4S1/2 state:
            self.D0S = S1_2[:,3]  # dipole matrix elements from S1/2 state
            self.D0P1 = P1_2[:,3] # dipole matrix elements from P1/2 state
            self.D0P3 = P3_2[:,3] # dipole matrix elements from P3/2 state
            self.nljS = S1_2[:,:3] # (n,l,j) quantum numbers for transitions
            self.nljP1 = P1_2[:,:3]
            self.nljP3 = P3_2[:,:3]
            self.rwS = S1_2[:,4]   # resonant wavelengths from S1/2 state (m)
            self.rwP1 = P1_2[:,4] # resonant wavelengths from P1/2 state (m)
            self.rwP3 = P3_2[:,4]  # resonant wavelengths from P3/2 state (m)
            self.w0S = 2*np.pi*c / S1_2[:,4]# resonant frequency (rad/s)
            self.w0P1 = 2*np.pi*c / P1_2[:,4]
            self.w0P3 = 2*np.pi*c / P3_2[:,4]
            self.lwS = S1_2[:,5]  # natural linewidth from S1/2 (rad/s)
            self.lwP1 = P1_2[:,5]  # natural linewidth from P1/2 (rad/s)
            self.lwP3 = P3_2[:,5] # natural linewidth from P3/2 (rad/s)
            self.m  =  23*amu
            self.I  = 3/2
            self.X  = self.atm
            # Values from https://steck.us/alkalidata/sodiumnumbers.1.6.pdf
            self.Ahfs_S = 885.8130644*1e6         # Magnetic Dipole Constant for S1/2 state (Hz)
            self.Ahfs_P1 = 94.44*1e6         # Magnetic Dipole Constant for P1/2 state (Hz)
            self.Ahfs_P3 = 18.534*1e6         # Magnetic Dipole Constant for P3/2 state (Hz)
            self.Bhfs_P3 = 2.724*1e6        #Electric Quadrupole Constant for for P3/2 state (Hz)
            self.gS = 2.0023193043737       #Electron spin g-factor
            self.gL = 0.9999761          #Electron orbital g-factor
            self.gI = -0.0008046108       #Nuclear spin g-factor
            self.gJS = 2.0022960         #Fine structure Lande g-factor for S1/2
            self.gJP1 = 0.66581         #Fine structure Lande g-factor for P1/2
            self.gJP3 = 1.3342         #Fine structure Lande g-factor for P3/2
            
        if self.atm != 'Cs133' and self.atm != 'Rb87' and self.atm != 'K41'and self.atm != 'Na23':
            raise SyntaxError('Please enter a valid atom. Available options are Cs133, Rb87, K41.')
            sys.exit(1)
        

#######################


class Gauss:
    """Properties and associated equations of a Gaussian beam"""
    def __init__(self, wavelength, power, beam_waist, polarization=(0,0,1)):
        self.lam = wavelength    # wavelength of the laser light (in metres)
        self.P   = power         # total power of the beam (in Watts)
        self.w0  = beam_waist    # the beam waist defines the laser mode (in metres)
        self.I   = 2 * power / np.pi / beam_waist**2 # intensity of beam (in Watts/metre squared)
        self.ehat= polarization  # the direction of polarization (assume linear)
        # note: we will mostly ignore polarization since the induced dipole 
        # moment will be proportional to the direction of the field
        
        # assume that the beam waist is positioned at z0 = 0
        
        # from these properties we can deduce:
        self.zR = np.pi * beam_waist**2 / wavelength # the Rayleigh range
        # average intensity of sinusoidal wave gives the factor of 2
        self.E0 = 2 * np.sqrt(power / eps0 / c / np.pi)/beam_waist  # field amplitude at the origin
        self.k  = 2 * np.pi / wavelength             # the wave vector
        
    def amplitude(self, x, y, z):
        """Calculate the amplitude of the Gaussian beam at a given position
        note that this function will not work if several coordinates are 1D arrays
        instead, loop over the other coordinates so that there is only ever one
        coordinate as an array."""
        rhosq = x**2 + y**2                     # radial coordinate squared    
        q     = z - 1.j * self.zR               # complex beam parameter
        
        # Gaussian beam equation (see Optics f2f Eqn 11.7)
        return self.zR /1.j /q * self.E0 * np.exp(1j * self.k * z) * np.exp(
                                                1j * self.k * rhosq / 2. / q)
        

#######################
        
        
class dipole:
    """Properties and equations of the dipole interaction between atom and field"""
    def __init__(self, ATOM, spin_state, field_properties):
        
        self.L, self.J, self.F, self.MF = spin_state           # spin quantum numbers L, J, F, M_F
        if self.L == 0: #S1/2 state
            self.Ahfs                       = ATOM.Ahfs_S       # magnetic dipole constant
            self.Bhfs                       = 0                 # electric quadrupole constant
            self.states = ATOM.nljS              # (n,l,j) quantum numbers for transitions
            self.omega0 = np.array(ATOM.w0S)    # resonant frequencies (rad/s)
            self.gam    = np.array(ATOM.lwS)      # spontaneous decay rate (s)
            self.D0s    = np.array(ATOM.D0S)   # D0 = -e <a|r|b> for displacement r along the polarization direction
            self.gJ  = ATOM.gJS
        else:
            if self.J == 1/2.:#P1/2 state
                self.Ahfs                       = ATOM.Ahfs_P1      # magnetic dipole constant
                self.Bhfs                       = 0                 # electric quadrupole constant
                self.states = ATOM.nljP1              # (n,l,j) quantum numbers for transitions
                self.omega0 = np.array(ATOM.w0P1)    # resonant frequencies (rad/s)
                self.gam    =  np.array(ATOM.lwP1)      # spontaneous decay rate (s)
                self.D0s    =  np.array(ATOM.D0P1)   # D0 = -e <a|r|b> for displacement r along the polarization direction
                self.gJ  = ATOM.gJP1
            else:           #P3/2 state
                self.Ahfs                       = ATOM.Ahfs_P3       # magnetic dipole constant
                self.Bhfs                       = ATOM.Bhfs_P3       # electric quadrupole constant
                self.states = ATOM.nljP3              # (n,l,j) quantum numbers for transitions
                self.omega0 = np.array(ATOM.w0P3)    # resonant frequencies (rad/s)
                self.gam    = np.array(ATOM.lwP3)      # spontaneous decay rate (s)
                self.D0s    = np.array(ATOM.D0P3)   # D0 = -e <a|r|b> for displacement r along the polarization direction
                self.gJ  = ATOM.gJP3
                
                
        self.m                          = ATOM.m                 # mass of the atom in kg
        self.gI  = ATOM.gI
        self.gS  = ATOM.gS
        self.gL  = ATOM.gL
        self.I                          = ATOM.I         # nuclear spin quantum number I
        self.field = Gauss(*field_properties)                  # combines all properties of the field
        self.X = ATOM.X
        if self.X == 'Cs133':
            self.Isats = np.array([24.981, 11.023]) # saturation intensities for D1, D2 transitions
            self.Dlws = np.array([ATOM.lwS[0], ATOM.lwS[35]]) # linewidths for D1, D2 lines
            self.Drws = np.array([ATOM.rwS[0], ATOM.rwS[35]]) # resonant wavelengths of D1, D2 lines
        elif self.X == 'Rb87':
            self.Isats = np.array([44.84, 25.03])   # saturation intensities for D1, D2 transitions
            self.Dlws = np.array([ATOM.lwS[0], ATOM.lwS[5]]) # linewidths for D1, D2 lines
            self.Drws = np.array([ATOM.rwS[0], ATOM.rwS[5]]) # resonant wavelengths of D1, D2 lines
        
        
        self.omegas = np.array(2*np.pi*c/self.field.lam)# laser frequencies (rad/s)
        
    def scatRate(self, wavel=[], I=[]):
        """Return the scattering rate at a given wavelength and intensity
        Default uses the dipole object's wavelength and intensity
        If wavelength and intensity are supplied, they should be the same length."""
        if np.size(wavel) != 0: 
            omegas = np.array(2*np.pi*c/wavel) # laser frequencies (rad/s)
        else:
            omegas = self.omegas
        if np.size(I) == 0: # use intensity from field
            I = 2 * self.field.P / np.pi / self.field.w0**2 # beam intensity

        Rsc = 0
        for i in range(len(self.Isats)):
            deltas = omegas - 2 * np.pi * c / self.Drws[i] # detuning from D line
            Rsc += self.Dlws[i]/2. * I/self.Isats[i] / (1 + 4*(deltas/self.Dlws[i])**2 + I/self.Isats[i])

        return Rsc
            
    def acStarkShift(self, x=0, y=0, z=0, wavel=[], mj=None, HF=False):
        """Return the potential from the dipole interaction 
        U = -<d>E = -1/2 Re[alpha] E^2
        Then taking the time average of the cos^2(wt) AC field term we get 
        U = -1/4 Re[alpha] E^2"""
        return -self.polarisability(wavel, mj, HF, split=False) /4. *np.abs( 
                            self.field.amplitude(x,y,z) )**2
    
            
    def polarisability(self, wavel=[], mj=None, HF=False, split=False):
        """wavel: wavelength (m) - default is self.field.lam
        mj: used when hyperfine splitting is negligible.
        HF: Boolean - include hyperfine structure
        split: Boolean - False gives total polarisability, True splits into
        scalar, vector, and tensor.
        Return the polarisability as given Arora 2007 (also see Cooper 2018,
        Mitroy 2010, Kein 2013) assuming that J and mj are good quantum 
        numbers when hyperfine splitting can be neglected, or that F and mf are
        good quantum numbers. Assumes linear polarisation so that the vector
        polarisability is zero."""
        if np.size(wavel) != 0:            
            omegas = np.array(2*np.pi*c/wavel) # laser frequencies (rad/s)
        else:
            omegas = self.omegas
         
        # initiate arrays for results
        empty = np.zeros(np.size(omegas))
        aSvals, aVvals, aTvals = empty.copy(), empty.copy(), empty.copy()
        
        for ii in range(np.size(omegas)):
            aS, aV, aT = 0, 0, 0
            #print([self.omega0.shape,self.gam.shape,omegas.shape])
            # loop over final states
            for i in range(len(self.states)):   
                if np.size(omegas) > 1:
                    Ep = hbar*(self.omega0[i] + omegas[ii] + 1j*self.gam[i])
                    Em = hbar*(self.omega0[i] - omegas[ii] - 1j*self.gam[i])
                
                else:
                    Ep = hbar*(self.omega0[i] + omegas + 1j*self.gam[i])
                    Em = hbar*(self.omega0[i] - omegas - 1j*self.gam[i])
                    
                aS += 1/3. /(2.*self.J + 1.) *self.D0s[i]**2 * (1/Ep + 1/Em)
                    
                aV += 0.5*(-1)**(self.J + 2 + self.states[i][2]) * np.sqrt(6*self.J
                    /(self.J + 1.) /(2*self.J + 1.)) * self.D0s[i]**2 * wigner6j(
                    1, 1, 1, self.J, self.states[i][2], self.J) * (1/Em - 1/Ep)
                    
                aT += 2*np.sqrt(5 * self.J * (2*self.J - 1) / 6. /(self.J + 1) /
                        (2*self.J + 1) / (2*self.J + 3)) * (-1)**(self.J + 
                        self.states[i][2]) * wigner6j(self.J, 1, self.states[i][2], 
                        1, self.J, 2) * self.D0s[i]**2 * (1/Ep + 1/Em)
 
            aSvals[ii] = aS.real  # scalar polarisability
            aVvals[ii] = aV.real  # vector polarisability
            aTvals[ii] = aT.real  # tensor polarisability

        # combine polarisabilities
        u = self.field.ehat
        if self.J > 0.5:
            if HF:  # hyperfine splitting is significant
                # from Kien 2013: when stark shift << hfs splitting so there isn't mixing of F levels
                # combine eq 16 and 18 to get the a_nJF in terms of the a_nJ
                # also assume stark shift << Zeeman splitting so we can use |F,MF> states.
                aVvals *= -(-1)**(self.J + self.I + self.F) * np.sqrt(self.F * (2*self.F + 1)
                    *(self.J + 1) *(2*self.J + 1) /self.J /(self.F + 1)) *wigner6j(self.F, 1, self.F, 
                    self.J, self.I, self.J)
                
                # from Arora 2007
                aTvals *= (-1)**(self.I + self.J - self.MF) * (2*self.F + 1
                    ) * np.sqrt((self.J + 1) *(2*self.J + 1) *(2*self.J + 3)
                    /self.J /(2*self.J - 1.)) * wigner3j(self.F, 2, self.F, 
                    self.MF, 0, -self.MF) * wigner6j(self.F, 2, self.F,
                    self.J, self.I, self.J)  
                
                if split:
                    return (aSvals, aVvals, aTvals)
                else:        
                    return aSvals + aTvals
                
            else: # hyperfine splitting is ignored
                # NB: currently ignoring vector polarisability as in Arora 2007
                if split:
                    return (aSvals, aVvals, aTvals)
                else:
                    # return aSvals + aTvals * (3*mj**2 - self.J*(self.J + 1)
                    #     ) / self.J / (2*self.J - 1)
                    # include a general polarisation of light:
                    return aSvals + mj/self.J * np.imag(np.conj(u[0])*u[1]
                    ) * aVvals + (3*abs(u[2])**2 - 1)/2. * (3*mj**2 - 
                    self.J*(self.J + 1)) / self.J / (2*self.J - 1) * aTvals
        else:
            if HF: # there is no tensor polarisability for the J=1/2 state
                aVvals *= -(-1)**(self.J + self.I + self.F) * np.sqrt(self.F * (2*self.F + 1)
                    *(self.J + 1) *(2*self.J + 1) /self.J /(self.F + 1)) *wigner6j(self.F, 1, self.F, 
                    self.J, self.I, self.J)
                if split:
                    return (aSvals, aVvals, aTvals)
                else:
                    return aSvals #+ aVvals
            else:
                if split:
                    return (aSvals, aVvals, aTvals)
                else:
                    if mj == None: # for compatability with old scripts
                        mj = 0
                    return aSvals+ mj/self.J*np.imag(np.conj(u[0])*u[1])*aVvals
          
    def matrix_element(self,f1,m1,f2,m2,q = 0):
        """
        Calculate the gI*I + gJ*J matrix element for the zeeman hamiltonian, using indices as arguments.
        :return:    The value of < f1,m1| gI*I + gJ*J |f2,m2>
        :param q:   Spherical tensor rank of the incoming field, should be in [-1, 0, 1]
        function taken from https://github.com/cyip92/microwave-dressing.git 
        The exact expression in F mF basis can be found in https://arxiv.org/pdf/1309.5775.pdf
        """
        I = self.I
        J = self.J
        g_I = self.gI
        g_J = self.gJ
        if abs(f1 - f2) > 1 or abs(m1 - m2) > 1:   # Selection rules cut out a lot of time
            return 0
        reduced = ((-1) ** (1 + I + J)) * np.sqrt(2*f1 + 1)
        elem_i = ((-1) ** f2) * np.sqrt(I * (I + 1) * (2*I + 1)) * wigner6j(I, J, f1, f2, 1, I)
        elem_j = ((-1) ** f1) * np.sqrt(J * (J + 1) * (2*J + 1)) * wigner6j(J, I, f1, f2, 1, J)
        elem = ((-1) ** q) * reduced * (g_I * elem_i + g_J * elem_j) * clebschgordan(f1, 1, f2, m1, q, m2)
        return elem        
              
    def diagH(self, x = 0, y = 0, z = 0, Bfield = 0):
        
        """Diagonalise the combined Hamiltonian of hyperfine splitting + the ac
        Stark Shift + the zeeman effect. This gives the eigenenergies and eigenstates in the |F,mF>
        basis at a particular wavelength. Currently assuming linear polarisation
        along the z direction as in Arora 2007."""
        wavel = self.field.lam
        omega = 2*np.pi*c/wavel   # laser frequency in rad/s
        
        # |I-J| <= F <= I+J
        Fs = np.arange(abs(int(self.I - self.J)), int(self.I + self.J + 1)) #List of F states
        num_states = sum(2*Fs + 1) #Total number of states.
        
        H = np.zeros((num_states, num_states)) # Initialize combined interaction Hamiltonian
        F_labels = np.concatenate([[F]*(2*F+1) for F in Fs]) #F label of each state
        
        MF_labels = np.concatenate([list(range(-F,F+1)) for F in Fs]) #m_F label of each state
        
        Hhfs = np.zeros(num_states)            # diagonal elements of the hfs Hamiltonian
        # state vector: (|F0 -F0>, |F0 -F0+1>, ..., |F0 F0>, |F1 -F1>, ..., |FN FN>)
        for F in Fs:
            for MF in range(-F, F+1):
                # hyperfine interaction is diagonal in F and mF:
                G = F*(F + 1) - self.I*(self.I + 1) - self.J*(self.J + 1)
                if self.J == 0.5:
                    Vhfs = h/2. * self.Ahfs * G # no quadrupole
                else:
                    Vhfs = h/2. * (self.Ahfs * G + self.Bhfs/4. * (3*G*(G + 1)
                    - 4*self.I*(self.I + 1) * self.J*(self.J + 1)) / self.I
                    /(2*self.I - 1.) /self.J /(2*self.J - 1.))
                
                # stark interaction is diagonal in mF
                # since the Hamiltonian is Hermitian, we only need to fill the lower triangle
                i = 2*F - min(Fs) + MF + np.sum(2*np.arange(min(Fs),F)) # horizontal index of Hamiltonian
                
                Fps = np.arange(min(Fs), F+1) # F'
                
                Hhfs[i] = Vhfs
                
                # not making the rotating wave approximation
                Ep = hbar*(self.omega0 + omega + 1j*self.gam)
                Em = hbar*(self.omega0 - omega - 1j*self.gam)
                
                aS = np.sum(self.D0s**2 /3. /(2.*self.J + 1.) * (1/Ep + 1/Em))
                
                aT = 0
                for ii in range(len(self.D0s)):  # wigner6j function takes scalars
                    aT += (-1)**(self.J + self.states[ii][2] + 1
                            ) * wigner6j(self.J, 1, self.states[ii][2], 1, 
                            self.J, 2) * self.D0s[ii]**2 * (1/Ep[ii] + 1/Em[ii])
                    
                for Fp in Fps:
                    if Fp >= abs(MF):
                        # due to symmetry, only some of the matrix elements need filling
                        j = Fp + MF + np.sum(2*np.arange(min(Fs),Fp) + 1)
                        
                        aT_F = aT * 4*np.sqrt(5/6. * (2*F + 1) * (2*Fp + 1)) * (-1)**(
                            self.J + self.I + F - Fp - MF) * wigner3j(F, 2, 
                            Fp, MF, 0, -MF) * wigner6j(F, 2, Fp, self.J, 
                            self.I, self.J)
                        if F == Fp: 
                            # The hyperfine splitting is diagonal in |F,MF>                   
                            H[i,j] = -0.25 * (aS.real + aT_F.real) * np.abs( 
                                        self.field.amplitude(x,y,z) )**2 + Vhfs
                        else: 
                            # state mixing is only from the anisotropic polarisability
                            H[i,j] = -0.25 * aT_F.real * np.abs( self.field.amplitude(x,y,z) )**2
                
                #Zeeman Hamiltonian
                for Fp in Fps:
                    for MFp in range(-Fp, Fp+1):
                        j = 2*Fp - min(Fs) + MFp + np.sum(2*np.arange(min(Fs),Fp)) # vertical index of Hamiltonian
                        #keep only lower triangle
                        if j <=i:
                            #print([F, MF], [Fp, MFp] , [i, j])
                            H[i,j] =H[i,j] + Bfield*bohr_magneton *self.matrix_element(F,MF,Fp,MFp)
                            #print(self.matrix_element(F,MF,Fp,MFp)/self.matrix_element(1,0,1,0))
                        
                            
        # could fill the rest of H from symmetry: # H = H + H.T - np.diagflat(np.diag(H))
        # diagonalise the Hamiltonian to find the combined shift
        # since it's hermitian np can diagonalise just from the lower traingle
        eigenvalues, eigenvectors = np.linalg.eigh(H, UPLO='L')
        
        # to get the Stark shift, subtract off the hyperfine shift
        Hac = eigenvalues - Hhfs
        
        # note: the diagonalisation in numpy will likely re-order the eigenvectors
        # assume the eigenvector is that closest to the original eigenvector
        indexes = np.argmax(abs(eigenvectors), axis=1)
        #return Hac[indexes], eigenvectors[:,indexes], Hhfs[indexes], F_labels, MF_labels
        return eigenvalues, eigenvectors
    
    def ufunc(self,K,q,u):
        """
        :param u:  Spherical tensor rank of the light field, should be in [-1, 0, 1].
        Expression is eqn 12 in https://doi.org/10.1140/epjd/e2013-30729-x
        """
        V = 0
        vec = [-1,0,1]
        for m in vec:
            for m1 in vec:
                V = V + (-1.00+0j)**(q+m1)*u[m+1]*np.conj(u[-m1+1]+0j)*np.sqrt(2*K+1)*wigner3j(1, K,1, m, -q, m1)
        return V

    def alpha_func(self,K):
        """
        Expression is eqn 11 in https://doi.org/10.1140/epjd/e2013-30729-x
        """
        
        I = self.I
        J = self.J
        C = (-1.00+0j)**(K+J+1) * np.sqrt(2*K+1)
        wavel = self.field.lam
        omega = 2*np.pi*c/wavel   # laser frequency in rad/s
        alpha = 0 
        for ii in range(len(self.D0s)):  # sum over all the transitions
            Jd = self.states[ii][2]
            # not making the rotating wave approximation
            Ep = hbar*(self.omega0[ii] + omega + 1j*self.gam[ii])
            Em = hbar*(self.omega0[ii] - omega - 1j*self.gam[ii])
            alpha += (-1.00+0j)**Jd * wigner6j(1, K, 1, J,Jd,J) * self.D0s[ii]**2 * (((-1)**K)/Ep + 1/Em).real
        return C*alpha
                
    def ac_stark_matrix_element(self,f1,m1,f2,m2,u):
        """
        Exact matrix element for the ac-stark shifts. Includes all contributions, Scalar, vector and tensor.
        :param u:  Spherical tensor rank of the light field, should be in [-1, 0, 1].
        :return:    The value of < f1,m1| gI*I + gJ*J |f2,m2>
        Expression is eqn 10 in https://doi.org/10.1140/epjd/e2013-30729-x
        """
        V = 0
        I = self.I
        J = self.J
        
        for K in range(3):
            for q in np.arange(-K,K+1):
                V = V + self.alpha_func(K)*self.ufunc(K,q,u)*(-1.00+0j)**(J+I+K+q-m1)*np.sqrt((2*f1+1)*(2*f1+1))*wigner3j(f1, K,f2, m1, q, -m2) * wigner6j(f1, q, f2, J, I, J)
        return V
        
    def diagHV(self, x = 0, y = 0, z = 0, Bfield = 0, u = [0,1,0]):
        wavel = self.field.lam
        """Diagonalise the combined Hamiltonian of hyperfine splitting + the ac
        Stark Shift + the zeeman effect. This gives the eigenenergies and eigenstates in the |F,mF>
        basis at a particular wavelength. This works for arbitrary polrization of the light. """
        omega = 2*np.pi*c/wavel   # laser frequency in rad/s
        
        # |I-J| <= F <= I+J
        Fs = np.arange(abs(int(self.I - self.J)), int(self.I + self.J + 1)) #List of F states
        num_states = sum(2*Fs + 1) #Total number of states.
        
        H = np.zeros((num_states, num_states)) + 0j # Initialize combined interaction Hamiltonian
        F_labels = np.concatenate([[F]*(2*F+1) for F in Fs]) #F label of each state
        
        MF_labels = np.concatenate([list(range(-F,F+1)) for F in Fs]) #m_F label of each state
        
        Hhfs = np.zeros(num_states)            # diagonal elements of the hfs Hamiltonian
        # state vector: (|F0 -F0>, |F0 -F0+1>, ..., |F0 F0>, |F1 -F1>, ..., |FN FN>)
        for F in Fs:
            # hyperfine interaction is diagonal in F and mF:
            G = F*(F + 1) - self.I*(self.I + 1) - self.J*(self.J + 1)
            if self.J == 0.5:
                Vhfs = h/2. * self.Ahfs * G # no quadrupole
                #print(self.Ahfs * G/2*1e-6)
            else:
                Vhfs = h/2. * (self.Ahfs * G + self.Bhfs/4. * (3*G*(G + 1)
                - 4*self.I*(self.I + 1) * self.J*(self.J + 1)) / self.I
                /(2*self.I - 1.) /self.J /(2*self.J - 1.))
                #print(Vhfs/h*1e-6)
            for MF in range(-F, F+1):
                # stark interaction is diagonal in mF
                # since the Hamiltonian is Hermitian, we only need to fill the lower triangle
                i = 2*F - min(Fs) + MF + np.sum(2*np.arange(min(Fs),F)) # horizontal index of Hamiltonian
                
                Fps = np.arange(min(Fs), F+1) # F'
                
                Hhfs[i] = Vhfs

                for Fp in Fps:
                    for MFp in range(-Fp, Fp+1):
                        j = 2*Fp - min(Fs) + MFp + np.sum(2*np.arange(min(Fs),Fp)) # vertical index of Hamiltonian
                        #keep only lower triangle
                        if j <=i:
                            
                            #Zeeman Hamiltonian
                            H[i,j] =H[i,j] + Bfield*bohr_magneton *self.matrix_element(F,MF,Fp,MFp)
                            
                            #AC stark hamiltonian
                            Hac =  (1/4.00)*self.ac_stark_matrix_element(F,MF,Fp,MFp,u)* np.abs(self.field.amplitude(x,y,z))**2 
                            H[i,j] = H[i,j] + Hac
                        
                            #print([F, MF], [Fp, MFp] , [i, j])
                            #print(Hac/h*1e-6)
                            
                             # Add hyperfine interaction
                            if F == Fp and MF == MFp: 
                            # The hyperfine splitting is diagonal in |F,MF>                   
                                H[i,j] =  H[i,j] + Vhfs
                                #print([F, MF], [Fp, MFp] , [i, j])
        # could fill the rest of H from symmetry: # H = H + H.T - np.diagflat(np.diag(H))
        # diagonalise the Hamiltonian to find the combined shift
        # since it's hermitian np can diagonalise just from the lower traingle
        eigenvalues, eigenvectors = np.linalg.eigh(H, UPLO='L')
        
        # to get the Stark shift, subtract off the hyperfine shift
        #Hac = eigenvalues - Hhfs
        #
        # note: the diagonalisation in numpy will likely re-order the eigenvectors
        # assume the eigenvector is that closest to the original eigenvector
        #indexes = np.argmax(abs(eigenvectors), axis=1)
        #return Hac[indexes], eigenvectors[:,indexes], Hhfs[indexes], F_labels, MF_labels
        return eigenvalues, eigenvectors
        
    def zeeman_map(self,Bfield_min=1,Bfield_max=2,Bfield_int=1):
        'This filed gives the zeeman map for the values specified in Bfield_array'
        # compare polarisability of excited states
        y = []
        Bfield_array  = np.arange(Bfield_min,Bfield_max,Bfield_int)
        for Bfield in Bfield_array:
            ev, evec = self.diagHV(Bfield = Bfield*1e-4)
            y.append(ev)
        X = []
        Y = []
        for i in range(len(Bfield_array)):
            for j in range(len(y[i])):
                X.append(Bfield_array[i])
                Y.append(y[i][j]/h*1e-9)
        plt.figure()
        plt.title("Zeeman structure of "+self.X + 'in state [L,J] = ' + str([self.L,self.J]))
        plt.scatter(X, Y, s=10, linewidth=0, zorder=2, picker=5)
        plt.xlabel("Magnetic field (Gauss)")
        plt.ylabel("Energy (GHz)")

if __name__ == "__main__":
    # run GUI by passing an arg:
    if np.size(sys.argv) > 1 and sys.argv[1] == 'rungui':
        AFF.runGUI()
        sys.exit() # don't run any of the other code below
    Rb = atom(atm = 'Rb87')
    AFF.vmfSS(Rb)

    # combinedTrap(Cswl = 1064e-9, # wavelength of the Cs tweezer trap in m
    #             Rbwl = 810e-9, # wavelength of the Rb tweezer trap in m
    #             power = 5e-3, # power of Cs tweezer beam in W
    #             Rbpower = 1e-3, # power of Rb tweezer beam in W 
    #             beamwaist = 1e-6)
    #check880Trap(wavels=np.linspace(795, 1100, 400)*1e-9, species='Rb')

    # getMFStarkShifts()
    # plotStarkShifts(wlrange=[800,1100])

    # for STATES in [[Rb5S, Rb5P],[Cs6S, Cs6P]]:
    #     plt.figure()
    #     plt.title("AC Stark Shift in "+STATES[0].X+"\nbeam power %.3g mW, beam waist %.3g $\mu$m"%(power*1e3,beamwaist*1e6))
    #     plt.plot(wavels*1e9, STATES[0].acStarkShift(0,0,0,wavels)/kB*1e3, 'tab:blue', label='Ground S$_{1/2}$')
    #     excited_shift = 0.5*(STATES[1].acStarkShift(0,0,0,wavels,mj=0.5) + STATES[1].acStarkShift(0,0,0,wavels,mj=1.5))
    #     plt.plot(wavels*1e9, excited_shift/kB*1e3, 'r-.', label='Excited P$_{3/2}$')
    #     plt.legend()
    #     plt.ylabel("Trap Depth (mK)")
    #     plt.xlabel("Wavelength (nm)")
    #     plt.xlim(wavels[0]*1e9, wavels[-1]*1e9)
    #     plt.ylim(-5,5)
    #     plt.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.25) # show zero crossing
    # plt.show()