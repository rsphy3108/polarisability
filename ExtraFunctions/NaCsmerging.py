"""Stefan Spence 40/4/19
Calculate the polarisability of Cs and Na while merging as in 
L. R. Liu et al., Science 360, 900–903 (2018)
assuming hyperfine transitions can be ignored
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('..')
from AtomFieldInt_V3 import (dipole, atom, c, eps0, h, hbar, a0, e, me, 
    kB, amu, Eh, au, atom)

Rb = atom(atm = 'Rb87')
Cs = atom(atm = 'Cs133')
Na = atom(atm = 'Na23')


Cswl = 976e-9           # wavelength of the Cs tweezer trap in m
Nawl = 700e-9           # wavelength of the Rb tweezer trap in m
Cspower = 5e-3          # power of Cs tweezer beam in W
Napower = 5e-3          # power of Rb tweezer beam in W 
Cswaist = 0.8e-6        # beam waist for Cs in m
Nawaist = 0.7e-6        # beam waist fir Na in m

# Cs in its own tweezer
Cs976 = dipole(Cs, (0,1/2.,4,4), [Cswl, Cspower, Cswaist])
# Cs in the Na tweezer
Cs700 = dipole(Cs, (0,1/2.,4,4), [Nawl, Napower, Nawaist])
# Na in the Cs tweezer
Na976 = dipole(Na, (0,1/2.,1,1), [Cswl, Cspower, Cswaist])
# Na in its own tweezer
Na700 = dipole(Na, (0,1/2.,1,1), [Nawl, Napower, Nawaist])

# separated tweezer trap depths in mK:
U_cs_976 = Cs976.acStarkShift(0,0,0) / kB * 1e3  
U_cs_700 = Cs700.acStarkShift(0,0,0) / kB * 1e3 
U_na_976 = Na976.acStarkShift(0,0,0) / kB * 1e3 
U_na_700 = Na700.acStarkShift(0,0,0) / kB * 1e3 

# combined trap depths in mK:
U_cs = U_cs_976 + U_cs_700
U_na = U_na_976 + U_na_700

# ratio of polarisabilities Na / Cs
a_cs_976 = Cs976.polarisability() / au
a_cs_700 = Cs700.polarisability() / au
a_na_976 = Na976.polarisability() / au
a_na_700 = Na700.polarisability() / au
aratio_976 = a_na_976 / a_cs_976
aratio_700 = a_na_700 / a_cs_700

print("""
                           %.0f nm      %.0f nm       Combined (ratio 700/976)
beam power (mW)            %.3g        %.3g
beam waist (microns)       %.3g         %.3g
Na:
polarisability (a_0^3)      %.4g        %.4g      (%.3g)
Trap depth (mK)            %.3g       %.3g      %.3g
Cs:
polarisability (a_0^3)     %.4g          %.4g      (%.3g)
Trap depth (mK)             %.3g        %.3g        %.3g
Ratio:
polarisability (Na/Cs)     %.3g        %.3g    (%.3g)
"""%(Nawl*1e9, Cswl*1e9, Napower*1e3, Cspower*1e3, Nawaist*1e6, Cswaist*1e6,
a_na_700, a_na_976, a_na_700 / a_na_976, U_na_700, U_na_976, U_na,
a_cs_700, a_cs_976, a_cs_700 / a_cs_976, U_cs_700, U_cs_976, U_cs,
aratio_700, aratio_976, aratio_700 / aratio_976))