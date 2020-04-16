import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from math import factorial 
from matplotlib.ticker import AutoLocator
from AtomFieldInt_V3 import atom, dipole

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

#############################

##### example functions:#####

#############################
   
def getMagicWavelengths(deltaE, E, wavelengths):
    """Find the magic wavelengths where the energy difference is zero.
    Define this where the fractional difference |deltaE/E| < 0.05 and the 
    difference deltaE changes sign"""

    magicWavelengths = []
    magicindexes = np.where(abs(deltaE/E)<0.05)[0]
    
    for mi in magicindexes:
        if np.sign(deltaE[mi]) == -np.sign(deltaE[mi+1]):
            magicWavelengths.append(wavelengths[mi])
    
    return magicWavelengths
     
def plotStarkShifts(ATOM1 = atom(atm = 'Rb87'),ATOM2 = atom(atm = 'Cs133'), wavelength = 880e-9,             # laser wavelength in nm
                    beamwaist = 1e-6,                # beam waist in m
                    power = 20e-3):                  # power in Watts
    """Find the ac Stark Shifts in Rb, Cs"""
    # typical optical tweezer parameters:
    bprop = [wavelength, power, beamwaist] # collect beam properties

    # mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
    # linewidths (rad/s), state labels, nuclear spin, atomic symbol.
    Rb5S = dipole(ATOM1, (0,1/2.,1,1), bprop)
    
    Rb5P = dipole(ATOM1, (1,3/2.,1,1), bprop)
                    
    Cs6S = dipole(ATOM2, (0,1/2.,3,3), bprop)
                    
    Cs6P = dipole(ATOM2, (1,3/2.,3,3), bprop)
    
    # need a small spacing to resolve the magic wavelengths - so it will run slow
    # to resolve magic wavelengths, take about 10,000 points.
    wavels = np.linspace(700e-9, 1100e-9, 500) 
    
    # ac Stark Shift in Joules:
    dE6S = Cs6S.acStarkShift(0,0,0,wavels, mj=0.5, HF=False)
    # average over mj states
    dE6P = 0.5*(Cs6P.acStarkShift(0,0,0,wavels, mj=1.5, HF=False) + 
        Cs6P.acStarkShift(0,0,0,wavels, mj=0.5, HF=False))
    dif6P = dE6P - dE6S
    
    magic6P = getMagicWavelengths(dif6P, dE6P, wavels)
    
    plt.figure()
    plt.title("AC Stark Shift in $^{133}$Cs")
    plt.plot(wavels*1e9, dE6S/h*1e-6, 'b--', label='Ground S$_{1/2}$')
    plt.plot(wavels*1e9, dE6P/h*1e-6, 'r-.', label='Excited P$_{3/2}$')
    plt.plot(wavels*1e9, (dif6P)/h*1e-6, 'k', label='Difference')
    plt.plot([magic6P[0]*1e9]*2, [min(dif6P/h/1e6),max(dif6P/h/1e6)], 'm:',
                label = 'Magic Wavelength')
    plt.legend()
    for mw in magic6P[1:]:
        plt.plot([mw*1e9]*2, [min(dif6P/h/1e6),max(dif6P/h/1e6)], 'm:')
    plt.ylabel("Stark Shift (MHz)")
    plt.xlabel("Wavelength (nm)")
    plt.xlim(wavels[0]*1e9, wavels[-1]*1e9)
    plt.ylim(-2200,2200)
    plt.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.25) # show zero crossing
    plt.show()
    print("Magic wavelengths at:\n", magic6P)
    
    
    # ac Stark Shift in Joules:
    dE5S = Rb5S.acStarkShift(0,0,0,wavels, mj=0.5, HF=False)
    # average over mj states
    dE5P = 0.5*(Rb5P.acStarkShift(0,0,0,wavels, mj=1.5, HF=False) + 
            Rb5P.acStarkShift(0,0,0,wavels, mj=0.5, HF=False))
    dif5P = dE5P - dE5S

    plt.figure()
    plt.title("AC Stark Shift in $^{87}$Rb")
    plt.plot(wavels*1e9, dE5S/h*1e-6, 'b--', label='Ground S$_{1/2}$')
    plt.plot(wavels*1e9, dE5P/h*1e-6, 'r-.', label='Excited P$_{3/2}$')
    plt.plot(wavels*1e9, (dif5P)/h*1e-6, 'k', label='Difference')
    plt.legend()
    plt.ylabel("Stark Shift (MHz)")
    plt.xlabel("Wavelength (nm)")
    plt.ylim(-500,500)
    plt.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.25) # show zero crossing
    plt.show()
    
def compareArora():
    """Plot Fig. 5 - 8 in Arora et al 2007 to show that the polarisabilities 
    of Rb and Cs without hyperfine levels are correct"""
    # beam properties: wavelength, power, beam waist
    # intensity set to 1e10 MW/cm^2
    bprop = [1064e-9, np.pi*0.5e-2, 1e-6]
    
    for ATOM in [atom(atm = 'Rb87'), atom(atm = 'Cs133')]:
        if ATOM.X == 'Rb87':
            wavel1 = np.linspace(780, 800, 200)*1e-9
            Ylim1 = (-8000, 8000)
            wavel2 = np.linspace(787,794, 200)*1e-9 
            Ylim2 = (-1000, 1000)
            FS, FP = 1, 3
        elif ATOM.X == 'Cs133':
            wavel1 = np.linspace(925, 1000, 200)*1e-9
            Ylim1 = (-1000, 5000)
            wavel2 = np.linspace(927, 945, 200)*1e-9
            Ylim2 = (-100, 100)
            FS, FP = 3, 5
            
        S = dipole(ATOM, (0,1/2.,FS,FS), bprop)
        
        P3 = dipole(ATOM, (1,3/2.,FP,FP), bprop)
                        
        # compare polarisability of excited states
        plt.figure()
        plt.title("Polarisability of "+ATOM.X)
        plt.plot(wavel1*1e9, S.polarisability(wavel1)/au, 'r', label='s')
        plt.plot(wavel1*1e9, P3.polarisability(wavel1,mj=0.5)/au, 'g--', label='p$_{3/2}$, mj=1/2')
        plt.plot(wavel1*1e9, P3.polarisability(wavel1,mj=1.5)/au, 'm:', label='p$_{3/2}$, mj=3/2')
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Polarisability (a.u.)")
        plt.ylim(Ylim1)
        plt.xlim(wavel1[0]*1e9, wavel1[-1]*1e9)
        
        # calculate stark shifts between F, MF states
        mfLS = ['r', 'g--', 'm:', 'c-.', 'k-.', 'y']  # line styles
        plt.figure()
        plt.title("AC Stark Shifts for transitions from P$_{3/2}$ m$_F$ to \nthe groundstate in "+ATOM.X)
        dES = S.acStarkShift(0,0,0, wavel2, HF=True)   # ground state stark shift
        for MF in range(FP+1):
            P3.MF = MF
            dEPMF = P3.acStarkShift(0,0,0, wavel2, HF=True) # excited MF state stark shift
            plt.plot(wavel2*1e9, (dEPMF - dES)/h/1e6, mfLS[MF], label=r'm$_F$ = $\pm$'+str(MF))
        xlims = [wavel2[0]*1e9, wavel2[-1]*1e9]
        plt.plot(xlims, [0,0], 'k:', alpha=0.4)  #  show where zero is
        plt.ylim(Ylim2)
        plt.xlim(xlims)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Stark Shift (MHz)")
        plt.legend()
        
    plt.show()
        
        
    
def getStarkShift(obj):
    """Print the ac Stark Shift for all of the hyperfine levels in a particular
    fine structure state of the atom in dipoleObject"""
    Lterms = ['S', 'P', 'D', 'F', 'G'] # labels for angular momentum states

    # show some important parameters
    outstring = obj.X + " " + Lterms[obj.L] + str(int(obj.J*2)
            ) + "/2 ac Stark Shift at %.0f nm for E field %.2g V/m:\n"%(
            obj.field.lam*1e9, obj.field.E0)
            
    outstring += "\nIf hyperfine splitting is insignificant:\n"
    for MJ in np.arange(1, 2*obj.J+1, 2).astype(int): # NB: this is 2*MJ
        outstring += "MJ = "+str(MJ)+"/2 : %.5g MHz\n"%(
            obj.acStarkShift(0,0,0, obj.field.lam, mj=MJ/2., HF=False)/h/1e6)
        a = np.zeros(4)  # make sure the results are always a size 3 array
        alpha = obj.polarisability(wavel=obj.field.lam, mj=MJ/2., HF=False, split=True)
        atot  = obj.polarisability(wavel=obj.field.lam, mj=MJ/2., HF=False, split=False)
        a[:np.size(alpha)] = alpha
        outstring += "polarisability components (scalar, vector, tensor) : \n(%.4g, %.4g, %.4g) a.u.\n"%(
            a[0]/au, a[1]/au, a[2]/au)
        outstring += "combined polarisability : %.4g a.u\n"%(atot/au)
            
        
    outstring += "\nIf hyperfine splitting is significant:\n"
    for F in range(int(abs(obj.I - obj.J)), int(obj.I + obj.J+1)):
        mfAveShift = 0
        for MF in range(-F, F+1):
            obj.F, obj.MF = F, MF
            mfAveShift += obj.acStarkShift(0,0,0, obj.field.lam, HF=True)/h/1e6
        outstring += "F = "+str(F)+ ", ave. mF  : %.5g MHz.\n"%(mfAveShift/(2.*F+1.))
        obj.F, obj.MF = F, F
        outstring += "|"+str(F)+","+str(F)+">  : %.5g MHz\n"%(
                obj.acStarkShift(0,0,0, obj.field.lam, HF=True)/h/1e6)

        a = np.zeros(4)  # make sure the results are always a size 3 array
        alpha = obj.polarisability(wavel=obj.field.lam, HF=True, split=True)
        atot  = obj.polarisability(wavel=obj.field.lam, HF=True, split=False)
        a[:np.size(alpha)] = alpha
        outstring += "polarisability components (scalar, vector, tensor) : \n(%.4g, %.4g, %.4g) a.u.\n"%(
            a[0]/au, a[1]/au, a[2]/au)
        outstring += "combined polarisability : %.4g a.u\n\n"%(atot/au)
                    
    return outstring
                                                    
def runGUI():
    """A UI to get the stark shift from a user supplied state"""
    import tkinter
    from tkinter import messagebox
    
    root = tkinter.Tk()  # main window
    frames, labels, entries = [], [], [] # the user must enter variables
    labeltext = ['Wavelength (m): ', 'Beam waist (m): ', 'Beam power (W): ', 
            'Atom (Rb87/Cs133): ', 'Orbital angular momentum L: ', 
            'Total angular momentum J: ']
    entrystrings = [tkinter.StringVar() for i in range(len(labeltext))]
    default = ['932e-9', '1e-6', '3.9e-3', 'Cs133', '0', '0.5']
    
    
    for i in range(len(labeltext)):
        frames.append(tkinter.Frame(root)) # frame for placing entries
        frames[-1].pack(side=tkinter.TOP)  # position in descending order
        labels.append(tkinter.Label(frames[-1], text=labeltext[i]))
        labels[-1].pack(side=tkinter.LEFT) # position label on left
        entries.append(tkinter.Entry(frames[-1], textvariable=entrystrings[i]))
        entrystrings[i].set(default[i])    # set default string text
        entries[-1].pack(side=tkinter.RIGHT)# text entry on right
    
    
    def showResult():
        wavelength = float(entrystrings[0].get())  # laser wavelength in nm
        beamwaist = float(entrystrings[1].get())   # beam waist in m
        power = float(entrystrings[2].get())       # power in Watts 
        bprop = [wavelength, power, beamwaist] # collect beam properties
        atomSymbol = entrystrings[3].get()         # choose Rb or Cs
        L = int(entrystrings[4].get())             # orbital angular momentum
        J = float(entrystrings[5].get())             # total angular momentum
        
        # choose element
        if atomSymbol == "Rb87":
            atomObj = atom(atm = 'Rb87')
            F = 1
        elif atomSymbol == "Cs133":
            atomObj = atom(atm = 'Cs133')
            F = 3
        elif atomSymbol == "K41":
            atomObj = atom(atm = 'K41')
            F = 1
        else:
            messagebox.showinfo("Error", "You must choose Rb87, Cs133 or K41")
            return 0
        
        # get transition data for the given state
        if L == 0:
            D0, w0, lw, nlj = atomObj.D0S, atomObj.w0S, atomObj.lwS, atomObj.nljS
        elif L == 1 and J == 0.5:
            D0, w0, lw, nlj = atomObj.D0P1, atomObj.w0P1, atomObj.lwP1, atomObj.nljP1
        elif L == 1 and J == 1.5:
            D0, w0, lw, nlj = atomObj.D0P3, atomObj.w0P3, atomObj.lwP3, atomObj.nljP3
        
        # construct the instance of the dipole class
        dipoleObj = dipole(atomObj, (L,J,F,F), bprop)
        
        messagebox.showinfo("Calculation Result", getStarkShift(dipoleObj))
        
    resultButton = tkinter.Button(root, text="Calculate Stark Shifts", 
                                    command=showResult)
    resultButton.pack(side = tkinter.BOTTOM)
    
    root.mainloop()
    
    
def combinedTrap(Cswl = 1064e-9, # wavelength of the Cs tweezer trap in m
                Rbwl = 880e-9, # wavelength of the Rb tweezer trap in m
                power = 6e-3, # power of Cs tweezer beam in W
                Rbpower = -1, # power of Rb tweezer beam in W 
                beamwaist = 1e-6): # beam waist in m
    """Model tweezer traps for Rb and Cs and find the potential each experiences
    when they're overlapping. Should fix the separate tweezer trap depths to >1mK.
    We also want Rb to experience a deeper trap from its tweezer than from the Cs
    tweezer so that there isn't too much heating during merging.
    args:
    Cswl = 1064e-9, # wavelength of the Cs tweezer trap in m
    Rbwl = 880e-9, # wavelength of the Rb tweezer trap in m
    power = 6e-3, # power of Cs tweezer beam in W
    Rbpower = -1, # power of Rb tweezer beam in W (if < 0 then choose a power
    such that both species experience the same trap depth when the tweezers are
    overlapping)
    beamwaist = 1e-6 # beam waist in m
    """
    bprop = [Cswl, power, beamwaist] # collect beam properties
    
    # For the 1064nm trap:
    # mass, (L,J,F,MF), bprop, dipole matrix elements (Cm), resonant frequencies (rad/s),
    # linewidths (rad/s), state labels, nuclear spin, atomic symbol.
    # groundstate rubidium
    Rb = atom(atm = 'Rb87')
    Rb1064 = dipole(Rb, (0,1/2.,1,1), bprop)
                    
    # groundstate caesium
    Cs = atom(atm = 'Cs133')
    Cs1064 = dipole(Cs, (0,1/2.,4,4), bprop)
                    
    CsP = dipole(Cs, (1,3/2.,5,5), bprop)
                    
    # set the power of the traps so that the trap depth experienced by each 
    # species in the overlapping trap is the same:
    if Rbpower < 0:
        Rbpower = (Cs1064.polarisability(Cswl,mj=0.5) - Rb1064.polarisability(Cswl, mj=0.5)) / (Rb1064.polarisability(Rbwl, mj=0.5) - Cs1064.polarisability(Rbwl, mj=0.5)) * power
    
    # for the 880nm trap:
    bprop = [Rbwl, abs(Rbpower), beamwaist]
    Rb880 = dipole(Rb, (0,1/2.,1,1), bprop)
                    
    Cs880 = dipole(Cs, (0,1/2.,3,3), bprop)
                    
    
    # in the trap with both tweezers overlapping: 
    U0 = abs(Rb1064.acStarkShift(0,0,0) + Rb880.acStarkShift(0,0,0))
    wrRb = np.sqrt(4*U0 / Rb.m / beamwaist**2) /2. /np.pi /1e3
    wrCs = np.sqrt(4*U0 / Cs.m / beamwaist**2) /2. /np.pi /1e3
    print("%.0f beam power: %.3g mW\t\t%.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3))
    print("""In the combined %.0fnm and %.0fnm trap with a depth of %.3g mK the radial trapping frequencies are: 
Rubidium: %.0f kHz \nCaesium: %.0f kHz"""%(Rbwl*1e9, Cswl*1e9, U0/kB*1e3, wrRb, wrCs))
    
    # with just the Cs tweezer trap:
    URb =abs(Rb1064.acStarkShift(0,0,0))
    wrRb1064 = np.sqrt(4*URb / Rb.m / beamwaist**2) /2. /np.pi /1e3
    UCs = abs(Cs1064.acStarkShift(0,0,0))
    wrCs1064 = np.sqrt(4*UCs / Cs.m / beamwaist**2) /2. /np.pi /1e3
    print("""\nIn just the %.0fnm trap:
    Rubidium has trap depth %.3g mK
                 radial trapping frequency %.0f kHz
    Caesium has trap depth %.3g mK
                radial trapping frequency %.0f kHz"""%(Cswl*1e9, URb/kB*1e3, wrRb1064, UCs/kB*1e3, wrCs1064))
    
    print(getStarkShift(Cs1064))
    print(getStarkShift(CsP))
    
    # plot merging traps:
    n = 5   # number of time steps in merging to plot
    sep = np.linspace(0, 10e-6, n)     # initial separation of the tweezer traps
    zs = np.linspace(-2, 10, 200)*1e-6 # positions along the beam axis
    
    for atoms in [[Rb1064, Rb880], [Cs1064, Cs880]]:
        plt.figure()
        plt.subplots_adjust(hspace=0.01)
        
        for i in range(n):
            ax = plt.subplot2grid((n,1), (i,0))
            
            U = (atoms[0].acStarkShift(0,0,zs) + atoms[1].acStarkShift(0,0,zs-sep[n-i-1]))/kB*1e3 # combined potential along the beam axis
            U1064 = atoms[0].acStarkShift(0,0,zs)/kB*1e3         # potential in the 1064 trap
            U880 = atoms[1].acStarkShift(0,0,zs-sep[n-i-1])/kB*1e3 # potential in the 880 trap
            plt.plot(zs*1e6, U, 'k')
            plt.plot(zs*1e6, U1064, color='tab:orange', alpha=0.6)
            plt.plot(zs*1e6, U880, color='tab:blue', alpha=0.6)
            plt.plot([0]*2, [min(U),0], color='tab:orange', linewidth=10, label='%.0f'%(Cswl*1e9), alpha=0.4)
            plt.plot([sep[n-i-1]*1e6]*2, [min(U),0], color='tab:blue', linewidth=10, label='%.0f'%(Rbwl*1e9), alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title("Optical potential experienced by "+atoms[0].X
        +"\n%.0f beam power: %.3g mW   %.0f beam power: %.3g mW"%(Cswl*1e9, power*1e3, Rbwl*1e9, Rbpower*1e3),
                    pad = 25)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        
            
        plt.xlabel(r'Position ($\mu$m)')
        ax.set_xticks(sep*1e6)
        plt.ylabel('Trap Depth (mK)')
        ax.yaxis.set_major_locator(AutoLocator())
        
    plt.show()
            
            
def getMFStarkShifts(ATOM, wavelength = 1064e-9, # laser wavelength in m
                    power = 0.00906143,    # laser power in W
                    beamwaist = 1e-6      # beam waist in m
                    ):
    """Return the Stark shifts of the MF states for cooling/repump transitions"""
    bprop = [wavelength, power, beamwaist] # collect beam properties
    if ATOM.atm == 'Cs133': # assign the relevant hyperfine transitions
        Fs = [3,4]
        l1 = [18,24] # index of lines for making legend
    elif ATOM.atm == 'Rb87':
        Fs = [1,2]
        l1 = [6,12] # index of lines for making legend
    
    # print("Stark shift of "+ATOM.X+" S1/2 F = %s, %s -> P3/2 F' = %s, %s for different MF states."%(Fs[0],Fs[0]+1,Fs[1],Fs[1]+1))
    
    plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for F in Fs:
        for MF in range(-F, F+1):
            print(" ----- |F = "+str(F)+", m_F = "+str(MF)+">")
            for MFp in range(MF-1, MF+2):
                S = dipole(ATOM, (0,1/2.,F,MF), bprop)
                P = dipole(ATOM, (1,3/2.,F+1,MFp), bprop)
                shift = (S.acStarkShift(0,0,0, bprop[0], HF=True) - P.acStarkShift(0,0,0, bprop[0], HF=True))/h/1e6
                if MF != 0:
                    deltaMF = (MFp-MF)*np.sign(MF)
                else:
                    deltaMF = (MFp-MF)
                plt.plot(MF, shift, '_', color=colors[F-1], alpha=0.33*(2+deltaMF), markersize=15, linewidth=10)
                print("|F' = "+str(F+1)+", m_F' = "+str(MFp)+"> : %.5g MHz"%shift)
                
    plt.xlabel("$M_F$")  
    plt.ylabel("AC Stark Shift (MHz)")
    lines = plt.gca().lines
    plt.legend(lines[l1[0]:l1[1]], ['F='+str(f)+r', $\Delta M_F=$'+str(-dmf) 
                for f in range(min(Fs),max(Fs)+1) for dmf in range(-1,2)])
    plt.show()

''' rvb 15.05.2019: I've just added this function so I can work out lightshifts for the LGM -
    haven't changed anything else! 
    Interested in the Cs (Rb) shifts of the F' = 4 (2) mF states relative to each other.
    
    '''
def vmfSS(ATOM):
    """Return the Stark shifts of the MF states for Cs cooling/repump transitions"""
    
    plt.figure()
    
    if ATOM.atm == 'Cs133':
        Cs = ATOM
        F = 5
        bprop = [938e-9, 8e-3, 1.7e-6]      # wavelength, beam power, beam waist
        for MFp in range(-F, F+1, 1):
            P = dipole(Cs, (1,3/2.,F,MFp), bprop)
            Pshift = P.acStarkShift(0,0,0, bprop[0], HF=True)/h/1e6    # interested in how the EStates shift relative to each other
            plt.plot(MFp, Pshift, '_', markersize=15, linewidth=10, color = '#7E317B')
            print("|F' = "+str(F)+", m_F' = "+str(MFp)+"> : %.5g MHz"%Pshift)
        
    elif ATOM.atm == 'Rb87':
        Rb = ATOM
        F = 3
        bprop = [940e-9, 35e-3, 1.7e-6]      # wavelength, beam power, beam waist
        for MFp in range(-F, F+1, 1):
            P = dipole(Rb, (1,3/2.,F,MFp), bprop)
            Pshift = P.acStarkShift(0,0,0, bprop[0], HF=True)/h/1e6    # interested in how the EStates shift relative to each other
            s, v, t = P.polarisability(bprop[0], mj=3/2, HF=False, split=True)
           # print('split ',s/au, t/au)
            s, v, t = P.polarisability(bprop[0], mj=3/2, HF=True, split=True)
         #   print('avrge ',s/au, t/au)
            plt.plot(MFp, Pshift, '_', markersize=15, linewidth=10, color = '#7E317B')
            print("|F' = "+str(F)+", m_F' = "+str(MFp)+"> : %.5g MHz"%Pshift)
    
    plt.title('Stark Shifts of ' + ATOM.atm  + " |F' = " + str(F) + ', $M_F$> states' )            
    plt.xlabel("$M_F$")  
    plt.ylabel("AC Stark Shift (MHz)")
    plt.show()


def compareKien():
    """compare Kien 2013 Fig 4,5"""
    bprop =[880e-9,20e-3,1e-6]
    Cs = atom(atm = 'Cs133')
    Cs880 = dipole(Cs, (0,1/2.,3,3), bprop)
                    
    CsP = dipole(Cs, (1,3/2.,3,3), bprop)               
    
    wls = [np.linspace(680, 690, 200)*1e-9, np.linspace(930, 940, 200)*1e-9]
    ylims = [(-1200, 300), (-3000, 6000)]
    for ii in range(2):
        plt.figure()
        plt.title("Cs Polarisabilities. Red: 6S$_{1/2}$, Blue: 6P$_{3/2}$.\nscalar: solid, vector: dashed, tensor: dotted")
        a1 = Cs880.polarisability(wls[ii],mj=0.5,split=True)
        a2 = 0.5*(np.array(CsP.polarisability(wls[ii],mj=1.5, split=True))+
            np.array(CsP.polarisability(wls[ii],mj=0.5, split=True)))
        ls = ['-', '--', ':']
        for i in range(3):
            plt.plot(wls[ii]*1e9, a1[i]/au, 'r', linestyle=ls[i], label="Cs")
            plt.plot(wls[ii]*1e9, a2[i]/au, 'b', linestyle=ls[i], label="$P_{3/2}$")
        #plt.legend()
        plt.ylim(ylims[ii])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Polarisablity (a.u.)")
        plt.show()
    

def check880Trap(wavelength = 880e-9,     # wavelength in m
                 wavels = np.linspace(795,930,500)*1e-9, # wavelengths in m to plot
                 power = 5e-3,            # beam power in W
                 beamwaist = 1e-6,        # beam waist in m
                 species = 'Rb'):         # which species to set a 1mK trap for
    """Plot graphs of the trap depth experienced by Cs around 880nm when 
    the ground state Rb trap depth is fixed at 1mK. Look at the scattering
    rates and hence trap lifetimes that are possible."""
    bprop = [wavelength, power, beamwaist]
    Rb = atom(atm = 'Rb87')
    Rb5S = dipole(Rb, (0,1/2.,1,1), bprop)
    
    Rb5P = dipole(Rb, (1,3/2.,1,1), bprop)
    
    Cs = atom(atm = 'Cs133')                
    Cs6S = dipole(Cs, (0,1/2.,3,3), bprop)
                    
    Cs6P = dipole(Cs, (1,3/2.,3,3), bprop)               
    
    # choose power so that Rb trap depth is fixed at 1 mK:
    if species == Cs.X:
        Powers = abs(1e-3*kB * np.pi * eps0 * c * beamwaist**2 / Cs6S.polarisability(wavels)) # in Watts
    else:
        Powers = abs(1e-3*kB * np.pi * eps0 * c * beamwaist**2 / Rb5S.polarisability(wavels)) # in Watts
    _, ax1 = plt.subplots()
    ax1.set_title('Fixing the trap depth of ground state '+species+' at 1 mK')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.plot(wavels*1e9, Powers*1e3, color='tab:blue')
    ax1.set_ylabel('Power (mW)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlim(wavels[0]*1e9, wavels[-1]*1e9)
    # ax1.set_ylim(min(Powers)*1e3-0.5, 15)

    ax2 = ax1.twinx()
    # now the power and the wavelength are varied:
    Llabels = ['$S_{1/2}$', '$P_{3/2}$']
    if species == Cs.X:
        colors = ['k', 'tab:orange', 'tab:orange', 'tab:orange']
        linestyles = ['--', '-.', '-', ':']
    else:
        colors = ['tab:orange', 'tab:orange', 'k', 'tab:orange']
        linestyles = ['-', '-.', '--', ':']
    
    trapdepths = []
    for obj in [Cs6S, Cs6P, Rb5S, Rb5P]:
        res = np.zeros(len(Powers))
        for i in range(len(Powers)):
            obj.field.E0 = 2 * np.sqrt(Powers[i] / eps0 / c / np.pi)/beamwaist
            # average mj states (doesn't have any effect on j=1/2 states)
            res[i] = 0.5*(obj.acStarkShift(0,0,0, wavels[i], mj=1.5) + 
                    obj.acStarkShift(0,0,0, wavels[i], mj=0.5))
        color = colors.pop(0)
        ls = linestyles.pop(0)
        ax2.plot(wavels*1e9, res*1e3/kB, color=color, label=obj.X+" "+Llabels[obj.L], linestyle=ls)
        trapdepths.append(res)

    ax2.plot(wavels*1e9, np.zeros(len(wavels)), 'k', alpha=0.1) # show zero crossing    
    ax2.set_ylabel('Trap Depth (mK)', color='tab:orange')
    ax2.legend()
    ax2.set_ylim(-3, 3)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    plt.tight_layout()

    I = 2*Powers / np.pi / beamwaist**2
    # scattering rate of Cs from the D2 line:
    deltaCsD1 = 2*np.pi*c * (1/wavels - 1/Cs.rwS[0]) # detuning from D1 (rad/s)
    deltaCsD2 = 2*np.pi*c * (1/wavels - 1/Cs.rwS[35]) # detuning from D2 (rad/s)
    IsatCsD1 = 2.4981 *1e-3 *1e4 # saturation intensity for D1 transition, sigma polarised
    IsatCsD2 = 1.1023 *1e-3 *1e4 # saturation intensity for D2 transition, pi polarised
    CsRsc = 0
    for vals in [[Cs.lwS[0], deltaCsD1, IsatCsD1], [Cs.lwS[35], deltaCsD2, IsatCsD2]]:
        CsRsc += vals[0]/2. * I/vals[2] / (1 + 4*(vals[1]/vals[0])**2 + I/vals[2])
    # Cstau = 1e-3*kB / (hbar*(2*np.pi/wavels))**2 * 2.*Cs.m / CsRsc # the lifetime is the trap depth / recoil energy / scattering rate
    Cst = 4*np.sqrt(Cs.m*abs(trapdepths[0])) / (2*np.pi/wavels)**2 /hbar /beamwaist /CsRsc # duration in vibrational ground state (s) = 1/Lamb-Dicke^2 /Rsc

    # scattering rate of Rb from the D1 line:
    deltaRbD1 = 2*np.pi*c * (1/wavels - 1/Rb.rwS[0]) # detuning from D1 (rad/s)
    IsatRbD1 = 4.484 *1e-3 *1e4 # saturation intensity for D1 transition, pi polarised
    RbRsc = Rb.lwS[0]/2. * I/IsatRbD1 / (1 + 4*(deltaRbD1/Rb.lwS[0])**2 + I/IsatRbD1) # per second
    # Rbtau = 1e-3*kB / (hbar*(2*np.pi/wavels))**2 * 2.*Rb.m / RbRsc # the lifetime is the trap depth / recoil energy / scattering rate
    Rbt = 4*np.sqrt(Rb.m*abs(trapdepths[2])) / (2*np.pi/wavels)**2 /hbar /beamwaist /RbRsc # duration in vibrational ground state (s) = 1/Lamb-Dicke^2 /Rsc

    # plot lifetime and scattering rate on the same axis:
    for Rsc, ts, X in [[RbRsc, Rbt, Rb.X], [CsRsc, Cst, Cs.X]]:
        fig, ax3 = plt.subplots()
        ax3.set_title('Scattering rate and lifetime of ground state '+X+' in a 1 mK trap (for '+species+')')
        ax3.set_xlabel('Wavelength (nm)')
        ax3.semilogy(wavels*1e9, Rsc, color='tab:blue')
        ax3.plot(wavels*1e9, np.zeros(len(wavels))+100, '--', color='tab:blue', alpha=0.25) # show acceptable region
        ax3.set_ylabel('Scattering rate ($s^{-1}$)', color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')
        ax3.set_xlim(wavels[0]*1e9, wavels[-1]*1e9)
        ax3.set_ylim(1, 1e5)

        ax4 = ax3.twinx()
        ax4.semilogy(wavels*1e9, ts, color='tab:orange')
        ax4.plot(wavels*1e9, np.ones(len(wavels))/2., '--', color='tab:orange', alpha=0.25) # show acceptable region
        ax4.set_ylabel('Time in the vibrational ground state (s)', color='tab:orange')
        ax4.tick_params(axis='y', labelcolor='tab:orange')
        ax4.set_ylim(0.001,10)
        plt.tight_layout()
    
    plt.show()
    
        
if __name__ == "__main__":
    # run GUI by passing an arg:
    if np.size(sys.argv) > 1 and sys.argv[1] == 'rungui':
        runGUI()
        sys.exit() # don't run any of the other code below
    atom  = atom(atm = 'Rb87')
    vmfSS(atom)

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