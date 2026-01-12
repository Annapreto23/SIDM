##################### cosmology-related functions #######################

# Arthur Fangzhou Jiang 2019 Hebrew University
# Arthur Fangzhou Jiang 2021 Caltech & Carnegie

# On 2021-05-04, added Benson+21 values of the PCH08 merger tree params
# Refactored 2025 to use Colossus instead of Cosmolopy

#########################################################################

import sidm.config as cfg

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# Colossus imports
from colossus.cosmology import cosmology
import colossus.lss.peaks as peaks
import colossus.lss.mass_function as mass_function
from colossus.halo import mass_so

# Helper to get current cosmology without repeatedly calling getCurrent() if possible,
# but calling getCurrent() is safe and fast in Colossus.

#########################################################################

#---basics 

def rhoc(z,h=0.7,Om=0.3,OL=0.7):
    """
    Critical density [Msun kpc^-3] at redshift z.
    """
    # Colossus returns M_sun * h^2 / kpc^3. To get physical M_sun / kpc^3, we multiply by h^2.
    # Note: Colossus density functions return values in (M_sun*h^2)/kpc^3 units.
    # If the user expects M_sun/kpc^3, we need to convert.
    # Physical density = Colossus density * h**2.
    cosmo = cosmology.getCurrent()
    return cosmo.rho_c(z) * cfg.h**2
    
def rhom(z,h=0.7,Om=0.3,OL=0.7):
    """
    Mean density [Msun kpc^-3] at redshift z.
    """
    cosmo = cosmology.getCurrent()
    return cosmo.rho_m(z) * cfg.h**2
    
def DeltaBN(z,Om=0.3,OL=0.7):
    """
    Virial overdensity of Bryan & Norman (1998).
    """
    # Colossus calculates virial overdensity relative to critical density?
    # mass_so.deltaVir returns the overdensity relative to CRITICAL or MEAN?
    # Colossus documentation: mass_so.deltaVir(z) returns delta_vir relative to rho_crit?
    # It usually depends.
    # Let's use the explicit approximation if Colossus doesn't match EXACTLY the BN formula the user used.
    # User formula: 18 pi^2 + 82x - 39x^2 where x = Omega(z)-1.
    # This is consistent with Colossus implementation for Flat LambdaCDM usually.
    # But to ensure 1:1 reproduction, we can keep the user's formula using colossus Omega(z).
    x = Omega(z) - 1.
    return 18.*np.pi**2 + 82.*x - 39.*x**2

def Omega(z,Om=0.3,OL=0.7):
    """
    Matter density in units of the critical density, at redshift z.
    """
    cosmo = cosmology.getCurrent()
    return cosmo.Om(z)

def tdyn(z,h=0.7,Om=0.3,OL=0.7):
    """
    Halo dynamical time [Gyr].
    """
    # User definition: sqrt(2/DeltaBN) / H(z)
    return np.sqrt(2./DeltaBN(z,Om,OL)) / H(z,h,Om,OL)

def Ndyn(z1,z2,h=0.7,Om=0.3,OL=0.7):
    """
    Number of halo dynamical times elapsed between redshift z1 and z2.
    """
    return quad(dNdz, z1,z2, args=(h,Om,OL,),
        epsabs=1.e-7, epsrel=1.e-6,limit=10000)[0]

def dNdz(z,h,Om,OL):
    """Auxiliary function for Ndyn"""
    return dtdz(z,h,Om,OL) / tdyn(z,h,Om,OL)

def dtdz(z,h,Om,OL):
    """dt / dz"""
    # Use finite difference or analytical derivative
    # Colossus has age(z).
    cosmo = cosmology.getCurrent()
    # approx derivative
    z1 = z*(1.-cfg.eps)
    z2 = z*(1.+cfg.eps)
    t1 = cosmo.age(z1) 
    t2 = cosmo.age(z2)
    return (t1-t2) / (z1-z2)

def H(z,h=0.7,Om=0.3,OL=0.7):
    """
    Hubble constant [Gyr^-1] at redshift z.
    """
    cosmo = cosmology.getCurrent()
    # Colossus Hz is in km/s/Mpc.
    # 1 km/s/Mpc = 1.02271e-3 Gyr^-1
    return cosmo.Hz(z) * 0.00102271

def E(z,Om=0.3,OL=0.7):
    """
    E(z) = H(z)/H0
    """
    cosmo = cosmology.getCurrent()
    return cosmo.Ez(z)
    
def t(z,h=0.7,Om=0.3,OL=0.7):
    """
    Cosmic time [Gyr].
    """
    cosmo = cosmology.getCurrent()
    return cosmo.age(z)

def tlkbk(z,h=0.7,Om=0.3,OL=0.7):
    """
    Lookback time [Gyr].
    """
    cosmo = cosmology.getCurrent()
    t = cosmo.lookbackTime(z)
    # Ensure output matches input shape preference (array vs scalar)
    if np.ndim(z) > 0 and np.shape(t) == ():
        # If input was array but output is scalar, broadcast it (though unlikely unless bug)
        # OR force vectorization if colossus fails
        # But assuming t is float:
        return np.full_like(z, t, dtype=float)
    return t

#------------------------- for EPS formalism ----------------------------

def deltac(z,Om=0.3):
    """Critical linearized overdensity for spherical collapse."""
    cosmo = cosmology.getCurrent()
    if np.ndim(z) > 0:
        return np.array([peaks.collapseOverdensity(z=zi) for zi in z])
    return peaks.collapseOverdensity(z=z)

def D(z,Om=0.3):
    """Linear growth rate D(z). normalized to 1 at z=0?"""
    # User's old cosmolopy function likely normalized D(0)=1.
    cosmo = cosmology.getCurrent()
    return cosmo.growthFactor(z) 

# transfer function    
def T(k, **cosmo_params):
    """Transfer function."""
    # Colossus transfer function
    cosmo = cosmology.getCurrent()
    # k in h/Mpc?
    # User: k in h Mpc^-1.
    # Colossus: k in h Mpc^-1.
    return cosmo.transferFunction(k)

# power spectrum
def P(k,z=0.,**cosmo_params):
    """Power spectrum."""
    cosmo = cosmology.getCurrent()
    return cosmo.matterPowerSpectrum(k, z)

def k0(**cosmo):
    """Dummy or calculated k0 - largely obsolete with Colossus"""
    # If forced to return something, return what Config calculated, or 1.0
    if 'k0' in cfg.cosmo:
        return cfg.cosmo['k0']
    return 1.0 

def sigmaR(R,**cosmo):
    """Variance in sphere R [Mpc/h]."""
    cosmo = cosmology.getCurrent()
    return cosmo.sigma(R, z=0.0)
    
def sigma(M,z=0.,**cosmo):
    """
    Variance sigma(M, z).
    M in M_sun.
    """
    cosmo = cosmology.getCurrent()
    
    # Calculate R from M
    # M_col = M * h [M_sun/h]
    # rho_col = rho_m(0) [M_sun h^2 / kpc^3]
    # R^3 = 3 * M_col / (4pi * rho_col) [kpc^3 / h^3]
    # R_kpch = (3 * M_col / (4pi * rho_col))^(1/3) [kpc/h]
    # R_Mpch = R_kpch / 1000.0 [Mpc/h]
    
    M_col = M * cfg.h
    rho_col = cosmo.rho_m(0.0)
    R_kpch = (3.0 * M_col / (4.0 * np.pi * rho_col))**(1.0/3.0)
    R_Mpch = R_kpch / 1000.0
    
    return cosmo.sigma(R_Mpch, z=z)

def nu(M,z=0,**cosmo):
    """Peak height"""
    # Colossus peakHeight
    cosmo = cosmology.getCurrent()
    M_h = M * cfg.h
    return peaks.peakHeight(M_h, z)

# Parkinson+08 algorithm logic
# Preserved mostly as is, but calling new primitives

def dlnSdlnM(M,**cosmo):
    return 2.* dlnsigmadlnM(M,**cosmo)

def dlnsigmadlnM(M,**cosmo):
    # Finite difference
    M1 = (1.+cfg.eps)*M
    M2 = (1.-cfg.eps)*M
    sigma1 = sigma(M1,0.,**cosmo)
    sigma2 = sigma(M2,0.,**cosmo)
    return (np.log(sigma1) - np.log(sigma2))/(np.log(M1) - np.log(M2))

def UpdateGlobalVariables(**cosmo):
    """
    Update a few intermediate global variables that are repeatedly used 
    by the functions for the Parkinson+08 algorithm.
    """
    cfg.W0 = deltac(cfg.z0,cosmo.get('omega_M_0', None))
    if cfg.M0>cfg.Mres:
        cfg.qres = min(cfg.Mres/cfg.M0,0.499)
    else:
        cfg.qres = min(cfg.Mmin/cfg.M0,0.499)
    cfg.sigmares = sigma(cfg.qres*cfg.M0,0.,**cosmo)
    cfg.sigma0 = sigma(cfg.M0,0.,**cosmo)
    cfg.sigmah = sigma(0.5*cfg.M0,0.,**cosmo)
    cfg.S0 = cfg.sigma0**2
    cfg.Sh = cfg.sigmah**2
    Sres = cfg.sigmares**2
    cfg.alphah = -dlnsigmadlnM(0.5*cfg.M0,**cosmo)
    cfg.ures = cfg.sigma0/np.sqrt(Sres-cfg.S0)
    Vres = Sres / (Sres - cfg.S0)**1.5
    Vh = cfg.Sh / (cfg.Sh - cfg.S0)**1.5
    cfg.beta = np.log(Vres/Vh) / np.log(2.*cfg.qres)
    cfg.B = 2.0**cfg.beta * Vh
    cfg.mu = cfg.alphah if cfg.gamma1>=0. else \
        - np.log(cfg.sigmares/cfg.sigmah) / np.log(2.*cfg.qres)
    cfg.eta = cfg.beta - 1. - cfg.gamma1*cfg.mu
    cfg.NupperOverdW = NupperOverdW()
    cfg.dW = dW()
   
def R(q,**cosmo): 
    M1 = q*cfg.M0
    S1 = sigma(M1,0.,**cosmo)**2
    V = S1 / (S1 - cfg.S0)**1.5
    V = V * (1.- cfg.S0/S1)**cfg.gamma3
    fac1 = -dlnsigmadlnM(M1,**cosmo) / cfg.alphah
    fac2 = V / (cfg.B * q**cfg.beta)
    fac3=((2.*q)**cfg.mu *sigma(M1,0.,**cosmo)/cfg.sigmah)**cfg.gamma1
    Rtmp = fac1 * fac2 * fac3
    return Rtmp
    
def dW():
    dW1 = 0.1 * cfg.Root2 * np.sqrt(cfg.Sh-cfg.S0)
    dW2 = 0.1 / cfg.NupperOverdW
    return min(dW1,dW2)

def NupperOverdW():
    A = cfg.Root2OverPi * cfg.B * cfg.alphah * cfg.G0 \
        / 2.**(cfg.mu*cfg.gamma1) * (cfg.W0/cfg.sigma0)**cfg.gamma2 \
        * (cfg.sigmah/cfg.sigma0)**cfg.gamma1
    if cfg.qres>=(0.5-cfg.eps):
        I = cfg.eps
    else:
        if np.abs(cfg.eta)>cfg.eps:
            I = (0.5**cfg.eta - cfg.qres**cfg.eta)/cfg.eta
        else:
            I = - np.log(2.*cfg.qres)
    return A * I

def J(ures):
    return quad(dJdu,0.,ures,epsabs=1e-7,epsrel=1e-6,limit=50)[0]
J_vec = np.vectorize(J, doc="Vectorized 'J(u_res)' function")

def dJdu(u):
    return (1.+1./u**2)**(cfg.gamma1/2.)
    
def F():
    return min(0.5, cfg.Root2OverPi * cfg.Jures_interp(cfg.ures) * \
           cfg.G0/cfg.sigma0 * (cfg.W0/cfg.sigma0)**cfg.gamma2 * cfg.dW)
           
def DrawProgenitors(**cosmo):
    r1 = np.random.random()
    Nupper = cfg.NupperOverdW * cfg.dW
    Np = 0 # initialize
    if r1 > Nupper:
        M1 = cfg.M0 * (1.-F())
        M2 = 0.
    else:
        r2 = np.random.random()
        q = (cfg.qres**cfg.eta + \
            r2*(2.**(-cfg.eta) - cfg.qres**cfg.eta))**(1./cfg.eta)
        r3 = np.random.random()
        if (r3<R(q,**cosmo)):
            Mtmp1 = cfg.M0 * (1.-F()-q)
            Mtmp2 = cfg.M0 * q
            M1 = max(Mtmp1,Mtmp2)
            M2 = min(Mtmp1,Mtmp2)
        else:
            M1 = cfg.M0 * (1.-F()) 
            M2 = 0.
    if M1>cfg.Mres: Np += 1
    if M2>cfg.Mres: Np += 1
    return M1,M2,Np

# EPS conditional mass function & progenitor mass function

def Masterisk(z=0.,height=1.,**cosmo):
    return brentq(FindMasterisk, 1e1, 1e17, args=(z,height,cosmo), 
        xtol=1e-5, rtol=1e-3, maxiter=100)
def FindMasterisk(M,z,height,cosmo):
    return nu(M,z,**cosmo) - height
    
def dNdlnM1(M1,z1,M0,z0,**cosmo):
    return M0/M1*dPdlnM1(M1,z1,M0,z0,**cosmo)
def dPdlnM1(M1,z1,M0,z0,**cosmo):
    Om = Omega(0) # z=0 Om for deltac ref? No, deltac(z) needs Om(z)?
    # Wait, user passed 'Om' from args to deltac usually.
    # deltac(z, Om) uses Om at z=0 usually?
    # User code used default Om=0.3.
    # Colossus handles this internally via getCurrent().
    # But we need to match user signature or adapt logic.
    S1 = sigma(M1,0.,**cosmo)**2.
    S0 = sigma(M0,0.,**cosmo)**2.
    W1 = deltac(z1)
    W0 = deltac(z0)
    return fEPS(S1,W1,S0,W0) * S1 * (-dlnSdlnM(M1,**cosmo))
def fEPS(S1,W1,S0,W0):
    DeltaS = S1-S0
    v10 = (W1-W0)/np.sqrt(DeltaS)
    return 0.2 *v10**0.75 /DeltaS *np.exp(-0.1 *v10**3) 
    
def NGTM1(M1,z1,M0,z0,**cosmo):
    a = np.log(M1)
    b = np.log(M0)
    return quad(dNGTM1dlnM1, a, b, args=(z1,M0,z0,cosmo),
        epsabs=1e-4, epsrel=1e-3,limit=100)[0]
def dNGTM1dlnM1(lnM1,z1,M0,z0,cosmo):
    M1 = np.exp(lnM1)
    return dNdlnM1(M1,z1,M0,z0,**cosmo)
    
def MGTM1(M1,z1,M0,z0,**cosmo):
    a = np.log(M1)
    b = np.log(M0)
    return quad(dMGTM1dlnM1, a, b, args=(z1,M0,z0,cosmo),
        epsabs=1e-4, epsrel=1e-3,limit=100)[0]
def dMGTM1dlnM1(lnM1,z1,M0,z0,cosmo):
    M1 = np.exp(lnM1)
    return M1*dNdlnM1(M1,z1,M0,z0,**cosmo)
 
def dNdlnmaM0_all(x,gamma,alpha,beta,zeta):
    return gamma* x**alpha * np.exp(-beta*x**zeta)

def dNdlnmaM0_1st(x,gamma1,gamma2,alpha1,alpha2,beta,zeta):
    return (gamma1*x**alpha1+gamma2*x**alpha2)*np.exp(-beta*x**zeta)
