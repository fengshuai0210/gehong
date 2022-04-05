from __future__ import print_function

from os import path
import glob

import sys
import numpy as np
from scipy import ndimage
from astropy.io import fits
from scipy import ndimage,constants

import numpy as np
import sys

def readcol(filename, **kwargs):
    
    f = np.genfromtxt(filename, dtype=None, **kwargs)

    t = type(f[0])
    if t == np.ndarray or t == np.void: # array or structured array
        f = map(np.array, zip(*f))

    # In Python 3.x all strings (e.g. name='NGC1023') are Unicode strings by defauls.
    # However genfromtxt() returns byte strings b'NGC1023' for non-numeric columns.
    # To have the same behaviour in Python 3 as in Python 2, I convert the Numpy
    # byte string 'S' type into Unicode strings, which behaves like normal strings.
    # With this change I can read the string a='NGC1023' from a text file and the
    # test a == 'NGC1023' will give True as expected.

    if sys.version >= '3':
        f = [v.astype(str) if v.dtype.char=='S' else v for v in f]

    return f

def log_rebin(lamRange, spec, oversample=False, velscale=None, flux=False):
    
    lamRange = np.asarray(lamRange)
    assert len(lamRange) == 2, 'lamRange must contain two elements'
    assert lamRange[0] < lamRange[1], 'It must be lamRange[0] < lamRange[1]'
    s = spec.shape
    assert len(s) == 1, 'input spectrum must be a vector'
    n = s[0]
    if oversample:
        m = int(n*oversample)
    else:
        m = int(n)

    dLam = np.diff(lamRange)/(n - 1.)        # Assume constant dLam
    lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
    borders = np.linspace(*lim, num=n+1)     # Linearly
    logLim = np.log(lim)

    c = 299792.458                           # Speed of light in km/s
    if velscale is None:                     # Velocity scale is set by user
        velscale = np.diff(logLim)/m*c       # Only for output
    else:
        logScale = velscale/c
        m = int(np.diff(logLim)/logScale)    # Number of output pixels
        logLim[1] = logLim[0] + m*logScale

    newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
    k = (newBorders - lim[0]).clip(0, n-1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0    # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k])*spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

    return specNew, logLam, velscale

def gaussian_filter1d(spec, sig):
    
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum

###############################################################################
# SSP Template

def age_metal(filename):
    
    s = path.basename(filename)
    age = float(s[s.find("T")+1:s.find("_iPp0.00_baseFe.fits")])
    metal = s[s.find("Z")+1:s.find("T")]
    if "m" in metal:
        metal = -float(metal[1:])
    elif "p" in metal:
        metal = float(metal[1:])
    else:
        raise ValueError("This is not a standard MILES filename")

    return age, metal

class emiles(object):

    def __init__(self, velscale=50, FWHM_inst = 0.1,
                 pathname='./data/EMILES_PADOVA00_BASE_CH_FITS/Ech*_baseFe.fits', 
                 normalize=False):
        
        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_range_temp = h2['CRVAL1'] + np.array([0, h2['CDELT1']*(h2['NAXIS1']-1)])
        sspNew, log_lam_temp = log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]
        #wave=((np.arange(hdr['NAXIS1'])+1.0)-hdr['CRPIX1'])*hdr['CDELT1']+hdr['CRVAL1']
        
        templates = np.empty((sspNew.size, n_ages, n_metal))
        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        
        # FWHM of Emiles templates
        Emile_wave = np.exp(log_lam_temp)
        Emile_FWHM = np.zeros(h2['NAXIS1'])
        Emile_FWHM[np.where(Emile_wave < 3060)] = 3.
        Emile_FWHM[np.where((Emile_wave >= 3060) & (Emile_wave < 3540))] = 3.
        Emile_FWHM[np.where((Emile_wave >= 3540) & (Emile_wave < 8950))] = 2.5
        Lwave = Emile_wave[np.where(Emile_wave >= 8950)]
        Emile_FWHM[np.where(Emile_wave >= 8950)]=60*2.35/3.e5*Lwave  # sigma=60km/s at lambda > 8950
        
        LSF = Emile_FWHM
        
        FWHM_eff=Emile_FWHM.copy()   # combined FWHM from stellar library and instrument(input)
        if np.isscalar(FWHM_inst):
            FWHM_eff[Emile_FWHM < FWHM_inst] = FWHM_inst
            LSF[Emile_FWHM < FWHM_inst] = FWHM_inst
        else:
            FWHM_eff[Emile_FWHM < FWHM_inst] = FWHM_inst[Emile_FWHM < FWHM_inst]
            LSF[Emile_FWHM < FWHM_inst] = FWHM_inst[Emile_FWHM < FWHM_inst]
        FWHM_dif = np.sqrt(FWHM_eff**2 - Emile_FWHM**2)
        sigma_dif = FWHM_dif/2.355/h2['CDELT1']   # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, metal in enumerate(metals):
                p = all.index((age, metal))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                if np.isscalar(FWHM_dif):
                    ssp = ndimage.gaussian_filter1d(ssp, sigma_dif)
                else:
                    ssp = gaussian_filter1d(ssp, sigma_dif)  # convolution with variable sigma
                sspNew = log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                if normalize:
                    sspNew /= np.mean(sspNew)
                templates[:, j, k] = sspNew
                age_grid[j, k] = age
                metal_grid[j, k] = metal

        self.templates = templates/np.median(templates)  # Normalize by a scalar
        self.log_lam_temp = log_lam_temp
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.LSF = log_rebin(lam_range_temp, LSF, velscale=velscale)[0]
        self.velscale = velscale
        
###############################################################################
# Stellar Template

class stellarlib():
        
    def __init__(self, velscale=50, FWHM_inst = 0.1,
                 pathname='MILES.StellarLib.v9.1.fits', 
                 normalize=False):
        
        hdulist = fits.open('MILES.StellarLib.v9.1.fits')
        par = hdulist[1].data
        lam = hdulist[2].data
        flux = hdulist[3].data
        
        lam_range_temp = [lam[0], lam[-1]]
        temNew, log_lam_temp = util.log_rebin(lam_range_temp, flux[1,:], velscale=velscale)[:2]

        temp = np.empty((temNew.size, par.size))
        for i in range(par.size):
            temp[:,i] = util.log_rebin(lam_range_temp, flux[i,:], velscale=velscale)[0]

        self.templates = temp
        self.log_lam_temp = log_lam_temp
        self.teff_grid = par['Teff']
        self.feh_grid = par['FeH']
        self.LSF = 2.5
        self.velscale = velscale