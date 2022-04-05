from __future__ import print_function
import numpy as np
import temp as temp
from scipy.interpolate import interp1d
import astropy.units as u

# ----------------
# Reddening Module

def Calzetti_Law(wave, Rv = 4.05):
    """
    Dust Extinction Curve by Calzetti et al. (2000)
    """
    wave_number = 1./(wave * 1e-4)
    reddening_curve = np.zeros(len(wave))
    
    idx = np.logical_and(wave > 1200, wave < 6300)
    reddening_curve[idx] = 2.659 * ( -2.156 + 1.509 * wave_number[idx] - 0.198 * \
                    (wave_number[idx] ** 2)) + 0.011 * (wave_number[idx] **3 ) + Rv
                                    
    idx = np.logical_and(wave > 6300, wave < 22000)
    reddening_curve[idx] = 2.659 * ( -1.857 + 1.040 * wave_number[idx]) + Rv
    return reddening_curve

def reddening(wave, flux, ebv = 0, law = 'calzetti', Rv = 4.05):
    """
    Redden an input spectra through a given reddening curve.
    """
    if law == 'calzetti':
        curve = Calzetti_Law(wave, Rv = Rv)
        fluxNew = flux / (10. ** (0.4 * ebv * curve))
    return fluxNew

def calibrate(wave, flux, mag, filtername='SLOAN_SDSS.r'):
    """
    Calibrate the spectra according to the magnitude.
    """
        
    # Loading response curve
    if filtername == '5100':
        wave0 = np.linspace(3000,10000,7000)
        response0 = np.zeros(7000)
        response0[(wave0 > 5050) & (wave0 < 5150)] = 1.
    else:
        filter_file = './data/filter/'+filtername+'.filter'
        wave0, response0 = temp.readcol(filter_file)
    
    # Setting the response
    func = interp1d(wave0, response0)
    response = np.copy(wave)
    ind_extra = (wave > max(wave0)) | (wave < min(wave0))
    response[ind_extra] = 0
    ind_inside = (wave < max(wave0)) & (wave > min(wave0))
    response[ind_inside] = func(wave[ind_inside])
        
    # Flux map of datacube for given filter band
    preflux = np.sum(flux * response * np.mean(np.diff(wave))) / np.sum(response * np.mean(np.diff(wave)))
    
    # Real flux from magnitude for given filter
    realflux = (mag * u.STmag).to(u.erg/u.s/u.cm**2/u.AA).value

    # Normalization
    flux_ratio = realflux / preflux
    flux_calibrate = flux * flux_ratio * 1e17                        # Units: 10^-17 erg/s/A/cm^2
    
    return flux_calibrate

class star():
    
    """
    The spectra of single star
    """
    
    def __init__(self, stellarlib, obswave, mag, Teff, FeH, vel, Ebv = 0, dFeH = 0.2):
        
        """
        Modelling the spectra of single stars
        
        Pars:
            stellarlib - The class of stellar library
            obswave    - Wavelength, 1D-array
            mag        - Magnitude in r-band used for the calibration of spectra, scalar
            Teff       - Effective tempareture used for determining the spectra profile, scalar (K)
            FeH        - Metallicity used for determining the spectra profile, scalar 
            vel        - Line of sight velocity used for determining the doppler effect, scalar (km/s)
            Ebv        - Extinction, scalar (mag), default 0 (no dust extinction)
            dFeH       - Range of FeH when searching the best templates, default 0.2 (searching the template
                         with -0.2 < FeH - dFeh < 0.2)
                         
        Attri:
            wave       - Wavelength is the same with obswave in Pars
            flux       - Flux of best model
        """
        
        # -------------------------
        # Spectra of single stellar
        
        temp = stellarlib.templates

        # Select FeH bins
        metals = stellarlib.feh_grid
        idx_feh = (metals > FeH - dFeH) & (metals < FeH + dFeH)
        tpls = temp[:, idx_feh]
        
        # Select Teff bins
        tt = stellarlib.teff_grid
        minloc = np.argmin(abs(tt[idx_feh] - Teff))
        flux0 = tpls[:, minloc]
        
        wave = np.exp(stellarlib.log_lam_temp)
        
        # Dust Reddening
        #if np.isscalar(Ebv):
        #    flux0 = reddening(wave, flux0, ebv = Ebv)
        
        # Calibaration
        flux = calibrate(wave, flux0, mag, filtername='SLOAN_SDSS.r')
        
        # Redshift
        redshift = vel / 3e5
        wave_r = wave * (1 + redshift)
        obsflux = np.interp(obswave, wave_r, flux)

        # Convert to input wavelength
        self.wave = obswave
        self.flux = obsflux
        
class StarPop():
    
    def __init__(self, ssp, obswave, mag = 14, Age = 3, FeH = 0, vel = 0, vdisp = 50, Ebv = 0):

        """
        Modelling the spectra of stellar population
        
        Pars:
            ssp        - The class of simple stellar population
            obswave    - Wavelength, 1D-array
            mag        - Magnitude in r-band used for the calibration of spectra, scalar
            Age        - Mean age of stellar population used for determining the spectra profile, scalar (Gyr)
            FeH        - Mean FeH of stellar population used for determining the spectra profile, scalar 
            vel        - Line of sight velocity used for determining the doppler effect, scalar (km/s)
            vdisp      - Line of sight velocity dispersion used for broadening the spectra, scalar (km/s)
            Ebv        - Extinction, scalar (mag), default 0 (no dust extinction)
            dFeH       - Range of FeH when searching the best templates, default 0.2 (searching the template
                         with -0.2 < FeH - dFeh < 0.2)
                         
        Attri:
            wave       - Wavelength is the same with obswave in Pars
            flux       - Flux of best model
        """
        
        # -----------------
        # Stellar Continuum
        
        SSP_temp=ssp.templates

        # Select metal bins
        metals = ssp.metal_grid[0,:]
        minloc = np.argmin(abs(FeH-metals))
        tpls = SSP_temp[:, :, minloc]
        #fmass = ssp.fmass_ssp()[:, minloc]
        
        # Select age bins
        Ages = ssp.age_grid[:,0]
        minloc = np.argmin(abs(Age-Ages))
        Stellar = tpls[:, minloc]
        
        wave = np.exp(ssp.log_lam_temp)
        
        # Broadening caused by Velocity Dispersion
        sigma_gal = vdisp / ssp.velscale                   # in pixel
        sigma_LSF = ssp.LSF / ssp.velscale                 # in pixel
        
        if sigma_gal>0: 
            sigma_dif = np.zeros(len(Stellar))
            idx = (sigma_gal > sigma_LSF)
            sigma_dif[idx] = np.sqrt(sigma_gal ** 2. - sigma_LSF[idx] ** 2.)
            idx = (sigma_gal <= sigma_LSF)
            sigma_dif[idx] = 0.1
            flux0 = temp.gaussian_filter1d(Stellar, sigma_dif)
        
        # Dust Reddening
        if np.isscalar(Ebv):
            flux0 = reddening(wave, flux0, ebv = Ebv)
            
        # Redshift
        redshift = vel / 3e5
        wave_r = wave * (1 + redshift)
        
        flux = np.interp(obswave, wave_r, flux0)
        
        # Calibration
        if np.isscalar(mag):
            flux = calibrate(obswave, flux, mag, filtername='SLOAN_SDSS.r')
        
        # Convert to input wavelength
        self.wave = obswave
        self.flux = flux
        
from scipy import special, fftpack
def emline(logLam_temp, line_wave, FWHM_inst, pixel=True):
    """
    Model of single emission line
    """
    if callable(FWHM_inst):
        FWHM_inst = FWHM_inst(line_wave)

    n = logLam_temp.size
    npad = fftpack.next_fast_len(n)
    nl = npad//2 + 1  # Expected length of rfft

    dx = (logLam_temp[-1] - logLam_temp[0])/(n - 1)
    x0 = (np.log(line_wave) - logLam_temp[0])/dx
    xsig = FWHM_inst/2.355/line_wave/dx    # sigma in pixels units
    w = np.linspace(0, np.pi, nl)[:, None]

    # Gaussian with sigma=xsig and center=x0,
    # optionally convolved with an unitary pixel UnitBox[]
    # analytically defined in frequency domain
    # and numerically transformed to time domain
    rfft = np.exp(-0.5*(w*xsig)**2 - 1j*w*x0)
    if pixel:
        rfft *= np.sinc(w/(2*np.pi))
    line = np.fft.irfft(rfft, n=npad, axis=0)

    return line[:n, :]

def emission_lines(logLam_temp, lamRange_gal, FWHM_gal):
    """
    Emission lines Model
    """
    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha
    line_wave = np.array([4101.76, 4340.47, 4861.33, 6562.80])  # air wavelengths
    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta', 'Halpha'])
    emission_lines = emline(logLam_temp, line_wave, FWHM_gal)

    #                 -----[OII]-----    -----[SII]-----
    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85])  # air wavelengths
    names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731'])
    gauss = emline(logLam_temp, lines, FWHM_gal)
    emission_lines = np.append(emission_lines, gauss, 1)
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                 -----[OIII]-----
    lines = np.array([4958.92, 5006.84])    # air wavelengths
    doublet = 0.33*emline(logLam_temp, lines[0], FWHM_gal) + emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                  -----[OI]-----
    lines = np.array([6300.30, 6363.67])    # air wavelengths
    doublet = emline(logLam_temp, lines[0], FWHM_gal) + 0.33*emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[OI]6300d') # single template for this doublet
    line_wave = np.append(line_wave, lines[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                 -----[NII]-----
    lines = np.array([6548.03, 6583.41])    # air wavelengths
    doublet = 0.33*emline(logLam_temp, lines[0], FWHM_gal) + emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[NII]6583d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the pPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    return emission_lines, line_names, line_wave

class Gas():
    
    def __init__(self, obswave, Haflux, FeH, vel, vdisp, velscale = 50, Ebv = 0.):
        
        """
        Modelling the spectra of stellar population
        
        Pars:
            obswave    - Wavelength, 1D-array
            Haflux     - Halpha flux used for the calibration of spectra, scalar (1e-17 erg/s/cm^2)
            FeH        - Mean FeH of ionized used for determining the emissiion line ratio, scalar 
            vel        - Line of sight velocity used for determining the doppler effect, scalar (km/s)
            vdisp      - Line of sight velocity dispersion used for broadening the spectra, scalar (km/s)
            Ebv        - Extinction, scalar (mag), default 0 (no dust extinction)
                         
        Attri:
            wave       - Wavelength is the same with obswave in Pars
            flux       - Flux of best model
        """
        
        _, logLam_Rest = temp.log_rebin([3000, 10000], np.zeros(4000), velscale=velscale)[:2]
        wave = np.exp(logLam_Rest)
        lamRange_Rest = [wave.min(), wave.max()]
        emlines, line_names, line_wave = emission_lines(logLam_Rest, lamRange_Rest, vdisp * 2.355 / velscale)

        # ------------------------------------------------------------------------
        # Flux Ratio, normalized at Ha, namely F(Ha)=1
        # Hdelta, Hgamma, Hbeta, Halpha, OII A, OII B, SII A, SII B, OIII, OI, NII

        if FeH == None:
            # Taken from the SDSS spectra of II Zw 072
            ratio = np.array([254 ,480, 1000, 2713, 1667, 2450, 378, 272, 3207, 87, 315]) / 1000. 
        else:
            Z = (10. ** FeH) * 0.02
            # Ref: Anders et al. (2003)
            if Z < 0.004:
                ratio = np.array([0.254, 0.48, 1, 2.73, 0.489*0.4, 0.489*0.6, 
                                  0.037, 0.029, 4.256, 0.008, 0.02])
            if (Z > 0.004) & (Z < 0.008):
                ratio = np.array([0.254, 0.48, 1, 2.73, 1.791*0.4, 1.791*0.6, 
                                  0.188, 0.138, 6.369, 0.041, 0.234])
            if Z > 0.008:
                ratio = np.array([0.254, 0.48, 1, 2.73, 3.010*0.4, 3.010*0.6, 
                                  0.3, 0.21, 5.48, 0.13, 0.54])

        emlines = emlines * ratio
        
        ha_flux = emlines[:, 3]
        ha_totflux = np.sum(ha_flux)
        flux0 = np.sum(emlines, axis = 1)
        
        # Calibration
        # Flux map of datacube for given filter band
        preflux = ha_totflux

        # Real flux #from magnitude for given filter
        realflux = Haflux * 1e-17 #(HaMag * u.STmag).to(u.erg/u.s/u.cm**2/u.AA).value

        # Normalization
        flux_ratio = realflux / preflux
        flux_calibrate = flux0 * flux_ratio * 1e17                        # Units: 10^-17 erg/s/A/cm^2
        
        # Dust attenuation
        if np.isscalar(Ebv):
            flux_dust = reddening(wave, flux_calibrate, ebv = 0.44 * Ebv)
            
        # Redshift
        redshift = vel / 3e5
        wave_r = wave * (1 + redshift)
    
        flux_red = np.interp(obswave, wave_r, flux_dust)
            
        self.wave = obswave
        self.flux = flux_red
        
class AGNlib():
        
    def __init__(self, velscale=50, FWHM_inst = 0.1,
                 pathname='AGN.Temp.v2.0.fits', 
                 normalize=False):
        
        hdulist = fits.open(pathname)
        par = hdulist[1].data
        lam = hdulist[2].data
        flux = hdulist[3].data
        
        lam_range_temp = [lam[0], lam[-1]]
        temNew, log_lam_temp = temp.log_rebin(lam_range_temp, flux, velscale=velscale)[:2]

        #temp = np.empty((temNew.size, par.size))
        #for i in range(par.size):
        temp = temp.log_rebin(lam_range_temp, flux, velscale=velscale)[0]

        self.templates = temp
        self.log_lam_temp = log_lam_temp
        self.stype_grid = par['Type']
        self.LSF = 2.5
        self.velscale = velscale
        
class AGN():
    
    """
    The spectra of AGN
    """
    
    def __init__(self, AGNlib, obswave, Mbh, Stype, vel, Dist = None, Ebv = 0, EddRatio = 1):
        
        """
        Modelling the spectra of AGN
        
        Pars:
            stellarlib - The class of stellar library
            obswave    - Wavelength, 1D-array
            Mbh        - Black hole mass, scalar (solar mass)
            Stype      - Seyfert type used for determining the spectra profile, scalar
            vel        - Line of sight velocity used for determining the doppler effect, scalar (km/s)
            Dist       - Distance of AGN, scalar (Mpc)
            Ebv        - Extinction, scalar (mag), default 0 (no dust extinction)
            EddRatio   - Eddington Ratio, scalar, default 1
                         
        Attri:
            wave       - Wavelength is the same with obswave in Pars
            flux       - Flux of best model
        """
        
        # -------------------------
        # Spectra of single stellar
        
        temp = AGNlib.templates

        # Select Template
        #stypes = AGNlib.stype_grid
        #minloc = np.argmin(abs(stypes - Stype))
        flux0 = temp#[:, minloc]
        
        wave = np.exp(AGNlib.log_lam_temp)
        
        # Dust Reddening
        if np.isscalar(Ebv):
            flux0 = reddening(wave, flux0, ebv = Ebv)
        
        
        # Calculate bolometric luminosity
        Ledd = 3e4 * Mbh
        Lbol = Ledd * EddRatio
        
        # Convert bolometric luminosity to 5100A luminosity (Marconi et al. 2004)
        L5100 = Lbol / 10.9
        M5100 = 4.86 - 2.5 * np.log10(L5100)
        m5100 = M5100 + 5. * np.log10(Dist * 1e5)
        
        # Calibaration
        flux = calibrate(wave, flux0, m5100, filtername='5100')
        
        # Redshift
        redshift = vel / 3e5
        wave_r = wave * (1 + redshift)
        obsflux = np.interp(obswave, wave_r, flux)

        # Convert to input wavelength
        self.wave = obswave
        self.flux = obsflux