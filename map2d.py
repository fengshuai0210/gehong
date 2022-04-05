from __future__ import division

import scipy.special as sp
from coords import Grid
import numpy as np
from astropy.io import fits
#import config

POS_EPSILON = 1e-8
SERSIC_OVERSAMPLE = 101

class Map2d():
    
    def __init__(self, src, inst=None, filename=None):
        
        if inst==None:
            inst=config.Instrument()
            
        self.pa = src.pa
        self.xoff = src.x0
        self.yoff = src.y0
        self.grid = inst.grid
        yrot, xrot = self.grid.shift_rotate(self.yoff, self.xoff, self.pa)
        self.xrot = xrot
        self.yrot = yrot
        self.pix_area_sqarcsec = inst.grid.xsamp * inst.grid.ysamp
        self.surf_area_units = src.surf_area_units
        
        self.filename=filename

    def pixelscale(self):
        if self.surf_area_units in ['sr']:
            arcsec2 = u.arcsec * u.arcsec
            normfactor = self.pix_area_sqarcsec / u.sr.to(arcsec2)  
        elif self.surf_area_units in ['arcsec^2', None]: 
            normfactor = self.pix_area_sqarcsec
        else:
            raise Exception("Unsupported surface area unit: %s" % self.surf_area_units)
        return normfactor
    
# ------------------
# Surface Brightness

class SBMap(Map2d):

    def __init__(self, src, inst=None, sbprof=None):
        
        Map2d.__init__(self, src, inst=inst)
        
        self.major = src.major
        self.minor = src.minor
        self.index = src.index
        self.surf_area_units = src.surf_area_units
        
        if np.ndim(sbprof)>0:
            if np.ndim(sbprof)!=2:
                raise Exception("If porf_type is NONE, sbprof should be 2D array!")
            else:
                self.sb = sbprof
                self.mag = self.sb - 2.5 * np.log10(self.pixelscale())
        else:        
            if src.prof_type=='sersic':
                self.total_mag = src.total_mag
                self.total_flux = 10. ** ((22.5 - self.total_mag) * 0.4)  # Unit: nanomaggy
                # the actual value of b. Formula taken from astropy's sersic2d shape.
                self.b = sp.gammaincinv(2 * self.index, 0.5)
                self.sersic_generator()
                self.normalize()
                self.sb = 22.5-2.5 * np.log10(self.norm_prof * self.total_flux / self.pixelscale())
                self.mag = self.sb - 2.5 * np.log10(self.pixelscale())
            else:
                raise Exception("Please define a surface brightness prifile by sbprof or src.prof_type!")

    def sersic_generator(self):
        center_grid = Grid(self.grid.xsamp/(SERSIC_OVERSAMPLE-1), self.grid.ysamp/(SERSIC_OVERSAMPLE-1),
                           SERSIC_OVERSAMPLE, SERSIC_OVERSAMPLE)
        yrot,xrot = center_grid.shift_rotate(0, 0, self.pa)
        center_pixel = self.sersic_func(yrot,xrot,self.major,self.minor,self.index)
        self.center_flux = np.sum(center_pixel) / (SERSIC_OVERSAMPLE**2)

        self.prof = np.zeros_like(self.grid.x)
        
        lx = (self.xoff // self.grid.xsamp) * self.grid.xsamp
        rx = (self.xoff % self.grid.xsamp) * self.grid.xsamp
        ly = (self.yoff // self.grid.ysamp) * self.grid.ysamp
        ry = (self.yoff % self.grid.ysamp) * self.grid.ysamp

        pixels = [[lx, ly, (1 - rx) * (1 - ry)], [lx, ly + self.grid.ysamp, (1 - rx) * ry],
                      [lx + self.grid.xsamp, ly, rx * (1 - ry)],
                      [lx + self.grid.xsamp, ly + self.grid.ysamp, rx * ry]]
        for tx,ty,frac in pixels:
            if frac != 0:
                yrot, xrot = self.grid.shift_rotate(ty, tx, self.pa)
                partial = self.sersic_func(yrot, xrot, self.major, self.minor, self.index)
                yzero = np.min(np.abs(yrot))
                xzero = np.min(np.abs(xrot))
                if np.abs(xzero) < POS_EPSILON * self.grid.xsamp and \
                np.abs(yzero) < POS_EPSILON * self.grid.ysamp:
                    xvals = np.where(np.abs(xrot.flatten()) < POS_EPSILON * self.grid.xsamp)[0]
                    yvals = np.where(np.abs(yrot.flatten()) < POS_EPSILON * self.grid.ysamp)[0]
                    xy = np.intersect1d(xvals,yvals)
                    xval = xy // self.grid.shape[0]
                    yval = xy % self.grid.shape[0]
                    partial[xval, yval] = self.center_flux
                self.prof += partial * frac

    def normalize(self):
        integral = self.major * self.minor * 2 * np.pi * self.index * np.exp(self.b)\
        /(self.b**(2*self.index))* sp.gamma(2 * self.index)
        self.norm_prof = self.prof / integral * self.pixelscale() 

    def sersic_func(self, y, x, major, minor, index):
        dist = np.sqrt((x / major)**2.0 + (y / minor)**2.0)
        profile = np.exp( -self.b * (dist**(1.0 / index) - 1) )
        return profile

# ---------------
# Kinematic Field

# Rotation curve function

def arctan_curve(r, vmax=150, rt=3):
    """
    Ref: Puech et al. 2008
    """
    vr=vmax*(2./np.pi)*np.arctan(r/rt)
    return vr

def exp_curve(r, vmax=150, rt=3):
    """
    Ref: Feng & Gallo 2013
    """
    vr=vmax*(1-np.exp(-r/rt))
    return vr

def tanh_curve(r, vmax=150, rt=3):
    """
    Ref: Andersen & Bershady 2013
    """
    vr=vmax*np.tanh(r/rt)
    return vr

class KinMap(Map2d):
    """
    Kinematic map, including velocity map and velocity dispersion map.
    
    
    """
    
    def __init__(self, src, inst=None, vel_map=None, veldisp_map=None,
                 curve='tanh', vmax=150, rt=None, sigma0=180, gredsigma=-5):
            
        Map2d.__init__(self, src, inst=inst)
    
        self.major = src.major
        self.minor = src.minor
        self.curve = curve
        self.vsys = src.redshift * 3e5
        
        if rt==None:
            rt=src.major
        
        if src.prof_type!='psf':
            self.velmap_generator(vmax=vmax,rt=rt)
            self.vel=self.prof + self.vsys
            self.sigma = np.copy(self.vel)
            self.sigma[:] = self.SigmaMap_generator(sigma0,gredsigma)
        else:
            prof=np.copy(self.xrot)
            prof[:]=0
            self.vel=prof + self.vsys
            self.sigma = self.SigmaMap_generator(sigma0,gredsigma)

    def velmap_generator(self,vmax=150,rt=3):
        center_grid = Grid(self.grid.xsamp/(SERSIC_OVERSAMPLE-1), self.grid.ysamp/(SERSIC_OVERSAMPLE-1),
                           SERSIC_OVERSAMPLE, SERSIC_OVERSAMPLE)
        yrot,xrot = center_grid.shift_rotate(0, 0, self.pa)
        center_pixel = self.velcurve_func(yrot,xrot,self.major,self.minor,vmax=vmax,rt=rt)
        self.center_flux = np.sum(center_pixel) / (SERSIC_OVERSAMPLE**2)

        self.prof = np.zeros_like(self.grid.x)

        lx = (self.xoff // self.grid.xsamp) * self.grid.xsamp
        rx = (self.xoff % self.grid.xsamp) * self.grid.xsamp
        ly = (self.yoff // self.grid.ysamp) * self.grid.ysamp
        ry = (self.yoff % self.grid.ysamp) * self.grid.ysamp

        pixels = [[lx, ly, (1 - rx) * (1 - ry)], [lx, ly + self.grid.ysamp, (1 - rx) * ry],
                      [lx + self.grid.xsamp, ly, rx * (1 - ry)],
                      [lx + self.grid.xsamp, ly + self.grid.ysamp, rx * ry]]
        for tx,ty,frac in pixels:
            if frac != 0:
                yrot, xrot = self.grid.shift_rotate(ty, tx, self.pa)
                partial = self.velcurve_func(yrot, xrot, self.major, self.minor,vmax=vmax,rt=rt)
                yzero = np.min(np.abs(yrot))
                xzero = np.min(np.abs(xrot))
                if np.abs(xzero) < POS_EPSILON * self.grid.xsamp and np.abs(yzero) < \
                POS_EPSILON * self.grid.ysamp:
                    xvals = np.where(np.abs(xrot.flatten()) < POS_EPSILON * self.grid.xsamp)[0]
                    yvals = np.where(np.abs(yrot.flatten()) < POS_EPSILON * self.grid.ysamp)[0]
                    xy = np.intersect1d(xvals,yvals)
                    xval = xy // self.grid.shape[0]
                    yval = xy % self.grid.shape[0]
                    partial[xval, yval] = self.center_flux
                self.prof += partial * frac
                
    def velcurve_func(self, y, x, major, minor, vmax=150, rt=1):
        dist = np.sqrt((x / major)**2.0 + (y / minor)**2.0) + 0.0001
        #dist = np.sqrt(x**2.0 + y**2.0) + 0.0001
        if self.curve.lower() in ['tanh', 'exp', 'arctan']:
            if self.curve.lower()=='tanh':
                vr = tanh_curve(dist, vmax=vmax, rt=rt) * ((x/major)/dist) 
            if self.curve.lower()=='exp':
                vr = exp_curve(dist, vmax=vmax, rt=rt) * (x/dist)
            if self.curve.lower()=='arctan':
                vr = arctan_curve(dist, vmax=vmax, rt=rt) * (x/dist)
        else:
            raise Exception("curve must be included in 'tanh', 'exp', 'arctan'")
        return vr
    
    def SigmaMap_generator(self, A0, gred):
        center_grid = Grid(self.grid.xsamp/(SERSIC_OVERSAMPLE-1), self.grid.ysamp/(SERSIC_OVERSAMPLE-1),
                           SERSIC_OVERSAMPLE, SERSIC_OVERSAMPLE)
        yrot,xrot = center_grid.shift_rotate(0, 0, self.pa)
        center_pixel = self.GredSigma_func(yrot, xrot, self.major, self.minor, A0, gred)
        self.center_flux = np.sum(center_pixel) / (SERSIC_OVERSAMPLE**2)

        self.prof = np.zeros_like(self.grid.x)
        
        lx = (self.xoff // self.grid.xsamp) * self.grid.xsamp
        rx = (self.xoff % self.grid.xsamp) * self.grid.xsamp
        ly = (self.yoff // self.grid.ysamp) * self.grid.ysamp
        ry = (self.yoff % self.grid.ysamp) * self.grid.ysamp

        pixels = [[lx, ly, (1 - rx) * (1 - ry)], [lx, ly + self.grid.ysamp, (1 - rx) * ry],
                      [lx + self.grid.xsamp, ly, rx * (1 - ry)],
                      [lx + self.grid.xsamp, ly + self.grid.ysamp, rx * ry]]
        for tx,ty,frac in pixels:
            if frac != 0:
                yrot, xrot = self.grid.shift_rotate(ty, tx, self.pa)
                partial = self.GredSigma_func(yrot, xrot, self.major, self.minor, A0, gred)
                yzero = np.min(np.abs(yrot))
                xzero = np.min(np.abs(xrot))
                if np.abs(xzero) < POS_EPSILON * self.grid.xsamp and \
                np.abs(yzero) < POS_EPSILON * self.grid.ysamp:
                    xvals = np.where(np.abs(xrot.flatten()) < POS_EPSILON * self.grid.xsamp)[0]
                    yvals = np.where(np.abs(yrot.flatten()) < POS_EPSILON * self.grid.ysamp)[0]
                    xy = np.intersect1d(xvals,yvals)
                    xval = xy // self.grid.shape[0]
                    yval = xy % self.grid.shape[0]
                    partial[xval, yval] = self.center_flux
                self.prof += partial * frac
        return self.prof

    def GredSigma_func(self, y, x, major, minor, A0, gred):
        dist = np.sqrt((x / major)**2.0 + (y / minor)**2.0) + 0.0001
        profile = A0 + dist * gred
        return profile
        
class GasMap(Map2d):
    """
    Maps of ionized gas.
    """
    
    def __init__(self, src, haflux=5, oh=None, ebv=0,
                 inst=None):
            
        Map2d.__init__(self, src, inst=inst)
    
        self.major = src.major
        self.minor = src.minor
        
        # Setting the equivelent width of Halpha
        if np.ndim(haflux)==0:
            self.haflux = np.copy(self.xrot)
            self.haflux[:] = haflux
        if np.ndim(haflux)==2:
            self.haflux=load_image(haflux,(inst.nx,inst.ny))
            
            
        # Setting the dust extinction
        if np.ndim(ebv)==0:
            self.ebv = np.copy(self.xrot)
            self.ebv[:] = ebv
        if np.ndim(ebv)==2:
            self.ebv=load_image(ebv,(inst.nx,inst.ny))
        
class StellarMap(Map2d):
    """
    Stellar population
    """
    
    def __init__(self, src, age=5, feh=0, inst=None, GredZ=0, GredAge=0):
        
        Map2d.__init__(self, src, inst=inst)
    
        self.major = src.major
        self.minor = src.minor
        
        # Age Distribution
        if np.ndim(age)==0:
            self.age = np.copy(self.xrot)
            self.age[:] = 10. ** self.GredMap_generator(np.log10(age), GredAge)  # Units: Gyr
        if np.ndim(age)==2:
            self.age=load_image(age,(inst.nx,inst.ny))
        
        # Metallicity Distribution
        if np.ndim(feh)==0:
            self.feh = np.copy(self.xrot)
            self.feh[:] = self.GredMap_generator(feh, GredZ)
        if np.ndim(feh)==2:
            self.feh=load_image(feh,(inst.nx,inst.ny))
        else:
            pass
        
    def GredMap_generator(self, A0, gred):
        center_grid = Grid(self.grid.xsamp/(SERSIC_OVERSAMPLE-1), self.grid.ysamp/(SERSIC_OVERSAMPLE-1),
                           SERSIC_OVERSAMPLE, SERSIC_OVERSAMPLE)
        yrot,xrot = center_grid.shift_rotate(0, 0, self.pa)
        center_pixel = self.Gred_func(yrot, xrot, self.major, self.minor, A0, gred)
        self.center_flux = np.sum(center_pixel) / (SERSIC_OVERSAMPLE**2)

        self.prof = np.zeros_like(self.grid.x)
        
        lx = (self.xoff // self.grid.xsamp) * self.grid.xsamp
        rx = (self.xoff % self.grid.xsamp) * self.grid.xsamp
        ly = (self.yoff // self.grid.ysamp) * self.grid.ysamp
        ry = (self.yoff % self.grid.ysamp) * self.grid.ysamp

        pixels = [[lx, ly, (1 - rx) * (1 - ry)], [lx, ly + self.grid.ysamp, (1 - rx) * ry],
                      [lx + self.grid.xsamp, ly, rx * (1 - ry)],
                      [lx + self.grid.xsamp, ly + self.grid.ysamp, rx * ry]]
        for tx,ty,frac in pixels:
            if frac != 0:
                yrot, xrot = self.grid.shift_rotate(ty, tx, self.pa)
                partial = self.Gred_func(yrot, xrot, self.major, self.minor, A0, gred)
                yzero = np.min(np.abs(yrot))
                xzero = np.min(np.abs(xrot))
                if np.abs(xzero) < POS_EPSILON * self.grid.xsamp and \
                np.abs(yzero) < POS_EPSILON * self.grid.ysamp:
                    xvals = np.where(np.abs(xrot.flatten()) < POS_EPSILON * self.grid.xsamp)[0]
                    yvals = np.where(np.abs(yrot.flatten()) < POS_EPSILON * self.grid.ysamp)[0]
                    xy = np.intersect1d(xvals,yvals)
                    xval = xy // self.grid.shape[0]
                    yval = xy % self.grid.shape[0]
                    partial[xval, yval] = self.center_flux
                self.prof += partial * frac
        return self.prof

    def Gred_func(self, y, x, major, minor, A0, gred):
        dist = np.sqrt((x / major)**2.0 + (y / minor)**2.0) + 0.0001
        profile = A0 + dist * gred
        return profile

from skimage.transform import resize
def load_image(array2d, shape):
    return resize(array2d,(shape[0],shape[1]))
    #return array2d