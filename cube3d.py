import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from sed import *

class Cube3d():
    """
    3D data cube
    """
    
    def __init__(self, src, inst, ssp, sbmap, kinmap, gasmap, stellarmap):
        
        self.nx = inst.npix
        self.ny = inst.npix
        self.nw = len(inst.wave)
        
        self.mag = sbmap.mag + 2.5 * np.log10(inst.dpix * inst.dpix)
        self.sb = sbmap.sb
        self.star_age = stellarmap.age
        self.star_feh = stellarmap.feh
        self.star_vel = kinmap.vel
        self.star_vdisp = kinmap.sigma
        
        self.haflux = gasmap.haflux
        self.gas_z = stellarmap.feh
        self.gas_vel = kinmap.vel
        self.gas_vdisp = kinmap.sigma
        
        self.wave = inst.wave
        fluxcube = np.zeros((self.nx,self.ny,self.nw))
        
        for i in range(self.nx):
            for j in range(self.ny):
                ss = StarPop(ssp, self.wave, self.mag[i,j], self.star_age[i,j], self.star_feh[i,j],
                             self.star_vel[i,j], self.star_vdisp[i,j])
                if self.haflux[i,j] > 0:
                    gg = Gas(self.wave, self.haflux[i,j], self.gas_z[i,j], self.gas_vel[i,j],
                             self.gas_vdisp[i,j], Ebv = self.ebv[i,j])
                    fluxcube[i,j,:] = ss.flux + gg.flux
                else:
                    fluxcube[i,j,:] = ss.flux
                    
        self.flux=fluxcube