from __future__ import print_function
import numpy as np

class Source():
    
    def __init__(self, x0=0.0, y0=0.0, total_mag=16.0, redshift=0.005):
        
        self.x0=x0
        self.y0=y0
        self.total_mag=total_mag
        self.redshift=redshift
        self.surf_area_units='arcsec^2'

        
class Sersic(Source):
    
    def __init__(self, pa=30.0, re=4.0, ba=1.5, index=1.5 ,**kwargs):
        
        Source.__init__(self,**kwargs)
        self.index=index
        self.pa=pa
        self.re=re
        self.major=re
        self.minor=self.major*ba
        self.prof_type='sersic'
        
class PSF(Source):
    
    def __init__(self, fwhm=0.2 ,**kwargs):
        
        Source.__init__(self,**kwargs)
        self.fwhm=fwhm
        self.prof_type='psf'