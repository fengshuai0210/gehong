from __future__ import division, absolute_import
import numpy as np

class Grid(object):

    """
    Generate an x, y grid in a rectangular region, sampled with xsamp and
    ysamp spacings in the x and y directions, respectively.  Origin is at the
    upper-left corner to match numpy and matplotlib convention. astropy.io.fits
    also assumes UL origin by default.

    Parameters
    ----------
    xsamp, ysamp : float, float
        The sampling spacing in the x and y directions.
    nx, ny: int, int
        Number of samples in the x and y directions.
    """

    def __init__(self, xsamp, ysamp, nx, ny):
        self.xsamp = np.abs(xsamp)
        self.ysamp = np.abs(ysamp)
        startx = -(nx - 1) / 2.0 * self.xsamp
        stopx = (nx - 1) / 2.0 * self.xsamp
        starty = -(ny - 1) / 2.0 * self.ysamp
        stopy = (ny - 1) / 2.0 * self.ysamp
        xvals = np.linspace(startx, stopx, num=nx)
        yvals = np.linspace(starty, stopy, num=ny)

        ones = np.ones((ny, nx))
        x = ones * xvals
        y = np.flipud(ones * yvals.reshape(int(ny), 1))

        self.nx = nx
        self.ny = ny
        self.x = x
        self.y = y
        self.row = xvals
        # flip Y axis because we use Y increasing from bottom to top
        self.col = yvals[::-1]

    @property
    def shape(self):
        """
        Provides array-like .shape functionality
        """
        sh = (self.ny, self.nx)
        return sh

    def wcs_info(self):
        """
        Define coordinate transform in WCS header format.

        Returns
        -------
        header: dict
            WCS keys defining coordinate transform for the 2 spatial axes
        """
        t = self.as_dict()
        header = {
            'ctype1': 'X offset',
            'crpix1': 1,
            'crval1': t['x_min'],
            'cdelt1': t['x_step'],
            'cunit1': 'arcsec',
            'cname1': 'X',
            'ctype2': 'Y offset',
            'crpix2': 1,
            'crval2': t['y_min'],
            'cdelt2': -t['y_step'],
            'cunit2': 'arcsec',
            'cname2': 'Y',
        }
        return header

    def __len__(self):
        """
        Provide array-like len() functionality
        """
        l = self.x.shape[0] * self.x.shape[1]
        return l

    def __getitem__(self, val):
        """
        Provide array-like indexing functionality.

        Parameters
        ----------
        val : slice object
            Valid python index or slice() specification.

        Returns
        -------
        pos : [y, x] list of floats or arrays
            Elements are floats in the case of specific indicies and arrays in
            the cases of slice()'s.
        """
        section = [self.y[val], self.x[val]]
        return section

    def bounds(self):
        return {'xmin': np.min(self.x),
                'xmax': np.max(self.x),
                'ymin': np.min(self.y),
                'ymax': np.max(self.y)}

    def dist(self, xcen=0.0, ycen=0.0):
        """
        Return a distance array where each element contains its distance from
        the center of the grid.
        """
        d = np.sqrt((self.x - xcen) ** 2 + (self.y - ycen) ** 2)
        return d

    def dist_xy(self, xcen=0.0, ycen=0.0):
        """
        Return a directional distance array where each element contains its (x,y) distance from
        the center of the grid.
        """
        dx = self.x - xcen
        dy = self.y - ycen
        return dx,dy

    def world_to_image(self, y, x):
        """
        Return fractional index coordinates yi,xi corresponding to given y,x position

        Parameters
        ----------
        y, x : float
            y, x position in world coordinates

        Returns
        -------
        pos : [yi, xi] list of floats
            yi, xi image coordinates that correspond to given position
        """
        t = self.as_dict()
        if x < t['x_min'] or x > t['x_max']:
            raise ValueError("X outside of Grid bounds.")
        if y < t['y_min'] or y > t['y_max']:
            raise ValueError("Y outside of Grid bounds.")
        x_image = x / self.xsamp + (self.shape[1] - 1) / 2.0
        # there's a minus sign here because y indices increase from up to down,
        # but y coordinates increase down to up.
        y_image = -y / self.ysamp + (self.shape[0] - 1) / 2.0
        pos = [y_image, x_image]
        return pos

    def world_to_index(self, y, x):
        """
        Return nearest index pair to given y,x position

        Parameters
        ----------
        y, x : float
            y,x position in world coordinates

        Returns
        -------
        pos : [y, x] list of ints
            y,x indices that most closely correspond to given position
        """
        y_index, x_index = self.world_to_image(y, x)
        y_index, x_index = self._index_bound(int(round(y_index)), self.ny), self._index_bound(int(round(x_index)), self.nx)
        pos = [y_index, x_index]
        return pos

    def _index_bound(self, index, nindex):
        """
        Apply bounds to indicies so the nearest index is either the first or last index for cases
        at edge or beyond Grid.

        Parameters
        ----------
        index: int
            Index to bounds-check
        nindex: int
            Length of axis being indexed

        Returns
        -------
        index: int
            Index with boundaries applied
        """
        if index < 0:
            index = 0
        if index > nindex - 1:
            index = nindex - 1
        return index

    def shift_rotate(self, yoff, xoff, rot):
        """
        Return shifted/rotated (y, x) given offsets (yoff, xoff) and rotation, rot (degrees)

        Parameters
        ----------
        yoff, xoff: float
            yoff, xoff offsets in world coordinates
        rot: float
            rotation angle in degrees

        Returns
        -------
        ysh_rot, xsh_rot: 2D numpy arrays
            rotated and shifted copies of Grid.x and Grid.y
        """
        pa_radians = np.pi * rot / 180.0
        xsh = self.x - xoff
        ysh = self.y - yoff
        xsh_rot = xsh * np.cos(pa_radians) + ysh * np.sin(pa_radians)
        ysh_rot = -xsh * np.sin(pa_radians) + ysh * np.cos(pa_radians)
        return ysh_rot, xsh_rot

    def get_aperture(self):
        """
        Return Grid parameters as an aperture specification
        """
        ap = {'width': self.nx * self.xsamp, 'height': self.ny * self.ysamp, 'offset': (0, 0)}
        return ap

class Instrument():
    
    def __init__(self, FoV = 12, dpix = 0.1, spec_FWHM=2.0, lamRange=[3500,10000], dlam = 0.1):
        
        self.fov=FoV
        self.dpix=dpix
        self.npix=np.int64(FoV/dpix)
        self.spec_FWHM=spec_FWHM
        self.lamrange=lamRange
        self.dlam=dlam
        self.wave=np.arange(self.lamrange[0],self.lamrange[1],dlam)
        self.grid=Grid(self.dpix,self.dpix,self.npix,self.npix)