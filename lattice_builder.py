import atomap.api as am
import atomap.testing_tools as tt
import numpy as np
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
from atomap.tools import rotate_points_around_signal_centre
import temul.api as tml
import hyperspy.api as hs
import cupy as cp


def RMNO3_FE_up(dim, num_x,  contrast, sigma, dist, dx =0.15):
   
    """
    The function creates lattices of h-ReMnO3 (or ReFeO3) along [110] zone axis, where the buckling of Re ions shows
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: Ferroelectroc domain with upward polarization
    
    """
    # Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
     
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c = 2a, then y = 2 * num_x (distance Mn planes)
        
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5, 4)

    #Creating second sublattice, the Re ions         
    # along c axis in unit cell, there are four atomic planes, 2 x Mn and 2 x Lu        

    Y_pos_dn = (dx-0.1) * (num_x)     # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)         # 10% buckling with respect to middle distance of Mn - Mn planes
    
    Y1x, Y1y = cp.mgrid[0           :dim:num_x*3,  2*(num_x//2)+Y_pos_dn   :dim:num_x*2]   # Re 2a Wyckoff, down 
    Y2x, Y2y = cp.mgrid[num_x       :dim:num_x*3,  2*(num_x//2)-Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, up
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim:num_x*3,  2*(num_x//2)-Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, up
    
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])

    Yx = cp.concatenate((Y1x, Y2x, Y3x), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y), axis=None ) 
    Yx, Yy = Yx.get(), Yy.get()


    t_0.add_atom_list(Yx, Yy, 6, 6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)

    return t_0



def RMNO3_FE_down(dim, num_x, contrast, sigma, dist, dx = 0.15):
    
    """
    The function creates lattices of h-ReMnO3 (or ReFeO3) along [110] zone axis, where the buckling of Re ions shows
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: Ferroelectroc domain with downward polarization
    
    """
    
    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c_lattice = 2a = 2 * num_x
    Mnx=cp.array([i+3*abs(cp.random.random(1)) for i in Mnx.flatten()])
    Mny=cp.array([i+3*abs(cp.random.random(1)) for i in Mny.flatten()])
    
    Mnx, Mny = Mnx.get(), Mny.get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5, 4)

    #Creating second sublattice, the Re ions 
         
    Y_pos_dn = (dx-0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)            # 10% buckling with respect to middle distance of Mn - Mn planes
    
    
    Y1x, Y1y = cp.mgrid[0           :dim:num_x*3,  2*(num_x//2)-Y_pos_dn   :dim:num_x*2]   # Re 2a Wyckoff, up 
    Y2x, Y2y = cp.mgrid[num_x       :dim:num_x*3,  2*(num_x//2)+Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, down
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim:num_x*3,  2*(num_x//2)+Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, down
     
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])

    Yx = cp.concatenate((Y1x, Y2x, Y3x), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y), axis=None ) 
    Yx, Yy = Yx.get(), Yy.get()

    t_0.add_atom_list(Yx, Yy, 6, 6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)
    
    return t_0  


def RMNO3_FE_head_head(dim, num_x, contrast, sigma, dist, dx = 0.15):
        
    """
    The function creates head to head FE domians along [110] direction.
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: two FE domains with head to head geometry
    
    """

    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c_lattice = 2a = 2 * num_x   
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5, 4)
    
    #Creating second sublattice, the Re ions 
    
    Y_pos_dn = (dx - 0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)              # 10% buckling with respect to middle distance of Mn - Mn planes
        
    Y1x, Y1y = cp.mgrid[0           :dim:num_x*3,  2*(num_x//2)-Y_pos_dn   :dim//2:num_x*2]   # Re 2a Wyckoff 
    Y2x, Y2y = cp.mgrid[num_x       :dim:num_x*3,  2*(num_x//2)+Y_pos_up   :dim//2:num_x*2]   # Re 4b Wyckoff
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim:num_x*3,  2*(num_x//2)+Y_pos_up   :dim//2:num_x*2]   # Re 4b Wyckoff
    
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])

    #building the second half of the lattice. The x values are the same, only along y the polarization switches
    Y1x1,Y2x1, Y3x1   =Y1x, Y2x, Y3x
    Y1y1, Y2y1, Y3y1 = Y1y + dim//2+2*Y_pos_dn, Y2y + dim//2-2*Y_pos_up, Y3y + dim//2-2*Y_pos_up
       
    Yx1 = cp.concatenate((Y1x, Y2x, Y3x, Y1x1, Y2x1, Y3x1), axis=None ) 
    Yy1 = cp.concatenate((Y1y, Y2y, Y3y, Y1y1, Y2y1, Y3y1), axis=None ) 
    Yx1, Yy1 = Yx1.get(), Yy1.get()

    t_0.add_atom_list(Yx1, Yy1, 6, 6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)
    
    return t_0    

def RMNO3_FE_tail_tail(dim, num_x, contrast, sigma, dist, dx = 0.15):
    
    """
    The function creates tail to tail FE domians along [110] direction.
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200 2400].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: two FE domains with tail to tail geometry
    
    """

    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c_lattice = 2a
        
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image = False, add_row_scan_distortion = dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5 ,4)
         
    Y_pos_dn = (dx - 0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)              # 10% buckling with respect to middle distance of Mn - Mn planes
    
    Y1x, Y1y = cp.mgrid[0           :dim:num_x*3,  2*(num_x//2)+Y_pos_dn   :dim//2:num_x*2]   # Re 2a Wyckoff, down 
    Y2x, Y2y = cp.mgrid[num_x       :dim:num_x*3,  2*(num_x//2)-Y_pos_up   :dim//2:num_x*2]   # Re 4b Wyckoff, up
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim:num_x*3,  2*(num_x//2)-Y_pos_up   :dim//2:num_x*2]   # Re 4b Wyckoff, up
       
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])
    
    #building the second half of the lattice. The x values are the same, only along y the polarization switches

    Y1x1,Y2x1, Y3x1   =Y1x, Y2x, Y3x
    Y1y1, Y2y1, Y3y1 = Y1y + dim//2-2*Y_pos_dn, Y2y + dim//2+2*Y_pos_up, Y3y + dim//2+2*Y_pos_up
    
    Yx = cp.concatenate((Y1x, Y2x, Y3x, Y1x1, Y2x1, Y3x1), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y, Y1y1, Y2y1, Y3y1), axis=None ) 
    
    Yx, Yy = Yx.get(), Yy.get()

    t_0.add_atom_list(Yx, Yy, 6,6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)

    return t_0

# develop it based on previous fucntions
def RMNO3_FE_head_shift(dim, num_x, contrast, sigma, dist, dx = 0.15):
    """
    The function creates head to head FE domians along [110] direction with one atomic plane shift along x.
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200 2400].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: lattice of two FE domains with head to head geometry with one atomic plane shift along x
    
    """
    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c_lattice = 2a
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5 ,4)
         
    # Build the first FE domain with upward polarization in first half of the image along x-axis
   
    Y_pos_dn = (dx - 0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)      # 10% buckling with respect to middle distance of Mn - Mn planes
    
    Y1x, Y1y = cp.mgrid[0           :dim:num_x*3, 2*(num_x//2)-Y_pos_dn  :dim//2:num_x*2]   # Re 2a Wyckoff, up 
    Y2x, Y2y = cp.mgrid[num_x       :dim:num_x*3, 2*(num_x//2)+Y_pos_up  :dim//2:num_x*2]   # Re 4b Wyckoff, down
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim:num_x*3, 2*(num_x//2)+Y_pos_up  :dim//2:num_x*2]   # Re 4b Wyckoff, down
    
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])
    
     #building the second half of the lattice. The x values are the same, only along y the polarization switches

    Y1x1, Y2x1, Y3X1 = Y1x , Y2x, Y3x
    Y1y1, Y2y1, Y3y1 = Y1y + dim//2, Y2y + dim//2-Y_pos_up - Y_pos_dn, Y3y + dim//2

    Yx = cp.concatenate((Y1x, Y2x, Y3x, Y1x1, Y2x1, Y3X1), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y, Y1y1, Y2y1, Y3y1), axis=None ) 
    Yx, Yy = Yx.get(), Yy.get()

    t_0.add_atom_list(Yx, Yy, 6,6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)

    return t_0

def RMNO3_FE_tail_shift(dim, num_x, contrast, sigma, dist, dx = 0.15):
    """
    The function creates tail to tail FE domians along [110] direction with one atomic plane shift along x.
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200 2400].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: lattice of two FE domains with tail to tail geometry with one atomic plane shift along x
    
    """
    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
       
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c_lattice = 2a
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5 ,4)
         
    # Build the first FE domain with upward polarization in first half of the image along x-axis
   
    Y_pos_dn =(dx - 0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)             # 10% buckling with respect to middle distance of Mn - Mn planes
    
    Y1x, Y1y = cp.mgrid[0           :dim:num_x*3, 2*(num_x//2)+Y_pos_dn  :dim//2:num_x*2]   # Re 2a Wyckoff, down
    Y2x, Y2y = cp.mgrid[num_x       :dim:num_x*3, 2*(num_x//2)-Y_pos_up  :dim//2:num_x*2]   # Re 4b Wyckoff, up
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim:num_x*3, 2*(num_x//2)-Y_pos_up  :dim//2:num_x*2]   # Re 4b Wyckoff, up
    
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])

    #building the second half of the lattice. The x values are the same, only along y the polarization switches

    Y1x1, Y2x1, Y3X1 = Y1x , Y2x, Y3x
    Y1y1, Y2y1, Y3y1 = Y1y + dim//2, Y2y + dim//2 + Y_pos_up + Y_pos_dn, Y3y + dim//2
    
    Yx = cp.concatenate((Y1x, Y2x, Y3x, Y1x1, Y2x1, Y3X1), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y, Y1y1, Y2y1, Y3y1), axis=None ) 
    Yx, Yy = Yx.get(), Yy.get()

    t_0.add_atom_list(Yx, Yy, 6,6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)
    
    return t_0

def RMNO3_FE_sideup_sidedn(dim, num_x, contrast, sigma, dist, dx = 0.15):
    """
    The function creates up and down FE domians along x.
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200 2400].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: lattice of two FE domains with tail to tail geometry with one atomic plane shift along x
    
    """
    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c = 2a, then y = 2 * num_x (distance Mn planes)
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5, 4)
         
    # along c axis in unit cell, there are four atomic planes, 2 x Mn and 2 x Lu        

    Y_pos_dn = (dx - 0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)      # 10% buckling with respect to middle distance of Mn - Mn planes
    
    Y1x, Y1y = cp.mgrid[0           :dim//2:num_x*3,  2*(num_x//2)+Y_pos_dn   :dim:num_x*2]   # Re 2a Wyckoff, down 
    Y2x, Y2y = cp.mgrid[num_x       :dim//2:num_x*3,  2*(num_x//2)-Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, up
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim//2:num_x*3,  2*(num_x//2)-Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, up
    
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])

    Y1x1, Y2x1, Y3X1 = Y1x + (dim//2) , Y2x + (dim//2), Y3x + (dim//2)
    Y1y1, Y2y1, Y3y1 = Y1y - Y_pos_dn - Y_pos_up, Y2y + Y_pos_up+Y_pos_dn, Y3y+Y_pos_up+Y_pos_dn
    
    Yx = cp.concatenate((Y1x, Y2x, Y3x, Y1x1, Y2x1, Y3X1), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y, Y1y1, Y2y1, Y3y1), axis=None ) 
    Yx, Yy = Yx.get(), Yy.get()

    t_0.add_atom_list(Yx, Yy, 6,6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)

    return t_0  

def RMNO3_FE_sided_sideu(dim, num_x, contrast, sigma, dist, dx = 0.15):
    """
    The function creates down and up FE domians along  x.
    Args:

        dim (int): Dimension of the lattice. Range is in [600 1200 2400].
        num_x (int): Number of unit cells along the x-axis. Range is in [30 60].
        dx (int): Buckling of Re ions with respect to Mn place. Values in range [0.12, 0.15, 0.18].
        contrast (int): Contrast of the lattice. Range is in [8 15 ].
        sigma (float): Standard deviation of the Gaussian noise added to the lattice.
        dist (float): Introduce distortion by shifting horizontal rows.
    
    Returns:
        np.ndarray: lattice of two FE domains with tail to tail geometry with one atomic plane shift along x
    
    """
    #Creating first sublattice, the Mn ions 
    # Number of Mn ions along x-axis = dim / num_x and along y_axis: dim / (num_x * 2)
    
    Mnx, Mny = cp.mgrid[0:dim:num_x, 0:dim:num_x*2]            #c = 2a, then y = 2 * num_x (distance Mn planes)
    Mnx, Mny = Mnx.flatten().get(), Mny.flatten().get()
    
    t_0=tt.MakeTestData(dim, dim, sublattice_generate_image=False, add_row_scan_distortion= dist)

    t_0.add_atom_list(Mnx, Mny, 5, 5, 4)
         
    # along c axis in unit cell, there are four atomic planes, 2 x Mn and 2 x Lu        

    Y_pos_dn = (dx - 0.1) * (num_x)      # 5% buckling with respect to middle distance of Mn - Mn planes
    Y_pos_up = dx * (num_x)      # 10% buckling with respect to middle distance of Mn - Mn planes
    
    Y1x, Y1y = cp.mgrid[0           :dim//2:num_x*3,  2*(num_x//2)-Y_pos_dn   :dim:num_x*2]   # Re 2a Wyckoff, down 
    Y2x, Y2y = cp.mgrid[num_x       :dim//2:num_x*3,  2*(num_x//2)+Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, up
    Y3x, Y3y = cp.mgrid[num_x+num_x :dim//2:num_x*3,  2*(num_x//2)+Y_pos_up   :dim:num_x*2]   # Re 4b Wyckoff, up
    
    Y1x=cp.array([i+3*abs(cp.random.random(1)) for i in Y1x.flatten()])
    Y1y=cp.array([i+3*abs(cp.random.random(1)) for i in Y1y.flatten()])
    Y2x=cp.array([i+3*abs(cp.random.random(1)) for i in Y2x.flatten()])
    Y2y=cp.array([i+3*abs(cp.random.random(1)) for i in Y2y.flatten()])
    Y3x=cp.array([i+3*abs(cp.random.random(1)) for i in Y3x.flatten()])
    Y3y=cp.array([i+3*abs(cp.random.random(1)) for i in Y3y.flatten()])

    Y1x1, Y2x1, Y3X1 = Y1x + (dim//2) , Y2x + (dim//2), Y3x + (dim//2)
    Y1y1, Y2y1, Y3y1 = Y1y + Y_pos_dn + Y_pos_up, Y2y - Y_pos_up - Y_pos_dn, Y3y - Y_pos_up - Y_pos_dn
    
    Yx = cp.concatenate((Y1x, Y2x, Y3x, Y1x1, Y2x1, Y3X1), axis=None ) 
    Yy = cp.concatenate((Y1y, Y2y, Y3y, Y1y1, Y2y1, Y3y1), axis=None ) 
    Yx, Yy = Yx.get(), Yy.get()

    t_0.add_atom_list(Yx, Yy, 6,6, contrast)
    t_0.add_image_noise(sigma=sigma, only_positive=True)
      
    return t_0  


