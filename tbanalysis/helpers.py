import numpy as np


def clong_contigious_cells(cell: str):
    contigious_cells = {
        # 'A3': [5, 8, 9, 10, 15, 18, 7, 6, 11, 12, 17, 16, 13, 14, 25, 26] # old
        'A3': [[5, 8, 9, 10, 15, 18], # LBC/M0C
               [11, 10, 21, 20, 32, 31]] # EBC
    }
    return contigious_cells[cell]



def ctot_contigious_cells(cell: str):
    contigious_cells = {
        #'A3':[[5, 8, 9, 10, 15, 18, 7, 6, 11, 12, 17, 16, 13, 14, 25, 24], # old
        #    [11, 10, 21, 20, 32, 31, 15, 14, 23, 22, 35, 30, 17, 16, 37, 38]]
        
        'A3': [[1, 4, 5, 8, 9, 10, 15, 18, 19, 20, 23, 24, 27, 30, 33, 36, 37, 38, 46, 47, 2, 3, 6, 7, 11, 12, 16, 17, 21, 22, 28, 29, 34, 35, 40, 41, 44, 45, 0, 13, 14, 25, 26, 39, 42],
               [6, 7, 10, 11, 20, 21, 31, 32, 40, 41, 8, 9, 14, 15, 22, 23, 30, 35, 36, 39, 2, 3, 4, 5, 16, 17, 37, 38]]
    }
    return contigious_cells[cell]


def get_clong(cell: str, long_cell_energy: np.ndarray, beam_energy: int) -> np.ndarray:
    """
    Computes C_long for a cell
    
    Parameters
    ----------
    cell:
        The name of cell in strings. Cell mapping can be found here:
        http://hep.dnp.fmph.uniba.sk/~zenis/tile
    beam_energy:
        energy of beam in MeV
    long_cell_energy"
        numpy ndarray of energy deposit in longitudinal neighbor cells. Dimension of this must be - 48 (PMTs) x N (number of events)
        
    Returns
    -------
    clong
    """
    long_cell_energy = np.clip(long_cell_energy, 0.000001, None)
    long_deposit_energy = long_cell_energy.sum(axis=0)
    clong = long_deposit_energy / beam_energy
    print(clong.min())
    
    return clong


def get_ctot(cell: str, ctot_cell_energy: np.ndarray) -> np.ndarray:
    ALPHA = 0.6
    N_CELL = 24
    
    # clip negative energies to zero
    ctot_cell_energy = np.clip(ctot_cell_energy, 0.000001, None)        
    
    # raise energies to power ALPHA
    e_raised_alpha = ctot_cell_energy ** ALPHA
    
    e_raised_alpha_sum = np.sum(e_raised_alpha, axis=0)
    e_raised_alpha_sum = np.where(e_raised_alpha_sum == 0, np.nan, e_raised_alpha_sum)
    
    mean_raised_alpha = e_raised_alpha_sum / N_CELL
    parenthesis_term = (e_raised_alpha - mean_raised_alpha) ** 2
    sqrt_term = np.sum(parenthesis_term, axis=0) / N_CELL
    inner = np.sqrt(sqrt_term)
    
    c_tot = inner / e_raised_alpha_sum 
    
    return c_tot
