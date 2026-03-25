import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
from typing import Tuple, List


def fetch_data_mc(data_type: str):
    if data_type.lower() == 'data':
        energy_branch = {'M0A': 'EfitA01', 'M0C': 'EfitC01' , 'LBA65': 'EfitA02', 'LBC65': 'EfitC02', 'EBC65': 'EfitE03'}
    elif data_type.lower() == 'mc':
        energy_branch = {'M0A': 'EfitA0', 'M0C': 'EfitC0' , 'LBA65': 'EfitA1', 'LBC65': 'EfitC1', 'EBC65': 'EfitE0'}
    else:
        print('Invalid choice ', data_type, ' choose from (data, MC)')
        exit() 
    return energy_branch       



def remove_dash(cell: str) -> str:
    return ''.join(cell.split('-'))


def mask_energy_threshold(arr: np.ndarray, threshold: float = 0) -> np.ma.MaskedArray:
    """
    Mask array values below threshold
    
    Parameters
    ----------
    arr : np.ndarray
        Input energy array
    threshold: float [default = 0]
        Threshold to mask the array
        
    Returns
    -------
    np.ma.MaskedArray
        Masked array where negative/zero values are masked
    """
    return np.ma.masked_where(arr <= threshold, arr)


def root_like_histogram(
    x: np.ndarray, 
    y: np.ndarray, 
    xmin: float = 0, 
    xmax: float = 0.2, 
    ymin: float = 0, 
    ymax: float = 1.6, 
    n_xbins: int = 300, 
    n_ybins: int = 300, 
    x_inc: float = 0.02, 
    y_inc: float = 0.2
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Makes root like TH2 histogram with their colorbar using TH2.Draw("colz")
    
    Parameters
    ----------
    x : np.ndarray
        Data for x-axis
    y : np.ndarray
        Data for y-axis
    xmin : float, optional
        Minimum x value, by default 0
    xmax : float, optional
        Maximum x value, by default 0.2
    ymin : float, optional
        Minimum y value, by default 0
    ymax : float, optional
        Maximum y value, by default 1.6
    n_xbins : int, optional
        Number of x bins, by default 300
    n_ybins : int, optional
        Number of y bins, by default 300
    x_inc : float, optional
        X-axis tick increment, by default 0.02
    y_inc : float, optional
        Y-axis tick increment, by default 0.2
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects of the plot
    """
    ctot = False
    if xmax==0.20 and ymax==1.6:
        ctot = True
    fig, ax = plt.subplots(figsize=(6,6))
    
    x_ticks = np.arange(xmin, xmax + x_inc, x_inc)
    y_ticks = np.arange(ymin, ymax + y_inc, y_inc)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[n_xbins, n_ybins],
                                      range=[[xmin, xmax], [ymin, ymax]])
    
    H = H.T
    H_masked = ma.masked_where(H == 0, H)
    nevt = len(x)
    
    colors = ["blue", "cyan", "green", "yellow", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("root_colz", colors)
    custom_cmap.set_bad(color='white')
    
    im = ax.imshow(H_masked, origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap=custom_cmap,
                   aspect='auto')
    
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    
    return fig, ax, nevt
    
    
def clong_contigious_cells(cell: str) -> List[List]:
    cell = remove_dash(cell)
    # TODO - possibly remove the inconsistency between this and ctot cell definition. List of int vs List of Tuple.
    contigious_cells = {
        'A3': [[5, 8, 9, 10, 15, 18], # LBC/M0C
               [6, 7, 11, 10, 20, 21]] # EBC
    }
    return contigious_cells[cell]


def ctot_contigious_cells(cell: str):
    cell = remove_dash(cell)
    # standard cells in TileTB analyses. Similar to Tigran 2021 paper.
    contigious_cells = {
        'A3':[[(5, 8), (9, 10), (15, 18), (6, 7), (11, 12), (16, 17), (13, 14), (25, 26)], # LBC/M0C (L, R) channels
            [(6, 7), (10, 11), (20, 21), (8, 9), (14, 15), (22, 23), (16, 17), (30,31)]] # EBC
    }
    return contigious_cells[cell]


def get_clong(cell: str, long_cell_energy: np.ndarray, beam_energy: int) -> np.ndarray:
    """
    Computes C_long for a cell
    
    Parameters
    ----------
    cell:
        The name of cell in strings. Cell mapping can be found here:
        https://twiki.cern.ch/twiki/pub/Atlas/TileCalTestbeamAnalysis2022/Tile-mapping-new-4.html
    beam_energy:
        energy of beam in MeV
    long_cell_energy"
        numpy ndarray of energy deposit in longitudinal neighbor cells. Dimension of this must be - 48 (PMTs) x N (number of events)
        
    Returns
    -------
    clong
    """
    energy_masked = np.ma.masked_where(long_cell_energy <= 0, long_cell_energy)
    long_deposit_energy = np.ma.sum(energy_masked, axis=0)
    clong = long_deposit_energy / beam_energy
    clong = np.ma.filled(clong, np.nan)
    
    return clong


def get_ctot(cell: str, ctot_cell_energy: np.ndarray) -> np.ndarray:
    """
    Computes C_tot for a cell using masked arrays to handle negative values
    
    Parameters
    ----------
    cell: str
        The cell name
    ctot_cell_energy: np.ndarray
        Energy deposits in cells, shape (PMTs x Events)
        
    Returns
    -------
    np.ndarray
        C_tot values for each event
    """
    ALPHA = 0.6
    N_CELL = 24
    
    # remove # from print statements for debug.
    valid_mask = ctot_cell_energy > 0
    energy_masked = np.ma.array(ctot_cell_energy, mask=~valid_mask)
    e_raised_alpha = energy_masked ** ALPHA
    # print('E raised alpha', e_raised_alpha)
    e_raised_alpha_sum = np.ma.sum(e_raised_alpha, axis=0)
    # print('E raised alpha sum', e_raised_alpha_sum)
    mean_raised_alpha = e_raised_alpha_sum / N_CELL
    # print('Mean raised alpha', mean_raised_alpha)
    parenthesis_term = (e_raised_alpha - mean_raised_alpha) ** 2
    # print('Parenthesis term', parenthesis_term)
    sqrt_term = np.sum(parenthesis_term, axis=0) / (N_CELL)
    # print('Sqrt term', sqrt_term)
    inner = np.ma.sqrt(sqrt_term)
    # print('Inner', inner)
    c_tot = inner / e_raised_alpha_sum
    # print('Ctot', c_tot)    
    return np.ma.filled(c_tot, np.nan)


def reco_energy(h1000: np.ndarray, modules:List[str]=None, events: list = None, module: str = None, cell: str = None, data_type='data', return_indices=False):
    """
    Computes the total energy deposited in 4 modules - LBA 65, LBC 65, EBC 65, M0C
    A cell's reading is assumed as noise if E_c = E_L + E_R < 2 * sigma_noise.
    Sigma_noise is taken as 30 MeV following 2021 hadron paper.
    This requirement could be different for later test beam. Consult with Tile TB
    operators for accurate value.
    
    Parameters
    ----------
    h1000: 
        TileTB h1000 ntuple in np.ndarray format
    modules:
        List of modules. Subset of - ['M0A', 'LBA65', 'M0C', 'LBC65', 'EBC65']
    module:

    events:
        List of events to calculate reconstructed energy for
        
    Returns
    ----------
        energy_evt:
            numpy array of total reconstructed energy in each event
    """
    
    from itertools import chain
    
    # Input validation
    if modules is None:
        modules = ['LBA65', 'LBC65', 'EBC65', 'M0C']  # Default modules (excluding M0A)
    
    # Validate modules
    valid_modules = ['M0A', 'LBA65', 'M0C', 'LBC65', 'EBC65']
    invalid_modules = [m for m in modules if m not in valid_modules]
    if invalid_modules:
        raise ValueError(f"Invalid modules: {invalid_modules}. Valid modules are: {valid_modules}")
    
    sigma_noise = 30 # MeV
    threshold = 2 * sigma_noise
    
    MZERO_A_BRANCH = fetch_data_mc(data_type=data_type)['M0A']
    MZERO_C_BRANCH = fetch_data_mc(data_type=data_type)['M0C']
    LBA_BRANCH = fetch_data_mc(data_type=data_type)['LBA65']
    LBC_BRANCH = fetch_data_mc(data_type=data_type)['LBC65']
    EBC_BRANCH = fetch_data_mc(data_type=data_type)['EBC65']
        
    lba_energy = h1000[LBA_BRANCH].arrays(library='np')[LBA_BRANCH].T
    lbc_energy = h1000[LBC_BRANCH].arrays(library='np')[LBC_BRANCH].T
    mzeroc_energy = h1000[MZERO_C_BRANCH].arrays(library='np')[MZERO_C_BRANCH].T
    ebc_energy = h1000[EBC_BRANCH].arrays(library='np')[EBC_BRANCH].T
    
    
    if cell is None:
        cell_map = {
            'LB': [
                (1, 4), (5, 8), (9, 10), (15, 18), (19, 20), (23, 24), (27, 30), (33, 36), (37, 38), (46, 47),
                (2, 3), (6, 7), (11, 12), (16, 17), (21, 22), (28, 29), (34, 35), (40, 41), (44, 45),
                (0, ), (13, 14), (25, 26), (39, 42)
                ],
            'EB': [
                (6, 7), (10, 11), (20, 21), (24, 25), (32, 33),
                (8, 9), (14, 15), (22, 23), (26, 27), (34, 35), (4, 5),
                (2, 3), (16, 17), (30, 31),
                (12, ), (13, ), (0, ), (1, )
            ] 
        }
        lb_cells = list(chain.from_iterable(cell_map['LB']))
        eb_cells = list(chain.from_iterable(cell_map['EB']))
        lba_chan_ene = np.array([lba_energy[list(c)].sum(axis=0) for c in cell_map['LB']])
        lbc_chan_ene = np.array([lbc_energy[list(c)].sum(axis=0) for c in cell_map['LB']])
        mzeroc_chan_ene = np.array([mzeroc_energy[list(c)].sum(axis=0) for c in cell_map['LB']])
        ebc_chan_ene = np.array([ebc_energy[list(c)].sum(axis=0) for c in cell_map['EB']])
        
        
        # mask energy less than 2 * sigma_noise
        lba_chan_ene = mask_energy_threshold(lba_chan_ene, threshold=threshold)
        lbc_chan_ene = mask_energy_threshold(lbc_chan_ene, threshold=threshold)
        ebc_chan_ene = mask_energy_threshold(ebc_chan_ene, threshold=threshold)
        mzeroc_chan_ene = mask_energy_threshold(mzeroc_chan_ene, threshold=threshold)
        
        # filter which module to add to total energy
        tot_energy = []
        if 'M0C' in modules:
            tot_energy.append(mzeroc_chan_ene)
        if 'LBC65' in modules:
            tot_energy.append(lbc_chan_ene)
        if 'EBC65' in modules:
            tot_energy.append(ebc_chan_ene)
        if 'LBA65' in modules:
            tot_energy.append(lba_chan_ene)
        # Note: M0A is loaded for completeness but not used in energy calculation
        
        if len(tot_energy) == 0:
            raise ValueError(f"No valid energy modules found in {modules}. Available modules for energy calculation: ['LBA65', 'LBC65', 'EBC65', 'M0C']")
        
        print(f'Used {len(tot_energy)} modules from given {len(modules)} modules to compute energy')
        #tot_chan_ene = np.vstack([lbc_chan_ene, mzeroc_chan_ene, ebc_chan_ene])
        tot_chan_ene = np.vstack(tot_energy)

        
    else: # energy for single cell
        tot_energy = []
        for mod in modules:
            if mod == 'LBC65':
                mod_energy = np.array([lbc_energy[list(c)].sum(axis=0) for c in cell])
                mod_energy = mask_energy_threshold(mod_energy, threshold=threshold)
                tot_energy.append(mod_energy)
            elif mod == 'LBA65':
                mod_energy = np.array([lba_energy[list(c)].sum(axis=0) for c in cell])
                mod_energy = mask_energy_threshold(mod_energy, threshold=threshold)
                tot_energy.append(mod_energy)
            elif mod == 'EBC65':
                mod_energy = np.array([ebc_energy[list(c)].sum(axis=0) for c in cell])
                mod_energy = mask_energy_threshold(mod_energy, threshold=threshold)
                tot_energy.append(mod_energy)
            elif mod == 'M0C':
                mod_energy = np.array([mzeroc_energy[list(c)].sum(axis=0) for c in cell])
                mod_energy = mask_energy_threshold(mod_energy, threshold=threshold)
                tot_energy.append(mod_energy)        
        tot_chan_ene = np.vstack(tot_energy)
    
    if events is not None:
        tot_chan_ene = tot_chan_ene[:,events]
    
    # keep track of events used for reco energy calc
    if events is not None:
        original_indices = np.array(events)
    else:
        original_indices = np.arange(tot_chan_ene.shape[1])
    
    # Properly sum masked arrays - use np.ma.sum to handle masked values correctly
    tot_chan_ene = np.ma.sum(tot_chan_ene, axis=0)
    # print(tot_chan_ene.min(), tot_chan_ene.max(), tot_chan_ene.mean(), tot_chan_ene.std())
    good_energy_events = np.where(tot_chan_ene > threshold)
    # Convert masked array to regular array, filling masked values with 0 (they won't pass threshold anyway)
    tot_chan_ene = np.ma.filled(tot_chan_ene, 0.0)
    tot_chan_ene = tot_chan_ene[good_energy_events]
    
    used_indices = original_indices[good_energy_events]
    if return_indices:
        return tot_chan_ene, used_indices
    return tot_chan_ene

class ParticleSeparator:
    
    
    def __init__(self, h1000, data_type='data'):
        self.h1000 = h1000
        
        self.MZERO_A_BRANCH = fetch_data_mc(data_type=data_type)['M0A']
        self.MZERO_C_BRANCH = fetch_data_mc(data_type=data_type)['M0C']
        self.LBA_BRANCH = fetch_data_mc(data_type=data_type)['LBA65']
        self.LBC_BRANCH = fetch_data_mc(data_type=data_type)['LBC65']
        self.EBC_BRANCH = fetch_data_mc(data_type=data_type)['EBC65']
    
        lba_energy = h1000[self.LBA_BRANCH].arrays(library='np')[self.LBA_BRANCH].T
        lbc_energy = h1000[self.LBC_BRANCH].arrays(library='np')[self.LBC_BRANCH].T
        mzeroc_energy = h1000[self.MZERO_C_BRANCH].arrays(library='np')[self.MZERO_C_BRANCH].T
        ebc_energy = h1000[self.EBC_BRANCH].arrays(library='np')[self.EBC_BRANCH].T
        
        self.lb_cells = [i for i in range(31)] + [i for i in range(33, 43)] + [44, 45, 46, 47]
        self.eb_cells = [i for i in range(0, 18)] + [20, 21, 22, 23, 24, 25, 26, 27] + [30, 31, 32, 33, 34, 35]
        self.lba_energy = lba_energy[self.lb_cells]
        self.lbc_energy = lbc_energy[self.lb_cells]
        self.ebc_energy = ebc_energy[self.eb_cells]
        self.mzeroc_energy = mzeroc_energy[self.lb_cells]
        
        # Use masking instead of clipping
        self.lba_energy = mask_energy_threshold(lba_energy[self.lb_cells])
        self.lbc_energy = mask_energy_threshold(lbc_energy[self.lb_cells])
        self.ebc_energy = mask_energy_threshold(ebc_energy[self.eb_cells])
        self.mzeroc_energy = mask_energy_threshold(mzeroc_energy[self.lb_cells])
    
    
    def good_beam_trajectory(self, h1000=None) -> np.ndarray:
        """
        Filters events where particle lie close to the beam center. Beam chamber BC1 is used to select
        beam close to the beam pipe axis.
        Good trajectory is defined as |x - x_mu | < 25 mm and |y - y_mu | 25 mm, where x_mu and y_mu are gaussian
        mean of beam coordinates in BC1.
        
        Parameters
        ----------
            h1000: np.ndarray of h1000 ntuple
        
        Returns
        ----------
            good_traj_indices: np.ndarray of events which satisfy the beam trajectory cut
        """
        
        from scipy.stats import norm
        
        if h1000 is None:
            h1000 = self.h1000
        
        # move to Ximp, Yimp coordinates in future as these coordinates are closer to the scanning table.
        x = h1000['Xcha2_0'].arrays(library='np')['Xcha2_0']
        y = h1000['Ycha2_0'].arrays(library='np')['Ycha2_0']
        
        # Remove bad tADC readout. Bad values at 4096 come from tADC misread. BC1/BC2 dimensions 50 mm x 50 mm.
        mask = (np.abs(x) < 50) & (np.abs(y)<50)
        # below prints use them for debug in future
        # print(f'Good events {mask.sum()}')
        # print(f'Total Bad events {len(x) - mask.sum()}')
        x_good = x[mask]
        y_good = y[mask]
        
        good_readout_indices = np.where(mask)[0]
        
        x_mu, x_sigma = norm.fit(x_good, loc=x_good.mean(), scale=x_good.std())
        y_mu, y_sigma = norm.fit(y_good, loc=y_good.mean(), scale=y_good.std())
        
        # Mean based selection. x_sigma, y_sigma can also be used.
        x_mask = np.abs(x_good - x_mu) < 25
        y_mask = np.abs(y_good - y_mu) < 25
        good_traj = (x_mask) & (y_mask)
        filtered_indices = np.where(good_traj)[0]        
        good_traj_indices = good_readout_indices[filtered_indices]
        
        # use those line for debug in future
        total_events = len(x)
        good_events = len(good_traj_indices)
        print(f"Total events: {total_events}")
        print(f"Events with good trajectory: {good_events} ({100*good_events/total_events:.1f}%)")
        print(f"Beam center: x = {x_mu:.4f} mm, y = {y_mu:.4f} mm")
        # print(f"Stds are {x_sigma} and {y_sigma}")
        # print(x_good.min(), x_good.max(), y_good.min(), y_good.max())
        
        return good_traj_indices
        
    
    def muon_events(self,  h1000=None) -> np.ndarray:
        
        """
        Finds out muon events from the TestBeam ntuple. Muon events are identified as Minimum Ionizing Particles(MIP).
        Similar to earlier analysis, an event is defined as muon event if it's total energy deposition in the detector
        modules is less than 5 GeV. Modules - LBA 65, LBC 65, M0C, EBC 65. M0A module is discarded for total energy computation.
        Parameters
        -----------
        h1000:
            TileCal h1000 ntuple in numpy.ndarray format.
        
        Returns
        ----------
        muon_indices:
            NumPy array of event number(or index) of muon events in h1000
        """
        
        if h1000 is None:
            h1000 = self.h1000
        
        lba_energy = h1000[self.LBA_BRANCH].arrays(library='np')[self.LBA_BRANCH].T[self.lb_cells]
        lbc_energy = h1000[self.LBC_BRANCH].arrays(library='np')[self.LBC_BRANCH].T[self.lb_cells]
        ebc_energy = h1000[self.EBC_BRANCH].arrays(library='np')[self.EBC_BRANCH].T[self.eb_cells]
        mzeroc_energy = h1000[self.MZERO_C_BRANCH].arrays(library='np')[self.MZERO_C_BRANCH].T[self.lb_cells]
        
        # mask negative energy
        lba_energy = mask_energy_threshold(lba_energy)
        lbc_energy = mask_energy_threshold(lbc_energy)
        ebc_energy = mask_energy_threshold(ebc_energy)
        mzeroc_energy = mask_energy_threshold(mzeroc_energy)
                        
        tot_energy = (np.ma.sum(lba_energy, axis=0) + 
                     np.ma.sum(lbc_energy, axis=0) + 
                     np.ma.sum(ebc_energy, axis=0) + 
                     np.ma.sum(mzeroc_energy, axis=0))
        
        # Convert to regular array for comparison
        tot_energy = tot_energy.filled(np.inf)  # Fill masked values with inf to exclude them
        muon_indices = np.where(tot_energy < 5000)[0]
        
        print(f"Total events: {len(tot_energy)}")
        print(f"Muon events: {len(muon_indices)} ({100*len(muon_indices)/len(tot_energy):.1f}%)")
        print(f"Mean energy of muon events: {tot_energy[muon_indices].mean():.1f} MeV")     
        return muon_indices
    
    def alter_muon_rejection(self, h1000=None) -> np.ndarray:
        """
        Alternative muon rejection method based on D-layer energy deposits.
        Muons are minimum ionizing particles that can penetrate through all layers
        and deposit energy in the back D-layer. Electrons and hadrons typically
        shower in the front layers (A, BC) and don't reach D-layer with significant energy.
        
        An event is identified as a muon if energy in any D-layer cell exceeds 2*sigma_noise.
        
        D-layer PMT channel mapping:
        - LBC65: D1 (13, 14), D2 (25, 26), D3 (39, 42)
        - LBA65: D1 (13, 14), D2 (25, 26), D3 (39, 42)
        - M0C: D1 (13, 14), D2 (25, 26), D3 (39, 42)
        - EBC65: D5 (16, 17), D6 (30, 31)
        
        Parameters
        ----------
        h1000 : np.ndarray, optional
            TileCal h1000 ntuple. If None, uses self.h1000
            
        Returns
        -------
        np.ndarray
            Array of event indices identified as muons
        """
        
        if h1000 is None:
            h1000 = self.h1000
        
        # Load energy data from all modules
        lbc_energy = h1000[self.LBC_BRANCH].arrays(library='np')[self.LBC_BRANCH].T
        lba_energy = h1000[self.LBA_BRANCH].arrays(library='np')[self.LBA_BRANCH].T
        m0c_energy = h1000[self.MZERO_C_BRANCH].arrays(library='np')[self.MZERO_C_BRANCH].T
        ebc_energy = h1000[self.EBC_BRANCH].arrays(library='np')[self.EBC_BRANCH].T
        
        # Define D-layer PMT channels for Long Barrel modules (LBC65, LBA65, M0C)
        lb_d1_cells = [13, 14]
        lb_d2_cells = [25, 26]
        lb_d3_cells = [39, 42]
        
        # Define D-layer PMT channels for Extended Barrel (EBC65)
        eb_d5_cells = [16, 17]
        eb_d6_cells = [30, 31]
        
        # Electronic noise threshold
        sigma_noise = 30  # MeV
        threshold = 2 * sigma_noise
        
        # Extract and sum energy for LBC65 D-layers
        lbc_d1_total = lbc_energy[lb_d1_cells].sum(axis=0)
        lbc_d2_total = lbc_energy[lb_d2_cells].sum(axis=0)
        lbc_d3_total = lbc_energy[lb_d3_cells].sum(axis=0)
        
        # Extract and sum energy for LBA65 D-layers
        lba_d1_total = lba_energy[lb_d1_cells].sum(axis=0)
        lba_d2_total = lba_energy[lb_d2_cells].sum(axis=0)
        lba_d3_total = lba_energy[lb_d3_cells].sum(axis=0)
        
        # Extract and sum energy for M0C D-layers
        m0c_d1_total = m0c_energy[lb_d1_cells].sum(axis=0)
        m0c_d2_total = m0c_energy[lb_d2_cells].sum(axis=0)
        m0c_d3_total = m0c_energy[lb_d3_cells].sum(axis=0)
        
        # Extract and sum energy for EBC65 D-layers
        ebc_d5_total = ebc_energy[eb_d5_cells].sum(axis=0)
        ebc_d6_total = ebc_energy[eb_d6_cells].sum(axis=0)
        
        # Identify muon events: any D-layer in any module with energy > threshold
        muon_mask = (
            (lbc_d1_total > threshold) | (lbc_d2_total > threshold) | (lbc_d3_total > threshold) |
            (lba_d1_total > threshold) | (lba_d2_total > threshold) | (lba_d3_total > threshold) |
            (m0c_d1_total > threshold) | (m0c_d2_total > threshold) | (m0c_d3_total > threshold) |
            (ebc_d5_total > threshold) | (ebc_d6_total > threshold)
        )
        muon_indices = np.where(muon_mask)[0]
        
        # Print statistics
        total_events = len(lbc_d1_total)
        n_muons = len(muon_indices)
        print(f"Total events: {total_events}")
        print(f"Muon events (D-layer method): {n_muons} ({100*n_muons/total_events:.1f}%)")
        print(f"\nLBC65 D-layer triggers:")
        print(f"  D1: {np.sum(lbc_d1_total > threshold)}")
        print(f"  D2: {np.sum(lbc_d2_total > threshold)}")
        print(f"  D3: {np.sum(lbc_d3_total > threshold)}")
        print(f"\nLBA65 D-layer triggers:")
        print(f"  D1: {np.sum(lba_d1_total > threshold)}")
        print(f"  D2: {np.sum(lba_d2_total > threshold)}")
        print(f"  D3: {np.sum(lba_d3_total > threshold)}")
        print(f"\nM0C D-layer triggers:")
        print(f"  D1: {np.sum(m0c_d1_total > threshold)}")
        print(f"  D2: {np.sum(m0c_d2_total > threshold)}")
        print(f"  D3: {np.sum(m0c_d3_total > threshold)}")
        print(f"\nEBC65 D-layer triggers:")
        print(f"  D5: {np.sum(ebc_d5_total > threshold)}")
        print(f"  D6: {np.sum(ebc_d6_total > threshold)}")
        
        return muon_indices
    
    
    def single_particle_cut(self, h1000=None):
        """
        Finds the indices(event numbers) of events where a potential debris in scintillator S1 and S2 is found.
        For single particle we require that the energy deposit in the two scintillator is < 2 * energy deposit of MIP particle.
        
        Parameters
        ----------
        h1000:
            TileCal h1000 ntuple in numpy.ndarray format.
        
        Returns
        ----------
        single particle indices:
            np.ndarray of event number(indices) where no debris in S1 & S2 scintillator was found.        
        """
        
        if h1000 is None:
            h1000 = self.h1000
    
        s_one = h1000['S1cou'].arrays(library='np')['S1cou']
        s_two = h1000['S2cou'].arrays(library='np')['S2cou']
        
        muon_indices = self.muon_events()
        muon_s1 = s_one[muon_indices]
        muon_s2 = s_two[muon_indices]
        good_muon_adc = (muon_s1 < 7500) & (muon_s2 < 7500)
        
        # get the mpv(mode) of S1cou and S2cou
        n_ent_s1, n_bins_s1 = np.histogram(muon_s1[good_muon_adc], bins=400, range=(0, 7500))
        n_ent_s2, n_bins_s2 = np.histogram(muon_s2[good_muon_adc], bins=400, range=(0, 7500))
        max_bin_s1, max_bin_s2 = np.argmax(n_ent_s1), np.argmax(n_ent_s2)
        mpv_bin_s1, mpv_bin_s2 = (n_bins_s1[max_bin_s1] + n_bins_s1[max_bin_s1 + 1]) / 2, (n_bins_s2[max_bin_s2] + n_bins_s2[max_bin_s2 + 1]) / 2
        
       
        s1_cut = (s_one < 7500) & (s_one < 2 * mpv_bin_s1)
        s2_cut = (s_two < 7500) & (s_two < 2 * mpv_bin_s2)
        single_particle_mask = s1_cut & s2_cut
        
        single_particle_indices = np.where(single_particle_mask)[0]
        
        print(f"Total events: {len(s_one)}")
        print(f"S1 MPV (MIP): {mpv_bin_s1:.2f}")
        print(f"S2 MPV (MIP): {mpv_bin_s2:.2f}")
        print(f"Events passing cuts: {len(single_particle_indices)} ({100*len(single_particle_indices)/len(s_one):.1f}%)")
        
        return single_particle_indices

    def compute_clong(self, cell: str, beam_energy: float, events: list):
        clong_cells = clong_contigious_cells(cell=cell)
        
        lbc_long = self.lbc_energy[clong_cells[0]]
        m0c_long = self.mzeroc_energy[clong_cells[0]]
        ebc_long = self.ebc_energy[clong_cells[1]]
        
        tot_long = np.vstack([lbc_long, m0c_long, ebc_long])
        tot_pass_long = tot_long[:, events]              
        clong = get_clong(cell=cell, long_cell_energy=tot_pass_long, beam_energy=beam_energy)
        return clong
    
    
    def compute_ctot(self,cell: str, events: list):
        """
        Computes ctot for the given cell and events
        
        Parameters
        ----------
            cell: Name of the cell. E.g- A-3 or A3
            events: List of events to compute ctot for
        Returns
        ----------
        Ctot value for given cell and events
        """
        
        import itertools
        
        ctot_cells = ctot_contigious_cells(cell=cell)
        lbc_chan_ene = np.array([self.lbc_energy[list(c)].sum(axis=0) for c in ctot_cells[0]])
        mzeroc_chan_ene = np.array([self.mzeroc_energy[list(c)].sum(axis=0) for c in ctot_cells[0]])
        ebc_chan_ene = np.array([self.ebc_energy[list(c)].sum(axis=0) for c in ctot_cells[1]])
        
        tot_chan_ene = np.vstack([lbc_chan_ene, mzeroc_chan_ene, ebc_chan_ene])
        tot_chan_ene_events = tot_chan_ene[:, events]
        ctot = get_ctot(cell=cell, ctot_cell_energy=tot_chan_ene_events)
        return ctot


    def pure_electron_events(self, 
                             cell: str,
                             ctot: np.ndarray,
                             clong: np.ndarray,
                             elec_had_indices: np.ndarray,
                             ellipse_center: tuple = (0.14, 1),
                             ellipse_axes: tuple = (0.03, 0.2)
        ) -> np.ndarray:
        """
        Selects electron events using elliptical cuts in Ctot-Clong space.
        
        Parameters
        ----------
        cell : str
            Cell name (e.g., 'A-3' or 'A3') - for validation purposes
        ctot: np.ndarray
            numpy array of ctot values (same length as elec_had_indices)
        clong: np.ndarray
            numpy array of clong values (same length as elec_had_indices)
        elec_had_indices : np.ndarray
            numpy array of electron and hadron event indices (event numbers)
        ellipse_center : tuple, optional
            (h, k) center of electron ellipse in (Ctot, Clong) space, by default (0.14, 0.9)
        ellipse_axes : tuple, optional
            (a, b) semi-major and semi-minor axes, by default (0.05, 0.3)
            
        Returns
        -------
        np.ndarray
            Array of original event indices that pass the electron elliptical cut
            
        References
        ----------
        Ellipse definition following Siruansh analysis:
        https://indico.cern.ch/event/1293030/contributions/5436570/attachments/2665654/4619155/TB%20week%2014%20June.pdf
        """
        if len(ctot) != len(clong) or len(ctot) != len(elec_had_indices):
            raise ValueError(f"Input arrays must have same length: ctot={len(ctot)}, clong={len(clong)}, indices={len(elec_had_indices)}")
        
        h, k = ellipse_center  # center (Ctot, Clong)
        a, b = ellipse_axes    # semi-major, semi-minor axes
        
        electron_indices = []
        
        for i, event_idx in enumerate(elec_had_indices):
            c_t = ctot[i]
            c_l = clong[i]
            
            if np.isnan(c_t) or np.isnan(c_l):
                continue
            
            # Ellipse equation: ((x-h)/a)² + ((y-k)/b)² ≤ 1
            lhs = ((c_t - h)/a) ** 2 + ((c_l - k)/b) ** 2
            
            # Check if point is inside ellipse
            if lhs < 1:
                electron_indices.append(event_idx)
        
        electron_indices = np.array(electron_indices)
        
        print(f"Events analyzed: {len(elec_had_indices)}")
        print(f"Electron events (elliptical cut): {len(electron_indices)} ({100*len(electron_indices)/len(elec_had_indices):.1f}%)")
        print(f"Ellipse center (Ctot, Clong): ({h}, {k})")
        print(f"Ellipse axes (a, b): ({a}, {b})")
        
        return electron_indices


    # not much useful - remove this func in future
    def single_particles(self):
        
        import matplotlib.pyplot as plt
        
        sig_scint_i = None
        sig_scint_ii = None
        
        # get muon signal in scintilators based on similar energy deposit in LBA 65/ LBC 65
        
        lba_event_energy = self.lba_energy.sum(axis=0)/1000 # GeV
        lbc_event_energy = self.lbc_energy.sum(axis=0)/1000
        ebc_event_energy = self.ebc_energy.sum(axis=0)/1000
        mzeroc_event_energy = self.mzeroc_energy.sum(axis=0)/1000
        
        s1_count = self.h1000['S1cou'].array(library='np')
        s2_count = self.h1000['S2cou'].array(library='np')        
        mask = (s1_count < 2 * self.s1mpv) & (s2_count < 2 * self.s2mpv)
        
        lba_event_energy = lba_event_energy[mask]
        lbc_event_energy = lbc_event_energy[mask]
        
        print(lba_event_energy.shape)
        n_evt = lba_event_energy.shape[0]
        print("LBC", "mean", lbc_event_energy.mean(), "min", lbc_event_energy.min(), "max", lbc_event_energy.max())
        print("LBA", "mean", lba_event_energy.mean(), "min", lba_event_energy.max(), "max", lba_event_energy.max())
        #exit()
        
        #LBC on x axis and LBA on y axis
        y_max = int(lba_event_energy.max()) + 5
        y_max = 5
        x_max = int(lbc_event_energy.max()) + 10
        fig, ax = root_like_histogram(x=lbc_event_energy, y=lba_event_energy, xmin=0, xmax=x_max, ymin=0, ymax=y_max, x_inc=10, y_inc=0.5)
        ax.set_xlabel(r'$\text{E}_{\text{LBC}}  \text{[GeV]}$')
        ax.set_ylabel(r'$\text{E}_{\text{LBA}}  \text{[GeV]}$')
        #plt.hist2d(lbc_event_energy, lba_event_energy, bins=(500, 500))
        
        #plt.colorbar()  
        #plt.savefig('test_muon.jpg')
        
        return fig, ax, n_evt
