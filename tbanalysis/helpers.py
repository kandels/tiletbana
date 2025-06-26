import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
from typing import Tuple, List


def remove_dash(cell: str) -> str:
    return ''.join(cell.split('-'))


def mask_negative_energy(arr: np.ndarray) -> np.ma.MaskedArray:
    """
    Mask negative and zero energy values
    
    Parameters
    ----------
    arr : np.ndarray
        Input energy array
        
    Returns
    -------
    np.ma.MaskedArray
        Masked array where negative/zero values are masked
    """
    return np.ma.masked_where(arr <= 0, arr)


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
    fig, ax = plt.subplots(figsize=(8,6))
    
    x_ticks = np.arange(0, xmax + x_inc, x_inc)
    y_ticks = np.arange(0, ymax + y_inc, y_inc)
    
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


class ParticleSeparator:
    
    
    def __init__(self, h1000):
        self.h1000 = h1000
        
        lba_energy = h1000['EfitA02'].arrays(library='np')['EfitA02'].T
        lbc_energy = h1000['EfitC02'].arrays(library='np')['EfitC02'].T
        ebc_energy = h1000['EfitE03'].arrays(library='np')['EfitE03'].T
        mzeroc_energy = h1000['EfitC01'].arrays(library='np')['EfitC01'].T
        
        self.lb_cells = [i for i in range(31)] + [i for i in range(33, 43)] + [44, 45, 46, 47]
        self.eb_cells = [i for i in range(0, 18)] + [20, 21, 22, 23, 24, 25, 26, 27] + [30, 31, 32, 33, 34, 35]
        self.lba_energy = lba_energy[self.lb_cells]
        self.lbc_energy = lbc_energy[self.lb_cells]
        self.ebc_energy = ebc_energy[self.eb_cells]
        self.mzeroc_energy = mzeroc_energy[self.lb_cells]
        
        # Use masking instead of clipping
        self.lba_energy = mask_negative_energy(lba_energy[self.lb_cells])
        self.lbc_energy = mask_negative_energy(lbc_energy[self.lb_cells])
        self.ebc_energy = mask_negative_energy(ebc_energy[self.eb_cells])
        self.mzeroc_energy = mask_negative_energy(mzeroc_energy[self.lb_cells])
    
    
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
        x = h1000['Xcha1_0'].arrays(library='np')['Xcha1_0']
        y = h1000['Ycha1_0'].arrays(library='np')['Ycha1_0']
        
        #μm to mm
        x, y = x * 1e-3, y * 1e-3
        x_mu, _ = norm.fit(x)
        y_mu, _ = norm.fit(y)
        
        x_mask = np.abs(x - x_mu) < 25
        y_mask = np.abs(y - y_mu) < 25
        good_traj = (x_mask) & (y_mask)
        good_traj_indices = np.where(good_traj)[0]
        
        # use those line for debug in future
        total_events = len(x)
        good_events = len(good_traj_indices)
        print(f"Total events: {total_events}")
        print(f"Events with good trajectory: {good_events} ({100*good_events/total_events:.1f}%)")
        print(f"Beam center: x = {x_mu:.4f} mm, y = {y_mu:.4f} mm")
        
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
        
        lba_energy = h1000['EfitA02'].arrays(library='np')['EfitA02'].T[self.lb_cells]
        lbc_energy = h1000['EfitC02'].arrays(library='np')['EfitC02'].T[self.lb_cells]
        ebc_energy = h1000['EfitE03'].arrays(library='np')['EfitE03'].T[self.eb_cells]
        mzeroc_energy = h1000['EfitC01'].arrays(library='np')['EfitC01'].T[self.lb_cells]
        
        # mask negative energy
        lba_energy = mask_negative_energy(lba_energy)
        lbc_energy = mask_negative_energy(lbc_energy)
        ebc_energy = mask_negative_energy(ebc_energy)
        mzeroc_energy = mask_negative_energy(mzeroc_energy)
                        
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
