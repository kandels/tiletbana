import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
from typing import Tuple, List


def remove_dash(cell: str) -> str:
    return ''.join(cell.split('-'))


def filter_indices(h1000: np.ndarray) -> np.ndarray:
    
    return

def clip_e_positive(arr) -> np.ndarray:
    return np.clip(arr, 0.000001, None)


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
    
    #if ctot:
    #    ax.set_xticklabels([f"{x:.0f}" if x == 0 else f"{x:.2f}" for x in x_ticks])
    #    ax.set_yticklabels([f"{y:.0f}" if y == 0 else f"{y:.1f}" for y in y_ticks])
    
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
    contigious_cells = {
        'A3': [[5, 8, 9, 10, 15, 18], # LBC/M0C
               [6, 7, 11, 10, 1]] # EBC
    }
    return contigious_cells[cell]



def ctot_contigious_cells(cell: str):
    cell = remove_dash(cell)
    # configuration used by stergios
    # https://indico.cern.ch/event/851769/contributions/3581404/attachments/1918746/3173263/TileCal_Analysis_Meeting2019_10_02_TileWeek.pdf
    contigious_cells = {
        'A3':[[5, 8, 9, 10, 15, 18, 6, 7, 11, 12, 16, 17, 13, 14, 25, 26], # LBC/M0C
            [6, 7, 10, 11, 8, 9, 14, 15, 4, 5, 2, 3, 16, 17, 1]] # EBC
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
    long_cell_energy = np.clip(long_cell_energy, 0.000001, None)
    long_deposit_energy = long_cell_energy.sum(axis=0)
    clong = long_deposit_energy / beam_energy    
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
        
        self.lba_energy = clip_e_positive(self.lba_energy)
        self.lbc_energy = clip_e_positive(self.lbc_energy)
        self.ebc_energy = clip_e_positive(self.ebc_energy)
        self.mzeroc_energy = clip_e_positive(self.mzeroc_energy)
    
    
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
        is less than 5 GeV. Detector - LBA 65, LBC 65, M0C, EBC 65. M0A module is discarded for total energy computation.
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
        
        lba_energy = clip_e_positive(lba_energy)
        lbc_energy = clip_e_positive(lbc_energy)
        ebc_energy = clip_e_positive(ebc_energy)
        mzeroc_energy = clip_e_positive(mzeroc_energy)
        
        
        tot_energy = lba_energy.sum(axis=0) + lbc_energy.sum(axis=0) + ebc_energy.sum(axis=0) + mzeroc_energy.sum(axis=0)
        muon_indices = np.where(tot_energy < 5000)[0] # MeV
        
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
        good_muon_adc = (muon_s1 < 8000) & (muon_s2 < 8000)
        s1_mpv = muon_s1[good_muon_adc].mean()
        s2_mpv = muon_s2[good_muon_adc].mean()
        
       
        s1_cut = (s_one < 8000) & (s_one < 2 * s1_mpv)
        s2_cut = (s_two < 8000) & (s_two < 2 * s2_mpv)
        single_particle_mask = s1_cut & s2_cut
        
        single_particle_indices = np.where(single_particle_mask)[0]
        
        print(f"Total events: {len(s_one)}")
        print(f"S1 MPV (MIP): {s1_mpv:.2f}")
        print(f"S2 MPV (MIP): {s2_mpv:.2f}")
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
        
        ctot_cells = ctot_contigious_cells(cell=cell)
    
        lbc_trans = self.lbc_energy[ctot_cells[0]]
        m0c_trans = self.mzeroc_energy[ctot_cells[0]]
        ebc_trans = self.ebc_energy[ctot_cells[1]]
        
        tot_trans = np.vstack([lbc_trans, m0c_trans, ebc_trans])        
        tot_pass_trans = tot_trans[:, events]
        
        ctot = get_ctot(cell=cell, ctot_cell_energy=tot_pass_trans)
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
    
    
