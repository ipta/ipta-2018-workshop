from __future__ import division
import numpy as np
import glob
import matplotlib.pyplot as plt

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


def create_gw_antenna_pattern(theta, phi, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians
    
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([np.sin(gwphi), -np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta)*np.cos(gwphi), 
                  -np.cos(gwtheta)*np.sin(gwphi),
                  np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), 
                      -np.sin(gwtheta)*np.sin(gwphi),
                      -np.cos(gwtheta)])

    phat = np.array([np.sin(theta)*np.cos(phi), 
                     np.sin(theta)*np.sin(phi), 
                     np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu

@signal_base.function
def cw_delay(toas, theta, phi, cos_gwtheta=0, gwphi=0, log10_mc=9, log10_dL=2, log10_fgw=-8, 
             phase0=0, psi=0, cos_inc=0, log10_h=None, p_dist=1, p_phase=None, 
             inc_psr_term=True, evolve=False, phase_approx=True, tref=53000*86400):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.
    :param toas: Pular toas in seconds
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param cos_gwtheta: Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param log10_mc: log10 of Chirp mass of SMBMB [solar masses]
    :param log10_dL: log10 of Luminosity distance to SMBMB [Mpc]
    :param log10_fgw: log10 of Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param cos_inc: cosine of Inclination of GW source [radians]
    :param p_dist: Pulsar distance to use other than those in psr [kpc]
    :param p_phase: Use pulsar phase to determine distance [radian]
    :param psrTerm: Option to include pulsar term [boolean] 
    :param evolve: Option to exclude full evolution [boolean]
    :param phase_approx: Option to not model phase evolution across observation time [default]
    :param tref: Reference time for phase and frequency [s]
    
    :return: Vector of induced residuals
    """
    
    # convert units
    mc = 10**log10_mc * const.Tsun               
    dist = 10**log10_dL * const.Mpc / const.c   
    p_dist *= const.kpc / const.c   
    gwtheta = np.arccos(cos_gwtheta)
    inc = np.arccos(cos_inc)
    fgw = 10**log10_fgw
    
    # is log10_h is given, use it
    if log10_h is not None:
        dist = 2 * mc**(5/3) * (np.pi*fgw)**(2/3) / 10**log10_h

    # get antenna pattern funcs and cosMu
    fplus, fcross, cosMu = create_gw_antenna_pattern(theta, phi, gwtheta, gwphi)
    
    # get pulsar time
    toas -= tref
    tp = toas-p_dist*(1-cosMu)

    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    omegadot = 96/5 * mc**(5/3) * w0**(11/3)

    # evolution
    if evolve:

        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)

        # calculate time dependent phase
        phase = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))
        phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3))
    

    elif phase_approx:
        
        # monochromatic
        omega = np.pi*fgw
        omega_p = w0 * (1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)
        
        # phases
        phase = phase0 + omega * toas
        if p_phase is not None:
            phase_p = phase0 + p_phase + omega_p * toas
        else:
            phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3)) + omega_p*toas
          
    # no evolution
    else: 
        
        # monochromatic
        omega = np.pi*fgw
        omega_p = omega
        
        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp
        

    # define time dependent coefficients
    At = -0.5*np.sin(2*phase)*(3+np.cos(2*inc))
    Bt = 2*np.cos(2*phase)*np.cos(inc)
    At_p = -0.5*np.sin(2*phase_p)*(3+np.cos(2*inc))
    Bt_p = 2*np.cos(2*phase_p)*np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))
    
    # define rplus and rcross
    rplus = alpha*(-At*np.cos(2*psi)+Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(-At_p*np.cos(2*psi)+Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))

    # residuals
    if inc_psr_term:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res


@signal_base.function
def free_spectrum(f, log10_rho=None):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return np.repeat(10**(2*log10_rho), 2)


def CWSignal(cw_wf, inc_psr_term=True):

    BaseClass = deterministic_signals.Deterministic(cw_wf, name='cgw')

    class CWSignal(BaseClass):

        def __init__(self, psr):
            super(CWSignal, self).__init__(psr)
            self._wf[''].add_kwarg(inc_psr_term=inc_psr_term)
            if inc_psr_term:
                pdist = parameter.Normal(psr.pdist[0], psr.pdist[1])('_'.join([psr.name, 'cgw', 'pdist']))
                pphase = parameter.Uniform(0, 2*np.pi)('_'.join([psr.name, 'cgw', 'pphase']))
                self._params['p_dist'] = pdist
                self._params['p_phase'] = pphase
                self._wf['']._params['p_dist'] = pdist
                self._wf['']._params['p_phase'] = pphase

    return CWSignal


def get_noise_from_pal2(noisefile):
        psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
        fin = open(noisefile, 'r')
        lines = fin.readlines()
        params = {}
        for line in lines:
            ln = line.split()
            if 'efac' in line:
                par = 'efac'
                flag = ln[0].split('efac-')[-1]
            elif 'equad' in line:
                par = 'log10_equad'
                flag = ln[0].split('equad-')[-1]
            elif 'jitter_q' in line:
                par = 'log10_ecorr'
                flag = ln[0].split('jitter_q-')[-1]
            elif 'RN-Amplitude' in line:
                par = 'log10_A'
                flag = ''
            elif 'RN-spectral-index' in line:
                par = 'gamma'
                flag = ''
            else:
                break
            if flag:
#                name = [psrname, flag.replace('-', '_'), par] # use this for 5yr noisefiles
                name = [psrname, flag, par]
            else:
                name = [psrname, par]
            pname = '_'.join(name)
            params.update({pname: float(ln[1])})
        return params


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)
    
    gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    ipars = np.array([p for p in pars if p not in gpars])

    return gpars, ipars


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names
    
    # get global and individual parameters
    gpars, ipars = get_global_parameters(pta)
    if any(gpars):
        groups.extend([[params.index(gp) for gp in gpars]])
    
    for sc in pta._signalcollections:
        for signal in sc._signals:
            ind = [params.index(p) for p in signal.param_names if p not in gpars]
            if ind:
                groups.extend([ind])

    return groups


class JumpProposal(object):
    
    def __init__(self, pta, snames=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)
        
        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size
        
        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct
        
        # collecting signal parameters across pta
        if snames is None:
            self.snames = dict.fromkeys(np.unique([[qq.signal_name for qq in pp._signals]
                                                   for pp in pta._signalcollections]))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = np.unique(self.snames[key]).tolist()
        else:
            self.snames = snames

    def draw_from_ephem_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        signal_name = 'phys_ephem'
        
        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]
    
        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)
    
    def draw_from_prior(self, x, iter, beta):
        """Prior draw.
            The function signature is specific to PTMCMCSampler.
            """
        
        q = x.copy()
        lqxy = 0
        
        # randomly choose parameter
        idx = np.random.randint(0, self.npar)
        
        # if vector parameter jump in random component
        param = self.params[idx]
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]
        
        # scalar parameter
        else:
            q[idx] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        # draw parameter from signal model
        idx = self.pnames.index('log10_h')
        q[idx] = np.random.uniform(-18, -11)
        
        return q, 0
