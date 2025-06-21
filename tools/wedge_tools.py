import math
import numpy as np
import scipy.signal

import os

from typing import Optional, Tuple, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_debug = bool(os.environ.get('DEBUG'))

def debug(*args):
    import sys
    if _debug:
        print('DEBUG:', *args, file = sys.stderr)

def phaserotate(trc, deg):
    spec = np.fft.rfft(trc)
    spec *=np.exp(1j * deg * np.pi / 180.)
    return np.fft.irfft(spec, trc.size)

def ricker(length, dt, f0, quad = False):
    f0 /= 1000.

    om2 = (2. * np.pi *f0)**2
    a2 = 4. / om2
    nt = int(length / dt + 1)
    t = np.linspace(0, nt-1, nt)*dt -nt//2*dt
    arg = t**2/a2
    trc = (1. - 2.*arg)*np.exp(-arg)

    if quad:
        trc = phaserotate(trc, -90.)

    return t, trc

def ormsby(length, dt, f1, f2, f3, f4):
    def numerator(f,t):
        return (np.sinc(f*t)**2)*((np.pi*f)**2)
    
    pf43 = (np.pi*f4) - (np.pi*f3)
    pf21 = (np.pi*f2) - (np.pi*f1)

    nt = int(length/dt) + 1
    t_ms = np.linspace(-length/2, length/2, nt)
    t = t_ms/1000.

    trc = (numerator(f4,t)/pf43 - numerator(f3,t)/pf43 - numerator(f2,t)/pf21 + numerator(f1,t)/pf21)
    trc /= np.max(trc)

    return t_ms, trc

def resample_prep(t, trc, dt_new):
    NPAD_MAX = 2000
    n = trc.size - 1
    dt = t[1] - t[0]

    npad = 0
    for i in range(NPAD_MAX):
        n +=1
        tmax = dt*n
        if math.isclose(int(round(tmax/dt_new))*dt_new, tmax):
            npad = n - trc.size
            break
    else:
        raise Exception('resample_prep: failed to find proper padding length')
        
    t_pad = np.hstack((t, np.arange(1, npad+1)*dt + t[-1]))
    trc_pad = np.hstack((trc, np.zeros(npad)))

    return t_pad, trc_pad
    

def resample(t, trc, dt_new):
    assert t.shape == trc.shape, 'resampleL t and trc need to have the same shape'

    dt = t[1] - t[0]
    if math.isclose(dt, dt_new):
        return t, trc
    
    _, trc_pad = resample_prep(t, trc, dt_new)
    tmax = trc_pad.size*dt

    nfft = trc_pad.size
    spec = np.fft.rfft(trc_pad, nfft)
    freq = np.fft.rfftfreq(nfft, dt*0.001)

    nfft_new = int(round(tmax/dt_new))
    freq_new = np.fft.rfftfreq(nfft_new, dt_new*0.001)

    re = np.interp(freq_new, freq, spec.real, left = 0, right =0)
    im = np.interp(freq_new, freq, spec.img, left =0, right = 0)

    spec_intp = re + 1j*im

    norm_factor = nfft_new/nfft
    trc_resamp = np.fft.irfft(spec_intp, nfft_new)*norm_factor
    t_resamp = np.arange(0, trc_resamp.size)*dt_new+t[0]

    return t_resamp, trc_resamp


def get_fraction(vals, i):
    PERCENT = 10
    EPS = 1e-8

    vals = np.abs(vals)
    vals_min = np.percentile(vals, PERCENT)
    vals_max = np.percentile(vals, 100-PERCENT)

    frac = (vals[i] - vals_min)/(vals_max-vals_min+EPS)
    frac = np.clip(frac, 0, 1)

    frac = frac**3

    return frac


def get_red_rgb(data, i):
    MAX_RED_RGB = np.array([1., 0.4, 0.4])
    MIN_RED_RGB = np.array([1., 0.8, 0.8])

    minvals = data.min(axis =1)
    fract = get_fraction(minvals, 1)

    return tuple(MIN_RED_RGB + fract*(MAX_RED_RGB-MIN_RED_RGB))


def get_blue_rgb(data, i):
    MAX_BLUE_RGB = np.array([0.4, 0.4, 1.])
    MIN_BLUE_RGB = np.array([0.8, 0.8, 1.])

    minvals = data.min(axis =1)
    fract = get_fraction(minvals, 1)

    return tuple(MIN_BLUE_RGB + fract*(MAX_BLUE_RGB-MIN_BLUE_RGB))


def plot_vawig(ax, data, t, z_min, dz, excursion):
    EPS = 1e-8

    [ntrc, nsamp] = data.shape

    data = data/(np.max(np.abs(data))+EPS)

    for i in range(ntrc):
        trace_base = z_min + i*dz
        trace = excursion *data[i] + trace_base

        ax.plot(trace, t, color = 'black', linewidth = 1)

def choose_pick_mode(data, interface_t, halfwin, t0, dt):
    ntraces = data.shape[1]

    last_n = 5
    avg_amp = 0.0

    AMP_THRESHOLD = 0.5

    for i in range(ntraces-last_n, ntraces):
        iwin_end = int(round((interface_t[i] + halfwin-t0)/dt))
        data_ = data[:, i]/np.max(np.abs(data[:iwin_end, i]))
        idx = int(round((interface_t[i]-t0)/dt))
        
        if 0 <= idx < len(data_):
            avg_amp += data_[idx]

    avg_amp /= last_n

    if abs(avg_amp) < AMP_THRESHOLD:
        return 'zero-crossing';
    elif avg_amp >= AMP_THRESHOLD:
        return 'peaks';
    else:
        return 'troughs';

def pick_zero_crossings(data, ref_interface, top_limit, base_limit, t0, dt):
    
    ntraces = data.shape[1] 
    tpicks = ref_interface.copy() 

    it= np.round((ref_interface - t0)/dt).astype('int')

    it_top = np.round((top_limit - t0)/dt).astype('int')
    it_base = np.round((base_limit - t0)/dt).astype('int')

    for itr in range(ntraces): 
        it0 = it[itr] 
        it1 = it[itr] 
        pick = None 

        while pick is None and (it0 > it_top[itr] or it1 < it_base[itr]): 
            for i in [it0, it1]: 
                ii = min(max(i, it_top[itr]), it_base[itr]) 
                product = data[ii - 1, itr] * data[ii, itr] 
                if product == 0.0 or product < 0: 
                    pick = t0+ii*dt
                    break
            it0 -= 1
            it1 += 1
        if pick:
            tpicks[itr] = pick


def peak_peaks_or_troughs(data, top_limit, base_limit, t0, dt, pickmode):
    it_top = np.round((top_limit - t0)/dt).astype('int')  
    it_base = np.round((base_limit - t0)/dt).astype('int') 

    tpicks = np.empty_like(top_limit)
    amp_picks = np.empty_like(top_limit)

    ntraces = data.shape[1]
    t_op = np.argmax if pickmode == 'peaks' else np.argmin
    amp_op = np.max if pickmode == 'peaks' else np.min

    for itr in range(ntraces):
        tpicks[itr] = t_op(data[it_top[itr]:it_base[itr], itr]) + it_top[itr]
        amp_picks[itr] = amp_op(data[it_top[itr]:it_base[itr], itr])

    tpicks = t0 + tpicks*dt
    return tpicks, amp_picks


def pick_interface_and_amp(data, interface1_t, interface2_t, t0, nt, dt):

    halfwin = (interface2_t[-1] - interface1_t[-1]) / 2 
    pickmode = choose_pick_mode(data, interface1_t, halfwin, t0, dt) #debug("pickmode = %s" % pickmode), 

    hor1_tpicks = interface1_t.copy() 
    hor2_tpicks = interface2_t.copy() 
    amp_picks = np.empty_like(interface1_t) 
    tmax = t0 + (nt - 1) * dt 
    ntraces = data.shape[1]
    
    if pickmode == 'zero-crossings':
        top_limit = np.full_like(interface1_t, t0) 
        base_limit = (interface1_t + interface2_t) / 2.0
        hor1_tpicks = pick_zero_crossings(data, interface1_t, top_limit, base_limit, t0, dt)
        top_limit = base_limit
        base_limit = np.full_like(interface1_t, tmax) 
        hor2_tpicks = pick_zero_crossings(data, interface2_t, top_limit, base_limit, t0, dt) 
        fract = 0.67

        top_limit = hor1_tpicks
        base_limit = hor1_tpicks+(hor2_tpicks-hor1_tpicks)*fract

        it_top = np.round((top_limit - t0)/dt).astype('int')
        it_base = np.round((hor1_tpicks + (hor2_tpicks - hor1_tpicks)*fract - t0)/dt).astype('int')
        
        last_n = 5
        sum_amp = 0.0
        for itr in range(1, last_n+1): 
            sum_amp += data[it_top[-itr]+2, -itr]

        amp_op = np.max if sum_amp > 0.0 else np.min
        t_op = np.argmax if sum_amp> 0.0 else np.argmin

        hor3_tpicks = np.empty_like(interface1_t)
        for itr in range(ntraces):
            hor3_tpicks[itr] = t_op(data[it_top[itr]:it_base[itr], itr]) + it_top[itr]
            amp_picks = amp_op(data[it_top[itr]:it_base[itr], itr])
        hor3_tpicks = t0 + hor3_tpicks*dt
    else:
        top_limit = np.full_like(interface1_t, t0)
        base_limit = (interface1_t + interface2_t)/2.0
        hor1_tpicks, amp_picks = peak_peaks_or_troughs(data, top_limit, base_limit, t0, dt, pickmode)
        top_limit = base_limit
        base_limit = np.full_like(interface1_t, tmax)
        reverse_pickmode = 'troughs' if pickmode == 'peaks' else 'peaks'
        hor2_tpicks, _ = peak_peaks_or_troughs(data, top_limit, base_limit, t0, dt, reverse_pickmode)
        hor3_tpicks = None

    return hor1_tpicks, hor2_tpicks, hor3_tpicks, amp_picks

def create_figure():
    params = {
        'legend.fontsize': 'x-large',
        'axes.labelsize': 16,
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
    }
    plt.rcParams.update(params)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    return fig, axes

def make_plot(zunit, data, wavelet_label, vp_layers, rho_layers, thickness, \
    interface1_t, interface2_t, t0, nt, dt, z_min, z_max, dz, gain, plotpadtime, thickness_domain,\
        fig_fname, csv_fname=''):
    
    hor1_tpicks, hor2_tpicks, hor3_tpicks, amp_picks = pick_interface_and_amp(data, interface1_t, interface2_t, t0, nt, dt)
    thickness_apparent_t = hor2_tpicks - hor1_tpicks
    thickness_apparent_z = thickness_apparent_t*vp_layers[1]/2000

    if thickness_domain == 'time':
        thickness_true = 2000. *thickness/vp_layers[1]
        thickness_unit = 'ms'

    else:
        thickness_true = thickness
        thickness_unit = zunit
    
    excursion = gain*dz
    fig, axes = create_figure()

    ax0, ax1, ax2 = axes
    layer_labels = []

    for i in range(3):
        layer_labels.append('Layer %d\n$V_P$=%.2f %s/s\n$\\rho$=%.2f $g/cc$' % (i+1, vp_layers[i], zunit, rho_layers[i]))

    ax0.plot(thickness, interface1_t, color = 'blue', lw=1.5)
    ax0.plot(thickness, interface2_t, color = 'red', lw=1.5)
    min_plot_time = interface1_t[0] - plotpadtime
    max_plot_time = interface2_t[-1] + plotpadtime
    ax0.set_ylim(min_plot_time, max_plot_time)
    ax0.set_xlim(z_min-excursion, z_max+excursion)
    ax0.invert_yaxis()

    xlabel = 'True Thickness (%s)' % zunit
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel('Time (ms)')
    ax0.tick_params(top=True, right = True, labelright = True)
    ax0.grid(linestyle= ':')

    ax0.text(2,
        min_plot_time+(interface1_t[0]-min_plot_time)*0.5,
        layer_labels[0],
        verticalalignment = 'center',
        fontsize = 16)
    ax0.text((z_min+z_max)*0.8,
        interface1_t[-1] + (interface2_t[-1]-interface1_t[-1])*0.5,
        layer_labels[1],
        verticalalignment = 'center',
        fontsize = 16
    )
    ax0.text(2,
        interface2_t[0] + (max_plot_time-interface1_t[0])*0.5,
        layer_labels[2],
        verticalalignment = 'center',
        fontsize = 16
    )

    t= t0+np.arange(nt)*dt
    plot_vawig(ax1, data.T, t, z_min, dz, excursion)
    ax1.plot(thickness, interface1_t, color = 'blue', lw = 1)
    ax1.plot(thickness, interface2_t, color = 'red', lw = 1)

    if hor3_tpicks is None:
        ax1.plot(thickness, hor1_tpicks, 'o', color = 'blue', lw  =2, linestyle = '--')
    else:
        ax1.plot(thickness, hor1_tpicks, 'o', color = 'black', lw = 2, linestyle = '--')
        ax1.plot(thickness, hor3_tpicks, 'o', color = 'blue', lw = 2, linestyle = ':')
    ax1.plot(thickness, hor2_tpicks, 'o', color = 'red', lw = 2, linestyle = '--')
    ax1.set_xlim(z_min-excursion, z_max+excursion)
    ax1.set_ylim(min_plot_time, max_plot_time)
    ax1.invert_yaxis()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(top = True, right = True, labelright = True)
    ax1.text(0,
        min_plot_time + (max_plot_time - min_plot_time)*0.9,
        wavelet_label,
        verticalalignment='center',
        fontsize=16,
        bbox = dict(facecolor = 'white')
        )

    itrc_tuning = np.argmax(np.abs(amp_picks))
    tuning_thickness = z_min+itrc_tuning*dz

    ax2.plot(thickness, amp_picks, color = 'blue')
    ax2.tick_params(top = True)

    ax2.tick_params(axis='y', labelcolor = 'blue')
    ax2.set_xlim(z_min- excursion, z_max + excursion)
    ax2.axvline(tuning_thickness, color = 'k', lw=2, linestyle = '--')
    ax2.plot(tuning_thickness, amp_picks[itrc_tuning], marker = 'o', markersize =8, color = 'k', linestyle = ':')
    ax2.set_xlabel('Upper Interface Amplitude', color = 'blue')
    ax2.grid(True, axis = 'x', linestyle = ':')

    if amp_picks[itrc_tuning] > 0:
        y = ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0])*0.1
    else:
        y = ax2.get_ylim()[1] + (ax2.get_ylim()[1] - ax2.get_ylim()[0])*0.17
    
    dx = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 60
    ax2.text(tuning_thickness + dx, y, 
            'peak tuning thickness:\n%.1f %s (%.1f ms)' % (tuning_thickness, zunit, tuning_thickness*2000/vp_layers[1]), 
             fontsize = 16)

    ax3 = ax2.twinx()
    ax3_color = 'magenta'
    ax3.plot(thickness, thickness_true, color = ax3_color, lw = 0.5, linestyle = '--')
    ax3.plot(thickness, thickness_apparent_t if thickness_domain == 'time' else
                thickness_apparent_z, color = ax3_color)
    ax3.tick_params(top = True)
    ax3.tick_params(axis = 'y', labelcolor = ax3_color)
    min_thickness_txt = 'minimum apparent thickness:\n%.1f %s (%.1f ms)' % (thickness_apparent_z.min(), zunit, thickness_apparent_t.min())


    # Fix variable name: xalbel -> xlabel
    ax2.set_xlabel(xlabel + '\n\n%s' % min_thickness_txt)
    ax3.set_ylabel('Apparent Thickness (%s)' % thickness_unit, color = ax3_color)
    ax3.grid(True, axis = 'y', linestyle = ':')

    plt.tight_layout()
    plt.savefig(fig_fname)
    plt.close()

    if csv_fname:
        curves = np.vstack((thickness, amp_picks, thickness_apparent_t, thickness_apparent_z)).T
        header = ('True_Thickness_%s, Upper_Interface_Amplitude, Apparent_Thickness_ms, Apparent_Thickness_%s' % (zunit, zunit))
        np.savetxt(csv_fname, curves, fmt = '%g',delimiter = ',', header = header, comments = '')

def make_symmetric_wavelet(t, wavelet):
    if np.alltrue(t<0) or np.alltrue(t>=0):
        raise Exception('Input wavelet needs to be sampled at both negative and positive time values.')
    
    nt_positive = (t>0).sum()
    nt_negative = (t<0).sum()
    dt = t[1] - t[0]
    if nt_positive > nt_negative:
        ndiff = nt_positive - nt_negative
        t = np.hstack((np.arange(-ndiff,0)*dt+t[0], t))
        wavelet = np.hstack([np.zeros(ndiff), wavelet])
    elif nt_negative > nt_positive:
        ndiff = nt_negative - nt_positive
        t = np.hstack((t, t[-1]+ np.arange(1, ndiff+1)*dt))
        wavelet = np.hstack([wavelet, np.zeros(ndiff)])

    return t, wavelet

def gen_wavelet(dt, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, wavelet_length=500):
    if wv_type == 'ricker':
        t, wavelet = ricker(wavelet_length, dt, ricker_freq)
        wavelet_label = 'Ricker %d Hz' % ricker_freq
    elif wv_type == 'ormsby':
        freqs = ormsby_freq.strip().split(',')
        if len(freqs)!=4:
            raise Exception('Need exactly four frequencies for Ormsby wavelet.')

        try:
            f1, f2, f3, f4 = map(float, freqs)
        except:
            raise Exception('Could not parse frequencies for Ormsby wavelet.')

        if f1 >= f2 or f2 >=f3 or f3>=f4:
            raise Exception('Ormsby wavelet frequencies must be strictly increasing.')

        if f1 < 0:
            raise Exception('Ormsby wavelet frequencies must be positive.')
        t, wavelet = ormsby(wavelet_length, dt, f1, f2, f3, f4)
        wavelet_label = 'Ormsby %s Hz' % ormsby_freq.replace(' ', '')

    else:
        t, wavelet = parse_and_prep_wavelet(wavelet_str, dt)
        wavelet_label = wavelet_fname if wavelet_fname else 'Custom Wavelet'
        if len(wavelet_label) >4 and wavelet_label[-4:].upper() in ['.TXT', '.CSV']:
            wavelet_label = wavelet_label[:-4]

    if phase_rot == 0:
        if wv_type in ['ricker', 'ormsby']:
            wavelet_label += ' (zero phase)'

        else:
            wavelet_label += ' with $%.0f^\\circ$ phase rotation' % phase_rot

    return t, wavelet, wavelet_label

def spectrum_analysis(t, wavelet):
    EPS = 1e-8
    NFFT_MIN = 8192
    PADFACTION = 4

    dt = t[1] - t[0]
    nfft = max(wavelet.size*PADFACTION, NFFT_MIN)
    spec = np.fft.rfft(wavelet, nfft)
    freq = np.fft.rfftfreq(nfft, dt*0.001)
    amp_spec = np.abs(spec)
    pow_spec = 20* np.log10(amp_spec / amp_spec.max() + EPS)

    return freq, amp_spec, pow_spec

def wavelet_trim_small_val(t, wavelet):
    i0=0
    i1=wavelet.size - 1
    EPS = 2e-4*np.abs(wavelet).max()
    while i0 < i1 and np.abs(wavelet[i0]) < EPS and np.abs(wavelet[i1]) < EPS:
        i0 += 1
        i1 -= 1
    return t[i0:i1+1], wavelet[i0:i1+1]

def spectrum_trim_small_val(freq, amp_spec, pow_spec):
    EPS = 2e-4*amp_spec.max()
    idx = amp_spec.size - 1
    while idx > 0 and amp_spec[idx] < EPS:
        idx -= 1
    return freq[:idx+1], amp_spec[:idx+1], pow_spec[:idx+1]

def plot_wavelet(dt, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, wavelet_length, fig_fname):

    t, wavelet, wavelet_label = gen_wavelet(dt, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, wavelet_length)
    t, wavelet = wavelet_trim_small_val(t, wavelet)
    freq, amp_spec, pow_spec = spectrum_analysis(t, wavelet)
    freq, amp_spec, pow_spec = spectrum_trim_small_val(freq, amp_spec, pow_spec)
    
    fig, axes = create_figure()
    ax0, ax1, ax2 = axes
    lw = 1.5
    ax0.plot(t, wavelet, color = 'black', lw = lw, label = 'wavelet')
    ax0.legend()
    ax0.fill_between(t, wavelet, 0, where = wavelet > 0, facecolor = [0.8, 0.8, 1.0], interpolate = True)
    ax0.fill_between(t, wavelet, 0, where = wavelet < 0, facecolor = [1.0, 0.8, 0.8], interpolate = True)
    ax0.set_xlabel('Time (ms)')
    ax0.set_ylabel('Amplitude')
    ax0.set_title(wavelet_label, fontsize = 18)
    ax0.tick_params(top = True, right = True, labelright = True)
    ax0.grid(linestyle = ':')

    ax1.plot(freq, amp_spec, color = 'green', lw = lw, label = 'Amplitude spectrum')
    ax1.legend()
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude (linear)')
    ax1.tick_params(top = True, right = True, labelright = True)
    ax1.grid(linestyle = ':')
    
    ax2.plot(freq, pow_spec, color = 'blue', lw = lw, label = 'Power spectrum (normalized)')
    ax2.legend()
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_ylim((-65, 5))
    ax2.tick_params(top = True, right = True, labelright = True)
    ax2.grid(linestyle = ':')
    
    plt.tight_layout()
    plt.savefig(fig_fname)
    plt.close()

def wedge_model(zunit, max_thickness, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, vp1, vp2, vp3, rho1, rho2, rho3, gain, plotpadtime, thickness_domain, fig_fname, csv_fname):
    """
    Creates a wedge model for seismic analysis.
    
    Parameters:
    - zunit: Unit for depth/thickness (e.g., 'm', 'ft')
    - max_thickness: Maximum thickness of the wedge
    - wv_type: Wavelet type ('ricker', 'ormsby', or custom)
    - ricker_freq: Frequency for Ricker wavelet (Hz)
    - ormsby_freq: Comma-separated frequencies for Ormsby wavelet
    - wavelet_str: Custom wavelet string representation
    - wavelet_fname: Filename for custom wavelet
    - phase_rot: Phase rotation in degrees
    - vp1, vp2, vp3: P-wave velocities for the three layers (units/s)
    - rho1, rho2, rho3: Densities for the three layers (g/cc)
    - gain: Gain factor for display
    - plotpadtime: Padding time for plots (ms)
    - thickness_domain: Domain for thickness calculation ('time' or 'depth')
    - fig_fname: Output figure filename
    - csv_fname: Output CSV filename for curves
    """
    # Create arrays for layer properties
    vp_layers = [vp1, vp2, vp3]
    rho_layers = [rho1, rho2, rho3]
    
    # Calculate acoustic impedance for each layer
    imp_layers = [vp_layers[i]*rho_layers[i] for i in range(3)]

    # Set up model geometry
    z_min = 0
    z_max = max_thickness
    ntraces = 61  # Number of traces in the model
    dz = (z_max - z_min)/(ntraces - 1)  # Trace spacing

    # Set time sampling interval
    dt = 0.1  # ms

    # Generate wavelet based on specified parameters
    t, wavelet, wavelet_label = gen_wavelet(dt, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, wavelet_length=256.0)
    wavelet_length = t[-1] - t[0] + dt
    
    # Calculate padding time to ensure model can fit the wavelet
    pad_time = plotpadtime
    model_time = 2*pad_time + 2000*(z_max - z_min)/vp_layers[1]  # Total model time in ms

    # Adjust padding if needed to accommodate wavelet length
    if model_time < wavelet_length:
        pad_time += (wavelet_length - model_time)/2.0 + dt
    
    # Calculate number of time samples
    nt = int(round(2*pad_time + 2000*(z_max - z_min)/vp_layers[1]/dt))

    # Initialize reflection coefficient model
    rc_model = np.zeros((nt, ntraces))

    # Calculate reflection coefficients at layer interfaces
    rc1 = (imp_layers[1] - imp_layers[0])/(imp_layers[1] + imp_layers[0])  # Upper interface
    rc2 = (imp_layers[2] - imp_layers[1])/(imp_layers[2] + imp_layers[1])  # Lower interface

    # Create thickness array for the wedge
    thickness = np.linspace(z_min, z_max, ntraces)

    # Set reference time and starting time
    t_ref = 300  # Reference time for first interface (ms)
    t0 = t_ref - pad_time  # Starting time of the model

    # Calculate time of interfaces
    interface1_t = t_ref + thickness*0  # Upper interface (constant time)
    interface2_t = t_ref + thickness*2000/vp_layers[1]  # Lower interface (varies with thickness)

    # Ensure model is large enough for the maximum interface time
    max_interface_time = max(interface1_t.max(), interface2_t.max())
    min_interface_time = min(interface1_t.min(), interface2_t.min())
    
    # Calculate required model size
    required_time_range = max_interface_time - min_interface_time + 2*pad_time
    nt = max(nt, int(round(required_time_range/dt)) + 100)  # Add some buffer
    
    # Recalculate t0 to ensure all interfaces fit
    t0 = min_interface_time - pad_time
    
    # Reinitialize reflection coefficient model with correct size
    rc_model = np.zeros((nt, ntraces))

    # Place reflection coefficients with bounds checking
    for itr in range(ntraces):
        idx1 = int(round((interface1_t[itr] - t0)/dt))
        idx2 = int(round((interface2_t[itr] - t0)/dt)) + 1
        
        # Check bounds before assignment
        if 0 <= idx1 < nt:
            rc_model[idx1, itr] = rc1
        if 0 <= idx2 < nt:
            rc_model[idx2, itr] = rc2

    # Convolve reflection coefficients with wavelet to create synthetic seismic data
    data = np.apply_along_axis(lambda _t: scipy.signal.convolve(_t, wavelet, mode = 'same'), axis = 0, arr = rc_model)

    # Save intermediate results for debugging if enabled
    if _debug:
        import pickle
        input_date = dict(
            data = data,
            wavelet_label = wavelet_label,
            vp_layers = vp_layers,
            rho_layers = rho_layers,
            thickness = thickness,
            interface1_t = interface1_t,
            interface2_t = interface2_t,
            t0 = t0,
            nt = nt,
            dt = dt,
            z_min = z_min,
            z_max = z_max,
            dz = dz,
            gain = gain,
            plotpadtime = plotpadtime,
            thickness_domain = thickness_domain,
            fig_fname = fig_fname,
            csv_fname = csv_fname,
        )
        pickle.dump(input_date, open('save.p', 'wb'))

    # Generate plot
    import tempfile
    fig_fd, fig_fname = tempfile.mkstemp(suffix=".png")
    os.close(fig_fd)
    
    make_plot(
        zunit, data, wavelet_label, vp_layers, rho_layers, thickness,
        interface1_t, interface2_t, t0, nt, dt, z_min, z_max, dz, gain,
        plotpadtime, thickness_domain, fig_fname, ''
    )

    return t, rc_model, data, {
        'max_thickness': max_thickness,
        'v1': vp1, 'v2': vp2, 'v3': vp3,
        'rho1': rho1, 'rho2': rho2, 'rho3': rho3,
        'rc1': rc1, 'rc2': rc2,
        'wavelet_freq': ricker_freq if wv_type == 'ricker' else float(ormsby_freq.split(',')[0]),
        'dt': dt,
        'num_traces': ntraces,
        'wavelet_label': wavelet_label,
        'zunit': zunit,
        'thickness_domain': thickness_domain,
        'interface1_t': interface1_t,
        'interface2_t': interface2_t,
        't0': t0,
        'nt': nt,
        'dz': dz,
        'gain': gain,
        'plotpadtime': plotpadtime
    }, fig_fname

def create_wedge_model(
    max_thickness: float,
    v1: float,
    v2: float,
    v3: float,
    rho1: float,
    rho2: float,
    rho3: float,
    num_traces: int = 61,
    dt: float = 0.1,
    wavelet_freq: float = 30.0,
    wavelet_length: float = 256.0,
    phase_rot: float = 0.0,
    wv_type: str = 'ricker',
    ormsby_freq: Optional[str] = None,
    gain: float = 1.0,
    plotpadtime: float = 50.0,
    thickness_domain: str = 'depth',
    zunit: str = 'm'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create a wedge model with specified parameters and return the path to the plot.
    """
    # Call the internal wedge_model function that does the work
    time_array, model, synthetic, parameters, _ = wedge_model(
        zunit=zunit,
        max_thickness=max_thickness,
        wv_type=wv_type,
        ricker_freq=wavelet_freq,
        ormsby_freq=ormsby_freq,
        wavelet_str='',
        wavelet_fname='',
        phase_rot=phase_rot,
        vp1=v1,
        vp2=v2,
        vp3=v3,
        rho1=rho1,
        rho2=rho2,
        rho3=rho3,
        gain=gain,
        plotpadtime=plotpadtime,
        thickness_domain=thickness_domain,
        fig_fname='',
        csv_fname=''
    )
    
    return time_array, model, synthetic, parameters

def plot_wedge_model(
    synthetic_data: np.ndarray,
    parameters: Dict
) -> str:
    """
    Plot a wedge model and return the path to the plot.
    """
    import tempfile
    
    fig_fd, fig_fname = tempfile.mkstemp(suffix=".png")
    os.close(fig_fd)
    
    make_plot(
        zunit=parameters['zunit'],
        data=synthetic_data,
        wavelet_label=parameters['wavelet_label'],
        vp_layers=[parameters['v1'], parameters['v2'], parameters['v3']],
        rho_layers=[parameters['rho1'], parameters['rho2'], parameters['rho3']],
        thickness=np.linspace(0, parameters['max_thickness'], parameters['num_traces']),
        interface1_t=parameters['interface1_t'],
        interface2_t=parameters['interface2_t'],
        t0=parameters['t0'],
        nt=parameters['nt'],
        dt=parameters['dt'],
        z_min=0,
        z_max=parameters['max_thickness'],
        dz=parameters['dz'],
        gain=parameters['gain'],
        plotpadtime=parameters['plotpadtime'],
        thickness_domain=parameters['thickness_domain'],
        fig_fname=fig_fname,
        csv_fname=''
    )
    
    return fig_fname

def analyze_wedge_model(
    time_array: np.ndarray,
    synthetic_data: np.ndarray,
    parameters: Dict,
    show_plot: bool = True
) -> Dict:
    """
    Analyze a wedge model and return its properties.
    
    Args:
        time_array: Array of time values
        synthetic_data: 2D array of synthetic data
        parameters: Dictionary of model parameters
        show_plot: Whether to show the analysis plot
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate tuning thickness
    v2 = parameters['v2']
    freq = parameters['wavelet_freq']
    tuning_thickness = v2 / (4 * freq)
    
    # Find maximum amplitude for each trace
    max_amplitudes = np.max(np.abs(synthetic_data), axis=0)
    
    # Calculate thicknesses
    thicknesses = np.linspace(0, parameters['max_thickness'], parameters['num_traces'])
    
    # Find tuning point
    tuning_idx = np.argmax(max_amplitudes)
    tuning_amplitude = max_amplitudes[tuning_idx]
    
    # Calculate resolution limit (1/4 wavelength)
    resolution_limit = tuning_thickness / 2
    
    # Store analysis results
    analysis = {
        'tuning_thickness': tuning_thickness,
        'tuning_amplitude': tuning_amplitude,
        'resolution_limit': resolution_limit,
        'max_amplitudes': max_amplitudes,
        'thicknesses': thicknesses
    }
    
    if show_plot:
        plot_wedge_analysis(time_array, synthetic_data, analysis, parameters)
    
    return analysis

def plot_wedge_analysis(
    time_array: np.ndarray,
    synthetic_data: np.ndarray,
    analysis: Dict,
    parameters: Dict
) -> None:
    """
    Plot the wedge model analysis.
    
    Args:
        time_array: Array of time values
        synthetic_data: 2D array of synthetic data
        analysis: Dictionary of analysis results
        parameters: Dictionary of model parameters
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot synthetic data
    im = ax1.imshow(synthetic_data, aspect='auto', 
                    extent=[0, parameters['max_thickness'], time_array[-1], 0],
                    cmap='seismic', 
                    vmin=-np.max(np.abs(synthetic_data)), 
                    vmax=np.max(np.abs(synthetic_data)))
    ax1.set_xlabel(f'Thickness ({parameters["zunit"]})')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Wedge Model Synthetic Data')
    plt.colorbar(im, ax=ax1, label='Amplitude')
    
    # Plot amplitude vs thickness
    ax2.plot(analysis['thicknesses'], analysis['max_amplitudes'], 'b-', label='Maximum Amplitude')
    ax2.axvline(analysis['tuning_thickness'], color='r', linestyle='--',
                label=f'Tuning Thickness: {analysis["tuning_thickness"]:.2f} {parameters["zunit"]}')
    ax2.axvline(analysis['resolution_limit'], color='g', linestyle='--',
                label=f'Resolution Limit: {analysis["resolution_limit"]:.2f} {parameters["zunit"]}')
    ax2.set_xlabel(f'Thickness ({parameters["zunit"]})')
    ax2.set_ylabel('Maximum Amplitude')
    ax2.set_title('Amplitude vs Thickness')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()