import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
import tempfile
import os


def phaserotate(trc, deg):
    spec = np.fft.rfft(trc)
    spec *=np.exp(1j * deg * np.pi / 180.)
    return np.fft.irfft(spec, trc.size)

def create_ricker_wavelet(
    frequency: float,
    time_length: float = 256.,
    dt: float = 0.001,
    quad: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Ricker wavelet with specified parameters.
    
    Args:
        frequency: The dominant frequency in Hz
        time_length: Total length of the wavelet in milliseconds (default: 256 ms)
        dt: Time sampling interval in seconds (default: 0.001)
        amplitude: Maximum amplitude of the wavelet
        
    Returns:
        Tuple of (time_array, wavelet_array)
    """
    # Create time array

    f0 = frequency/1000.
    om2 = (2. * np.pi *f0)**2
    a2 = 4. / om2
    
    # Convert time_length from ms to seconds to match dt
    time_length_s = time_length / 1000.0
    
    nt = int(time_length_s / dt + 1)
    t = np.linspace(0, nt-1, nt)*dt - (nt//2)*dt
    
    # The time t is now in seconds. To match f0 in kHz, convert time to ms for arg
    arg = (t*1000)**2/a2
    wavelet = (1. - 2.*arg)*np.exp(-arg)

    if quad:
        wavelet = phaserotate(wavelet, -90.)

    return t*1000, wavelet

def analyze_wavelet(
    time_array: np.ndarray,
    wavelet: np.ndarray,
    dt: float = 0.001
) -> dict:
    """
    Analyze a wavelet and return its properties.
    
    Args:
        time_array: Array of time values
        wavelet: Array of wavelet amplitudes
        dt: Time sampling interval in seconds
        
    Returns:
        Dictionary containing wavelet properties
    """
    EPS = 1e-8
    NFFT_MIN = 8192
    PADFACTION = 4
    
    # Calculate frequency spectrum
    time_array = time_array/1000.
    dt = time_array[1] - time_array[0]
    nfft = max(wavelet.size*PADFACTION, NFFT_MIN)
    spec = np.fft.rfft(wavelet, nfft)
    freq = np.fft.rfftfreq(nfft, dt*0.001)
    amp_spec = np.abs(spec)
    pow_spec = 20* np.log10(amp_spec / amp_spec.max() + EPS)
    
    # Calculate wavelet properties
    properties = {
        'peak_amplitude': np.max(np.abs(wavelet)),
        'rms_amplitude': np.sqrt(np.mean(wavelet**2)),
        'amplitude_spectrum': amp_spec,
        'power_spectrum': pow_spec,
        'frequencies': freq
    }
    
    return properties

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


def plot_wavelet(
    time_array: np.ndarray,
    wavelet: np.ndarray,
    properties: Optional[dict] = None,
    show_spectrum: bool = True,
    output_path: Optional[str] = None
) -> str:
    """
    Plot the wavelet and its spectrum, save to file, and return the file path.
    """
    if show_spectrum:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Plot time domain
    ax1.plot(time_array, wavelet, 'b-', label='Wavelet')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Wavelet')
    ax1.grid(True)
    
    if properties and show_spectrum:
        # Plot frequency domain
        ax2.plot(properties['frequencies'], properties['amplitude_spectrum'], 'r-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Frequency Spectrum')
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    plt.savefig(output_path)
    plt.close()
    return output_path

def apply_wavelet_to_trace(
    wavelet: np.ndarray,
    trace: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Convolve a wavelet with a seismic trace.
    
    Args:
        wavelet: Array of wavelet amplitudes
        trace: Input seismic trace
        dt: Time sampling interval in seconds
        
    Returns:
        Convolved trace
    """
    return signal.convolve(trace, wavelet, mode='same')

def create_synthetic_trace(
    wavelet: np.ndarray,
    reflectivity: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Create a synthetic seismic trace by convolving a wavelet with reflectivity.
    
    Args:
        wavelet: Array of wavelet amplitudes
        reflectivity: Array of reflection coefficients
        dt: Time sampling interval in seconds
        
    Returns:
        Synthetic seismic trace
    """
    return apply_wavelet_to_trace(wavelet, reflectivity, dt)
