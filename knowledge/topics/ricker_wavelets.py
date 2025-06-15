RICKER_KNOWLEDGE = {
    'overview': """A **Ricker wavelet** is a zero-phase wavelet commonly used in seismic modeling and processing. It's mathematically defined as the second derivative of a Gaussian function.

**Mathematical Definition:**
```
w(t) = (1 - 2π²f²t²) × exp(-π²f²t²)
```
Where f is the dominant frequency and t is time.

**Key Characteristics:**
- **Zero-phase**: Symmetric shape with peak at time zero
- **Finite duration**: Compact in time with minimal side lobes  
- **Known frequency content**: Dominant frequency easily controlled
- **Causal**: Can be made causal by time shifting
- **Bandwidth**: Approximately 1.5 octaves at -3dB points

**Applications:**
- Synthetic seismogram generation
- Seismic forward modeling
- Wavelet processing and deconvolution
- Resolution studies and thin bed analysis

**Advantages:**
- Simple analytical form
- Zero-phase preserves timing relationships
- Good approximation of many seismic sources
- Computationally efficient

The Ricker wavelet's shape and bandwidth are entirely determined by its dominant frequency, making it an ideal choice for controlled seismic modeling experiments.""",

    'frequency': """The **frequency characteristics** of a Ricker wavelet are fundamental to its behavior:

**Dominant Frequency (f₀):**
- The peak frequency in the amplitude spectrum
- Controls both temporal width and spectral bandwidth
- Relationship: temporal width ∝ 1/f₀

**Bandwidth Characteristics:**
- **-3dB bandwidth**: ~1.5 octaves (frequency range where amplitude > 70%)
- **Effective bandwidth**: ~0.3f₀ to 2.5f₀  
- **Peak frequency**: Exactly at the specified dominant frequency

**Frequency-Domain Properties:**
- Smooth, bell-shaped amplitude spectrum
- No energy at DC (0 Hz) due to zero-mean property
- Rapid amplitude decay beyond 3×f₀

**Practical Frequency Selection:**
- **High-resolution shallow surveys**: 100-300 Hz
- **Land seismic exploration**: 20-80 Hz
- **Marine seismic**: 15-120 Hz
- **Deep exploration**: 5-40 Hz

**Trade-offs:**
- **Higher frequency**: Better temporal resolution, less penetration, more noise sensitive
- **Lower frequency**: Better penetration, lower resolution, less noise sensitive

**Sampling Considerations:**
- Nyquist frequency: f_sample ≥ 2 × f_max
- Typical sampling: dt = 1/(20×f₀) for good representation""",

    'creation': """To **create a Ricker wavelet**, you need to specify several parameters:

**Required Parameters:**
- **Frequency (f)**: Dominant frequency in Hz
  - Typical range: 5-300 Hz depending on application
  - Controls wavelet width and spectral content

**Optional Parameters:**
- **Sampling interval (dt)**: Time between samples
  - Typical range: 0.0001 to 0.01 seconds
  - Rule of thumb: dt ≤ 1/(20×f) for accurate representation
  - Example: For 50 Hz → dt ≤ 0.001 s

- **Duration**: Total wavelet length in seconds
  - Typical range: 0.1 to 2.0 seconds
  - Rule of thumb: duration ≥ 6/f for complete wavelet
  - Example: For 30 Hz → duration ≥ 0.2 s

**Creation Process:**
1. Define time vector: t = [-duration/2 : dt : duration/2]
2. Apply Ricker formula: w(t) = (1 - 2π²f²t²) × exp(-π²f²t²)
3. Normalize if desired (optional)

**Quality Checks:**
- Zero-mean: sum(w) ≈ 0
- Symmetric: w(t) = w(-t)
- Peak at t=0: max(w) occurs at center

**Example Parameters:**
- 30 Hz exploration: f=30, dt=0.001, duration=0.3
- 100 Hz high-res: f=100, dt=0.0005, duration=0.1"""
}
