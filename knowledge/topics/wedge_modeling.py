WEDGE_KNOWLEDGE = {
    'overview': """A **wedge model** is a fundamental seismic modeling tool that simulates a layer with continuously varying thickness, creating a "wedge" shape.

**Model Structure:**
```
Layer 1: Overburden (constant thickness)
Layer 2: Wedge (0 → max thickness)
Layer 3: Basement (semi-infinite)
```

**Physical Setup:**
- **Geometry**: Triangular wedge layer between two uniform layers
- **Thickness variation**: Linear from 0 to maximum thickness
- **Lateral extent**: Typically 50-100 traces
- **Layer properties**: Each layer has distinct velocity and density

**Key Applications:**
1. **Thin bed analysis**: Study layers below seismic resolution
2. **Tuning studies**: Understand amplitude vs thickness relationships
3. **Resolution limits**: Determine minimum detectable thickness
4. **Interference patterns**: Analyze composite waveforms
5. **Attribute behavior**: Study seismic attributes vs geology

**Observable Phenomena:**
- **Tuning thickness**: Maximum amplitude at ~λ/4
- **Composite reflections**: Single event for thin beds
- **Phase rotation**: Systematic phase changes with thickness
- **Frequency content**: Apparent frequency variations

**Modeling Parameters:**
- **Velocities**: Vp for each layer (m/s or ft/s)
- **Densities**: ρ for each layer (g/cc)
- **Maximum thickness**: Upper limit of wedge
- **Source wavelet**: Usually Ricker with specified frequency

**Analysis Outputs:**
- Synthetic seismic section showing thickness effects
- Amplitude vs thickness curves
- Phase vs thickness relationships
- Frequency content analysis

This model is essential for understanding seismic resolution and thin bed detection capabilities.""",

    'tuning': """**Thickness effects** in wedge models reveal fundamental seismic principles:

**Tuning Phenomena:**
The tuning thickness is where **constructive interference** produces maximum amplitude.

**Mathematical Relationship:**
- **Tuning thickness**: t_tuning ≈ λ/4 = V/(4×f)
- Where: V = layer velocity, f = dominant frequency
- Example: V=2500 m/s, f=30 Hz → t_tuning ≈ 21 m

**Thickness Regimes:**

1. **Thin Bed Regime** (t < λ/8):
   - Single composite reflection
   - Amplitude ∝ thickness
   - Cannot resolve top and bottom separately
   - Phase ≈ constant

2. **Tuning Zone** (λ/8 < t < λ/2):
   - **Maximum amplitude** at t = λ/4
   - **Rapid phase rotation** (90° to 180°)
   - **Amplitude variation** follows sin(2πt/λ) pattern
   - Critical for seismic interpretation

3. **Thick Bed Regime** (t > λ/2):
   - **Separate reflections** from top and bottom
   - **Amplitude decreases** with thickness
   - **Phase approaches** interface values
   - **Time separation** allows direct thickness measurement

**Practical Implications:**
- **Detection limit**: ~λ/8 for amplitude methods
- **Resolution limit**: ~λ/4 for separation methods
- **Optimal imaging**: Thickness near tuning maximum
- **Attribute analysis**: Different regimes show different sensitivities

**Factors Affecting Tuning:**
- **Wavelet frequency**: Higher f → thinner tuning thickness
- **Velocity contrast**: Affects reflection strength
- **Impedance contrast**: Controls reflection polarity
- **Wavelet phase**: Zero-phase vs minimum-phase effects

**Applications:**
- Reservoir thickness mapping
- Pay zone identification
- Seismic resolution studies
- Processing parameter optimization""",

    'parameters': """**Wedge Model Parameters** are crucial for accurate seismic modeling:

**Required Parameters:**

1. **Maximum Thickness**:
   - Controls wedge geometry
   - Should exceed tuning thickness
   - Typical range: 50-500 m
   - Units: meters or feet

2. **Layer Velocities**:
   - **Vp1**: Overburden velocity
   - **Vp2**: Wedge layer velocity
   - **Vp3**: Basement velocity
   - Units: m/s or ft/s
   - Must be positive values

3. **Layer Densities**:
   - **ρ1**: Overburden density
   - **ρ2**: Wedge layer density
   - **ρ3**: Basement density
   - Units: g/cc
   - Must be positive values

**Optional Parameters:**

1. **Wavelet Parameters**:
   - **wv_type**: 'ricker' (default) or 'ormsby'
   - **ricker_freq**: Dominant frequency (Hz)
   - **phase_rot**: Phase rotation in degrees
   - **gain**: Amplitude scaling factor

2. **Output Settings**:
   - **zunit**: 'm' or 'ft' for depth units
   - **fig_fname**: Output figure filename
   - **csv_fname**: Output data filename
   - **plotpadtime**: Time padding for plots

3. **Modeling Options**:
   - **thickness_domain**: 'depth' or 'time'
   - **plotpadtime**: Time padding for plots

**Parameter Selection Guidelines:**

1. **Velocity Selection**:
   - Use realistic values for target geology
   - Consider velocity trends with depth
   - Account for compaction effects
   - Include velocity anisotropy if significant

2. **Density Selection**:
   - Use Gardner's relation if data unavailable
   - Consider lithology variations
   - Account for fluid effects
   - Include pressure effects if significant

3. **Thickness Range**:
   - Start with λ/8 to 2λ range
   - Include tuning thickness
   - Consider target resolution
   - Account for depth of interest

4. **Wavelet Selection**:
   - Match survey characteristics
   - Consider target depth
   - Account for attenuation
   - Include phase effects

**Quality Checks:**
- Verify parameter ranges
- Check unit consistency
- Validate physical relationships
- Test sensitivity to parameters"""
}
