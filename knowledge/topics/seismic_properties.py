SEISMIC_PROPERTIES_KNOWLEDGE = {
    'overview': """**Seismic Properties** are fundamental to understanding wave propagation in the Earth:

**Key Properties:**

1. **Velocity (V)**:
   - P-wave velocity (Vp)
   - S-wave velocity (Vs)
   - Units: m/s or ft/s
   - Controls wave propagation speed

2. **Density (ρ)**:
   - Mass per unit volume
   - Units: g/cc or kg/m³
   - Affects wave amplitude

3. **Impedance (Z)**:
   - Z = ρ × V
   - Controls reflection strength
   - Units: (g/cc) × (m/s)

4. **Quality Factor (Q)**:
   - Measures energy loss
   - Higher Q = less attenuation
   - Affects frequency content

**Typical Values:**
```
Material        Vp (m/s)    Vs (m/s)    ρ (g/cc)    Q
Air            330         0           0.001       -
Water          1500        0           1.0         -
Shale          2000-2500   800-1200    2.2-2.65    20-100
Sandstone      2000-4000   1000-2000   2.2-2.6    50-150
Limestone      3000-6000   1500-3000   2.4-2.8    100-200
Salt           4000-5500   2500-3000   2.1-2.3    1000+
```

**Applications:**
- Seismic interpretation
- Reservoir characterization
- Fluid detection
- Lithology identification
- Pore pressure prediction""",

    'velocity': """**Seismic Velocity** is a fundamental property controlling wave propagation:

**Types of Velocity:**

1. **P-wave Velocity (Vp)**:
   - Primary/compressional waves
   - Fastest seismic wave
   - Travels through all media
   - Most commonly measured

2. **S-wave Velocity (Vs)**:
   - Secondary/shear waves
   - Slower than P-waves
   - Only travels through solids
   - Important for rock strength

**Factors Affecting Velocity:**

1. **Lithology**:
   - Mineral composition
   - Grain size and shape
   - Cementation
   - Porosity

2. **Fluid Content**:
   - Water saturation
   - Oil/gas presence
   - Pore pressure
   - Temperature

3. **Stress State**:
   - Confining pressure
   - Differential stress
   - Fracture presence
   - Anisotropy

**Velocity Trends:**

1. **Depth Trends**:
   - Generally increases with depth
   - Due to compaction
   - Mineral phase changes
   - Pressure effects

2. **Lithology Trends**:
   - Carbonates > Sandstones > Shales
   - Igneous > Metamorphic > Sedimentary
   - Consolidated > Unconsolidated

**Measurement Methods:**

1. **Laboratory**:
   - Ultrasonic measurements
   - Core analysis
   - High-pressure tests
   - Temperature studies

2. **Field Methods**:
   - Well logging
   - Seismic surveys
   - VSP surveys
   - Cross-well tomography

**Applications:**
- Time-to-depth conversion
- Pore pressure prediction
- Fluid detection
- Lithology identification
- Reservoir characterization""",

    'impedance': """**Acoustic Impedance** is a crucial property in seismic analysis:

**Definition:**
- Z = ρ × V
- Product of density and velocity
- Controls reflection strength
- Units: (g/cc) × (m/s)

**Physical Meaning:**
- Measures resistance to wave propagation
- Higher Z = "harder" rock
- Lower Z = "softer" rock
- Controls energy partitioning

**Reflection Coefficient:**
- R = (Z₂ - Z₁)/(Z₂ + Z₁)
- Range: -1 to +1
- Positive R: impedance increase
- Negative R: impedance decrease

**Typical Values:**
```
Material        Z (×10⁶)
Air            0.0003
Water          1.5
Sand (dry)     0.7-2.4
Sand (wet)     3.0-4.4
Shale          4.4-7.0
Sandstone      4.4-10.4
Limestone      7.2-16.8
Salt           9.5-12.7
```

**Applications:**

1. **Seismic Interpretation:**
   - Identify lithology changes
   - Detect fluid contacts
   - Map geological boundaries
   - Characterize reservoirs

2. **Inversion Studies:**
   - Convert seismic to impedance
   - Estimate rock properties
   - Predict lithology
   - Detect hydrocarbons

3. **Forward Modeling:**
   - Generate synthetic seismograms
   - Test interpretation hypotheses
   - Validate processing
   - Design surveys

**Factors Affecting Impedance:**

1. **Lithology:**
   - Mineral composition
   - Grain size
   - Cementation
   - Porosity

2. **Fluids:**
   - Water saturation
   - Oil/gas presence
   - Pore pressure
   - Temperature

3. **Stress:**
   - Confining pressure
   - Differential stress
   - Fracture presence
   - Anisotropy

**Quality Considerations:**
- Accurate velocity measurement
- Precise density determination
- Proper unit conversion
- Consistent sampling"""
}
