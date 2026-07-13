ROCK_PHYSICS_KNOWLEDGE = {
    'overview': """**Rock Physics** is the study of the relationship between physical properties of rocks and their response to seismic waves:

**Key Concepts:**

1. **Elastic Properties**:
   - Bulk modulus (K): Resistance to volume change
   - Shear modulus (μ): Resistance to shape change
   - Young's modulus (E): Stiffness
   - Poisson's ratio (ν): Lateral strain to axial strain ratio

2. **Velocity-Porosity Relationships**:
   - Wyllie time-average equation
   - Raymer-Hunt-Gardner equation
   - Nur's critical porosity model
   - Gassmann fluid substitution

3. **Fluid Effects**:
   - Biot-Gassmann theory
   - Patchy saturation
   - Squirt flow
   - Velocity dispersion

4. **Rock Typing**:
   - Lithology identification
   - Pore geometry classification
   - Reservoir quality assessment
   - Facies analysis

**Applications:**
- Seismic reservoir characterization
- Amplitude variation with offset (AVO) analysis
- Time-lapse (4D) seismic monitoring
- Pore pressure prediction
- Geomechanical modeling""",

    'elastic_properties': """**Elastic Properties** are fundamental to understanding rock behavior under stress:

**Key Elastic Constants:**

1. **Bulk Modulus (K)**:
   - Measures resistance to volume change
   - K = Δp / (ΔV/V)
   - Units: GPa or psi
   - Higher for stiffer materials
   - Sensitive to fluid content

2. **Shear Modulus (μ)**:
   - Measures resistance to shape change
   - μ = shear stress / shear strain
   - Units: GPa or psi
   - Zero for fluids
   - Insensitive to fluid content

3. **Young's Modulus (E)**:
   - Measures stiffness
   - E = stress / strain
   - Units: GPa or psi
   - Higher for more rigid materials
   - Important for geomechanics

4. **Poisson's Ratio (ν)**:
   - Ratio of lateral strain to axial strain
   - Typically 0.1-0.4 for rocks
   - 0.5 for incompressible materials
   - Negative for auxetic materials

**Relationships to Seismic Velocities:**

- Vp = √[(K + 4μ/3) / ρ]
- Vs = √(μ / ρ)
- Vp/Vs = √[(K + 4μ/3) / μ]

**Typical Values:**
```
Rock Type       K (GPa)    μ (GPa)    E (GPa)    ν
Sandstone       10-40      5-20       15-50      0.15-0.30
Shale           5-30       2-15       10-40      0.25-0.40
Limestone       30-70      15-35      40-80      0.20-0.35
Granite         40-60      20-30      50-70      0.20-0.25
```

**Factors Affecting Elastic Properties:**
- Mineralogy
- Porosity
- Pore geometry
- Fluid content
- Pressure
- Temperature
- Anisotropy""",

    'velocity_porosity': """**Velocity-Porosity Relationships** are critical for reservoir characterization:

**Key Models:**

1. **Wyllie Time-Average Equation**:
   - 1/V = φ/Vfluid + (1-φ)/Vmatrix
   - Linear relationship between slowness and porosity
   - Works best for consolidated rocks
   - Assumes uniform pore distribution
   - Limited accuracy in unconsolidated sediments

2. **Raymer-Hunt-Gardner Equation**:
   - V = (1-φ)² × Vmatrix + φ × Vfluid
   - Empirical improvement over Wyllie
   - Better for wide porosity ranges
   - Accounts for rock frame effects
   - More accurate in carbonates

3. **Nur's Critical Porosity Model**:
   - Based on critical porosity concept (φc)
   - The MODULI (not velocity) scale linearly to zero at φc:
     Kdry = Kmin × (1 - φ/φc) and μdry = μmin × (1 - φ/φc), for 0 ≤ φ ≤ φc
   - Velocity then follows from V = √(M/ρ) — it is the modulus that is linear in φ
   - φc typically 0.36-0.40 for sandstones
   - Modified Voigt (upper) bound between the mineral point and φc
   - Good for granular media

4. **Gardner's Relation**:
   - ρ = a × Vᵇ
   - Typically a = 0.31, b = 0.25 (when V in m/s, ρ in g/cm³)
   - Empirical density-velocity relationship
   - Widely used in seismic processing
   - Different coefficients for different lithologies

**Porosity Effects on Velocity:**
- Generally, velocity decreases with increasing porosity
- Rate of decrease depends on pore geometry
- Aspect ratio of pores is critical
- Crack-like pores cause steeper velocity reduction
- Cementation increases velocity at given porosity

**Practical Applications:**
- Porosity estimation from seismic velocities
- Lithology discrimination
- Reservoir quality assessment
- Time-to-depth conversion
- Synthetic seismogram generation""",

    'fluid_effects': """**Fluid Effects** on rock properties are essential for hydrocarbon detection:

**Gassmann Fluid Substitution:**

1. **Theory Basics**:
   - Predicts changes in bulk modulus with fluid changes
   - Assumes isotropic, homogeneous rock
   - Low frequency (static) approximation
   - Shear modulus unaffected by fluids

2. **Key Equation** (low-frequency Gassmann):
   - Ratio form: Ksat/(Kmin − Ksat) = Kdry/(Kmin − Kdry) + Kfl/[φ(Kmin − Kfl)]
   - Forward form: Ksat = Kdry + (1 − Kdry/Kmin)² / [φ/Kfl + (1−φ)/Kmin − Kdry/Kmin²]
   - Shear modulus is unchanged by fluid: μsat = μdry
   - Ksat: saturated-rock (frame + fluid) bulk modulus
   - Kdry: dry-frame (drained) bulk modulus
   - Kmin: mineral/grain bulk modulus
   - Kfl: fluid bulk modulus
   - φ: porosity

3. **Fluid Mixing**:
   - Wood's equation (homogeneous mixing)
   - 1/Kfl = Sw/Kw + Sg/Kg + So/Ko
   - Patchy saturation (heterogeneous mixing)
   - Hill's equation for patchy saturation

**Velocity Changes with Fluid:**

1. **Gas Effect**:
   - Significant Vp decrease (10-30%)
   - Minimal Vs change
   - Increased Vp/Vs ratio
   - "Bright spots" in seismic data

2. **Oil vs Water**:
   - Smaller Vp contrast (2-8%)
   - Negligible Vs difference
   - Density contrast important
   - API gravity affects response

3. **Partial Saturation**:
   - Small gas saturation (5-10%) causes large Vp drop
   - Further gas increase has minimal effect
   - "Gas fizz" vs commercial gas distinction difficult

**Frequency Effects:**

1. **Dispersion**:
   - Velocity increases with frequency
   - Caused by wave-induced fluid flow
   - More pronounced in partially saturated rocks
   - Squirt flow at grain contacts
   - Patchy saturation effects

2. **Attenuation**:
   - Higher in partially saturated rocks
   - Frequency-dependent Q factor
   - Related to fluid mobility
   - Important for seismic resolution""",

    'rock_typing': """**Rock Typing** classifies rocks based on similar properties for reservoir characterization:

**Classification Methods:**

1. **Lithology-Based**:
   - Based on mineral composition
   - Sandstone, shale, limestone, etc.
   - XRD analysis for mineral percentages
   - Thin section petrography
   - Elemental analysis (XRF)

2. **Petrophysical**:
   - Based on porosity-permeability relationships
   - Hydraulic flow units
   - Winland R35 method
   - Flow Zone Indicator (FZI)
   - Lucia classification for carbonates

3. **Elastic Property-Based**:
   - Vp-Vs relationships
   - Impedance crossplots
   - λρ-μρ discrimination
   - Rock physics templates
   - AVO classification

4. **Pore Geometry**:
   - Kozeny-Carman relation
   - Aspect ratio distribution
   - Pore throat size
   - Specific surface area
   - Thomeer hyperbola parameters

**Applications in Reservoir Characterization:**

1. **Seismic Inversion**:
   - Constrains possible solutions
   - Reduces non-uniqueness
   - Guides facies classification
   - Improves property prediction

2. **Reservoir Modeling**:
   - Populates 3D grid cells
   - Preserves petrophysical relationships
   - Maintains geological realism
   - Honors spatial trends

3. **Fluid Substitution**:
   - Different rock types have different fluid responses
   - Calibrates Gassmann equations
   - Improves 4D seismic interpretation
   - Accounts for pressure effects

4. **Production Prediction**:
   - Flow unit identification
   - Sweet spot detection
   - Completion optimization
   - Enhanced recovery planning""",

    'avo_analysis': """**AVO (Amplitude Variation with Offset) Analysis** uses angle-dependent reflectivity for fluid and lithology prediction:

**Theoretical Foundation:**

1. **Zoeppritz Equations**:
   - Full elastic solution for plane wave reflection/transmission
   - Complex, non-intuitive equations
   - Accounts for all wave conversions
   - Computationally intensive

2. **Approximations**:
   - **Aki-Richards**: Three-term approximation
   - **Shuey**: Intercept (A) and gradient (B) formulation
   - **Fatti**: P and S reflectivity separation
   - **Verm & Hilterman**: Density term isolation

**AVO Classes:**

1. **Class I (High Impedance)**:
   - Positive zero-offset reflection
   - Decreasing AVO (negative gradient)
   - Example: Hard sand in soft shale
   - Polarity reversal possible

2. **Class II (Near-Zero)**:
   - Near-zero reflection at normal incidence
   - Can be positive or negative
   - High sensitivity to noise
   - Polarity reversals common

3. **Class III (Low Impedance)**:
   - Negative zero-offset reflection
   - Increasing negative amplitude with offset
   - "Bright spots" in seismic data
   - Classic gas sand response

4. **Class IV (Low Impedance)**:
   - Negative zero-offset reflection
   - Decreasing negative amplitude with offset
   - Uncommon but important
   - Often in thin beds or laminated sequences

**Crossplot Analysis:**

1. **Intercept-Gradient**:
   - A vs B crossplot
   - Background trend identification
   - Fluid anomaly detection
   - Lithology discrimination

2. **Extended Elastic Impedance**:
   - Projection to optimal angle
   - Enhanced fluid/lithology separation
   - Calibration to well data
   - Inversion target

3. **λρ-μρ Analysis**:
   - Lamé parameters × density
   - Fluid sensitivity in λρ
   - Lithology sensitivity in μρ
   - Better discrimination than Vp/Vs

**Practical Considerations:**

1. **Data Requirements**:
   - Preserved amplitude processing
   - Multiple angle/offset stacks
   - Careful NMO correction
   - Robust gather conditioning

2. **Limitations**:
   - Thin bed tuning effects
   - Anisotropy impacts
   - Attenuation distortions
   - Processing artifacts
   - Non-uniqueness of solutions""",

    'geomechanics': """**Geomechanics** links rock physics to stress, strain, and failure behavior:

**Key Concepts:**

1. **Stress and Strain**:
   - Stress tensor components
   - Principal stresses (σ₁, σ₂, σ₃)
   - Effective stress principle
   - Elastic vs plastic deformation
   - Strain hardening/softening

2. **Rock Strength**:
   - Uniaxial compressive strength (UCS)
   - Tensile strength
   - Cohesion and friction angle
   - Failure envelopes (Mohr-Coulomb, Hoek-Brown)
   - Brittleness index

3. **Pore Pressure**:
   - Normal vs abnormal pressure
   - Overpressure mechanisms
   - Terzaghi's effective stress
   - Biot's coefficient
   - Pressure prediction methods

4. **Wellbore Stability**:
   - Stress concentration around wellbore
   - Breakouts and tensile fractures
   - Mud weight window
   - Safe drilling trajectory
   - Collapse and fracture gradients

**Elastic Properties Relationships:**

1. **Dynamic vs Static**:
   - Dynamic: from seismic/sonic measurements
   - Static: from core tests
   - Typically dynamic > static
   - Conversion relationships
   - Frequency dependence

2. **Stress Dependence**:
   - Velocity increases with confining pressure
   - Exponential relationship
   - Crack closure effects
   - Hysteresis during loading/unloading
   - Stress-induced anisotropy

**Applications:**

1. **Drilling**:
   - Wellbore stability analysis
   - Mud weight optimization
   - Casing design
   - Drill bit selection
   - Stuck pipe prevention

2. **Completions**:
   - Hydraulic fracture design
   - Sand production prediction
   - Perforation strategy
   - Frac hit mitigation
   - Completion hardware selection

3. **Reservoir Management**:
   - Compaction and subsidence
   - Fault reactivation risk
   - Cap rock integrity
   - 4D seismic interpretation
   - Enhanced recovery planning""",

    'anisotropy': """**Anisotropy** refers to directional dependence of rock properties:

**Types of Anisotropy:**

1. **Intrinsic**:
   - Mineral alignment
   - Layering/bedding
   - Aligned fractures
   - Depositional fabric
   - Clay particle alignment

2. **Stress-Induced**:
   - Preferential crack closure
   - Non-uniform stress field
   - Differential horizontal stresses
   - Borehole stress concentration

3. **Symmetry Systems**:
   - Isotropic (no directional dependence)
   - VTI (Vertical Transverse Isotropy)
   - HTI (Horizontal Transverse Isotropy)
   - Orthorhombic (three perpendicular planes)
   - Monoclinic and Triclinic (lower symmetry)

**Thomsen Parameters:**

1. **For VTI Media**:
   - ε: P-wave anisotropy
   - γ: S-wave anisotropy
   - δ: Wavefront ellipticity
   - Typical values: 0.1-0.3 for shales
   - Near-zero for sandstones

2. **For HTI Media**:
   - Similar parameters but rotated coordinate system
   - Related to fracture density and orientation
   - Azimuthal variation in properties

**Seismic Manifestations:**

1. **Velocity Variations**:
   - Azimuthal velocity differences
   - Elliptical NMO patterns
   - Non-hyperbolic moveout
   - Apparent VTI in layered media

2. **Shear Wave Splitting**:
   - Fast and slow S-wave polarizations
   - Time delay proportional to anisotropy
   - Diagnostic of fracture orientation
   - Used in fracture characterization

3. **Amplitude Effects**:
   - AVAZ (Amplitude Variation with Azimuth)
   - Elliptical AVO gradients
   - Azimuthal reflectivity variations
   - Fracture density estimation

**Measurement Methods:**

1. **Laboratory**:
   - Ultrasonic velocity vs direction
   - Core plugs in multiple orientations
   - Strain measurements

2. **Borehole**:
   - Cross-dipole sonic
   - Walkaway VSP
   - Stoneley wave analysis
   - Image log fracture analysis

3. **Surface Seismic**:
   - Wide-azimuth surveys
   - Multi-component recording
   - AVAZ analysis
   - Shear wave splitting"""
}