# NeuroMLDocument

**The main NeuroML container class, and other associated types that do not fit into the other categories**

---


Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---

## NeuroMLDocument

`````
```` Schema
``` xml
  <xs:element name="neuroml" type="NeuroMLDocument">
    <xs:annotation>
      <xs:documentation>The root NeuroML element.</xs:documentation>
    </xs:annotation>
  </xs:element>

  <xs:complexType name="NeuroMLDocument">
    <xs:complexContent>
      <xs:extension base="Standalone">
        <xs:sequence>
          <xs:element name="include" type="IncludeType" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="extracellularProperties" type="ExtracellularProperties" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="intracellularProperties" type="IntracellularProperties" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="morphology" type="Morphology" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="ionChannel" type="IonChannel" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="ionChannelHH" type="IonChannelHH" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="ionChannelVShift" type="IonChannelVShift" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="ionChannelKS" type="IonChannelKS" minOccurs="0" maxOccurs="unbounded"/>
          <xs:group ref="ConcentrationModelTypes"/>
          <xs:group ref="SynapseTypes"/>
          <xs:element name="biophysicalProperties" type="BiophysicalProperties" minOccurs="0" maxOccurs="unbounded"/>
          <xs:group ref="CellTypes"/>
          <xs:group ref="InputTypes"/>
          <xs:group ref="PyNNCellTypes"/>
          <xs:group ref="PyNNSynapseTypes"/>
          <xs:group ref="PyNNInputTypes"/>
          <xs:element name="network" type="Network" minOccurs="0" maxOccurs="unbounded"/>
          <xs:element name="ComponentType" type="ComponentType" minOccurs="0" maxOccurs="unbounded"/>
        </xs:sequence>
      </xs:extension>
    </xs:complexContent>
  </xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=NeuroMLDocument" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import NeuroMLDocument
from neuroml.utils import component_factory

variable = component_factory(
    NeuroMLDocument,
    id: 'a NmlId (required)' = None
    metaid: 'a MetaId (optional)' = None
    notes: 'a string (optional)' = None
    properties: 'list of Property(s) (optional)' = None
    annotation: 'a Annotation (optional)' = None
    includes: 'list of IncludeType(s) (optional)' = None
    extracellular_properties: 'list of ExtracellularProperties(s) (optional)' = None
    intracellular_properties: 'list of IntracellularProperties(s) (optional)' = None
    morphology: 'list of Morphology(s) (optional)' = None
    ion_channel: 'list of IonChannel(s) (optional)' = None
    ion_channel_hhs: 'list of IonChannelHH(s) (optional)' = None
    ion_channel_v_shifts: 'list of IonChannelVShift(s) (optional)' = None
    ion_channel_kses: 'list of IonChannelKS(s) (optional)' = None
    decaying_pool_concentration_models: 'list of DecayingPoolConcentrationModel(s) (optional)' = None
    fixed_factor_concentration_models: 'list of FixedFactorConcentrationModel(s) (optional)' = None
    alpha_current_synapses: 'list of AlphaCurrentSynapse(s) (optional)' = None
    alpha_synapses: 'list of AlphaSynapse(s) (optional)' = None
    exp_one_synapses: 'list of ExpOneSynapse(s) (optional)' = None
    exp_two_synapses: 'list of ExpTwoSynapse(s) (optional)' = None
    exp_three_synapses: 'list of ExpThreeSynapse(s) (optional)' = None
    blocking_plastic_synapses: 'list of BlockingPlasticSynapse(s) (optional)' = None
    double_synapses: 'list of DoubleSynapse(s) (optional)' = None
    gap_junctions: 'list of GapJunction(s) (optional)' = None
    silent_synapses: 'list of SilentSynapse(s) (optional)' = None
    linear_graded_synapses: 'list of LinearGradedSynapse(s) (optional)' = None
    graded_synapses: 'list of GradedSynapse(s) (optional)' = None
    biophysical_properties: 'list of BiophysicalProperties(s) (optional)' = None
    cells: 'list of Cell(s) (optional)' = None
    cell2_ca_poolses: 'list of Cell2CaPools(s) (optional)' = None
    base_cells: 'list of BaseCell(s) (optional)' = None
    iaf_tau_cells: 'list of IafTauCell(s) (optional)' = None
    iaf_tau_ref_cells: 'list of IafTauRefCell(s) (optional)' = None
    iaf_cells: 'list of IafCell(s) (optional)' = None
    iaf_ref_cells: 'list of IafRefCell(s) (optional)' = None
    izhikevich_cells: 'list of IzhikevichCell(s) (optional)' = None
    izhikevich2007_cells: 'list of Izhikevich2007Cell(s) (optional)' = None
    ad_ex_ia_f_cells: 'list of AdExIaFCell(s) (optional)' = None
    fitz_hugh_nagumo_cells: 'list of FitzHughNagumoCell(s) (optional)' = None
    fitz_hugh_nagumo1969_cells: 'list of FitzHughNagumo1969Cell(s) (optional)' = None
    pinsky_rinzel_ca3_cells: 'list of PinskyRinzelCA3Cell(s) (optional)' = None
    hindmarshRose1984Cell: 'list of HindmarshRose1984Cell(s) (optional)' = None
    pulse_generators: 'list of PulseGenerator(s) (optional)' = None
    pulse_generator_dls: 'list of PulseGeneratorDL(s) (optional)' = None
    sine_generators: 'list of SineGenerator(s) (optional)' = None
    sine_generator_dls: 'list of SineGeneratorDL(s) (optional)' = None
    ramp_generators: 'list of RampGenerator(s) (optional)' = None
    ramp_generator_dls: 'list of RampGeneratorDL(s) (optional)' = None
    compound_inputs: 'list of CompoundInput(s) (optional)' = None
    compound_input_dls: 'list of CompoundInputDL(s) (optional)' = None
    voltage_clamps: 'list of VoltageClamp(s) (optional)' = None
    voltage_clamp_triples: 'list of VoltageClampTriple(s) (optional)' = None
    spike_arrays: 'list of SpikeArray(s) (optional)' = None
    timed_synaptic_inputs: 'list of TimedSynapticInput(s) (optional)' = None
    spike_generators: 'list of SpikeGenerator(s) (optional)' = None
    spike_generator_randoms: 'list of SpikeGeneratorRandom(s) (optional)' = None
    spike_generator_poissons: 'list of SpikeGeneratorPoisson(s) (optional)' = None
    spike_generator_ref_poissons: 'list of SpikeGeneratorRefPoisson(s) (optional)' = None
    poisson_firing_synapses: 'list of PoissonFiringSynapse(s) (optional)' = None
    transient_poisson_firing_synapses: 'list of TransientPoissonFiringSynapse(s) (optional)' = None
    IF_curr_alpha: 'list of IF_curr_alpha(s) (optional)' = None
    IF_curr_exp: 'list of IF_curr_exp(s) (optional)' = None
    IF_cond_alpha: 'list of IF_cond_alpha(s) (optional)' = None
    IF_cond_exp: 'list of IF_cond_exp(s) (optional)' = None
    EIF_cond_exp_isfa_ista: 'list of EIF_cond_exp_isfa_ista(s) (optional)' = None
    EIF_cond_alpha_isfa_ista: 'list of EIF_cond_alpha_isfa_ista(s) (optional)' = None
    HH_cond_exp: 'list of HH_cond_exp(s) (optional)' = None
    exp_cond_synapses: 'list of ExpCondSynapse(s) (optional)' = None
    alpha_cond_synapses: 'list of AlphaCondSynapse(s) (optional)' = None
    exp_curr_synapses: 'list of ExpCurrSynapse(s) (optional)' = None
    alpha_curr_synapses: 'list of AlphaCurrSynapse(s) (optional)' = None
    SpikeSourcePoisson: 'list of SpikeSourcePoisson(s) (optional)' = None
    networks: 'list of Network(s) (optional)' = None
    ComponentType: 'list of ComponentType(s) (optional)' = None
```
````

```` Usage: XML
``` xml
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"  xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2.3.xsd" id="HL23PYR">
    <include href="HL23PYR.cell.nml"/>
    <pulseGenerator id="pg_HL23PYR" delay="50ms" duration="200ms" amplitude="0.2nA">
        <notes>Simple pulse generator</notes>
    </pulseGenerator>
    <network id="HL23PYRNet" type="networkWithTemperature" temperature="34 degC">
        <population id="HL23PYR_pop" component="HL23PYR" type="populationList">
            <property tag="color" value="0.3220229197377431 0.19279726280626452 0.37635392246534727"/>
            <property tag="region" value="L23"/>
            <instance id="0">
                <location x="0.0" y="0.0" z="0.0"/>
            </instance>
        </population>
        <inputList id="stim_iclamp_HL23PYR" population="HL23PYR_pop" component="pg_HL23PYR">
            <input id="0" target="../HL23PYR_pop/0" destination="synapses"/>
        </inputList>
    </network>
</neuroml>
```
````
`````

## IncludeType


Used to include other documents into each other.

`````
```` Schema
``` xml
  <xs:complexType name="IncludeType">
    <xs:attribute name="href" use="required" type="xs:anyURI"/>
  </xs:complexType>
```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IncludeType" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IncludeType
from neuroml.utils import component_factory

variable = component_factory(
    IncludeType,
    href: 'a anyURI (required)' = None)
```
````

```` Usage: XML
``` xml
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"  xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2.3.xsd" id="HL23PYR">
    <include href="A.cell.nml"/>
    ..
</neuroml>
```
````
`````

# NeuroMLCoreDimensions




Original ComponentType definitions: [NeuroMLCoreDimensions.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//NeuroMLCoreDimensions.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---

## Dimensions

### area

`````


```` Dimensions
L^2 
````

```` Units

- Defined unit: cm2

- Defined unit: m2

- Defined unit: um2

````



`````

### capacitance

`````


```` Dimensions
M^-1 L^-2 T^4 I^2 
````

```` Units

- Defined unit: F

- Defined unit: nF

- Defined unit: pF

- Defined unit: uF

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_capacitance">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(F|uF|nF|pF)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### charge

`````


```` Dimensions
T^1 I^1 
````

```` Units

- Defined unit: C

- Defined unit: e

````



`````

### charge\_per\_mole

`````


```` Dimensions
T^1 I^1 N^-1 
````

```` Units

- Defined unit: C_per_mol

- Defined unit: nA_ms_per_amol

- Defined unit: pC_per_umol

````



`````

### concentration

`````


```` Dimensions
L^-3 N^1 
````

```` Units

- Defined unit: M

- Defined unit: mM

- Defined unit: mol_per_cm3

- Defined unit: mol_per_m3

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_concentration">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(mol_per_m3|mol_per_cm3|M|mM)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### conductance

`````


```` Dimensions
M^-1 L^-2 T^3 I^2 
````

```` Units

- Defined unit: S

- Defined unit: mS

- Defined unit: nS

- Defined unit: pS

- Defined unit: uS

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_conductance">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(S|mS|uS|nS|pS)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### conductanceDensity

`````


```` Dimensions
M^-1 L^-4 T^3 I^2 
````

```` Units

- Defined unit: S_per_cm2

- Defined unit: S_per_m2

- Defined unit: mS_per_cm2

- Defined unit: uS_per_cm2

````



`````

### conductance\_per\_voltage

`````


```` Dimensions
M^-2 L^-4 T^6 I^3 
````

```` Units

- Defined unit: S_per_V

- Defined unit: nS_per_mV

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_conductancePerVoltage">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(S_per_V|nS_per_mV)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### current

`````


```` Dimensions
I^1 
````

```` Units

- Defined unit: A

- Defined unit: nA

- Defined unit: pA

- Defined unit: uA

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_current">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(A|uA|nA|pA)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### currentDensity

`````


```` Dimensions
L^-2 I^1 
````

```` Units

- Defined unit: A_per_m2

- Defined unit: mA_per_cm2

- Defined unit: uA_per_cm2

````



`````

### idealGasConstantDims

`````


```` Dimensions
M^1 L^2 T^-2 K^-1 N^-1 
````

```` Units

- Defined unit: J_per_K_per_mol

- Defined unit: fJ_per_K_per_umol

````



`````

### length

`````


```` Dimensions
L^1 
````

```` Units

- Defined unit: cm

- Defined unit: m__

- Defined unit: um

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_length">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(m|cm|um)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### per\_time

`````


```` Dimensions
T^-1 
````

```` Units

- Defined unit: Hz

- Defined unit: per_hour

- Defined unit: per_min

- Defined unit: per_ms

- Defined unit: per_s

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_pertime">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(per_s|per_ms|Hz)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### per\_voltage

`````


```` Dimensions
M^-1 L^-2 T^3 I^1 
````

```` Units

- Defined unit: per_V

- Defined unit: per_mV

````



`````

### permeability

`````


```` Dimensions
L^1 T^-1 
````

```` Units

- Defined unit: cm_per_ms

- Defined unit: cm_per_s

- Defined unit: m_per_s

- Defined unit: um_per_ms

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_permeability">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(m_per_s|um_per_ms|cm_per_s|cm_per_ms)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### resistance

`````


```` Dimensions
M^1 L^2 T^-3 I^-2 
````

```` Units

- Defined unit: Mohm

- Defined unit: kohm

- Defined unit: ohm

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_resistance">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(ohm|kohm|Mohm)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### resistivity

`````


```` Dimensions
M^2 L^2 T^-3 I^-2 
````

```` Units

- Defined unit: kohm_cm

- Defined unit: ohm_cm

- Defined unit: ohm_m

````


```` Schema
``` xml
<xs:complexType name="Resistivity">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="value" type="Nml2Quantity_resistivity" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

`````

### rho\_factor

`````


```` Dimensions
L^-1 T^-1 I^-1 N^1 
````

```` Units

- Defined unit: mol_per_cm_per_uA_per_ms

- Defined unit: mol_per_m_per_A_per_s

- Defined unit: umol_per_cm_per_nA_per_ms

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_rhoFactor">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(mol_per_m_per_A_per_s|mol_per_cm_per_uA_per_ms)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### specificCapacitance

`````


```` Dimensions
M^-1 L^-4 T^4 I^2 
````

```` Units

- Defined unit: F_per_m2

- Defined unit: uF_per_cm2

````



`````

### substance

`````


```` Dimensions
N^1 
````

```` Units

- Defined unit: mol

````



`````

### temperature

`````


```` Dimensions
K^1 
````

```` Units

- Defined unit: K

- Defined unit: degC

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_temperature">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(degC)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### time

`````


```` Dimensions
T^1 
````

```` Units

- Defined unit: hour

- Defined unit: min

- Defined unit: ms__

- Defined unit: s__

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_time">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(s|ms)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### voltage

`````


```` Dimensions
M^1 L^2 T^-3 I^-1 
````

```` Units

- Defined unit: V

- Defined unit: mV

````


```` Schema
``` xml
<xs:simpleType name="Nml2Quantity_voltage">
  <xs:restriction base="xs:string">
    <xs:pattern value="-?([0-9]*(\.[0-9]+)?)([eE]-?[0-9]+)?[\s]*(V|mV)"/>
  </xs:restriction>
</xs:simpleType>

```
````

`````

### volume

`````


```` Dimensions
L^3 
````

```` Units

- Defined unit: cm3

- Defined unit: litre

- Defined unit: m3

- Defined unit: um3

````



`````



## Units

### A

````

``` Summary
- Dimension: dimensions:current
- Power of 10: 0



```

``` Conversions

- 1 A = 1.00e+09 nA
- 1 A = 1.00e+12 pA
- 1 A = 1.00e+06 uA

```
````

### A_per_m2

````

``` Summary
- Dimension: dimensions:currentDensity
- Power of 10: 0



```

``` Conversions

- 1 A_per_m2 = 0.1 mA_per_cm2
- 1 A_per_m2 = 100 uA_per_cm2

```
````

### C

````

``` Summary
- Dimension: dimensions:charge
- Power of 10: 0



```

``` Conversions

- 1 C = 6.24e+18 e

```
````

### C_per_mol

````

``` Summary
- Dimension: dimensions:charge_per_mole
- Power of 10: 0



```

``` Conversions

- 1 C_per_mol = 1e-06 nA_ms_per_amol
- 1 C_per_mol = 1.00e+06 pC_per_umol

```
````

### F

````

``` Summary
- Dimension: dimensions:capacitance
- Power of 10: 0



```

``` Conversions

- 1 F = 1.00e+09 nF
- 1 F = 1.00e+12 pF
- 1 F = 1.00e+06 uF

```
````

### F_per_m2

````

``` Summary
- Dimension: dimensions:specificCapacitance
- Power of 10: 0



```

``` Conversions

- 1 F_per_m2 = 100 uF_per_cm2

```
````

### Hz

````

``` Summary
- Dimension: dimensions:per_time
- Power of 10: 0



```

``` Conversions

- 1 Hz = 3600 per_hour
- 1 Hz = 60 per_min
- 1 Hz = 0.001 per_ms
- 1 Hz = 1 per_s

```
````

### J_per_K_per_mol

````

``` Summary
- Dimension: dimensions:idealGasConstantDims
- Power of 10: 0



```

``` Conversions

- 1 J_per_K_per_mol = 1.00e+09 fJ_per_K_per_umol

```
````

### K

````

``` Summary
- Dimension: dimensions:temperature
- Power of 10: 0



```

``` Conversions

- 1 K = -272.15 degC

```
````

### M

````

``` Summary
- Dimension: dimensions:concentration
- Power of 10: 3



```

``` Conversions

- 1 M = 1000 mM
- 1 M = 0.001 mol_per_cm3
- 1 M = 1000 mol_per_m3

```
````

### Mohm

````

``` Summary
- Dimension: dimensions:resistance
- Power of 10: 6



```

``` Conversions

- 1 Mohm = 1000 kohm
- 1 Mohm = 1.00e+06 ohm

```
````

### S

````

``` Summary
- Dimension: dimensions:conductance
- Power of 10: 0



```

``` Conversions

- 1 S = 1000 mS
- 1 S = 1.00e+09 nS
- 1 S = 1.00e+12 pS
- 1 S = 1.00e+06 uS

```
````

### S_per_V

````

``` Summary
- Dimension: dimensions:conductance_per_voltage
- Power of 10: 0



```

``` Conversions

- 1 S_per_V = 1.00e+06 nS_per_mV

```
````

### S_per_cm2

````

``` Summary
- Dimension: dimensions:conductanceDensity
- Power of 10: 4



```

``` Conversions

- 1 S_per_cm2 = 10000 S_per_m2
- 1 S_per_cm2 = 1000 mS_per_cm2
- 1 S_per_cm2 = 1.00e+06 uS_per_cm2

```
````

### S_per_m2

````

``` Summary
- Dimension: dimensions:conductanceDensity
- Power of 10: 0



```

``` Conversions

- 1 S_per_m2 = 0.0001 S_per_cm2
- 1 S_per_m2 = 0.1 mS_per_cm2
- 1 S_per_m2 = 100 uS_per_cm2

```
````

### V

````

``` Summary
- Dimension: dimensions:voltage
- Power of 10: 0



```

``` Conversions

- 1 V = 1000 mV

```
````

### cm

````

``` Summary
- Dimension: dimensions:length
- Power of 10: -2



```

``` Conversions

- 1 cm = 0.01 m__
- 1 cm = 10000 um

```
````

### cm2

````

``` Summary
- Dimension: dimensions:area
- Power of 10: -4



```

``` Conversions

- 1 cm2 = 0.0001 m2
- 1 cm2 = 1.00e+08 um2

```
````

### cm3

````

``` Summary
- Dimension: dimensions:volume
- Power of 10: -6



```

``` Conversions

- 1 cm3 = 0.001 litre
- 1 cm3 = 1e-06 m3
- 1 cm3 = 1.00e+12 um3

```
````

### cm_per_ms

````

``` Summary
- Dimension: dimensions:permeability
- Power of 10: 1



```

``` Conversions

- 1 cm_per_ms = 1000 cm_per_s
- 1 cm_per_ms = 10 m_per_s
- 1 cm_per_ms = 10000 um_per_ms

```
````

### cm_per_s

````

``` Summary
- Dimension: dimensions:permeability
- Power of 10: -2



```

``` Conversions

- 1 cm_per_s = 0.001 cm_per_ms
- 1 cm_per_s = 0.01 m_per_s
- 1 cm_per_s = 10 um_per_ms

```
````

### degC

````

``` Summary
- Dimension: dimensions:temperature
- Power of 10: 0
- Offset: 273.15



```

``` Conversions

- 1 degC = 274.15 K

```
````

### e

````

``` Summary
- Dimension: dimensions:charge
- Power of 10: 0


- Scale: 1.602176634e-19


```

``` Conversions

- 1 e = 1.6022e-19 C

```
````

### fJ_per_K_per_umol

````

``` Summary
- Dimension: dimensions:idealGasConstantDims
- Power of 10: -9



```

``` Conversions

- 1 fJ_per_K_per_umol = 1e-09 J_per_K_per_mol

```
````

### hour

````

``` Summary
- Dimension: dimensions:time
- Power of 10: 0


- Scale: 3600.0


```

``` Conversions

- 1 hour = 60 min
- 1 hour = 3.60e+06 ms__
- 1 hour = 3600 s__

```
````

### kohm

````

``` Summary
- Dimension: dimensions:resistance
- Power of 10: 3



```

``` Conversions

- 1 kohm = 0.001 Mohm
- 1 kohm = 1000 ohm

```
````

### kohm_cm

````

``` Summary
- Dimension: dimensions:resistivity
- Power of 10: 1



```

``` Conversions

- 1 kohm_cm = 1000 ohm_cm
- 1 kohm_cm = 10 ohm_m

```
````

### litre

````

``` Summary
- Dimension: dimensions:volume
- Power of 10: -3



```

``` Conversions

- 1 litre = 1000 cm3
- 1 litre = 0.001 m3
- 1 litre = 1.00e+15 um3

```
````

### m

````

``` Summary
- Dimension: dimensions:length
- Power of 10: 0



```

``` Conversions

- 1 m = 100 cm
- 1 m = 1.00e+06 um

```
````

### m2

````

``` Summary
- Dimension: dimensions:area
- Power of 10: 0



```

``` Conversions

- 1 m2 = 10000 cm2
- 1 m2 = 1.00e+12 um2

```
````

### m3

````

``` Summary
- Dimension: dimensions:volume
- Power of 10: 0



```

``` Conversions

- 1 m3 = 1.00e+06 cm3
- 1 m3 = 1000 litre
- 1 m3 = 1.00e+18 um3

```
````

### mA_per_cm2

````

``` Summary
- Dimension: dimensions:currentDensity
- Power of 10: 1



```

``` Conversions

- 1 mA_per_cm2 = 10 A_per_m2
- 1 mA_per_cm2 = 1000 uA_per_cm2

```
````

### mM

````

``` Summary
- Dimension: dimensions:concentration
- Power of 10: 0



```

``` Conversions

- 1 mM = 0.001 M
- 1 mM = 1e-06 mol_per_cm3
- 1 mM = 1 mol_per_m3

```
````

### mS

````

``` Summary
- Dimension: dimensions:conductance
- Power of 10: -3



```

``` Conversions

- 1 mS = 0.001 S
- 1 mS = 1.00e+06 nS
- 1 mS = 1.00e+09 pS
- 1 mS = 1000 uS

```
````

### mS_per_cm2

````

``` Summary
- Dimension: dimensions:conductanceDensity
- Power of 10: 1



```

``` Conversions

- 1 mS_per_cm2 = 0.001 S_per_cm2
- 1 mS_per_cm2 = 10 S_per_m2
- 1 mS_per_cm2 = 1000 uS_per_cm2

```
````

### mV

````

``` Summary
- Dimension: dimensions:voltage
- Power of 10: -3



```

``` Conversions

- 1 mV = 0.001 V

```
````

### m_per_s

````

``` Summary
- Dimension: dimensions:permeability
- Power of 10: 0



```

``` Conversions

- 1 m_per_s = 0.1 cm_per_ms
- 1 m_per_s = 100 cm_per_s
- 1 m_per_s = 1000 um_per_ms

```
````

### min

````

``` Summary
- Dimension: dimensions:time
- Power of 10: 0


- Scale: 60.0


```

``` Conversions

- 1 min = 0.016667 hour
- 1 min = 6.00e+04 ms__
- 1 min = 60 s__

```
````

### mol

````

``` Summary
- Dimension: dimensions:substance
- Power of 10: 0



```
````

### mol_per_cm3

````

``` Summary
- Dimension: dimensions:concentration
- Power of 10: 6



```

``` Conversions

- 1 mol_per_cm3 = 1000 M
- 1 mol_per_cm3 = 1.00e+06 mM
- 1 mol_per_cm3 = 1.00e+06 mol_per_m3

```
````

### mol_per_cm_per_uA_per_ms

````

``` Summary
- Dimension: dimensions:rho_factor
- Power of 10: 11



```

``` Conversions

- 1 mol_per_cm_per_uA_per_ms = 1.00e+11 mol_per_m_per_A_per_s
- 1 mol_per_cm_per_uA_per_ms = 1000 umol_per_cm_per_nA_per_ms

```
````

### mol_per_m3

````

``` Summary
- Dimension: dimensions:concentration
- Power of 10: 0



```

``` Conversions

- 1 mol_per_m3 = 0.001 M
- 1 mol_per_m3 = 1 mM
- 1 mol_per_m3 = 1e-06 mol_per_cm3

```
````

### mol_per_m_per_A_per_s

````

``` Summary
- Dimension: dimensions:rho_factor
- Power of 10: 0



```

``` Conversions

- 1 mol_per_m_per_A_per_s = 1e-11 mol_per_cm_per_uA_per_ms
- 1 mol_per_m_per_A_per_s = 1e-08 umol_per_cm_per_nA_per_ms

```
````

### ms

````

``` Summary
- Dimension: dimensions:time
- Power of 10: -3



```

``` Conversions

- 1 ms = 2.7778e-07 hour
- 1 ms = 1.6667e-05 min
- 1 ms = 0.001 s__

```
````

### nA

````

``` Summary
- Dimension: dimensions:current
- Power of 10: -9



```

``` Conversions

- 1 nA = 1e-09 A
- 1 nA = 1000 pA
- 1 nA = 0.001 uA

```
````

### nA_ms_per_amol

````

``` Summary
- Dimension: dimensions:charge_per_mole
- Power of 10: 6



```

``` Conversions

- 1 nA_ms_per_amol = 1.00e+06 C_per_mol
- 1 nA_ms_per_amol = 1.00e+12 pC_per_umol

```
````

### nF

````

``` Summary
- Dimension: dimensions:capacitance
- Power of 10: -9



```

``` Conversions

- 1 nF = 1e-09 F
- 1 nF = 1000 pF
- 1 nF = 0.001 uF

```
````

### nS

````

``` Summary
- Dimension: dimensions:conductance
- Power of 10: -9



```

``` Conversions

- 1 nS = 1e-09 S
- 1 nS = 1e-06 mS
- 1 nS = 1000 pS
- 1 nS = 0.001 uS

```
````

### nS_per_mV

````

``` Summary
- Dimension: dimensions:conductance_per_voltage
- Power of 10: -6



```

``` Conversions

- 1 nS_per_mV = 1e-06 S_per_V

```
````

### ohm

````

``` Summary
- Dimension: dimensions:resistance
- Power of 10: 0



```

``` Conversions

- 1 ohm = 1e-06 Mohm
- 1 ohm = 0.001 kohm

```
````

### ohm_cm

````

``` Summary
- Dimension: dimensions:resistivity
- Power of 10: -2



```

``` Conversions

- 1 ohm_cm = 0.001 kohm_cm
- 1 ohm_cm = 0.01 ohm_m

```
````

### ohm_m

````

``` Summary
- Dimension: dimensions:resistivity
- Power of 10: 0



```

``` Conversions

- 1 ohm_m = 0.1 kohm_cm
- 1 ohm_m = 100 ohm_cm

```
````

### pA

````

``` Summary
- Dimension: dimensions:current
- Power of 10: -12



```

``` Conversions

- 1 pA = 1e-12 A
- 1 pA = 0.001 nA
- 1 pA = 1e-06 uA

```
````

### pC_per_umol

````

``` Summary
- Dimension: dimensions:charge_per_mole
- Power of 10: -6



```

``` Conversions

- 1 pC_per_umol = 1e-06 C_per_mol
- 1 pC_per_umol = 1e-12 nA_ms_per_amol

```
````

### pF

````

``` Summary
- Dimension: dimensions:capacitance
- Power of 10: -12



```

``` Conversions

- 1 pF = 1e-12 F
- 1 pF = 0.001 nF
- 1 pF = 1e-06 uF

```
````

### pS

````

``` Summary
- Dimension: dimensions:conductance
- Power of 10: -12



```

``` Conversions

- 1 pS = 1e-12 S
- 1 pS = 1e-09 mS
- 1 pS = 0.001 nS
- 1 pS = 1e-06 uS

```
````

### per_V

````

``` Summary
- Dimension: dimensions:per_voltage
- Power of 10: 0



```

``` Conversions

- 1 per_V = 0.001 per_mV

```
````

### per_hour

````

``` Summary
- Dimension: dimensions:per_time
- Power of 10: 0


- Scale: 0.00027777777778


```

``` Conversions

- 1 per_hour = 0.00027778 Hz
- 1 per_hour = 0.016667 per_min
- 1 per_hour = 2.7778e-07 per_ms
- 1 per_hour = 0.00027778 per_s

```
````

### per_mV

````

``` Summary
- Dimension: dimensions:per_voltage
- Power of 10: 3



```

``` Conversions

- 1 per_mV = 1000 per_V

```
````

### per_min

````

``` Summary
- Dimension: dimensions:per_time
- Power of 10: 0


- Scale: 0.01666666667


```

``` Conversions

- 1 per_min = 0.016667 Hz
- 1 per_min = 60 per_hour
- 1 per_min = 1.6667e-05 per_ms
- 1 per_min = 0.016667 per_s

```
````

### per_ms

````

``` Summary
- Dimension: dimensions:per_time
- Power of 10: 3



```

``` Conversions

- 1 per_ms = 1000 Hz
- 1 per_ms = 3.60e+06 per_hour
- 1 per_ms = 6.00e+04 per_min
- 1 per_ms = 1000 per_s

```
````

### per_s

````

``` Summary
- Dimension: dimensions:per_time
- Power of 10: 0



```

``` Conversions

- 1 per_s = 1 Hz
- 1 per_s = 3600 per_hour
- 1 per_s = 60 per_min
- 1 per_s = 0.001 per_ms

```
````

### s

````

``` Summary
- Dimension: dimensions:time
- Power of 10: 0



```

``` Conversions

- 1 s = 0.00027778 hour
- 1 s = 0.016667 min
- 1 s = 1000 ms__

```
````

### uA

````

``` Summary
- Dimension: dimensions:current
- Power of 10: -6



```

``` Conversions

- 1 uA = 1e-06 A
- 1 uA = 1000 nA
- 1 uA = 1.00e+06 pA

```
````

### uA_per_cm2

````

``` Summary
- Dimension: dimensions:currentDensity
- Power of 10: -2



```

``` Conversions

- 1 uA_per_cm2 = 0.01 A_per_m2
- 1 uA_per_cm2 = 0.001 mA_per_cm2

```
````

### uF

````

``` Summary
- Dimension: dimensions:capacitance
- Power of 10: -6



```

``` Conversions

- 1 uF = 1e-06 F
- 1 uF = 1000 nF
- 1 uF = 1.00e+06 pF

```
````

### uF_per_cm2

````

``` Summary
- Dimension: dimensions:specificCapacitance
- Power of 10: -2



```

``` Conversions

- 1 uF_per_cm2 = 0.01 F_per_m2

```
````

### uS

````

``` Summary
- Dimension: dimensions:conductance
- Power of 10: -6



```

``` Conversions

- 1 uS = 1e-06 S
- 1 uS = 0.001 mS
- 1 uS = 1000 nS
- 1 uS = 1.00e+06 pS

```
````

### uS_per_cm2

````

``` Summary
- Dimension: dimensions:conductanceDensity
- Power of 10: -2



```

``` Conversions

- 1 uS_per_cm2 = 1e-06 S_per_cm2
- 1 uS_per_cm2 = 0.01 S_per_m2
- 1 uS_per_cm2 = 0.001 mS_per_cm2

```
````

### um

````

``` Summary
- Dimension: dimensions:length
- Power of 10: -6



```

``` Conversions

- 1 um = 0.0001 cm
- 1 um = 1e-06 m__

```
````

### um2

````

``` Summary
- Dimension: dimensions:area
- Power of 10: -12



```

``` Conversions

- 1 um2 = 1e-08 cm2
- 1 um2 = 1e-12 m2

```
````

### um3

````

``` Summary
- Dimension: dimensions:volume
- Power of 10: -18



```

``` Conversions

- 1 um3 = 1e-12 cm3
- 1 um3 = 1e-15 litre
- 1 um3 = 1e-18 m3

```
````

### um_per_ms

````

``` Summary
- Dimension: dimensions:permeability
- Power of 10: -3



```

``` Conversions

- 1 um_per_ms = 0.0001 cm_per_ms
- 1 um_per_ms = 0.1 cm_per_s
- 1 um_per_ms = 0.001 m_per_s

```
````

### umol_per_cm_per_nA_per_ms

````

``` Summary
- Dimension: dimensions:rho_factor
- Power of 10: 8



```

``` Conversions

- 1 umol_per_cm_per_nA_per_ms = 0.001 mol_per_cm_per_uA_per_ms
- 1 umol_per_cm_per_nA_per_ms = 1.00e+08 mol_per_m_per_A_per_s

```
````



# NeuroMLCoreCompTypes




Original ComponentType definitions: [NeuroMLCoreCompTypes.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//NeuroMLCoreCompTypes.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## notes




Human readable notes/description for a Component.


`````
```` Schema
``` xml
<xs:simpleType name="Notes">
  <xs:annotation>
    <xs:documentation>Textual human readable notes related to the element in question. It's useful to put these into
         the NeuroML files instead of XML comments, as the notes can be extracted and repeated in the files to which the NeuroML is mapped.
            </xs:documentation>
  </xs:annotation>
  <xs:restriction base="xs:string"/>
</xs:simpleType>

```
````


```` Usage: XML
``` xml
<notes>A Simple Spiking cell for testing purposes</notes>
```
``` xml
<notes>Multicompartmental cell</notes>
```
``` xml
<notes>Leak conductance</notes>
```
````
`````


## annotation




A structured annotation containing metadata, specifically RDF or  property elements.


`````
```` Child list
```Table:

**rdf:RDF**$  $ rdf_rdf

```
````

```` Children list
```Table:

**property**$  $ property

```
````

```` Schema
``` xml
<xs:complexType name="Annotation">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:sequence>
        <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Annotation" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Annotation
from neuroml.utils import component_factory

variable = component_factory(
    Annotation,
    anytypeobjs_=None,
)
```
````
```` Usage: XML
``` xml
<annotation>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
        <rdf:Description rdf:about="HippoCA1Cell">
            <bqbiol:is>
                <rdf:Bag>
                    <rdf:li rdf:resource="urn:miriam:neurondb:258"/>
                </rdf:Bag>
            </bqbiol:is>
        </rdf:Description>
    </rdf:RDF>
</annotation>
```
````
`````


## property




A property ( a **tag** and **value** pair ), which can be on any  basestandalone either as a direct child, or within an  annotation. Generally something which helps the visual display or facilitates simulation of a Component, but is not a core physiological property. Common examples include: **numberInternalDivisions,** equivalent of nseg in NEURON; **radius,** for a radius to use in graphical displays for abstract cells ( i.e. without defined morphologies ); **color,** the color to use for a  population or  populationlist of cells; **recommended_dt_ms,** the recommended timestep to use for simulating a  network, **recommended_duration_ms** the recommended duration to use when running a  network.


`````
```` Text fields
```Table:

**tag**$ Name of the property
**value**$ Value of the property

````

```` Schema
``` xml
<xs:complexType name="Property">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="tag" type="xs:string" use="required"/>
      <xs:attribute name="value" type="xs:string" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Property" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Property
from neuroml.utils import component_factory

variable = component_factory(
    Property,
    tag: 'a string (required)' = None,
    value: 'a string (required)' = None,
)
```
````
```` Usage: XML
``` xml
<property tag="numberInternalDivisions" value="9"/>
```
````
`````


## *baseStandalone*




Base type of any Component which can have  notes,  annotation, or a  property list.


`````
```` Child list
```Table:

**notes**$  $ notes
**annotation**$  $ annotation

```
````

```` Children list
```Table:

**property**$  $ property

```
````

```` Schema
``` xml
<xs:complexType name="Standalone">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="property" type="Property" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="annotation" type="Annotation" minOccurs="0"/>
      </xs:sequence>
      <xs:attribute name="metaid" type="MetaId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## rdf_RDF




Structured block in an  annotation based on RDF. See https://github.com/OpenSourceBrain/OSB_API/blob/master/python/examples/grancelllayer.xml.


`````
```` Text fields
```Table:

**xmlns:rdf**$ 

````

```` Child list
```Table:

**rdf:Description**$  $ rdf_description

```
````
`````


## rdf_Description




Structured block in an  annotation based on RDF.


`````
```` Text fields
```Table:

**rdf:about**$ 

````

```` Child list
```Table:

**bqbiol:encodes**$  $ bqbiol_encodes
**bqbiol:hasPart**$  $ bqbiol_haspart
**bqbiol:hasProperty**$  $ bqbiol_hasproperty
**bqbiol:hasVersion**$  $ bqbiol_hasversion
**bqbiol:is**$  $ bqbiol_is
**bqbiol:isDescribedBy**$  $ bqbiol_isdescribedby
**bqbiol:isEncodedBy**$  $ bqbiol_isencodedby
**bqbiol:isHomologTo**$  $ bqbiol_ishomologto
**bqbiol:isPartOf**$  $ bqbiol_ispartof
**bqbiol:isPropertyOf**$  $ bqbiol_ispropertyof
**bqbiol:isVersionOf**$  $ bqbiol_isversionof
**bqbiol:occursIn**$  $ bqbiol_occursin
**bqbiol:hasTaxon**$  $ bqbiol_hastaxon
**bqmodel:is**$  $ bqmodel_is
**bqmodel:isDescribedBy**$  $ bqmodel_isdescribedby
**bqmodel:isDerivedFrom**$  $ bqmodel_isderivedfrom

```
````
`````


## *baseBqbiol*




Structured block in an  annotation based on RDF.


`````
```` Child list
```Table:

**rdf:Bag**$  $ rdf_bag

```
````
`````


## bqbiol_encodes




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_hasPart




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_hasProperty




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_hasVersion




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_is




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_isDescribedBy




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_isEncodedBy




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_isHomologTo




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_isPartOf




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_isPropertyOf




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_isVersionOf




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.


`````
```` Text fields
```Table:

**xmlns:bqbiol**$ 

````
`````


## bqbiol_occursIn




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqbiol_hasTaxon




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqmodel_is




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## bqmodel_isDescribedBy




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.


`````
```` Text fields
```Table:

**xmlns:bqmodel**$ 

````
`````


## bqmodel_isDerivedFrom




extends *basebqbiol*



See http://co.mbine.org/standards/qualifiers.




## rdf_Bag




Structured block in an  annotation based on RDF.


`````
```` Children list
```Table:

**rdf:li**$  $ rdf:li

```
````
`````


## rdf_li




Structured block in an  annotation based on RDF.


`````
```` Text fields
```Table:

**rdf:resource**$ 

````
`````


## point3DWithDiam




Base type for ComponentTypes which specify an ( **x,** **y,** **z** ) coordinate along with a **diameter.** Note: no dimension used in the attributes for these coordinates! These are assumed to have dimension micrometer ( 10^-6 m ). This is due to micrometers being the default option for the majority of neuronal morphology formats, and dimensions are omitted here to facilitate reading and writing of morphologies in NeuroML.


`````
```` Parameters
```Table:

**diameter**$ Diameter of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. $Dimensionless
**x**$ x coordinate of the point. Note: no dimension used, see description of _point3DWithDiam_ for details. $Dimensionless
**y**$ y coordinate of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. $Dimensionless
**z**$ z coordinate of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. $Dimensionless

```
````

```` Constants
```Table:

**MICRON** = 1um$  $ dimensions:length

```
````

```` Derived parameters
```Table:

**radius**$ A dimensional quantity given by half the _diameter. $dimensions:length
```
**radius** = MICRON * diameter / 2
```Table:

**xLength**$ A version of _x with dimension length. $dimensions:length
```
**xLength** = MICRON * x
```Table:

**yLength**$ A version of _y with dimension length. $dimensions:length
```
**yLength** = MICRON * y
```Table:

**zLength**$ A version of _z with dimension length. $dimensions:length
```
**zLength** = MICRON * z

````

```` Schema
``` xml
<xs:complexType name="Point3DWithDiam">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="x" type="xs:double" use="required"/>
      <xs:attribute name="y" type="xs:double" use="required"/>
      <xs:attribute name="z" type="xs:double" use="required"/>
      <xs:attribute name="diameter" type="DoubleGreaterThanZero" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Point3DWithDiam" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Point3DWithDiam
from neuroml.utils import component_factory

variable = component_factory(
    Point3DWithDiam,
    x: 'a double (required)' = None,
    y: 'a double (required)' = None,
    z: 'a double (required)' = None,
    diameter: 'a DoubleGreaterThanZero (required)' = None,
)
```
````
`````

# Cells

**Defines both abstract cell models ( e.g.  izhikevichcell, adaptive exponential integrate and fire cell,  adexiafcell ), point conductance based cell models (  pointcellcondbased,  pointcellcondbasedca ) and cells models (  cell ) which specify the  morphology ( containing  segments ) and  biophysicalproperties separately.**

---


Original ComponentType definitions: [Cells.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//Cells.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## *baseCell*




extends *basestandalone*



Base type of any cell ( e.g. point neuron like  izhikevich2007cell, or a morphologically detailed  cell with  segments ) which can be used in a  population.


`````
```` Schema
``` xml
<xs:complexType name="BaseCell">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseCell
from neuroml.utils import component_factory

variable = component_factory(
    BaseCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    extensiontype_=None,
)
```
````
`````


## *baseSpikingCell*




extends *basecell*



Base type of any cell which can emit **spike** events.


`````
```` Event Ports
```Table:

**spike**$ Spike event$Direction: out

```
````
`````


## *baseCellMembPot*




extends *basespikingcell*



Any spiking cell which has a membrane potential **v** with units of voltage ( as opposed to a dimensionless membrane potential used in  basecellmembpotdl ).


`````
```` Exposures
```Table:

**v**$ Membrane potential $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````
`````


## *baseCellMembPotDL*




extends *basespikingcell*



Any spiking cell which has a dimensioness membrane potential, **V** ( as opposed to a membrane potential units of voltage,  basecellmembpot ).


`````
```` Exposures
```Table:

**V**$ Membrane potential $Dimensionless

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````
`````


## *baseChannelPopulation*




extends *basevoltagedeppointcurrent*



Base type for any current produced by a population of channels, all of which are of type **ionChannel**.


`````
```` Component References
```Table:

**ionChannel**$  $ baseionchannel

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````
`````


## channelPopulation




extends *basechannelpopulation*



Population of a **number** of ohmic ion channels. These each produce a conductance **channelg** across a reversal potential **erev,** giving a total current **i.** Note that active membrane currents are more frequently specified as a density over an area of the  cell using  channeldensity.


`````
```` Parameters
```Table:

**erev**$ The reversal potential of the current produced $dimensions:voltage
**number**$ The number of channels present. This will be multiplied by the time varying conductance of the individual ion channel (which extends _baseIonChannel_) to produce the total conductance $Dimensionless

```
````

```` Constants
```Table:

**vShift** = 0mV$ Set to a constant 0mV here to allow ion channels which use _vShift in their rate variable expressions to be used with _channelPopulation_, not just with _channelDensityVShift_ (where _vShift would be explicitly set) $ dimensions:voltage

```
````

```` Text fields
```Table:

**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **channelg** =&nbsp;ionChannel->g
    : **geff** =&nbsp;channelg * number
    : **i** =&nbsp;geff * (erev - v)(exposed as **i**)
    





````

```` Schema
``` xml
<xs:complexType name="ChannelPopulation">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="variableParameter" type="VariableParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="number" type="NonNegativeInteger" use="required"/>
      <xs:attribute name="erev" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
      <xs:attribute name="segment" type="NonNegativeInteger" use="optional"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelPopulation" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelPopulation
from neuroml.utils import component_factory

variable = component_factory(
    ChannelPopulation,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    number: 'a NonNegativeInteger (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<channelPopulation id="naChansDend" ionChannel="NaConductance" segment="2" number="120000" erev="50mV" ion="na"/>
```
````
`````


## channelPopulationNernst




extends *basechannelpopulation*



Population of a **number** of channels with a time varying reversal potential **erev** determined by Nernst equation. Note: hard coded for Ca only!


`````
```` Parameters
```Table:

**number**$ The number of channels present. This will be multiplied by the time varying conductance of the individual ion channel (which extends _baseIonChannel_) to produce the total conductance $Dimensionless

```
````

```` Constants
```Table:

**R** = 8.3144621 J_per_K_per_mol$ $ dimensions:idealGasConstantDims
**zCa** = 2$ $ Dimensionless
**F** = 96485.3 C_per_mol$ $ dimensions:charge_per_mole
**vShift** = 0mV$ Set to a constant 0mV here to allow ion channels which use _vShift in their rate variable expressions to be used with _channelPopulation_, not just with _channelDensityVShift_ (where _vShift would be explicitly set) $ dimensions:voltage

```
````

```` Text fields
```Table:

**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**erev**$ The reversal potential of the current produced, calculated from _caConcExt and _caConc $dimensions:voltage
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**caConc**$ The internal Ca2+ concentration, as calculated/exposed by the parent Component $dimensions:concentration
**caConcExt**$ The external Ca2+ concentration, as calculated/exposed by the parent Component $dimensions:concentration
**temperature**$ The temperature to use in the calculation of _erev. Note this is generally exposed by a _networkWithTemperature_. $dimensions:temperature
**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **singleChannelConductance** =&nbsp;ionChannel->g
    : **totalConductance** =&nbsp;singleChannelConductance * number
    : **erev** =&nbsp;(R * temperature / (zCa * F)) * log(caConcExt / caConc)(exposed as **erev**)
    : **i** =&nbsp;totalConductance * (erev - v)(exposed as **i**)
    





````
`````


## *baseChannelDensity*




Base type for a current of density **iDensity** distributed on an area of a  cell, flowing through the specified **ionChannel.** Instances of this ( normally  channeldensity ) are specified in the  membraneproperties of the  cell.


`````
```` Component References
```Table:

**ionChannel**$  $ baseionchannel

```
````

```` Exposures
```Table:

**iDensity**$  $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````
`````


## *baseChannelDensityCond*




extends *basechanneldensity*



Base type for distributed conductances on an area of a cell producing a ( not necessarily ohmic ) current.


`````
```` Parameters
```Table:

**condDensity**$  $dimensions:conductanceDensity

```
````

```` Exposures
```Table:

**gDensity**$  $dimensions:conductanceDensity
**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````
`````


## variableParameter




Specifies a **parameter** ( e.g. condDensity ) which can vary its value across a **segmentGroup.** The value is calculated from **value** attribute of the  inhomogeneousvalue subelement. This element is normally a child of  channeldensitynonuniform,  channeldensitynonuniformnernst or  channeldensitynonuniformghk and is used to calculate the value of the conductance, etc. which will vary on different parts of the cell. The **segmentGroup** specified here needs to define an  inhomogeneousparameter ( referenced from **inhomogeneousParameter** in the  inhomogeneousvalue ), which calculates a **variable** ( e.g. p ) varying across the cell ( e.g. based on the path length from soma ), which is then used in the **value** attribute of the  inhomogeneousvalue ( so for example condDensity = f( p ) ).


`````
```` Text fields
```Table:

**parameter**$ 
**segmentGroup**$ 

````

```` Child list
```Table:

**inhomogeneousValue**$  $ inhomogeneousvalue

```
````

```` Schema
``` xml
<xs:complexType name="VariableParameter">
  <xs:sequence>
    <xs:element name="inhomogeneousValue" type="InhomogeneousValue" minOccurs="0"/>
  </xs:sequence>
  <xs:attribute name="parameter" type="xs:string" use="required"/>
  <xs:attribute name="segmentGroup" type="xs:string" use="required"/>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=VariableParameter" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import VariableParameter
from neuroml.utils import component_factory

variable = component_factory(
    VariableParameter,
    parameter: 'a string (required)' = None,
    segment_groups: 'a string (required)' = None,
    inhomogeneous_value: 'a InhomogeneousValue (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<variableParameter parameter="condDensity" segmentGroup="dendrite_group">
    <inhomogeneousValue inhomogeneousParameter="dendrite_group_x1" value="5e-7 * exp(-p/200)"/>
</variableParameter>
```
````
`````


## inhomogeneousValue




Specifies the **value** of an **inhomogeneousParameter.** For usage see  variableparameter.


`````
```` Text fields
```Table:

**inhomogeneousParameter**$ 
**value**$ 

````

```` Schema
``` xml
<xs:complexType name="InhomogeneousValue">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="inhomogeneousParameter" type="xs:string" use="required"/>
      <xs:attribute name="value" type="xs:string" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=InhomogeneousValue" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import InhomogeneousValue
from neuroml.utils import component_factory

variable = component_factory(
    InhomogeneousValue,
    inhomogeneous_parameters: 'a string (required)' = None,
    value: 'a string (required)' = None,
)
```
````
```` Usage: XML
``` xml
<inhomogeneousValue inhomogeneousParameter="dendrite_group_x1" value="5e-7 * exp(-p/200)"/>
```
````
`````


## channelDensityNonUniform




extends *basechanneldensity*



Specifies a time varying ohmic conductance density, which is distributed on a region of the **cell.** The conductance density of the channel is not uniform, but is set using the  variableparameter. Note, there is no dynamical description of this in LEMS yet, as this type only makes sense for multicompartmental cells. A ComponentType for this needs to be present to enable export of NeuroML 2 multicompartmental cells via LEMS/jNeuroML to NEURON.


`````
```` Parameters
```Table:

**erev**$ The reversal potential of the current produced $dimensions:voltage

```
````

```` Constants
```Table:

**ZERO_CURR_DENS** = 0 A_per_m2$  $ dimensions:currentDensity

```
````

```` Text fields
```Table:

**segmentGroup**$ 
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Child list
```Table:

**variableParameter**$  $ variableparameter

```
````

```` Exposures
```Table:

**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **iDensity** =&nbsp;ZERO_CURR_DENS(exposed as **iDensity**)
    





````

```` Schema
``` xml
<xs:complexType name="ChannelDensityNonUniform">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="variableParameter" type="VariableParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="erev" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityNonUniform" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityNonUniform
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityNonUniform,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<channelDensityNonUniform id="nonuniform_na_chans" ionChannel="NaConductance" erev="50mV" ion="na">
    <variableParameter parameter="condDensity" segmentGroup="dendrite_group">
        <inhomogeneousValue inhomogeneousParameter="dendrite_group_x1" value="5e-7 * exp(-p/200)"/>
    </variableParameter>
</channelDensityNonUniform>
```
````
`````


## channelDensityNonUniformNernst




extends *basechanneldensity*



Specifies a time varying conductance density, which is distributed on a region of the **cell,** and whose reversal potential is calculated from the Nernst equation. Hard coded for Ca only!. The conductance density of the channel is not uniform, but is set using the  variableparameter. Note, there is no dynamical description of this in LEMS yet, as this type only makes sense for multicompartmental cells. A ComponentType for this needs to be present to enable export of NeuroML 2 multicompartmental cells via LEMS/jNeuroML to NEURON.


`````
```` Constants
```Table:

**ZERO_CURR_DENS** = 0 A_per_m2$  $ dimensions:currentDensity

```
````

```` Text fields
```Table:

**segmentGroup**$ 
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Child list
```Table:

**variableParameter**$  $ variableparameter

```
````

```` Exposures
```Table:

**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **iDensity** =&nbsp;ZERO_CURR_DENS(exposed as **iDensity**)
    





````

```` Schema
``` xml
<xs:complexType name="ChannelDensityNonUniformNernst">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="variableParameter" type="VariableParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityNonUniformNernst" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityNonUniformNernst
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityNonUniformNernst,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
)
```
````
`````


## channelDensityNonUniformGHK




extends *basechanneldensity*



Specifies a time varying conductance density, which is distributed on a region of the **cell,** and whose current is calculated from the Goldman-Hodgkin-Katz equation. Hard coded for Ca only!. The conductance density of the channel is not uniform, but is set using the  variableparameter. Note, there is no dynamical description of this in LEMS yet, as this type only makes sense for multicompartmental cells. A ComponentType for this needs to be present to enable export of NeuroML 2 multicompartmental cells via LEMS/jNeuroML to NEURON.


`````
```` Constants
```Table:

**ZERO_CURR_DENS** = 0 A_per_m2$  $ dimensions:currentDensity

```
````

```` Text fields
```Table:

**segmentGroup**$ 
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Child list
```Table:

**variableParameter**$  $ variableparameter

```
````

```` Exposures
```Table:

**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **iDensity** =&nbsp;ZERO_CURR_DENS(exposed as **iDensity**)
    





````

```` Schema
``` xml
<xs:complexType name="ChannelDensityNonUniformGHK">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="variableParameter" type="VariableParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityNonUniformGHK" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityNonUniformGHK
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityNonUniformGHK,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
)
```
````
`````


## channelDensity




extends *basechanneldensitycond*



Specifies a time varying ohmic conductance density, **gDensity,** which is distributed on an area of the **cell** ( specified in  membraneproperties ) with fixed reversal potential **erev** producing a current density **iDensity**.


`````
```` Parameters
```Table:

**condDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**erev**$ The reversal potential of the current produced $dimensions:voltage

```
````

```` Constants
```Table:

**vShift** = 0mV$ Set to a constant 0mV here to allow ion channels which use _vShift in their rate variable expressions to be used with _channelDensity_, not just with _channelDensityVShift_ (where _vShift would be explicitly set) $ dimensions:voltage

```
````

```` Text fields
```Table:

**segmentGroup**$ Which _segmentGroup_ the channelDensity is placed on. If this is missing, it implies it is placed on all _segment_s of the _cell_
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**gDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **channelf** =&nbsp;ionChannel->fopen
    : **gDensity** =&nbsp;condDensity * channelf(exposed as **gDensity**)
    : **iDensity** =&nbsp;gDensity * (erev - v)(exposed as **iDensity**)
    





````

```` Schema
``` xml
<xs:complexType name="ChannelDensity">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="variableParameter" type="VariableParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="condDensity" type="Nml2Quantity_conductanceDensity" use="optional"/>
      <xs:attribute name="erev" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
      <xs:attribute name="segment" type="NonNegativeInteger" use="optional"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensity" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensity
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensity,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    cond_density: 'a Nml2Quantity_conductanceDensity (optional)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific"/>
```
``` xml
<channelDensity id="naChansSoma" ionChannel="NaConductance" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" erev="50mV" ion="na"/>
```
``` xml
<channelDensity id="naChans" ionChannel="HH_Na" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" ion="na" erev="50mV"/>
```
````
`````


## channelDensityVShift




extends channeldensity



Same as  channeldensity, but with a **vShift** parameter to change voltage activation of gates. The exact usage of **vShift** in expressions for rates is determined by the individual gates.


`````
```` Parameters
```Table:

**condDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**erev**$ The reversal potential of the current produced *(from channeldensity)* $dimensions:voltage
**vShift**$  $dimensions:voltage

```
````

```` Text fields
```Table:

**segmentGroup**$ Which _segmentGroup_ the channelDensity is placed on. If this is missing, it implies it is placed on all _segment_s of the _cell_
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**gDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Schema
``` xml
<xs:complexType name="ChannelDensityVShift">
  <xs:complexContent>
    <xs:extension base="ChannelDensity">
      <xs:attribute name="vShift" type="Nml2Quantity_voltage" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityVShift" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityVShift
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityVShift,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    cond_density: 'a Nml2Quantity_conductanceDensity (optional)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
    v_shift: 'a Nml2Quantity_voltage (required)' = None,
)
```
````
`````


## channelDensityNernst




extends *basechanneldensitycond*



Specifies a time varying conductance density, **gDensity,** which is distributed on an area of the **cell,** producing a current density **iDensity** and whose reversal potential is calculated from the Nernst equation. Hard coded for Ca only! See https://github.com/OpenSourceBrain/ghk-nernst.


`````
```` Parameters
```Table:

**condDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity

```
````

```` Constants
```Table:

**R** = 8.3144621 J_per_K_per_mol$ $ dimensions:idealGasConstantDims
**zCa** = 2$ $ Dimensionless
**F** = 96485.3 C_per_mol$ $ dimensions:charge_per_mole

```
````

```` Text fields
```Table:

**segmentGroup**$ Which _segmentGroup_ the channelDensityNernst is placed on. If this is missing, it implies it is placed on all _segment_s of the _cell_
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**erev**$ The reversal potential of the current produced, calculated from caConcExt and caConc $dimensions:voltage
**gDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**caConcExt**$  $dimensions:concentration
**temperature**$  $dimensions:temperature
**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **channelf** =&nbsp;ionChannel->fopen
    



**Conditional Derived Variables**
    
: IF caConcExt &gt; 0 THEN
:  **gDensity** = condDensity \* channelf (exposed as **gDensity**)
: IF caConcExt &lt;= 0 THEN
:  **gDensity** = 0 (exposed as **gDensity**)
: IF caConcExt &gt; 0 THEN
:  **erev** = (R \* temperature / (zCa \* F)) \* log(caConcExt / caConc) (exposed as **erev**)
: IF caConcExt &lt;= 0 THEN
:  **erev** = 0 (exposed as **erev**)
: IF caConcExt &gt; 0 THEN
:  **iDensity** = gDensity \* (erev - v) (exposed as **iDensity**)
: IF caConcExt &lt;= 0 THEN
:  **iDensity** = 0 (exposed as **iDensity**)


````

```` Schema
``` xml
<xs:complexType name="ChannelDensityNernst">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="variableParameter" type="VariableParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="condDensity" type="Nml2Quantity_conductanceDensity" use="optional"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
      <xs:attribute name="segment" type="NmlId" use="optional"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityNernst" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityNernst
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityNernst,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    cond_density: 'a Nml2Quantity_conductanceDensity (optional)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
    extensiontype_=None,
)
```
````
`````


## channelDensityNernstCa2




extends *basechanneldensitycond*



This component is similar to the original component type  channeldensitynernst but it is changed in order to have a reversal potential that depends on a second independent Ca++ pool ( ca2 ). See https://github.com/OpenSourceBrain/ghk-nernst.


`````
```` Parameters
```Table:

**condDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity

```
````

```` Constants
```Table:

**R** = 8.3144621 J_per_K_per_mol$ $ dimensions:idealGasConstantDims
**zCa** = 2$ $ Dimensionless
**F** = 96485.3 C_per_mol$ $ dimensions:charge_per_mole

```
````

```` Text fields
```Table:

**segmentGroup**$ Which _segmentGroup_ the channelDensityNernstCa2 is placed on. If this is missing, it implies it is placed on all _segment_s of the _cell_
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**erev**$ The reversal potential of the current produced $dimensions:voltage
**gDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**caConc2**$  $dimensions:concentration
**caConcExt2**$  $dimensions:concentration
**temperature**$  $dimensions:temperature
**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **channelf** =&nbsp;ionChannel->fopen
    



**Conditional Derived Variables**
    
: IF caConcExt2 &gt; 0 THEN
:  **gDensity** = condDensity \* channelf (exposed as **gDensity**)
: IF caConcExt2 &lt;= 0 THEN
:  **gDensity** = 0 (exposed as **gDensity**)
: IF caConcExt2 &gt; 0 THEN
:  **erev** = (R \* temperature / (zCa \* F)) \* log(caConcExt2 / caConc2) (exposed as **erev**)
: IF caConcExt2 &lt;= 0 THEN
:  **erev** = 0 (exposed as **erev**)
: IF caConcExt2 &gt; 0 THEN
:  **iDensity** = gDensity \* (erev - v) (exposed as **iDensity**)
: IF caConcExt2 &lt;= 0 THEN
:  **iDensity** = 0 (exposed as **iDensity**)


````

```` Schema
``` xml
<xs:complexType name="ChannelDensityNernstCa2">
  <xs:complexContent>
    <xs:extension base="ChannelDensityNernst">
      </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityNernstCa2" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityNernstCa2
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityNernstCa2,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    cond_density: 'a Nml2Quantity_conductanceDensity (optional)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
    variable_parameters: 'list of VariableParameter(s) (optional)' = None,
)
```
````
`````


## channelDensityGHK




extends *basechanneldensity*



Specifies a time varying conductance density, **gDensity,** which is distributed on an area of the cell, producing a current density **iDensity** and whose reversal potential is calculated from the Goldman Hodgkin Katz equation. Hard coded for Ca only! See https://github.com/OpenSourceBrain/ghk-nernst.


`````
```` Parameters
```Table:

**permeability**$  $dimensions:permeability

```
````

```` Constants
```Table:

**R** = 8.3144621 J_per_K_per_mol$ $ dimensions:idealGasConstantDims
**zCa** = 2$ $ Dimensionless
**F** = 96485.3 C_per_mol$ $ dimensions:charge_per_mole

```
````

```` Text fields
```Table:

**segmentGroup**$ Which _segmentGroup_ the channelDensityGHK is placed on. If this is missing, it implies it is placed on all _segment_s of the _cell_
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**caConcExt**$  $dimensions:concentration
**temperature**$  $dimensions:temperature
**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **K** =&nbsp;(zCa * F) / (R * temperature)
    : **expKv** =&nbsp;exp(-1 * K * v)
    : **channelf** =&nbsp;ionChannel->fopen
    



**Conditional Derived Variables**
    
: IF caConcExt &gt; 0 THEN
:  **iDensity** = -1 \* channelf \* permeability \* zCa \* F \* K \* v \* ( caConc - (caConcExt \* expKv) ) / (1 - expKv) (exposed as **iDensity**)
: IF caConcExt &lt;= 0 THEN
:  **iDensity** = 0 (exposed as **iDensity**)


````

```` Schema
``` xml
<xs:complexType name="ChannelDensityGHK">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="permeability" type="Nml2Quantity_permeability" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
      <xs:attribute name="segment" type="NmlId" use="optional"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityGHK" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityGHK
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityGHK,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    permeability: 'a Nml2Quantity_permeability (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
)
```
````
`````


## channelDensityGHK2




extends *basechanneldensitycond*



Time varying conductance density, **gDensity,** which is distributed on an area of the cell, producing a current density **iDensity.** Modified version of Jaffe et al. 1994 ( used also in Lawrence et al. 2006 ). See https://github.com/OpenSourceBrain/ghk-nernst.


`````
```` Parameters
```Table:

**condDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity

```
````

```` Constants
```Table:

**VOLT_SCALE** = 1 mV$  $ dimensions:voltage
**CONC_SCALE** = 1 mM$  $ dimensions:concentration
**TEMP_SCALE** = 1 K$  $ dimensions:temperature

```
````

```` Text fields
```Table:

**segmentGroup**$ Which _segmentGroup_ the channelDensityGHK2 is placed on. If this is missing, it implies it is placed on all _segment_s of the _cell_
**ion**$ Which ion flows through the channel. Note: ideally this needs to be a property of ionChannel only, but it's here as it makes it easier to select channelPopulations transmitting specific ions.

````

```` Exposures
```Table:

**gDensity**$  *(from basechanneldensitycond)* $dimensions:conductanceDensity
**iDensity**$  *(from basechanneldensity)* $dimensions:currentDensity

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**caConcExt**$  $dimensions:concentration
**temperature**$  $dimensions:temperature
**v**$  *(from basechanneldensity)* $dimensions:voltage

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **ionChannel**









**Derived Variables**
    : **V** =&nbsp;v / VOLT_SCALE
    : **ca_conc_i** =&nbsp;caConc / CONC_SCALE
    : **ca_conc_ext** =&nbsp;caConcExt / CONC_SCALE
    : **T** =&nbsp;temperature / TEMP_SCALE
    : **channelf** =&nbsp;ionChannel->fopen
    : **gDensity** =&nbsp;condDensity * channelf(exposed as **gDensity**)
    : **tmp** =&nbsp;(25 * T) / (293.15 * 2)
    



**Conditional Derived Variables**
    
: IF V/tmp = 0. THEN
:  **pOpen** = tmp \* 1e-3 \* (1 - ((ca_conc_i/ca_conc_ext) \* exp(V/tmp))) \* (1 - (V/tmp)/2) 
: IF V/tmp != 0. THEN
:  **pOpen** = tmp \* 1e-3 \* (1 - ((ca_conc_i/ca_conc_ext) \* exp(V/tmp))) \* ((V/tmp) / (exp(V/tmp) - 1)) 
: IF ca_conc_ext &gt; 0 THEN
:  **iDensity** = gDensity \* pOpen (exposed as **iDensity**)
: IF ca_conc_ext &lt;= 0 THEN
:  **iDensity** = 0 (exposed as **iDensity**)


````

```` Schema
``` xml
<xs:complexType name="ChannelDensityGHK2">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:attribute name="ionChannel" type="NmlId" use="required"/>
      <xs:attribute name="condDensity" type="Nml2Quantity_conductanceDensity" use="optional"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
      <xs:attribute name="segment" type="NmlId" use="optional"/>
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ChannelDensityGHK2" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ChannelDensityGHK2
from neuroml.utils import component_factory

variable = component_factory(
    ChannelDensityGHK2,
    id: 'a NmlId (required)' = None,
    ion_channel: 'a NmlId (required)' = None,
    cond_density: 'a Nml2Quantity_conductanceDensity (optional)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
    segments: 'a NonNegativeInteger (optional)' = None,
    ion: 'a NmlId (required)' = None,
)
```
````
`````


## pointCellCondBased




extends *basecellmembpotcap*



Simple model of a conductance based cell, with no separate  morphology element, just an absolute capacitance **C,** and a set of channel **populations.** Note: use of  cell is generally preferable ( and more widely supported ), even for a single compartment cell.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**thresh**$ The voltage threshold above which the cell is considered to be _spiking $dimensions:voltage
**v0**$ The initial membrane potential of the cell $dimensions:voltage

```
````

```` Children list
```Table:

**populations**$  $ basechannelpopulation

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **spiking**: Dimensionless 









**On Start**
: **v** = v0
: **spiking** = 0



**On Conditions**

: IF v &gt; thresh AND spiking &lt; 0.5 THEN
: **spiking** = 1
: EVENT OUT on port: **spike**

: IF v &lt; thresh THEN
: **spiking** = 0





**Derived Variables**
    : **iChannels** =&nbsp;populations[*]->i(reduce method: add)
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iMemb** =&nbsp;iChannels + iSyn(exposed as **iMemb**)
    





**Time Derivatives**
    : d **v** /dt = iMemb / C
    

````
`````


## pointCellCondBasedCa




extends *basecellmembpotcap*



TEMPORARY: Point cell with conductances and Ca concentration info. Not yet fully tested!!!


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**thresh**$ The voltage threshold above which the cell is considered to be _spiking $dimensions:voltage
**v0**$ The initial membrane potential of the cell $dimensions:voltage

```
````

```` Children list
```Table:

**populations**$  $ basechannelpopulation
**concentrationModels**$  $ concentrationmodel

```
````

```` Exposures
```Table:

**caConc**$  $dimensions:concentration
**iCa**$  $dimensions:current
**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **spiking**: Dimensionless 









**On Start**
: **v** = v0
: **spiking** = 0



**On Conditions**

: IF v &gt; thresh AND spiking &lt; 0.5 THEN
: **spiking** = 1
: EVENT OUT on port: **spike**

: IF v &lt; thresh THEN
: **spiking** = 0





**Derived Variables**
    : **iChannels** =&nbsp;populations[*]->i(reduce method: add)
    : **iCa** =&nbsp;populations[ion='ca']->i(reduce method: add)(exposed as **iCa**)
    : **caConc** =&nbsp;concentrationModels[species='ca']->concentration(reduce method: add)(exposed as **caConc**)
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iMemb** =&nbsp;iChannels + iSyn(exposed as **iMemb**)
    





**Time Derivatives**
    : d **v** /dt = iMemb / C
    

````
`````


## distal




extends point3dwithdiam



Point on a  segment furthest from the soma. Should always be present in the description of a  segment, unlike  proximal.


`````
```` Parameters
```Table:

**diameter**$ Diameter of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless
**x**$ x coordinate of the point. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless
**y**$ y coordinate of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless
**z**$ z coordinate of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless

```
````

```` Derived parameters
```Table:

**radius**$ A dimensional quantity given by half the _diameter. *(from point3dwithdiam)* $dimensions:length
```
**radius** = MICRON * diameter / 2
```Table:

**xLength**$ A version of _x with dimension length. *(from point3dwithdiam)* $dimensions:length
```
**xLength** = MICRON * x
```Table:

**yLength**$ A version of _y with dimension length. *(from point3dwithdiam)* $dimensions:length
```
**yLength** = MICRON * y
```Table:

**zLength**$ A version of _z with dimension length. *(from point3dwithdiam)* $dimensions:length
```
**zLength** = MICRON * z

````


```` Usage: XML
``` xml
<distal x="10" y="0" z="0" diameter="10"/>
```
``` xml
<distal x="20" y="0" z="0" diameter="3"/>
```
``` xml
<distal x="30" y="0" z="0" diameter="1"/>
```
````
`````


## proximal




extends point3dwithdiam



Point on a  segment closest to the soma. Note, the proximal point can be omitted, and in this case is defined as being the point **fractionAlong** between the proximal and  distal point of the  parent, i.e. if **fractionAlong** = 1 ( as it is by default ) it will be the **distal** on the parent, or if **fractionAlong** = 0, it will be the proximal point. If between 0 and 1, it is the linear interpolation between the two points.


`````
```` Parameters
```Table:

**diameter**$ Diameter of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless
**x**$ x coordinate of the point. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless
**y**$ y coordinate of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless
**z**$ z coordinate of the ppoint. Note: no dimension used, see description of _point3DWithDiam_ for details. *(from point3dwithdiam)* $Dimensionless

```
````

```` Derived parameters
```Table:

**radius**$ A dimensional quantity given by half the _diameter. *(from point3dwithdiam)* $dimensions:length
```
**radius** = MICRON * diameter / 2
```Table:

**xLength**$ A version of _x with dimension length. *(from point3dwithdiam)* $dimensions:length
```
**xLength** = MICRON * x
```Table:

**yLength**$ A version of _y with dimension length. *(from point3dwithdiam)* $dimensions:length
```
**yLength** = MICRON * y
```Table:

**zLength**$ A version of _z with dimension length. *(from point3dwithdiam)* $dimensions:length
```
**zLength** = MICRON * z

````


```` Usage: XML
``` xml
<proximal x="0" y="0" z="0" diameter="10"/>
```
``` xml
<proximal x="25" y="0" z="0" diameter="0.1"/>
```
``` xml
<proximal x="0" y="0" z="0" diameter="10"/>
```
````
`````


## parent




Specifies the  segment which is this segment's parent. The **fractionAlong** specifies where it is connected, usually 1 ( the default value ), meaning the  distal point of the parent, or 0, meaning the  proximal point. If it is between these, a linear interpolation between the 2 points should be used.


`````
```` Text fields
```Table:

**segment**$ The id of the parent segment
**fractionAlong**$ The fraction along the the parent segment at which this segment is attached. For usage see _proximal_

````

```` Schema
``` xml
<xs:complexType name="SegmentParent">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="segment" type="NonNegativeInteger" use="required"/>
      <xs:attribute name="fractionAlong" type="ZeroToOne" use="optional" default="1"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````


```` Usage: XML
``` xml
<parent segment="0"/>
```
``` xml
<parent segment="1"/>
```
``` xml
<parent segment="2" fractionAlong="0.5"/>
```
````
`````


## segment




A segment defines the smallest unit within a possibly branching structure (  morphology ), such as a dendrite or axon. Its **id** should be a nonnegative integer ( usually soma/root = 0 ). Its end points are given by the  proximal and  distal points. The  proximal point can be omitted, usually because it is the same as a point on the  parent segment, see  proximal for details.  parent specifies the parent segment. The first segment of a  cell ( with no  parent ) usually represents the soma. The shape is normally a cylinder ( radii of the  proximal and  distal equal, but positions different ) or a conical frustum ( radii and positions different ). If the x, y, x positions of the  proximal and  distal are equal, the segment can be interpreted as a sphere, and in this case the radii of these points must be equal. NOTE: LEMS does not yet support multicompartmental modelling, so the Dynamics here is only appropriate for single compartment modelling.


`````
```` Constants
```Table:

**LEN** = 1m$  $ dimensions:length

```
````

```` Text fields
```Table:

**name**$ An optional name for the segment. Convenient for providing a suitable variable name for generated code, e.g. soma, dend0

````

```` Child list
```Table:

**parent**$  $ parent
**distal**$  $ distal
**proximal**$  $ proximal

```
````

```` Exposures
```Table:

**length**$  $dimensions:length
**radDist**$  $dimensions:length
**surfaceArea**$  $dimensions:area

```
````

```` Dynamics








**Derived Variables**
    : **radDist** =&nbsp;distal->radius(exposed as **radDist**)
    : **dx** =&nbsp;distal->xLength
    : **dy** =&nbsp;distal->yLength
    : **dz** =&nbsp;distal->zLength
    : **px** =&nbsp;proximal->xLength
    : **py** =&nbsp;proximal->yLength
    : **pz** =&nbsp;proximal->zLength
    : **length** =&nbsp;sqrt(((dx - px) * (dx - px) + (dy - py) * (dy - py) + (dz - pz) * (dz - pz))/(LEN * LEN)) * LEN(exposed as **length**)
    



**Conditional Derived Variables**
    
: IF length = 0 * LEN THEN
:  **surfaceArea** = 4 \* radDist \* radDist \* 3.14159265 (exposed as **surfaceArea**)
: IF length &gt; 0 * LEN THEN
:  **surfaceArea** = 2 \* radDist \* 3.14159265 \* length (exposed as **surfaceArea**)


````

```` Schema
``` xml
<xs:complexType name="Segment">
  <xs:complexContent>
    <xs:extension base="BaseNonNegativeIntegerId">
      <xs:sequence>
        <xs:element name="parent" type="SegmentParent" minOccurs="0"/>
        <xs:element name="proximal" type="Point3DWithDiam" minOccurs="0"/>
        <xs:element name="distal" type="Point3DWithDiam" minOccurs="1"/>
      </xs:sequence>
      <xs:attribute name="name" type="xs:string" use="optional"/>
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Segment" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Segment
from neuroml.utils import component_factory

variable = component_factory(
    Segment,
    id: 'a NonNegativeInteger (required)' = None,
    name: 'a string (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    parent: 'a SegmentParent (optional)' = None,
    proximal: 'a Point3DWithDiam (optional)' = None,
    distal: 'a Point3DWithDiam (required)' = None,
)
```
````
```` Usage: XML
``` xml
<segment id="3" name="Spine1">
    <parent segment="2" fractionAlong="0.5"/>
    <proximal x="25" y="0" z="0" diameter="0.1"/>
    <distal x="25" y="0.2" z="0" diameter="0.1"/>
</segment>
```
``` xml
<segment id="0" name="Soma">
    <proximal x="0" y="0" z="0" diameter="10"/>
    <distal x="10" y="0" z="0" diameter="10"/>
</segment>
```
``` xml
<segment id="1" name="Dendrite1">
    <parent segment="0"/>
    <distal x="20" y="0" z="0" diameter="3"/>
</segment>
```
````
`````


## segmentGroup




A method to describe a group of  segments in a  morphology, e.g. soma_group, dendrite_group, axon_group. While a name is useful to describe the group, the **neuroLexId** attribute can be used to explicitly specify the meaning of the group, e.g. sao1044911821 for 'Neuronal Cell Body', sao1211023249 for 'Dendrite'. The  segments in this group can be specified as: a list of individual  member segments; a  path, all of the segments along which should be included; a  subtree of the  cell to include; other segmentGroups to  include ( so all segments from those get included here ). An  inhomogeneousparameter can be defined on the region of the cell specified by this group ( see  variableparameter for usage ).


`````
```` Text fields
```Table:

**neuroLexId**$ An id string for pointing to an entry in the NeuroLex ontology. Use of this attribute is a shorthand for a full         RDF based reference to the MIRIAM Resource urn:miriam:neurolex, with an bqbiol:is qualifier.

````

```` Child list
```Table:

**notes**$  $ notes
**annotation**$  $ annotation

```
````

```` Children list
```Table:

**property**$  $ property
**members**$  $ member
**paths**$  $ path
**subTrees**$  $ subtree
**includes**$  $ include
**inhomogeneousParameter**$  $ inhomogeneousparameter

```
````

```` Schema
``` xml
<xs:complexType name="SegmentGroup">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="property" type="Property" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="annotation" type="Annotation" minOccurs="0"/>
        <xs:element name="member" type="Member" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="include" type="Include" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="path" type="Path" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="subTree" type="SubTree" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="inhomogeneousParameter" type="InhomogeneousParameter" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SegmentGroup" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SegmentGroup
from neuroml.utils import component_factory

variable = component_factory(
    SegmentGroup,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    members: 'list of Member(s) (optional)' = None,
    includes: 'list of Include(s) (optional)' = None,
    paths: 'list of Path(s) (optional)' = None,
    sub_trees: 'list of SubTree(s) (optional)' = None,
    inhomogeneous_parameters: 'list of InhomogeneousParameter(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<segmentGroup id="dendrite_group" neuroLexId="sao1211023249">
    <member segment="1"/>
    <member segment="2"/>
    <member segment="3"/>
</segmentGroup>
```
``` xml
<segmentGroup id="soma_group" neuroLexId="sao1044911821">
    <member segment="0"/>
</segmentGroup>
```
``` xml
<segmentGroup id="spines" neuroLexId="sao1145756102">
    <member segment="3"/>
</segmentGroup>
```
````
`````


## member




A single identified **segment** which is part of the  segmentgroup.


`````
```` Text fields
```Table:

**segment**$ 

````

```` Schema
``` xml
<xs:complexType name="Member">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="segment" type="NonNegativeInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Member" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Member
from neuroml.utils import component_factory

variable = component_factory(
    Member,
    segments: 'a NonNegativeInteger (required)' = None,
)
```
````
```` Usage: XML
``` xml
<member segment="0"/>
```
``` xml
<member segment="1"/>
```
``` xml
<member segment="2"/>
```
````
`````


## from




In a  path or  subtree, specifies which **segment** ( inclusive ) from which to calculate the  segmentgroup.


`````
```` Text fields
```Table:

**segment**$ 

````

```` Schema
``` xml
<xs:complexType name="SegmentEndPoint">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="segment" type="NonNegativeInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````


```` Usage: XML
``` xml
<from segment="1"/>
```
``` xml
<from segment="1"/>
```
````
`````


## to




In a  path, specifies which **segment** ( inclusive ) up to which to calculate the  segmentgroup.


`````
```` Text fields
```Table:

**segment**$ 

````

```` Schema
``` xml
<xs:complexType name="SegmentEndPoint">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="segment" type="NonNegativeInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````


```` Usage: XML
``` xml
<to segment="2"/>
```
````
`````


## include




Include all members of another  segmentgroup in this group.


`````
```` Text fields
```Table:

**href**$ 
**segmentGroup**$ 

````

```` Schema
``` xml
<xs:complexType name="Include">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="segmentGroup" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Include" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Include
from neuroml.utils import component_factory

variable = component_factory(
    Include,
    segment_groups: 'a NmlId (required)' = None,
)
```
````
```` Usage: XML
``` xml
<include href="NML2_SingleCompHHCell.nml"/>
```
``` xml
<include href="NML2_SimpleIonChannel.nml"/>
```
``` xml
<include href="NML2_SimpleIonChannel.nml"/>
```
````
`````


## path




Include all the  segments between those specified by  from and  to, inclusive.


`````
```` Child list
```Table:

**from**$  $ from
**to**$  $ to

```
````

```` Schema
``` xml
<xs:complexType name="Path">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:sequence>
        <xs:element name="from" type="SegmentEndPoint" minOccurs="0"/>
        <xs:element name="to" type="SegmentEndPoint" minOccurs="0"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Path" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Path
from neuroml.utils import component_factory

variable = component_factory(
    Path,
    from_: 'a SegmentEndPoint (optional)' = None,
    to: 'a SegmentEndPoint (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<path>
    <from segment="1"/>
    <to segment="2"/>
</path>
```
````
`````


## subTree




Include all the  segments distal to that specified by  from in the  segmentgroup.


`````
```` Child list
```Table:

**from**$  $ from

```
````

```` Schema
``` xml
<xs:complexType name="SubTree">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:choice>
        <xs:element name="from" type="SegmentEndPoint" minOccurs="0"/>
        <xs:element name="to" type="SegmentEndPoint" minOccurs="0"/>
      </xs:choice>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SubTree" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SubTree
from neuroml.utils import component_factory

variable = component_factory(
    SubTree,
    from_: 'a SegmentEndPoint (optional)' = None,
    to: 'a SegmentEndPoint (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<subTree>
    <from segment="1"/>
</subTree>
```
````
`````


## inhomogeneousParameter




An inhomogeneous parameter specified across the  segmentgroup ( see  variableparameter for usage ).


`````
```` Text fields
```Table:

**variable**$ 
**metric**$ 

````

```` Child list
```Table:

**proximal**$  $ proximaldetails
**distal**$  $ distaldetails

```
````

```` Schema
``` xml
<xs:complexType name="InhomogeneousParameter">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="proximal" type="ProximalDetails" minOccurs="0"/>
        <xs:element name="distal" type="DistalDetails" minOccurs="0"/>
      </xs:sequence>
      <xs:attribute name="variable" type="xs:string" use="required"/>
      <xs:attribute name="metric" type="Metric" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=InhomogeneousParameter" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import InhomogeneousParameter
from neuroml.utils import component_factory

variable = component_factory(
    InhomogeneousParameter,
    id: 'a NmlId (required)' = None,
    variable: 'a string (required)' = None,
    metric: 'a Metric (required)' = None,
    proximal: 'a ProximalDetails (optional)' = None,
    distal: 'a DistalDetails (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<inhomogeneousParameter id="dendrite_group_x2" variable="r" metric="Path Length from root">
    <proximal translationStart="0"/>
    <distal normalizationEnd="1"/>
</inhomogeneousParameter>
```
``` xml
<inhomogeneousParameter id="dendrite_group_x1" variable="p" metric="Path Length from root"/>
```
````
`````


## proximalDetails




What to do at the proximal point when creating an inhomogeneous parameter.


`````
```` Text fields
```Table:

**translationStart**$ 

````

```` Schema
``` xml
<xs:complexType name="ProximalDetails">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="translationStart" type="xs:double" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ProximalDetails" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ProximalDetails
from neuroml.utils import component_factory

variable = component_factory(
    ProximalDetails,
    translation_start: 'a double (required)' = None,
)
```
````
`````


## distalDetails




What to do at the distal point when creating an inhomogeneous parameter.


`````
```` Text fields
```Table:

**normalizationEnd**$ 

````

```` Schema
``` xml
<xs:complexType name="DistalDetails">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="normalizationEnd" type="xs:double" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=DistalDetails" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import DistalDetails
from neuroml.utils import component_factory

variable = component_factory(
    DistalDetails,
    normalization_end: 'a double (required)' = None,
)
```
````
`````


## morphology




The collection of  segments which specify the 3D structure of the cell, along with a number of  segmentgroups.


`````
```` Children list
```Table:

**segments**$  $ segment
**segmentGroups**$  $ segmentgroup

```
````

```` Schema
``` xml
<xs:complexType name="Morphology">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="segment" type="Segment" maxOccurs="unbounded"/>
        <xs:element name="segmentGroup" type="SegmentGroup" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Morphology" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Morphology
from neuroml.utils import component_factory

variable = component_factory(
    Morphology,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    segments: 'list of Segment(s) (required)' = None,
    segment_groups: 'list of SegmentGroup(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<morphology id="SpikingCell_morphology">
    <segment id="0" name="Soma">
        <proximal x="0" y="0" z="0" diameter="10"/>
        <distal x="10" y="0" z="0" diameter="10"/>
    </segment>
    <segment id="1" name="Dendrite1">
        <parent segment="0"/>
        <distal x="20" y="0" z="0" diameter="3"/>
    </segment>
    <segment id="2" name="Dendrite2">
        <parent segment="1"/>
        <distal x="30" y="0" z="0" diameter="1"/>
    </segment>
    <segment id="3" name="Spine1">
        <parent segment="2" fractionAlong="0.5"/>
        <proximal x="25" y="0" z="0" diameter="0.1"/>
        <distal x="25" y="0.2" z="0" diameter="0.1"/>
    </segment>
    <segmentGroup id="soma_group" neuroLexId="sao1044911821">
        <member segment="0"/>
    </segmentGroup>
    <segmentGroup id="dendrite_group" neuroLexId="sao1211023249">
        <member segment="1"/>
        <member segment="2"/>
        <member segment="3"/>
    </segmentGroup>
    <segmentGroup id="spines" neuroLexId="sao1145756102">
        <member segment="3"/>
    </segmentGroup>
</morphology>
```
``` xml
<morphology id="NeuroMorpho_PyrCell123">
    <segment id="0" name="Soma">
        <proximal x="0" y="0" z="0" diameter="10"/>
        <distal x="10" y="0" z="0" diameter="10"/>
    </segment>
</morphology>
```
``` xml
<morphology id="SimpleCell_Morphology">
    <segment id="0" name="Soma">
        <proximal x="0" y="0" z="0" diameter="10"/>
        <distal x="10" y="0" z="0" diameter="10"/>
    </segment>
    <segment id="1" name="MainDendrite1">
        <parent segment="0"/>
        <proximal x="10" y="0" z="0" diameter="3"/>
        <distal x="20" y="0" z="0" diameter="3"/>
    </segment>
    <segment id="2" name="MainDendrite2">
        <parent segment="1"/>
        <distal x="30" y="0" z="0" diameter="1"/>
    </segment>
    <segmentGroup id="soma_group" neuroLexId="sao1044911821">
        <member segment="0"/>
    </segmentGroup>
    <segmentGroup id="dendrite_group" neuroLexId="sao1211023249">
        <member segment="1"/>
        <member segment="2"/>
        <inhomogeneousParameter id="dendrite_group_x1" variable="p" metric="Path Length from root"/>
        <inhomogeneousParameter id="dendrite_group_x2" variable="r" metric="Path Length from root">
            <proximal translationStart="0"/>
            <distal normalizationEnd="1"/>
        </inhomogeneousParameter>
    </segmentGroup>
</morphology>
```
````
`````


## specificCapacitance




Capacitance per unit area.


`````
```` Parameters
```Table:

**value**$  $dimensions:specificCapacitance

```
````

```` Text fields
```Table:

**segmentGroup**$ 

````

```` Exposures
```Table:

**specCap**$  $dimensions:specificCapacitance

```
````

```` Dynamics








**Derived Variables**
    : **specCap** =&nbsp;value(exposed as **specCap**)
    





````

```` Schema
``` xml
<xs:complexType name="SpecificCapacitance">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="value" type="Nml2Quantity_specificCapacitance" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpecificCapacitance" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpecificCapacitance
from neuroml.utils import component_factory

variable = component_factory(
    SpecificCapacitance,
    value: 'a Nml2Quantity_specificCapacitance (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
)
```
````
```` Usage: XML
``` xml
<specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
```
``` xml
<specificCapacitance segmentGroup="dendrite_group" value="2.0 uF_per_cm2"/>
```
``` xml
<specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
```
````
`````


## initMembPotential




Explicitly set initial membrane potential for the cell.


`````
```` Parameters
```Table:

**value**$  $dimensions:voltage

```
````

```` Schema
``` xml
<xs:complexType name="InitMembPotential">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="value" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=InitMembPotential" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import InitMembPotential
from neuroml.utils import component_factory

variable = component_factory(
    InitMembPotential,
    value: 'a Nml2Quantity_voltage (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
)
```
````
```` Usage: XML
``` xml
<initMembPotential value="-65mV"/>
```
``` xml
<initMembPotential value="-65mV"/>
```
````
`````


## spikeThresh




Membrane potential at which to emit a spiking event. Note, usually the spiking event will not be emitted again until the membrane potential has fallen below this value and rises again to cross it in a positive direction.


`````
```` Parameters
```Table:

**value**$  $dimensions:voltage

```
````

```` Schema
``` xml
<xs:complexType name="SpikeThresh">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="value" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeThresh" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeThresh
from neuroml.utils import component_factory

variable = component_factory(
    SpikeThresh,
    value: 'a Nml2Quantity_voltage (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
)
```
````
```` Usage: XML
``` xml
<spikeThresh value="-20mV"/>
```
``` xml
<spikeThresh value="-20mV"/>
```
````
`````


## membraneProperties




Properties specific to the membrane, such as the **populations** of channels, **channelDensities,** **specificCapacitance,** etc.


`````
```` Child list
```Table:

**initMembPotential**$  $ initmembpotential
**spikeThresh**$  $ spikethresh

```
````

```` Children list
```Table:

**specificCapacitances**$  $ specificcapacitance
**populations**$  $ basechannelpopulation
**channelDensities**$  $ basechanneldensity

```
````

```` Exposures
```Table:

**iCa**$  $dimensions:current
**totChanCurrent**$  $dimensions:current
**totSpecCap**$  $dimensions:specificCapacitance

```
````

```` Requirements
```Table:

**surfaceArea**$  $dimensions:area

```
````

```` Dynamics








**Derived Variables**
    : **totSpecCap** =&nbsp;specificCapacitances[*]->specCap(reduce method: add)(exposed as **totSpecCap**)
    : **totChanPopCurrent** =&nbsp;populations[*]->i(reduce method: add)
    : **totChanDensCurrentDensity** =&nbsp;channelDensities[*]->iDensity(reduce method: add)
    : **totChanCurrent** =&nbsp;totChanPopCurrent + (totChanDensCurrentDensity * surfaceArea)(exposed as **totChanCurrent**)
    : **totChanPopCurrentCa** =&nbsp;populations[ion='ca']->i(reduce method: add)
    : **totChanDensCurrentDensityCa** =&nbsp;channelDensities[ion='ca']->iDensity(reduce method: add)
    : **iCa** =&nbsp;totChanPopCurrentCa + (totChanDensCurrentDensityCa * surfaceArea)(exposed as **iCa**)
    





````

```` Schema
``` xml
<xs:complexType name="MembraneProperties">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:sequence>
        <xs:element name="channelPopulation" type="ChannelPopulation" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensity" type="ChannelDensity" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityVShift" type="ChannelDensityVShift" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityNernst" type="ChannelDensityNernst" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityGHK" type="ChannelDensityGHK" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityGHK2" type="ChannelDensityGHK2" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityNonUniform" type="ChannelDensityNonUniform" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityNonUniformNernst" type="ChannelDensityNonUniformNernst" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="channelDensityNonUniformGHK" type="ChannelDensityNonUniformGHK" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="spikeThresh" type="SpikeThresh" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="specificCapacitance" type="SpecificCapacitance" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="initMembPotential" type="InitMembPotential" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=MembraneProperties" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import MembraneProperties
from neuroml.utils import component_factory

variable = component_factory(
    MembraneProperties,
    channel_populations: 'list of ChannelPopulation(s) (optional)' = None,
    channel_densities: 'list of ChannelDensity(s) (optional)' = None,
    channel_density_v_shifts: 'list of ChannelDensityVShift(s) (optional)' = None,
    channel_density_nernsts: 'list of ChannelDensityNernst(s) (optional)' = None,
    channel_density_ghks: 'list of ChannelDensityGHK(s) (optional)' = None,
    channel_density_ghk2s: 'list of ChannelDensityGHK2(s) (optional)' = None,
    channel_density_non_uniforms: 'list of ChannelDensityNonUniform(s) (optional)' = None,
    channel_density_non_uniform_nernsts: 'list of ChannelDensityNonUniformNernst(s) (optional)' = None,
    channel_density_non_uniform_ghks: 'list of ChannelDensityNonUniformGHK(s) (optional)' = None,
    spike_threshes: 'list of SpikeThresh(s) (required)' = None,
    specific_capacitances: 'list of SpecificCapacitance(s) (required)' = None,
    init_memb_potentials: 'list of InitMembPotential(s) (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<membraneProperties>
    <channelPopulation id="naChansDend" ionChannel="NaConductance" segment="2" number="120000" erev="50mV" ion="na"/>
    <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific"/>
    <channelDensity id="naChansSoma" ionChannel="NaConductance" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" erev="50mV" ion="na"/>
    <specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
    <specificCapacitance segmentGroup="dendrite_group" value="2.0 uF_per_cm2"/>
</membraneProperties>
```
``` xml
<membraneProperties>
    <channelDensity id="naChans" ionChannel="HH_Na" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" ion="na" erev="50mV"/>
    <!-- Ions present inside the cell. Note: a fixed reversal potential is specified here  
            <reversalPotential species="na" value="50mV"/>
            <reversalPotential species="k" value="-77mV"/>-->
</membraneProperties>
```
``` xml
<membraneProperties>
    <channelDensityNonUniform id="nonuniform_na_chans" ionChannel="NaConductance" erev="50mV" ion="na">
        <variableParameter parameter="condDensity" segmentGroup="dendrite_group">
            <inhomogeneousValue inhomogeneousParameter="dendrite_group_x1" value="5e-7 * exp(-p/200)"/>
        </variableParameter>
    </channelDensityNonUniform>
    <specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
</membraneProperties>
```
````
`````


## membraneProperties2CaPools




extends membraneproperties



Variant of membraneProperties with 2 independent Ca pools.


`````
```` Child list
```Table:

**initMembPotential**$  $ initmembpotential
**spikeThresh**$  $ spikethresh

```
````

```` Children list
```Table:

**specificCapacitances**$  $ specificcapacitance
**populations**$  $ basechannelpopulation
**channelDensities**$  $ basechanneldensity

```
````

```` Exposures
```Table:

**iCa**$  *(from membraneproperties)* $dimensions:current
**iCa2**$  $dimensions:current
**totChanCurrent**$  *(from membraneproperties)* $dimensions:current
**totSpecCap**$  *(from membraneproperties)* $dimensions:specificCapacitance

```
````

```` Requirements
```Table:

**surfaceArea**$  $dimensions:area
**surfaceArea**$  *(from membraneproperties)* $dimensions:area

```
````

```` Dynamics








**Derived Variables**
    : **totSpecCap** =&nbsp;specificCapacitances[*]->specCap(reduce method: add)(exposed as **totSpecCap**)
    : **totChanPopCurrent** =&nbsp;populations[*]->i(reduce method: add)
    : **totChanDensCurrentDensity** =&nbsp;channelDensities[*]->iDensity(reduce method: add)
    : **totChanCurrent** =&nbsp;totChanPopCurrent + (totChanDensCurrentDensity * surfaceArea)(exposed as **totChanCurrent**)
    : **totChanPopCurrentCa** =&nbsp;populations[ion='ca']->i(reduce method: add)
    : **totChanDensCurrentDensityCa** =&nbsp;channelDensities[ion='ca']->iDensity(reduce method: add)
    : **iCa** =&nbsp;totChanPopCurrentCa + (totChanDensCurrentDensityCa * surfaceArea)(exposed as **iCa**)
    : **totChanPopCurrentCa2** =&nbsp;populations[ion='ca2']->i(reduce method: add)
    : **totChanDensCurrentDensityCa2** =&nbsp;channelDensities[ion='ca2']->iDensity(reduce method: add)
    : **iCa2** =&nbsp;totChanPopCurrentCa2 + (totChanDensCurrentDensityCa2 * surfaceArea)(exposed as **iCa2**)
    





````

```` Schema
``` xml
<xs:complexType name="MembraneProperties2CaPools">
  <xs:complexContent>
    <xs:extension base="MembraneProperties">
      <xs:sequence>
        <xs:element name="channelDensityNernstCa2" type="ChannelDensityNernstCa2" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=MembraneProperties2CaPools" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import MembraneProperties2CaPools
from neuroml.utils import component_factory

variable = component_factory(
    MembraneProperties2CaPools,
    channel_populations: 'list of ChannelPopulation(s) (optional)' = None,
    channel_densities: 'list of ChannelDensity(s) (optional)' = None,
    channel_density_v_shifts: 'list of ChannelDensityVShift(s) (optional)' = None,
    channel_density_nernsts: 'list of ChannelDensityNernst(s) (optional)' = None,
    channel_density_ghks: 'list of ChannelDensityGHK(s) (optional)' = None,
    channel_density_ghk2s: 'list of ChannelDensityGHK2(s) (optional)' = None,
    channel_density_non_uniforms: 'list of ChannelDensityNonUniform(s) (optional)' = None,
    channel_density_non_uniform_nernsts: 'list of ChannelDensityNonUniformNernst(s) (optional)' = None,
    channel_density_non_uniform_ghks: 'list of ChannelDensityNonUniformGHK(s) (optional)' = None,
    spike_threshes: 'list of SpikeThresh(s) (required)' = None,
    specific_capacitances: 'list of SpecificCapacitance(s) (required)' = None,
    init_memb_potentials: 'list of InitMembPotential(s) (required)' = None,
    channel_density_nernst_ca2s: 'list of ChannelDensityNernstCa2(s) (optional)' = None,
)
```
````
`````


## biophysicalProperties




The biophysical properties of the  cell, including the  membraneproperties and the  intracellularproperties.


`````
```` Child list
```Table:

**membraneProperties**$  $ membraneproperties
**intracellularProperties**$  $ intracellularproperties

```
````

```` Exposures
```Table:

**totSpecCap**$  $dimensions:specificCapacitance

```
````

```` Dynamics








**Derived Variables**
    : **totSpecCap** =&nbsp;membraneProperties->totSpecCap(exposed as **totSpecCap**)
    





````

```` Schema
``` xml
<xs:complexType name="BiophysicalProperties">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="membraneProperties" type="MembraneProperties"/>
        <xs:element name="intracellularProperties" type="IntracellularProperties" minOccurs="0"/>
        <xs:element name="extracellularProperties" type="ExtracellularProperties" minOccurs="0"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BiophysicalProperties" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BiophysicalProperties
from neuroml.utils import component_factory

variable = component_factory(
    BiophysicalProperties,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    membrane_properties: 'a MembraneProperties (required)' = None,
    intracellular_properties: 'a IntracellularProperties (optional)' = None,
    extracellular_properties: 'a ExtracellularProperties (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<biophysicalProperties id="bio_cell">
    <membraneProperties>
        <channelPopulation id="naChansDend" ionChannel="NaConductance" segment="2" number="120000" erev="50mV" ion="na"/>
        <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific"/>
        <channelDensity id="naChansSoma" ionChannel="NaConductance" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" erev="50mV" ion="na"/>
        <specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
        <specificCapacitance segmentGroup="dendrite_group" value="2.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
        <resistivity value="0.1 kohm_cm"/>
    </intracellularProperties>
</biophysicalProperties>
```
``` xml
<biophysicalProperties id="PyrCellChanDist">
    <membraneProperties>
        <channelDensity id="naChans" ionChannel="HH_Na" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" ion="na" erev="50mV"/>
        <!-- Ions present inside the cell. Note: a fixed reversal potential is specified here  
            <reversalPotential species="na" value="50mV"/>
            <reversalPotential species="k" value="-77mV"/>-->
    </membraneProperties>
    <intracellularProperties>
        <resistivity value="0.1 kohm_cm"/>
        <!-- REMOVED UNTIL WE CHECK HOW THE USAGE OF LEMS IMPACTS THIS...
            <biochemistry reactionScheme="InternalCaDynamics"/>  Ref to earlier pathway -->
    </intracellularProperties>
</biophysicalProperties>
```
``` xml
<biophysicalProperties id="biophys">
    <membraneProperties>
        <channelDensityNonUniform id="nonuniform_na_chans" ionChannel="NaConductance" erev="50mV" ion="na">
            <variableParameter parameter="condDensity" segmentGroup="dendrite_group">
                <inhomogeneousValue inhomogeneousParameter="dendrite_group_x1" value="5e-7 * exp(-p/200)"/>
            </variableParameter>
        </channelDensityNonUniform>
        <specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
        <resistivity value="0.1 kohm_cm"/>
    </intracellularProperties>
</biophysicalProperties>
```
````
`````


## biophysicalProperties2CaPools




The biophysical properties of the  cell, including the  membraneproperties2capools and the  intracellularproperties2capools for a cell with two Ca pools.


`````
```` Child list
```Table:

**membraneProperties2CaPools**$  $ membraneproperties2capools
**intracellularProperties2CaPools**$  $ intracellularproperties2capools

```
````

```` Exposures
```Table:

**totSpecCap**$  $dimensions:specificCapacitance

```
````

```` Dynamics








**Derived Variables**
    : **totSpecCap** =&nbsp;membraneProperties2CaPools->totSpecCap(exposed as **totSpecCap**)
    





````

```` Schema
``` xml
<xs:complexType name="BiophysicalProperties2CaPools">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="membraneProperties2CaPools" type="MembraneProperties2CaPools"/>
        <xs:element name="intracellularProperties2CaPools" type="IntracellularProperties2CaPools" minOccurs="0"/>
        <xs:element name="extracellularProperties" type="ExtracellularProperties" minOccurs="0"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BiophysicalProperties2CaPools" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BiophysicalProperties2CaPools
from neuroml.utils import component_factory

variable = component_factory(
    BiophysicalProperties2CaPools,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    membrane_properties2_ca_pools: 'a MembraneProperties2CaPools (required)' = None,
    intracellular_properties2_ca_pools: 'a IntracellularProperties2CaPools (optional)' = None,
    extracellular_properties: 'a ExtracellularProperties (optional)' = None,
)
```
````
`````


## intracellularProperties




Biophysical properties related to the intracellular space within the  cell, such as the  resistivity and the list of ionic  species present. **caConc** and **caConcExt** are explicitly exposed here to facilitate accessing these values from other Components, even though **caConcExt** is clearly not an intracellular property.


`````
```` Children list
```Table:

**resistivity**$  $ resistivity
**speciesList**$  $ species

```
````

```` Exposures
```Table:

**caConc**$  $dimensions:concentration
**caConcExt**$  $dimensions:concentration

```
````

```` Dynamics








**Derived Variables**
    : **caConc** =&nbsp;speciesList[ion='ca']->concentration(reduce method: add)(exposed as **caConc**)
    : **caConcExt** =&nbsp;speciesList[ion='ca']->extConcentration(reduce method: add)(exposed as **caConcExt**)
    





````

```` Schema
``` xml
<xs:complexType name="IntracellularProperties">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:sequence>
        <xs:element name="species" type="Species" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="resistivity" type="Resistivity" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IntracellularProperties" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IntracellularProperties
from neuroml.utils import component_factory

variable = component_factory(
    IntracellularProperties,
    species: 'list of Species(s) (optional)' = None,
    resistivities: 'list of Resistivity(s) (optional)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<intracellularProperties>
    <resistivity value="0.1 kohm_cm"/>
</intracellularProperties>
```
``` xml
<intracellularProperties>
    <resistivity value="0.1 kohm_cm"/>
    <!-- REMOVED UNTIL WE CHECK HOW THE USAGE OF LEMS IMPACTS THIS...
            <biochemistry reactionScheme="InternalCaDynamics"/>  Ref to earlier pathway -->
</intracellularProperties>
```
``` xml
<intracellularProperties>
    <resistivity value="0.1 kohm_cm"/>
</intracellularProperties>
```
````
`````


## intracellularProperties2CaPools




extends intracellularproperties



Variant of intracellularProperties with 2 independent Ca pools.


`````
```` Children list
```Table:

**speciesList**$  $ species
**resistivity**$  $ resistivity

```
````

```` Exposures
```Table:

**caConc**$  *(from intracellularproperties)* $dimensions:concentration
**caConc2**$  $dimensions:concentration
**caConcExt**$  *(from intracellularproperties)* $dimensions:concentration
**caConcExt2**$  $dimensions:concentration

```
````

```` Dynamics








**Derived Variables**
    : **caConc2** =&nbsp;speciesList[ion='ca2']->concentration(reduce method: add)(exposed as **caConc2**)
    : **caConcExt2** =&nbsp;speciesList[ion='ca2']->extConcentration(reduce method: add)(exposed as **caConcExt2**)
    : **caConc** =&nbsp;speciesList[ion='ca']->concentration(reduce method: add)(exposed as **caConc**)
    : **caConcExt** =&nbsp;speciesList[ion='ca']->extConcentration(reduce method: add)(exposed as **caConcExt**)
    





````

```` Schema
``` xml
<xs:complexType name="IntracellularProperties2CaPools">
  <xs:complexContent>
    <xs:extension base="IntracellularProperties">
      </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IntracellularProperties2CaPools" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IntracellularProperties2CaPools
from neuroml.utils import component_factory

variable = component_factory(
    IntracellularProperties2CaPools,
    species: 'list of Species(s) (optional)' = None,
    resistivities: 'list of Resistivity(s) (optional)' = None,
)
```
````
`````


## resistivity




The resistivity, or specific axial resistance, of the cytoplasm.


`````
```` Parameters
```Table:

**value**$  $dimensions:resistivity

```
````

```` Text fields
```Table:

**segmentGroup**$ 

````

```` Schema
``` xml
<xs:complexType name="Resistivity">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="value" type="Nml2Quantity_resistivity" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Resistivity" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Resistivity
from neuroml.utils import component_factory

variable = component_factory(
    Resistivity,
    value: 'a Nml2Quantity_resistivity (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
)
```
````
```` Usage: XML
``` xml
<resistivity value="0.1 kohm_cm"/>
```
``` xml
<resistivity value="0.1 kohm_cm"/>
```
``` xml
<resistivity value="0.1 kohm_cm"/>
```
````
`````


## concentrationModel




Base for any model of an **ion** concentration which changes with time. Internal ( **concentration** ) and external ( **extConcentration** ) values for the concentration of the ion are given.


`````
```` Text fields
```Table:

**ion**$ 

````

```` Exposures
```Table:

**concentration**$  $dimensions:concentration
**extConcentration**$  $dimensions:concentration

```
````

```` Requirements
```Table:

**initialConcentration**$  $dimensions:concentration
**initialExtConcentration**$  $dimensions:concentration
**surfaceArea**$  $dimensions:area

```
````

```` Dynamics



**State Variables**
: **concentration**: dimensions:concentration (exposed as **concentration**)
: **extConcentration**: dimensions:concentration (exposed as **extConcentration**)









**On Start**
: **concentration** = initialConcentration
: **extConcentration** = initialExtConcentration








````
`````


## decayingPoolConcentrationModel




extends concentrationmodel



Model of an intracellular buffering mechanism for **ion** ( currently hard Coded to be calcium, due to requirement for **iCa** ) which has a baseline level **restingConc** and tends to this value with time course **decayConstant.** The ion is assumed to occupy a shell inside the membrane of thickness **shellThickness.**.


`````
```` Parameters
```Table:

**decayConstant**$  $dimensions:time
**restingConc**$  $dimensions:concentration
**shellThickness**$  $dimensions:length

```
````

```` Constants
```Table:

**Faraday** = 96485.3C_per_mol$  $ dimensions:charge_per_mole
**AREA_SCALE** = 1m2$  $ dimensions:area
**LENGTH_SCALE** = 1m$  $ dimensions:length

```
````

```` Text fields
```Table:

**ion**$ 

````

```` Exposures
```Table:

**concentration**$  *(from concentrationmodel)* $dimensions:concentration
**extConcentration**$  *(from concentrationmodel)* $dimensions:concentration

```
````

```` Requirements
```Table:

**iCa**$  $dimensions:current
**initialConcentration**$  *(from concentrationmodel)* $dimensions:concentration
**initialExtConcentration**$  *(from concentrationmodel)* $dimensions:concentration
**surfaceArea**$  *(from concentrationmodel)* $dimensions:area

```
````

```` Dynamics



**State Variables**
: **concentration**: dimensions:concentration (exposed as **concentration**)
: **extConcentration**: dimensions:concentration (exposed as **extConcentration**)









**On Start**
: **concentration** = initialConcentration
: **extConcentration** = initialExtConcentration



**On Conditions**

: IF concentration &lt; 0 THEN
: **concentration** = 0





**Derived Variables**
    : **effectiveRadius** =&nbsp;LENGTH_SCALE * sqrt(surfaceArea/(AREA_SCALE * (4 * 3.14159)))
    : **innerRadius** =&nbsp;effectiveRadius - shellThickness
    : **shellVolume** =&nbsp;(4 * (effectiveRadius * effectiveRadius * effectiveRadius) * 3.14159 / 3) - (4 * (innerRadius * innerRadius * innerRadius) * 3.14159 / 3)
    





**Time Derivatives**
    : d **concentration** /dt = iCa / (2 * Faraday * shellVolume) - ((concentration - restingConc) / decayConstant)
    

````

```` Schema
``` xml
<xs:complexType name="DecayingPoolConcentrationModel">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
      <xs:attribute name="restingConc" type="Nml2Quantity_concentration" use="required"/>
      <xs:attribute name="decayConstant" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="shellThickness" type="Nml2Quantity_length" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=DecayingPoolConcentrationModel" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import DecayingPoolConcentrationModel
from neuroml.utils import component_factory

variable = component_factory(
    DecayingPoolConcentrationModel,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    ion: 'a NmlId (required)' = None,
    resting_conc: 'a Nml2Quantity_concentration (required)' = None,
    decay_constant: 'a Nml2Quantity_time (required)' = None,
    shell_thickness: 'a Nml2Quantity_length (required)' = None,
    extensiontype_=None,
)
```
````
`````


## fixedFactorConcentrationModel




extends concentrationmodel



Model of buffering of concentration of an ion ( currently hard coded to be calcium, due to requirement for **iCa** ) which has a baseline level **restingConc** and tends to this value with time course **decayConstant.** A fixed factor **rho** is used to scale the incoming current *independently of the size of the compartment* to produce a concentration change.


`````
```` Parameters
```Table:

**decayConstant**$  $dimensions:time
**restingConc**$  $dimensions:concentration
**rho**$  $dimensions:rho_factor

```
````

```` Text fields
```Table:

**ion**$ 

````

```` Exposures
```Table:

**concentration**$  *(from concentrationmodel)* $dimensions:concentration
**extConcentration**$  *(from concentrationmodel)* $dimensions:concentration

```
````

```` Requirements
```Table:

**iCa**$  $dimensions:current
**initialConcentration**$  *(from concentrationmodel)* $dimensions:concentration
**initialExtConcentration**$  *(from concentrationmodel)* $dimensions:concentration
**surfaceArea**$  $dimensions:area
**surfaceArea**$  *(from concentrationmodel)* $dimensions:area

```
````

```` Dynamics



**State Variables**
: **concentration**: dimensions:concentration (exposed as **concentration**)
: **extConcentration**: dimensions:concentration (exposed as **extConcentration**)









**On Start**
: **concentration** = initialConcentration
: **extConcentration** = initialExtConcentration



**On Conditions**

: IF concentration &lt; 0 THEN
: **concentration** = 0








**Time Derivatives**
    : d **concentration** /dt = (iCa/surfaceArea) * rho - ((concentration - restingConc) / decayConstant)
    

````

```` Schema
``` xml
<xs:complexType name="FixedFactorConcentrationModel">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="ion" type="NmlId" use="required">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
      <xs:attribute name="restingConc" type="Nml2Quantity_concentration" use="required"/>
      <xs:attribute name="decayConstant" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="rho" type="Nml2Quantity_rhoFactor" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=FixedFactorConcentrationModel" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import FixedFactorConcentrationModel
from neuroml.utils import component_factory

variable = component_factory(
    FixedFactorConcentrationModel,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    ion: 'a NmlId (required)' = None,
    resting_conc: 'a Nml2Quantity_concentration (required)' = None,
    decay_constant: 'a Nml2Quantity_time (required)' = None,
    rho: 'a Nml2Quantity_rhoFactor (required)' = None,
)
```
````
`````


## fixedFactorConcentrationModelTraub




extends concentrationmodel



Model of buffering of concentration of an ion ( currently hard coded to be calcium, due to requirement for **iCa** ) which has a baseline level **restingConc** and tends to this value with time course 1 / **beta.** A fixed factor **phi** is used to scale the incoming current *independently of the size of the compartment* to produce a concentration change. Not recommended for use in models other than Traub et al. 2005!


`````
```` Parameters
```Table:

**beta**$  $dimensions:per_time
**phi**$  $dimensions:rho_factor
**restingConc**$  $dimensions:concentration

```
````

```` Text fields
```Table:

**species**$ 

````

```` Exposures
```Table:

**concentration**$  *(from concentrationmodel)* $dimensions:concentration
**extConcentration**$  *(from concentrationmodel)* $dimensions:concentration

```
````

```` Requirements
```Table:

**iCa**$  $dimensions:current
**initialConcentration**$  *(from concentrationmodel)* $dimensions:concentration
**initialExtConcentration**$  *(from concentrationmodel)* $dimensions:concentration
**surfaceArea**$  $dimensions:area
**surfaceArea**$  *(from concentrationmodel)* $dimensions:area

```
````

```` Dynamics



**State Variables**
: **concentration**: dimensions:concentration (exposed as **concentration**)
: **extConcentration**: dimensions:concentration (exposed as **extConcentration**)









**On Start**
: **concentration** = initialConcentration
: **extConcentration** = initialExtConcentration



**On Conditions**

: IF concentration &lt; 0 THEN
: **concentration** = 0








**Time Derivatives**
    : d **concentration** /dt = (iCa/surfaceArea) * 1e-9 * phi - ((concentration - restingConc) * beta)
    

````
`````


## species




Description of a chemical species identified by **ion,** which has internal, **concentration,** and external, **extConcentration** values for its concentration.


`````
```` Parameters
```Table:

**initialConcentration**$  $dimensions:concentration
**initialExtConcentration**$  $dimensions:concentration

```
````

```` Text fields
```Table:

**ion**$ 
**segmentGroup**$ 

````

```` Component References
```Table:

**concentrationModel**$  $ concentrationmodel

```
````

```` Exposures
```Table:

**concentration**$  $dimensions:concentration
**extConcentration**$  $dimensions:concentration

```
````

```` Dynamics

**Structure**
: CHILD INSTANCE: **concentrationModel**









**Derived Variables**
    : **concentration** =&nbsp;concentrationModel->concentration(exposed as **concentration**)
    : **extConcentration** =&nbsp;concentrationModel->extConcentration(exposed as **extConcentration**)
    





````

```` Schema
``` xml
<xs:complexType name="Species">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:attribute name="concentrationModel" type="NmlId" use="required"/>
      <xs:attribute name="ion" type="NmlId" use="optional">
        <xs:annotation>
        </xs:annotation>
      </xs:attribute>
      <xs:attribute name="initialConcentration" type="Nml2Quantity_concentration" use="required"/>
      <xs:attribute name="initialExtConcentration" type="Nml2Quantity_concentration" use="required"/>
      <xs:attribute name="segmentGroup" type="NmlId" use="optional" default="all"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Species" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Species
from neuroml.utils import component_factory

variable = component_factory(
    Species,
    id: 'a NmlId (required)' = None,
    concentration_model: 'a NmlId (required)' = None,
    ion: 'a NmlId (optional)' = None,
    initial_concentration: 'a Nml2Quantity_concentration (required)' = None,
    initial_ext_concentration: 'a Nml2Quantity_concentration (required)' = None,
    segment_groups: 'a NmlId (optional)' = 'all',
)
```
````
`````


## cell




extends *basecellmembpot*



Cell with  segments specified in a  morphology element along with details on its  biophysicalproperties. NOTE: this can only be correctly simulated using jLEMS when there is a single segment in the cell, and **v** of this cell represents the membrane potential in that isopotential segment.


`````
```` Text fields
```Table:

**neuroLexId**$ 

````

```` Child list
```Table:

**morphology**$ Should only be used if morphology element is outside the cell. This points to the id of the morphology. $ morphology
**biophysicalProperties**$ Should only be used if biophysicalProperties element is outside the cell.  This points to the id of the biophysicalProperties $ biophysicalproperties

```
````

```` Exposures
```Table:

**caConc**$  $dimensions:concentration
**caConcExt**$  $dimensions:concentration
**iCa**$  $dimensions:current
**iChannels**$  $dimensions:current
**iSyn**$  $dimensions:current
**spiking**$  $Dimensionless
**surfaceArea**$  $dimensions:area
**totSpecCap**$  $dimensions:specificCapacitance
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **spiking**: Dimensionless (exposed as **spiking**)









**On Start**
: **spiking** = 0
: **v** = initMembPot



**On Conditions**

: IF v &gt; thresh AND spiking &lt; 0.5 THEN
: **spiking** = 1
: EVENT OUT on port: **spike**

: IF v &lt; thresh THEN
: **spiking** = 0





**Derived Variables**
    : **initMembPot** =&nbsp;biophysicalProperties->membraneProperties->initMembPotential->value
    : **thresh** =&nbsp;biophysicalProperties->membraneProperties->spikeThresh->value
    : **surfaceArea** =&nbsp;morphology->segments[*]->surfaceArea(reduce method: add)(exposed as **surfaceArea**)
    : **totSpecCap** =&nbsp;biophysicalProperties->totSpecCap(exposed as **totSpecCap**)
    : **totCap** =&nbsp;totSpecCap * surfaceArea 
    : **iChannels** =&nbsp;biophysicalProperties->membraneProperties->totChanCurrent(exposed as **iChannels**)
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iCa** =&nbsp;biophysicalProperties->membraneProperties->iCa(exposed as **iCa**)
    : **caConc** =&nbsp;biophysicalProperties->intracellularProperties->caConc(exposed as **caConc**)
    : **caConcExt** =&nbsp;biophysicalProperties->intracellularProperties->caConcExt(exposed as **caConcExt**)
    





**Time Derivatives**
    : d **v** /dt = (iChannels + iSyn) / totCap
    

````

```` Schema
``` xml
<xs:complexType name="Cell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:sequence>
        <xs:element name="morphology" type="Morphology" minOccurs="0"/>
        <xs:element name="biophysicalProperties" type="BiophysicalProperties" minOccurs="0"/>
      </xs:sequence>
      <xs:attribute name="morphology" type="NmlId" use="optional">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
      <xs:attribute name="biophysicalProperties" type="NmlId" use="optional">
        <xs:annotation>
            </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Cell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Cell
from neuroml.utils import component_factory

variable = component_factory(
    Cell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    morphology_attr: 'a NmlId (optional)' = None,
    biophysical_properties_attr: 'a NmlId (optional)' = None,
    morphology: 'a Morphology (optional)' = None,
    biophysical_properties: 'a BiophysicalProperties (optional)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<cell id="SpikingCell" metaid="HippoCA1Cell">
    <notes>A Simple Spiking cell for testing purposes</notes>
    <annotation>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
            <rdf:Description rdf:about="HippoCA1Cell">
                <bqbiol:is>
                    <rdf:Bag>
                        <rdf:li rdf:resource="urn:miriam:neurondb:258"/>
                    </rdf:Bag>
                </bqbiol:is>
            </rdf:Description>
        </rdf:RDF>
    </annotation>
    <morphology id="SpikingCell_morphology">
        <segment id="0" name="Soma">
            <proximal x="0" y="0" z="0" diameter="10"/>
            <distal x="10" y="0" z="0" diameter="10"/>
        </segment>
        <segment id="1" name="Dendrite1">
            <parent segment="0"/>
            <distal x="20" y="0" z="0" diameter="3"/>
        </segment>
        <segment id="2" name="Dendrite2">
            <parent segment="1"/>
            <distal x="30" y="0" z="0" diameter="1"/>
        </segment>
        <segment id="3" name="Spine1">
            <parent segment="2" fractionAlong="0.5"/>
            <proximal x="25" y="0" z="0" diameter="0.1"/>
            <distal x="25" y="0.2" z="0" diameter="0.1"/>
        </segment>
        <segmentGroup id="soma_group" neuroLexId="sao1044911821">
            <member segment="0"/>
        </segmentGroup>
        <segmentGroup id="dendrite_group" neuroLexId="sao1211023249">
            <member segment="1"/>
            <member segment="2"/>
            <member segment="3"/>
        </segmentGroup>
        <segmentGroup id="spines" neuroLexId="sao1145756102">
            <member segment="3"/>
        </segmentGroup>
    </morphology>
    <biophysicalProperties id="bio_cell">
        <membraneProperties>
            <channelPopulation id="naChansDend" ionChannel="NaConductance" segment="2" number="120000" erev="50mV" ion="na"/>
            <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific"/>
            <channelDensity id="naChansSoma" ionChannel="NaConductance" segmentGroup="soma_group" condDensity="120.0 mS_per_cm2" erev="50mV" ion="na"/>
            <specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
            <specificCapacitance segmentGroup="dendrite_group" value="2.0 uF_per_cm2"/>
        </membraneProperties>
        <intracellularProperties>
            <resistivity value="0.1 kohm_cm"/>
        </intracellularProperties>
    </biophysicalProperties>
</cell>
```
``` xml
<cell id="PyrCell" morphology="NeuroMorpho_PyrCell123" biophysicalProperties="PyrCellChanDist"/>
```
``` xml
<cell id="SimpleCell">
    <morphology id="SimpleCell_Morphology">
        <segment id="0" name="Soma">
            <proximal x="0" y="0" z="0" diameter="10"/>
            <distal x="10" y="0" z="0" diameter="10"/>
        </segment>
        <segment id="1" name="MainDendrite1">
            <parent segment="0"/>
            <proximal x="10" y="0" z="0" diameter="3"/>
            <distal x="20" y="0" z="0" diameter="3"/>
        </segment>
        <segment id="2" name="MainDendrite2">
            <parent segment="1"/>
            <distal x="30" y="0" z="0" diameter="1"/>
        </segment>
        <segmentGroup id="soma_group" neuroLexId="sao1044911821">
            <member segment="0"/>
        </segmentGroup>
        <segmentGroup id="dendrite_group" neuroLexId="sao1211023249">
            <member segment="1"/>
            <member segment="2"/>
            <inhomogeneousParameter id="dendrite_group_x1" variable="p" metric="Path Length from root"/>
            <inhomogeneousParameter id="dendrite_group_x2" variable="r" metric="Path Length from root">
                <proximal translationStart="0"/>
                <distal normalizationEnd="1"/>
            </inhomogeneousParameter>
        </segmentGroup>
    </morphology>
    <biophysicalProperties id="biophys">
        <membraneProperties>
            <channelDensityNonUniform id="nonuniform_na_chans" ionChannel="NaConductance" erev="50mV" ion="na">
                <variableParameter parameter="condDensity" segmentGroup="dendrite_group">
                    <inhomogeneousValue inhomogeneousParameter="dendrite_group_x1" value="5e-7 * exp(-p/200)"/>
                </variableParameter>
            </channelDensityNonUniform>
            <specificCapacitance segmentGroup="soma_group" value="1.0 uF_per_cm2"/>
        </membraneProperties>
        <intracellularProperties>
            <resistivity value="0.1 kohm_cm"/>
        </intracellularProperties>
    </biophysicalProperties>
</cell>
```
````
`````


## cell2CaPools




extends cell



Variant of cell with two independent Ca2+ pools. Cell with  segments specified in a  morphology element along with details on its  biophysicalproperties. NOTE: this can only be correctly simulated using jLEMS when there is a single segment in the cell, and **v** of this cell represents the membrane potential in that isopotential segment.


`````
```` Text fields
```Table:

**neuroLexId**$ 

````

```` Child list
```Table:

**biophysicalProperties2CaPools**$  $ biophysicalproperties2capools

```
````

```` Exposures
```Table:

**caConc**$  *(from cell)* $dimensions:concentration
**caConc2**$  $dimensions:concentration
**caConcExt**$  *(from cell)* $dimensions:concentration
**caConcExt2**$  $dimensions:concentration
**iCa**$  *(from cell)* $dimensions:current
**iCa2**$  $dimensions:current
**iChannels**$  *(from cell)* $dimensions:current
**iSyn**$  *(from cell)* $dimensions:current
**spiking**$  *(from cell)* $Dimensionless
**surfaceArea**$  *(from cell)* $dimensions:area
**totSpecCap**$  *(from cell)* $dimensions:specificCapacitance
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **spiking**: Dimensionless (exposed as **spiking**)









**On Start**
: **spiking** = 0
: **v** = initMembPot



**On Conditions**

: IF v &gt; thresh AND spiking &lt; 0.5 THEN
: **spiking** = 1
: EVENT OUT on port: **spike**

: IF v &lt; thresh THEN
: **spiking** = 0





**Derived Variables**
    : **initMembPot** =&nbsp;biophysicalProperties2CaPools->membraneProperties2CaPools->initMembPotential->value
    : **thresh** =&nbsp;biophysicalProperties2CaPools->membraneProperties2CaPools->spikeThresh->value
    : **surfaceArea** =&nbsp;morphology->segments[*]->surfaceArea(reduce method: add)(exposed as **surfaceArea**)
    : **totSpecCap** =&nbsp;biophysicalProperties2CaPools->totSpecCap(exposed as **totSpecCap**)
    : **totCap** =&nbsp;totSpecCap * surfaceArea 
    : **iChannels** =&nbsp;biophysicalProperties2CaPools->membraneProperties2CaPools->totChanCurrent(exposed as **iChannels**)
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iCa** =&nbsp;biophysicalProperties2CaPools->membraneProperties2CaPools->iCa(exposed as **iCa**)
    : **caConc** =&nbsp;biophysicalProperties2CaPools->intracellularProperties2CaPools->caConc(exposed as **caConc**)
    : **caConcExt** =&nbsp;biophysicalProperties2CaPools->intracellularProperties2CaPools->caConcExt(exposed as **caConcExt**)
    : **iCa2** =&nbsp;biophysicalProperties2CaPools->membraneProperties2CaPools->iCa2(exposed as **iCa2**)
    : **caConc2** =&nbsp;biophysicalProperties2CaPools->intracellularProperties2CaPools->caConc2(exposed as **caConc2**)
    : **caConcExt2** =&nbsp;biophysicalProperties2CaPools->intracellularProperties2CaPools->caConcExt2(exposed as **caConcExt2**)
    





**Time Derivatives**
    : d **v** /dt = (iChannels + iSyn) / totCap
    

````

```` Schema
``` xml
<xs:complexType name="Cell2CaPools">
  <xs:complexContent>
    <xs:extension base="Cell">
      <xs:sequence>
        <xs:element name="biophysicalProperties2CaPools" type="BiophysicalProperties2CaPools" minOccurs="0"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Cell2CaPools" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Cell2CaPools
from neuroml.utils import component_factory

variable = component_factory(
    Cell2CaPools,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    morphology_attr: 'a NmlId (optional)' = None,
    biophysical_properties_attr: 'a NmlId (optional)' = None,
    morphology: 'a Morphology (optional)' = None,
    biophysical_properties: 'a BiophysicalProperties (optional)' = None,
    biophysical_properties2_ca_pools: 'a BiophysicalProperties2CaPools (optional)' = None,
)
```
````
`````


## *baseCellMembPotCap*




extends *basecellmembpot*



Any cell with a membrane potential **v** with voltage units and a membrane capacitance **C.** Also defines exposed value **iSyn** for current due to external synapses and **iMemb** for total transmembrane current ( usually channel currents plus **iSyn** ).


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane $dimensions:capacitance

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane $dimensions:current
**iSyn**$ Total current due to synaptic inputs $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Schema
``` xml
<xs:complexType name="BaseCellMembPotCap">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="C" type="Nml2Quantity_capacitance" use="required">
        <xs:annotation>
          <xs:appinfo>
            <jxb:property name="Cap"/>
          </xs:appinfo>
        </xs:annotation>
      </xs:attribute>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseCellMembPotCap" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseCellMembPotCap
from neuroml.utils import component_factory

variable = component_factory(
    BaseCellMembPotCap,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    C: 'a Nml2Quantity_capacitance (required)' = None,
    extensiontype_=None,
)
```
````
`````


## *baseIaf*




extends *basecellmembpot*



Base ComponentType for an integrate and fire cell which emits a spiking event at membrane potential **thresh** and and resets to **reset**.


`````
```` Parameters
```Table:

**reset**$ The value the membrane potential is reset to on spiking $dimensions:voltage
**thresh**$ The membrane potential at which to emit a spiking event and reset voltage $dimensions:voltage

```
````

```` Exposures
```Table:

**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````
`````


## iafTauCell




extends *baseiaf*



Integrate and fire cell which returns to its leak reversal potential of **leakReversal** with a time constant **tau**.


`````
```` Parameters
```Table:

**leakReversal**$  $dimensions:voltage
**reset**$ The value the membrane potential is reset to on spiking *(from baseiaf)* $dimensions:voltage
**tau**$  $dimensions:time
**thresh**$ The membrane potential at which to emit a spiking event and reset voltage *(from baseiaf)* $dimensions:voltage

```
````

```` Exposures
```Table:

**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)









**On Start**
: **v** = leakReversal



**On Conditions**

: IF v &gt; thresh THEN
: **v** = reset
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **v** /dt = (leakReversal - v) / tau
    

````

```` Schema
``` xml
<xs:complexType name="IafTauCell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="leakReversal" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="thresh" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="reset" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="tau" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IafTauCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IafTauCell
from neuroml.utils import component_factory

variable = component_factory(
    IafTauCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    leak_reversal: 'a Nml2Quantity_voltage (required)' = None,
    thresh: 'a Nml2Quantity_voltage (required)' = None,
    reset: 'a Nml2Quantity_voltage (required)' = None,
    tau: 'a Nml2Quantity_time (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<iafTauCell id="iafTau" leakReversal="-50mV" thresh="-55mV" reset="-70mV" tau="30ms"/>
```
````
`````


## iafTauRefCell




extends iaftaucell



Integrate and fire cell which returns to its leak reversal potential of **leakReversal** with a time course **tau.** It has a refractory period of **refract** after spiking.


`````
```` Parameters
```Table:

**leakReversal**$  *(from iaftaucell)* $dimensions:voltage
**refract**$  $dimensions:time
**reset**$ The value the membrane potential is reset to on spiking *(from baseiaf)* $dimensions:voltage
**tau**$  *(from iaftaucell)* $dimensions:time
**thresh**$ The membrane potential at which to emit a spiking event and reset voltage *(from baseiaf)* $dimensions:voltage

```
````

```` Exposures
```Table:

**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = leakReversal









**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = reset
: **On Conditions**
:  IF t &gt; lastSpikeTime + refract THEN
: TRANSITION to REGIME **integrating**

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; thresh THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (leakReversal - v) / tau
````

```` Schema
``` xml
<xs:complexType name="IafTauRefCell">
  <xs:complexContent>
    <xs:extension base="IafTauCell">
      <xs:attribute name="refract" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IafTauRefCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IafTauRefCell
from neuroml.utils import component_factory

variable = component_factory(
    IafTauRefCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    leak_reversal: 'a Nml2Quantity_voltage (required)' = None,
    thresh: 'a Nml2Quantity_voltage (required)' = None,
    reset: 'a Nml2Quantity_voltage (required)' = None,
    tau: 'a Nml2Quantity_time (required)' = None,
    refract: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<iafTauRefCell id="iafTauRef" leakReversal="-50mV" thresh="-55mV" reset="-70mV" tau="30ms" refract="5ms"/>
```
````
`````


## *baseIafCapCell*




extends *basecellmembpotcap*



Base Type for all Integrate and Fire cells with a capacitance **C,** threshold **thresh** and reset membrane potential **reset**.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**reset**$  $dimensions:voltage
**thresh**$  $dimensions:voltage

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````
`````


## iafCell




extends *baseiafcapcell*



Integrate and fire cell with capacitance **C,** **leakConductance** and **leakReversal**.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**leakConductance**$  $dimensions:conductance
**leakReversal**$  $dimensions:voltage
**reset**$  *(from baseiafcapcell)* $dimensions:voltage
**thresh**$  *(from baseiafcapcell)* $dimensions:voltage

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)









**On Start**
: **v** = leakReversal



**On Conditions**

: IF v &gt; thresh THEN
: **v** = reset
: EVENT OUT on port: **spike**





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iMemb** =&nbsp;leakConductance * (leakReversal - v) + iSyn(exposed as **iMemb**)
    





**Time Derivatives**
    : d **v** /dt = iMemb / C
    

````

```` Schema
``` xml
<xs:complexType name="IafCell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="leakReversal" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="thresh" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="reset" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="C" type="Nml2Quantity_capacitance" use="required"/>
      <xs:attribute name="leakConductance" type="Nml2Quantity_conductance" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IafCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IafCell
from neuroml.utils import component_factory

variable = component_factory(
    IafCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    leak_reversal: 'a Nml2Quantity_voltage (required)' = None,
    thresh: 'a Nml2Quantity_voltage (required)' = None,
    reset: 'a Nml2Quantity_voltage (required)' = None,
    C: 'a Nml2Quantity_capacitance (required)' = None,
    leak_conductance: 'a Nml2Quantity_conductance (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<iafCell id="iaf" leakReversal="-50mV" thresh="-55mV" reset="-70mV" C="0.2nF" leakConductance="0.01uS"/>
```
``` xml
<iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
```
``` xml
<iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
```
````
`````


## iafRefCell




extends iafcell



Integrate and fire cell with capacitance **C,** **leakConductance,** **leakReversal** and refractory period **refract**.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**leakConductance**$  *(from iafcell)* $dimensions:conductance
**leakReversal**$  *(from iafcell)* $dimensions:voltage
**refract**$  $dimensions:time
**reset**$  *(from baseiafcapcell)* $dimensions:voltage
**thresh**$  *(from baseiafcapcell)* $dimensions:voltage

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = leakReversal





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iMemb** =&nbsp;leakConductance * (leakReversal - v) + iSyn(exposed as **iMemb**)
    






**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = reset
: **On Conditions**
:  IF t &gt; lastSpikeTime + refract THEN
: TRANSITION to REGIME **integrating**

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; thresh THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = iMemb / C
````

```` Schema
``` xml
<xs:complexType name="IafRefCell">
  <xs:complexContent>
    <xs:extension base="IafCell">
      <xs:attribute name="refract" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IafRefCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IafRefCell
from neuroml.utils import component_factory

variable = component_factory(
    IafRefCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    leak_reversal: 'a Nml2Quantity_voltage (required)' = None,
    thresh: 'a Nml2Quantity_voltage (required)' = None,
    reset: 'a Nml2Quantity_voltage (required)' = None,
    C: 'a Nml2Quantity_capacitance (required)' = None,
    leak_conductance: 'a Nml2Quantity_conductance (required)' = None,
    refract: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<iafRefCell id="iafRef" leakReversal="-50mV" thresh="-55mV" reset="-70mV" C="0.2nF" leakConductance="0.01uS" refract="5ms"/>
```
````
`````


## izhikevichCell




extends *basecellmembpot*



Cell based on the 2003 model of Izhikevich, see http://izhikevich.org/publications/spikes.htm.


`````
```` Parameters
```Table:

**a**$ Time scale of the recovery variable U $Dimensionless
**b**$ Sensitivity of U to the subthreshold fluctuations of the membrane potential V $Dimensionless
**c**$ After-spike reset value of V $Dimensionless
**d**$ After-spike increase to U $Dimensionless
**thresh**$ Spike threshold $dimensions:voltage
**v0**$ Initial membrane potential $dimensions:voltage

```
````

```` Constants
```Table:

**MSEC** = 1ms$  $ dimensions:time
**MVOLT** = 1mV$  $ dimensions:voltage

```
````

```` Exposures
```Table:

**U**$ Membrane recovery variable $Dimensionless
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrentdl

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **U**: Dimensionless (exposed as **U**)









**On Start**
: **v** = v0
: **U** = v0 * b / MVOLT



**On Conditions**

: IF v &gt; thresh THEN
: **v** = c * MVOLT
: **U** = U + d
: EVENT OUT on port: **spike**





**Derived Variables**
    : **ISyn** =&nbsp;synapses[*]->I(reduce method: add)
    





**Time Derivatives**
    : d **v** /dt = (0.04 * v^2 / MVOLT + 5 * v + (140.0 - U + ISyn) * MVOLT)/MSEC
    : d **U** /dt = a * (b * v / MVOLT - U) / MSEC
    

````

```` Schema
``` xml
<xs:complexType name="IzhikevichCell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="v0" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="thresh" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="a" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="b" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="c" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="d" type="Nml2Quantity_none" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IzhikevichCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IzhikevichCell
from neuroml.utils import component_factory

variable = component_factory(
    IzhikevichCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    v0: 'a Nml2Quantity_voltage (required)' = None,
    thresh: 'a Nml2Quantity_voltage (required)' = None,
    a: 'a Nml2Quantity_none (required)' = None,
    b: 'a Nml2Quantity_none (required)' = None,
    c: 'a Nml2Quantity_none (required)' = None,
    d: 'a Nml2Quantity_none (required)' = None,
)
```
````
```` Usage: XML
``` xml
<izhikevichCell id="izBurst" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-50.0" d="2"/>
```
````
`````


## izhikevich2007Cell




extends *basecellmembpotcap*



Cell based on the modified Izhikevich model in Izhikevich 2007, Dynamical systems in neuroscience, MIT Press.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**a**$ Time scale of recovery variable u $dimensions:per_time
**b**$ Sensitivity of recovery variable u to subthreshold fluctuations of membrane potential v $dimensions:conductance
**c**$ After-spike reset value of v $dimensions:voltage
**d**$ After-spike increase to u $dimensions:current
**k**$  $dimensions:conductance_per_voltage
**v0**$ Initial membrane potential $dimensions:voltage
**vpeak**$ Peak action potential value $dimensions:voltage
**vr**$ Resting membrane potential $dimensions:voltage
**vt**$ Spike threshold $dimensions:voltage

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**u**$ Membrane recovery variable $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **u**: dimensions:current (exposed as **u**)









**On Start**
: **v** = v0
: **u** = 0



**On Conditions**

: IF v &gt; vpeak THEN
: **v** = c
: **u** = u + d
: EVENT OUT on port: **spike**





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iMemb** =&nbsp;k * (v-vr) * (v-vt) + iSyn - u(exposed as **iMemb**)
    





**Time Derivatives**
    : d **v** /dt = iMemb / C
    : d **u** /dt = a * (b * (v-vr) - u)
    

````

```` Schema
``` xml
<xs:complexType name="Izhikevich2007Cell">
  <xs:complexContent>
    <xs:extension base="BaseCellMembPotCap">
      <xs:attribute name="v0" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="k" type="Nml2Quantity_conductancePerVoltage" use="required"/>
      <xs:attribute name="vr" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="vt" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="vpeak" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="a" type="Nml2Quantity_pertime" use="required"/>
      <xs:attribute name="b" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="c" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="d" type="Nml2Quantity_current" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Izhikevich2007Cell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Izhikevich2007Cell
from neuroml.utils import component_factory

variable = component_factory(
    Izhikevich2007Cell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    C: 'a Nml2Quantity_capacitance (required)' = None,
    v0: 'a Nml2Quantity_voltage (required)' = None,
    k: 'a Nml2Quantity_conductancePerVoltage (required)' = None,
    vr: 'a Nml2Quantity_voltage (required)' = None,
    vt: 'a Nml2Quantity_voltage (required)' = None,
    vpeak: 'a Nml2Quantity_voltage (required)' = None,
    a: 'a Nml2Quantity_pertime (required)' = None,
    b: 'a Nml2Quantity_conductance (required)' = None,
    c: 'a Nml2Quantity_voltage (required)' = None,
    d: 'a Nml2Quantity_current (required)' = None,
)
```
````
```` Usage: XML
``` xml
<izhikevich2007Cell id="iz2007RS" v0="-60mV" C="100 pF" k="0.7 nS_per_mV" vr="-60 mV" vt="-40 mV" vpeak="35 mV" a="0.03 per_ms" b="-2 nS" c="-50 mV" d="100 pA"/>
```
````
`````


## adExIaFCell




extends *basecellmembpotcap*



Model based on Brette R and Gerstner W ( 2005 ) Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**EL**$ Leak reversal potential $dimensions:voltage
**VT**$ Spike threshold $dimensions:voltage
**a**$ Sub-threshold adaptation variable $dimensions:conductance
**b**$ Spike-triggered adaptation variable $dimensions:current
**delT**$ Slope factor $dimensions:voltage
**gL**$ Leak conductance $dimensions:conductance
**refract**$ Refractory period $dimensions:time
**reset**$ Reset potential $dimensions:voltage
**tauw**$ Adaptation time constant $dimensions:time
**thresh**$ Spike detection threshold $dimensions:voltage

```
````

```` Exposures
```Table:

**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage
**w**$ Adaptation current $dimensions:current

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **w**: dimensions:current (exposed as **w**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = EL
: **w** = 0





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iMemb** =&nbsp;-1 * gL * (v - EL) + gL * delT * exp((v - VT) / delT) - w + iSyn(exposed as **iMemb**)
    






**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = reset
:  **w** = w + b
: **On Conditions**
:  IF t &gt; lastSpikeTime + refract THEN
: TRANSITION to REGIME **integrating**
: **Time Derivatives**
:  d **w** /dt = (a * (v - EL) - w) / tauw

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; thresh THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = iMemb / C
:  d **w** /dt = (a * (v - EL) - w) / tauw
````

```` Schema
``` xml
<xs:complexType name="AdExIaFCell">
  <xs:complexContent>
    <xs:extension base="BaseCellMembPotCap">
      <xs:attribute name="gL" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="EL" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="reset" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="VT" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="thresh" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="delT" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="tauw" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="refract" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="a" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="b" type="Nml2Quantity_current" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=AdExIaFCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import AdExIaFCell
from neuroml.utils import component_factory

variable = component_factory(
    AdExIaFCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    C: 'a Nml2Quantity_capacitance (required)' = None,
    g_l: 'a Nml2Quantity_conductance (required)' = None,
    EL: 'a Nml2Quantity_voltage (required)' = None,
    reset: 'a Nml2Quantity_voltage (required)' = None,
    VT: 'a Nml2Quantity_voltage (required)' = None,
    thresh: 'a Nml2Quantity_voltage (required)' = None,
    del_t: 'a Nml2Quantity_voltage (required)' = None,
    tauw: 'a Nml2Quantity_time (required)' = None,
    refract: 'a Nml2Quantity_time (required)' = None,
    a: 'a Nml2Quantity_conductance (required)' = None,
    b: 'a Nml2Quantity_current (required)' = None,
)
```
````
```` Usage: XML
``` xml
<adExIaFCell id="adExBurst" C="281pF" gL="30nS" EL="-70.6mV" reset="-48.5mV" VT="-50.4mV" thresh="-40.4mV" refract="0ms" delT="2mV" tauw="40ms" a="4nS" b="0.08nA"/>
```
````
`````


## fitzHughNagumoCell




extends *basecellmembpotdl*



Simple dimensionless model of spiking cell from FitzHugh and Nagumo. Superseded by **fitzHughNagumo1969Cell** ( See https://github.com/NeuroML/NeuroML2/issues/42 ).


`````
```` Parameters
```Table:

**I**$  $Dimensionless

```
````

```` Constants
```Table:

**SEC** = 1s$  $ dimensions:time

```
````

```` Exposures
```Table:

**V**$ Membrane potential *(from basecellmembpotdl)* $Dimensionless
**W**$  $Dimensionless

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **V**: Dimensionless (exposed as **V**)
: **W**: Dimensionless (exposed as **W**)










**Time Derivatives**
    : d **V** /dt = ( (V - ((V^3) / 3)) - W + I) / SEC
    : d **W** /dt = (0.08 * (V + 0.7 - 0.8 * W)) / SEC
    

````

```` Schema
``` xml
<xs:complexType name="FitzHughNagumoCell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="I" type="Nml2Quantity_none" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=FitzHughNagumoCell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import FitzHughNagumoCell
from neuroml.utils import component_factory

variable = component_factory(
    FitzHughNagumoCell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    I: 'a Nml2Quantity_none (required)' = None,
)
```
````
```` Usage: XML
``` xml
<fitzHughNagumoCell id="fn1" I="0.8"/>
```
````
`````


## pinskyRinzelCA3Cell




extends *basecellmembpot*



Reduced CA3 cell model from Pinsky, P.F., Rinzel, J. Intrinsic and network rhythmogenesis in a reduced traub model for CA3 neurons. J Comput Neurosci 1, 39-60 ( 1994 ). See https://github.com/OpenSourceBrain/PinskyRinzelModel.


`````
```` Parameters
```Table:

**alphac**$  $Dimensionless
**betac**$  $Dimensionless
**cm**$  $dimensions:specificCapacitance
**eCa**$  $dimensions:voltage
**eK**$  $dimensions:voltage
**eL**$  $dimensions:voltage
**eNa**$  $dimensions:voltage
**gAmpa**$  $dimensions:conductanceDensity
**gCa**$  $dimensions:conductanceDensity
**gKC**$  $dimensions:conductanceDensity
**gKahp**$  $dimensions:conductanceDensity
**gKdr**$  $dimensions:conductanceDensity
**gLd**$  $dimensions:conductanceDensity
**gLs**$  $dimensions:conductanceDensity
**gNa**$  $dimensions:conductanceDensity
**gNmda**$  $dimensions:conductanceDensity
**gc**$  $dimensions:conductanceDensity
**iDend**$  $dimensions:currentDensity
**iSoma**$  $dimensions:currentDensity
**pp**$  $Dimensionless
**qd0**$  $Dimensionless

```
````

```` Constants
```Table:

**MSEC** = 1 ms$  $ dimensions:time
**MVOLT** = 1 mV$  $ dimensions:voltage
**UAMP_PER_CM2** = 1 uA_per_cm2$  $ dimensions:currentDensity
**Smax** = 125.0$  $ Dimensionless
**Vsyn** = 60.0 mV$  $ dimensions:voltage
**betaqd** = 0.001$  $ Dimensionless

```
````

```` Exposures
```Table:

**Cad**$  $Dimensionless
**ICad**$  $dimensions:currentDensity
**Si**$  $Dimensionless
**Vd**$ Dendritic membrane potential $dimensions:voltage
**Vs**$ Somatic membrane potential $dimensions:voltage
**Wi**$  $Dimensionless
**cd**$  $Dimensionless
**hs**$  $Dimensionless
**ns**$  $Dimensionless
**qd**$  $Dimensionless
**sd**$  $Dimensionless
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **Vs**: dimensions:voltage (exposed as **Vs**)
: **Vd**: dimensions:voltage (exposed as **Vd**)
: **Cad**: Dimensionless (exposed as **Cad**)
: **hs**: Dimensionless (exposed as **hs**)
: **ns**: Dimensionless (exposed as **ns**)
: **sd**: Dimensionless (exposed as **sd**)
: **cd**: Dimensionless (exposed as **cd**)
: **qd**: Dimensionless (exposed as **qd**)
: **Si**: Dimensionless (exposed as **Si**)
: **Wi**: Dimensionless (exposed as **Wi**)
: **Sisat**: Dimensionless 









**On Start**
: **Vs** = eL
: **Vd** = eL
: **qd** = qd0





**Derived Variables**
    : **v** =&nbsp;Vs(exposed as **v**)
    : **ICad** =&nbsp;gCa*sd*sd*(Vd-eCa)(exposed as **ICad**)
    : **alphams_Vs** =&nbsp;0.32*(-46.9-Vs/MVOLT)/(exp((-46.9-Vs/MVOLT)/4.0)-1.0)
    : **betams_Vs** =&nbsp;0.28*(Vs/MVOLT+19.9)/(exp((Vs/MVOLT+19.9)/5.0)-1.0)
    : **Minfs_Vs** =&nbsp;alphams_Vs/(alphams_Vs+betams_Vs)
    : **alphans_Vs** =&nbsp;0.016*(-24.9-Vs/MVOLT)/(exp((-24.9-Vs/MVOLT)/5.0)-1.0)
    : **betans_Vs** =&nbsp;0.25*exp(-1.0-0.025*Vs/MVOLT)
    : **alphahs_Vs** =&nbsp;0.128*exp((-43.0-Vs/MVOLT)/18.0)
    : **betahs_Vs** =&nbsp;4.0/(1.0+exp((-20.0-Vs/MVOLT)/5.0))
    : **alphasd_Vd** =&nbsp;1.6/(1.0+exp(-0.072*(Vd/MVOLT-5.0)))
    : **betasd_Vd** =&nbsp;0.02*(Vd/MVOLT+8.9)/(exp((Vd/MVOLT+8.9)/5.0)-1.0)
    : **Iampa** =&nbsp;gAmpa*Wi*(Vd-Vsyn)
    : **Inmda** =&nbsp;gNmda*Sisat*(Vd-Vsyn)/(1.0+0.28*exp(-0.062*(Vd/MVOLT-60.0)))
    : **Isyn** =&nbsp;Iampa+Inmda
    



**Conditional Derived Variables**
    
: IF 0.00002*Cad &gt; 0.01 THEN
:  **alphaqd** = 0.01 
: OTHERWISE
:  **alphaqd** = 0.00002\*Cad 
: IF Cad/250 &gt; 1 THEN
:  **chid** = 1 
: OTHERWISE
:  **chid** = Cad/250 
: IF Vd &lt; -10*MVOLT THEN
:  **alphacd_Vd** = exp((Vd/MVOLT+50.0)/11-(Vd/MVOLT+53.5)/27)/18.975 
: OTHERWISE
:  **alphacd_Vd** = 2.0\*exp((-53.5-Vd/MVOLT)/27.0) 
: IF Vd &lt; -10*MVOLT THEN
:  **betacd_Vd** = (2.0\*exp((-53.5-Vd/MVOLT)/27.0)-alphacd_Vd) 
: OTHERWISE
:  **betacd_Vd** = 0 
: IF Si &gt; Smax THEN
:  **Sisat** = Smax 
: OTHERWISE
:  **Sisat** = Si 


**Time Derivatives**
    : d **Vs** /dt = (-gLs*(Vs-eL)-gNa*(Minfs_Vs^2)*hs*(Vs-eNa)-gKdr*ns*(Vs-eK)+(gc/pp)*(Vd-Vs)+iSoma/pp) / cm
    : d **Vd** /dt = (iDend/(1.0-pp)-Isyn/(1.0-pp)-gLd*(Vd-eL)-ICad-gKahp*qd*(Vd-eK)-gKC*cd*chid*(Vd-eK)+(gc*(Vs-Vd))/(1.0-pp)) / cm
    : d **Cad** /dt = (-0.13*ICad/UAMP_PER_CM2-0.075*Cad) / MSEC
    : d **hs** /dt = (alphahs_Vs-(alphahs_Vs+betahs_Vs)*hs) / MSEC
    : d **ns** /dt = (alphans_Vs-(alphans_Vs+betans_Vs)*ns) / MSEC
    : d **sd** /dt = (alphasd_Vd-(alphasd_Vd+betasd_Vd)*sd) / MSEC
    : d **cd** /dt = (alphacd_Vd-(alphacd_Vd+betacd_Vd)*cd) / MSEC
    : d **qd** /dt = (alphaqd-(alphaqd+betaqd)*qd) / MSEC
    : d **Si** /dt = -Si/150.0
    : d **Wi** /dt = -Wi/2.0
    

````

```` Schema
``` xml
<xs:complexType name="PinskyRinzelCA3Cell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="iSoma" type="Nml2Quantity_currentDensity" use="required"/>
      <xs:attribute name="iDend" type="Nml2Quantity_currentDensity" use="required"/>
      <xs:attribute name="gc" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gLs" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gLd" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gNa" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gKdr" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gCa" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gKahp" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gKC" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gNmda" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="gAmpa" type="Nml2Quantity_conductanceDensity" use="required"/>
      <xs:attribute name="eNa" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="eCa" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="eK" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="eL" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="qd0" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="pp" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="alphac" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="betac" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="cm" type="Nml2Quantity_specificCapacitance" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=PinskyRinzelCA3Cell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import PinskyRinzelCA3Cell
from neuroml.utils import component_factory

variable = component_factory(
    PinskyRinzelCA3Cell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    i_soma: 'a Nml2Quantity_currentDensity (required)' = None,
    i_dend: 'a Nml2Quantity_currentDensity (required)' = None,
    gc: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_ls: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_ld: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_na: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_kdr: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_ca: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_kahp: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_kc: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_nmda: 'a Nml2Quantity_conductanceDensity (required)' = None,
    g_ampa: 'a Nml2Quantity_conductanceDensity (required)' = None,
    e_na: 'a Nml2Quantity_voltage (required)' = None,
    e_ca: 'a Nml2Quantity_voltage (required)' = None,
    e_k: 'a Nml2Quantity_voltage (required)' = None,
    e_l: 'a Nml2Quantity_voltage (required)' = None,
    qd0: 'a Nml2Quantity_none (required)' = None,
    pp: 'a Nml2Quantity_none (required)' = None,
    alphac: 'a Nml2Quantity_none (required)' = None,
    betac: 'a Nml2Quantity_none (required)' = None,
    cm: 'a Nml2Quantity_specificCapacitance (required)' = None,
)
```
````
```` Usage: XML
``` xml
<pinskyRinzelCA3Cell id="pr2A" iSoma="0.75 uA_per_cm2" iDend="0 uA_per_cm2" gc="2.1 mS_per_cm2" qd0="0" gLs="0.1 mS_per_cm2" gLd="0.1 mS_per_cm2" gNa="30 mS_per_cm2" gKdr="15 mS_per_cm2" gCa="10 mS_per_cm2" gKahp="0.8 mS_per_cm2" gKC="15 mS_per_cm2" eNa="60 mV" eCa="80 mV" eK="-75 mV" eL="-60 mV" pp="0.5" cm="3 uF_per_cm2" alphac="2" betac="0.1" gNmda="0 mS_per_cm2" gAmpa="0 mS_per_cm2"/>
```
````
`````


## hindmarshRose1984Cell




extends *basecellmembpotcap*



The Hindmarsh Rose model is a simplified point cell model which captures complex firing patterns of single neurons, such as periodic and chaotic bursting. It has a fast spiking subsystem, which is a generalization of the FitzHugh-Nagumo system, coupled to a slower subsystem which allows the model to fire bursts. The dynamical variables x, y, z correspond to the membrane potential, a recovery variable, and a slower adaptation current, respectively. See Hindmarsh J. L., and Rose R. M. ( 1984 ) A model of neuronal bursting using three coupled first order differential equations. Proc. R. Soc. London, Ser. B 221:87–102.


`````
```` Parameters
```Table:

**C**$ Total capacitance of the cell membrane *(from basecellmembpotcap)* $dimensions:capacitance
**a**$ cubic term in x nullcline $Dimensionless
**b**$ quadratic term in x nullcline $Dimensionless
**c**$ constant term in y nullcline $Dimensionless
**d**$ quadratic term in y nullcline $Dimensionless
**r**$ timescale separation between slow and fast subsystem (r greater than 0; r much less than 1) $Dimensionless
**s**$ related to adaptation $Dimensionless
**v_scaling**$ scaling of x for physiological membrane potential $dimensions:voltage
**x0**$  $Dimensionless
**x1**$ related to the system's resting potential $Dimensionless
**y0**$  $Dimensionless
**z0**$  $Dimensionless

```
````

```` Constants
```Table:

**MSEC** = 1ms$  $ dimensions:time

```
````

```` Exposures
```Table:

**chi**$  $Dimensionless
**iMemb**$ Total current crossing the cell membrane *(from basecellmembpotcap)* $dimensions:current
**iSyn**$ Total current due to synaptic inputs *(from basecellmembpotcap)* $dimensions:current
**phi**$  $Dimensionless
**rho**$  $Dimensionless
**spiking**$  $Dimensionless
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage
**x**$  $Dimensionless
**y**$  $Dimensionless
**z**$  $Dimensionless

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out

```
````

```` Attachments
```Table:

**synapses**$  $ basepointcurrent

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **y**: Dimensionless (exposed as **y**)
: **z**: Dimensionless (exposed as **z**)
: **spiking**: Dimensionless (exposed as **spiking**)









**On Start**
: **v** = x0 * v_scaling
: **y** = y0
: **z** = z0



**On Conditions**

: IF v &gt; 0 AND spiking &lt; 0.5 THEN
: **spiking** = 1
: EVENT OUT on port: **spike**

: IF v &lt; 0 THEN
: **spiking** = 0





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **x** =&nbsp;v / v_scaling(exposed as **x**)
    : **phi** =&nbsp;y - a * x^3 + b * x^2(exposed as **phi**)
    : **chi** =&nbsp;c - d * x^2 - y(exposed as **chi**)
    : **rho** =&nbsp;s * ( x - x1 ) - z(exposed as **rho**)
    : **iMemb** =&nbsp;(C * (v_scaling * (phi - z) / MSEC)) + iSyn(exposed as **iMemb**)
    





**Time Derivatives**
    : d **v** /dt = iMemb/C
    : d **y** /dt = chi / MSEC
    : d **z** /dt = r * rho / MSEC
    

````

```` Schema
``` xml
<xs:complexType name="HindmarshRose1984Cell">
  <xs:complexContent>
    <xs:extension base="BaseCellMembPotCap">
      <xs:attribute name="a" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="b" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="c" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="d" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="s" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="x1" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="r" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="x0" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="y0" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="z0" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="v_scaling" type="Nml2Quantity_voltage" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=HindmarshRose1984Cell" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import HindmarshRose1984Cell
from neuroml.utils import component_factory

variable = component_factory(
    HindmarshRose1984Cell,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    C: 'a Nml2Quantity_capacitance (required)' = None,
    a: 'a Nml2Quantity_none (required)' = None,
    b: 'a Nml2Quantity_none (required)' = None,
    c: 'a Nml2Quantity_none (required)' = None,
    d: 'a Nml2Quantity_none (required)' = None,
    s: 'a Nml2Quantity_none (required)' = None,
    x1: 'a Nml2Quantity_none (required)' = None,
    r: 'a Nml2Quantity_none (required)' = None,
    x0: 'a Nml2Quantity_none (required)' = None,
    y0: 'a Nml2Quantity_none (required)' = None,
    z0: 'a Nml2Quantity_none (required)' = None,
    v_scaling: 'a Nml2Quantity_voltage (required)' = None,
)
```
````
`````

# Channels

**Defines voltage ( and concentration ) gated ion channel models. Ion channels will generally extend  baseionchannel. The most commonly used voltage dependent gate will extend  basegate.**

---


Original ComponentType definitions: [Channels.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//Channels.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## *baseVoltageDepRate*




Base ComponentType for voltage dependent rate. Produces a time varying rate **r** which depends on **v.**.


`````
```` Exposures
```Table:

**r**$  $dimensions:per_time

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````
`````


## *baseVoltageConcDepRate*




extends *basevoltagedeprate*



Base ComponentType for voltage and concentration dependent rate. Produces a time varying rate **r** which depends on **v** and **caConc.**.


`````
```` Exposures
```Table:

**r**$  *(from basevoltagedeprate)* $dimensions:per_time

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**v**$  *(from basevoltagedeprate)* $dimensions:voltage

```
````
`````


## *baseHHRate*




extends *basevoltagedeprate*



Base ComponentType for rate which follow one of the typical forms for rate equations in the standard HH formalism, using the parameters **rate,** **midpoint** and **scale**.


`````
```` Parameters
```Table:

**midpoint**$  $dimensions:voltage
**rate**$  $dimensions:per_time
**scale**$  $dimensions:voltage

```
````

```` Exposures
```Table:

**r**$  *(from basevoltagedeprate)* $dimensions:per_time

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedeprate)* $dimensions:voltage

```
````

```` Schema
``` xml
<xs:complexType name="HHRate">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="Nml2Quantity_pertime" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## HHExpRate




extends *basehhrate*



Exponential form for rate equation ( Q: Should these be renamed hhExpRate, etc? ).


`````
```` Parameters
```Table:

**midpoint**$  *(from basehhrate)* $dimensions:voltage
**rate**$  *(from basehhrate)* $dimensions:per_time
**scale**$  *(from basehhrate)* $dimensions:voltage

```
````

```` Exposures
```Table:

**r**$  *(from basevoltagedeprate)* $dimensions:per_time

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedeprate)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **r** =&nbsp;rate * exp((v - midpoint)/scale)(exposed as **r**)
    





````

```` Schema
``` xml
<xs:complexType name="HHRate">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="Nml2Quantity_pertime" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## HHSigmoidRate




extends *basehhrate*



Sigmoidal form for rate equation.


`````
```` Parameters
```Table:

**midpoint**$  *(from basehhrate)* $dimensions:voltage
**rate**$  *(from basehhrate)* $dimensions:per_time
**scale**$  *(from basehhrate)* $dimensions:voltage

```
````

```` Exposures
```Table:

**r**$  *(from basevoltagedeprate)* $dimensions:per_time

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedeprate)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **r** =&nbsp;rate / (1 + exp(0 - (v - midpoint)/scale))(exposed as **r**)
    





````

```` Schema
``` xml
<xs:complexType name="HHRate">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="Nml2Quantity_pertime" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## HHExpLinearRate




extends *basehhrate*



Exponential linear form for rate equation. Linear for large positive **v,** exponentially decays for large negative **v.**.


`````
```` Parameters
```Table:

**midpoint**$  *(from basehhrate)* $dimensions:voltage
**rate**$  *(from basehhrate)* $dimensions:per_time
**scale**$  *(from basehhrate)* $dimensions:voltage

```
````

```` Exposures
```Table:

**r**$  *(from basevoltagedeprate)* $dimensions:per_time

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedeprate)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **x** =&nbsp;(v - midpoint) / scale
    



**Conditional Derived Variables**
    
: IF x != 0 THEN
:  **r** = rate \* x / (1 - exp(0 - x)) (exposed as **r**)
: IF x = 0 THEN
:  **r** = rate (exposed as **r**)


````

```` Schema
``` xml
<xs:complexType name="HHRate">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="Nml2Quantity_pertime" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## *baseVoltageDepVariable*




Base ComponentType for voltage dependent variable **x,** which depends on **v.** Can be used for inf/steady state of rate variable.


`````
```` Exposures
```Table:

**x**$  $Dimensionless

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````
`````


## *baseVoltageConcDepVariable*




extends *basevoltagedepvariable*



Base ComponentType for voltage and calcium concentration dependent variable **x,** which depends on **v** and **caConc.**.


`````
```` Exposures
```Table:

**x**$  *(from basevoltagedepvariable)* $Dimensionless

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**v**$  *(from basevoltagedepvariable)* $dimensions:voltage

```
````
`````


## *baseHHVariable*




extends *basevoltagedepvariable*



Base ComponentType for voltage dependent dimensionless variable which follow one of the typical forms for variable equations in the standard HH formalism, using the parameters **rate,** **midpoint,** **scale**.


`````
```` Parameters
```Table:

**midpoint**$  $dimensions:voltage
**rate**$  $Dimensionless
**scale**$  $dimensions:voltage

```
````

```` Exposures
```Table:

**x**$  *(from basevoltagedepvariable)* $Dimensionless

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedepvariable)* $dimensions:voltage

```
````

```` Schema
``` xml
<xs:complexType name="HHVariable">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="xs:float" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## HHExpVariable




extends *basehhvariable*



Exponential form for variable equation.


`````
```` Parameters
```Table:

**midpoint**$  *(from basehhvariable)* $dimensions:voltage
**rate**$  *(from basehhvariable)* $Dimensionless
**scale**$  *(from basehhvariable)* $dimensions:voltage

```
````

```` Exposures
```Table:

**x**$  *(from basevoltagedepvariable)* $Dimensionless

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedepvariable)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **x** =&nbsp;rate * exp((v - midpoint)/scale)(exposed as **x**)
    





````

```` Schema
``` xml
<xs:complexType name="HHVariable">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="xs:float" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## HHSigmoidVariable




extends *basehhvariable*



Sigmoidal form for variable equation.


`````
```` Parameters
```Table:

**midpoint**$  *(from basehhvariable)* $dimensions:voltage
**rate**$  *(from basehhvariable)* $Dimensionless
**scale**$  *(from basehhvariable)* $dimensions:voltage

```
````

```` Exposures
```Table:

**x**$  *(from basevoltagedepvariable)* $Dimensionless

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedepvariable)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **x** =&nbsp;rate / (1 + exp(0 - (v - midpoint)/scale))(exposed as **x**)
    





````

```` Schema
``` xml
<xs:complexType name="HHVariable">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="xs:float" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## HHExpLinearVariable




extends *basehhvariable*



Exponential linear form for variable equation. Linear for large positive **v,** exponentially decays for large negative **v.**.


`````
```` Parameters
```Table:

**midpoint**$  *(from basehhvariable)* $dimensions:voltage
**rate**$  *(from basehhvariable)* $Dimensionless
**scale**$  *(from basehhvariable)* $dimensions:voltage

```
````

```` Exposures
```Table:

**x**$  *(from basevoltagedepvariable)* $Dimensionless

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedepvariable)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **a** =&nbsp;(v - midpoint) / scale
    : **x** =&nbsp;rate * a / (1 - exp(0 - a))(exposed as **x**)
    





````

```` Schema
``` xml
<xs:complexType name="HHVariable">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="type" type="NmlId" use="required"/>
      <xs:attribute name="rate" type="xs:float" use="optional"/>
      <xs:attribute name="midpoint" type="Nml2Quantity_voltage" use="optional"/>
      <xs:attribute name="scale" type="Nml2Quantity_voltage" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## *baseVoltageDepTime*




Base ComponentType for voltage dependent ComponentType producing value **t** with dimension time ( e.g. for time course of rate variable ). Note: time course would not normally be fit to exp/sigmoid etc.


`````
```` Exposures
```Table:

**t**$  $dimensions:time

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````
`````


## *baseVoltageConcDepTime*




extends *basevoltagedeptime*



Base type for voltage and calcium concentration dependent ComponentType producing value **t** with dimension time ( e.g. for time course of rate variable ).


`````
```` Exposures
```Table:

**t**$  *(from basevoltagedeptime)* $dimensions:time

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**v**$  *(from basevoltagedeptime)* $dimensions:voltage

```
````
`````


## fixedTimeCourse




extends *basevoltagedeptime*



Time course of a fixed magnitude **tau** which can be used for the time course in  gatehhtauinf,  gatehhratestau or  gatehhratestauinf.


`````
```` Parameters
```Table:

**tau**$  $dimensions:time

```
````

```` Exposures
```Table:

**t**$  *(from basevoltagedeptime)* $dimensions:time

```
````

```` Requirements
```Table:

**v**$  *(from basevoltagedeptime)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **t** =&nbsp;tau(exposed as **t**)
    





````
`````


## *baseQ10Settings*




Base ComponentType for a scaling to apply to gating variable time course, usually temperature dependent.


`````
```` Exposures
```Table:

**q10**$  $Dimensionless

```
````

```` Requirements
```Table:

**temperature**$  $dimensions:temperature

```
````

```` Schema
``` xml
<xs:complexType name="Q10Settings">
  <xs:attribute name="type" type="NmlId" use="required"/>
  <xs:attribute name="fixedQ10" type="Nml2Quantity_none" use="optional"/>
  <xs:attribute name="q10Factor" type="Nml2Quantity_none" use="optional"/>
  <xs:attribute name="experimentalTemp" type="Nml2Quantity_temperature" use="optional"/>
</xs:complexType>

```
````
`````


## q10Fixed




extends *baseq10settings*



A fixed value, **fixedQ10,** for the scaling of the time course of the gating variable.


`````
```` Parameters
```Table:

**fixedQ10**$  $Dimensionless

```
````

```` Exposures
```Table:

**q10**$  *(from baseq10settings)* $Dimensionless

```
````

```` Requirements
```Table:

**temperature**$  *(from baseq10settings)* $dimensions:temperature

```
````

```` Dynamics








**Derived Variables**
    : **q10** =&nbsp;fixedQ10(exposed as **q10**)
    





````
`````


## q10ExpTemp




extends *baseq10settings*



A value for the Q10 scaling which varies as a standard function of the difference between the current temperature, **temperature,** and the temperature at which the gating variable equations were determined, **experimentalTemp**.


`````
```` Parameters
```Table:

**experimentalTemp**$  $dimensions:temperature
**q10Factor**$  $Dimensionless

```
````

```` Constants
```Table:

**TENDEGREES** = 10K$  $ dimensions:temperature

```
````

```` Exposures
```Table:

**q10**$  *(from baseq10settings)* $Dimensionless

```
````

```` Requirements
```Table:

**temperature**$  *(from baseq10settings)* $dimensions:temperature

```
````

```` Dynamics








**Derived Variables**
    : **q10** =&nbsp;q10Factor^((temperature - experimentalTemp)/TENDEGREES)(exposed as **q10**)
    





````
`````


## *baseConductanceScaling*




Base ComponentType for a scaling to apply to a gate's conductance, e.g. temperature dependent scaling.


`````
```` Exposures
```Table:

**factor**$  $Dimensionless

```
````

```` Requirements
```Table:

**temperature**$  $dimensions:temperature

```
````
`````


## q10ConductanceScaling




extends *baseconductancescaling*



A value for the conductance scaling which varies as a standard function of the difference between the current temperature, **temperature,** and the temperature at which the conductance was originally determined, **experimentalTemp**.


`````
```` Parameters
```Table:

**experimentalTemp**$  $dimensions:temperature
**q10Factor**$  $Dimensionless

```
````

```` Constants
```Table:

**TENDEGREES** = 10K$  $ dimensions:temperature

```
````

```` Exposures
```Table:

**factor**$  *(from baseconductancescaling)* $Dimensionless

```
````

```` Requirements
```Table:

**temperature**$  *(from baseconductancescaling)* $dimensions:temperature

```
````

```` Dynamics








**Derived Variables**
    : **factor** =&nbsp;q10Factor^((temperature - experimentalTemp)/TENDEGREES)(exposed as **factor**)
    





````

```` Schema
``` xml
<xs:complexType name="Q10ConductanceScaling">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="q10Factor" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="experimentalTemp" type="Nml2Quantity_temperature" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Q10ConductanceScaling" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Q10ConductanceScaling
from neuroml.utils import component_factory

variable = component_factory(
    Q10ConductanceScaling,
    q10_factor: 'a Nml2Quantity_none (required)' = None,
    experimental_temp: 'a Nml2Quantity_temperature (required)' = None,
)
```
````
`````


## *baseConductanceScalingCaDependent*




extends *baseconductancescaling*



Base ComponentType for a scaling to apply to a gate's conductance which depends on Ca concentration. Usually a generic expression of **caConc** ( so no standard, non-base form here ).


`````
```` Exposures
```Table:

**factor**$  *(from baseconductancescaling)* $Dimensionless

```
````

```` Requirements
```Table:

**caConc**$  $dimensions:concentration
**temperature**$  *(from baseconductancescaling)* $dimensions:temperature

```
````
`````


## *baseGate*




Base ComponentType for a voltage and/or concentration dependent gate.


`````
```` Parameters
```Table:

**instances**$  $Dimensionless

```
````

```` Child list
```Table:

**notes**$  $ notes

```
````

```` Exposures
```Table:

**fcond**$  $Dimensionless
**q**$  $Dimensionless

```
````

```` Schema
``` xml
<xs:complexType name="GateHHUndetermined">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="forwardRate" type="HHRate" minOccurs="0"/>
        <xs:element name="reverseRate" type="HHRate" minOccurs="0"/>
        <xs:element name="timeCourse" type="HHTime" minOccurs="0"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="0"/>
        <xs:element name="subGate" type="GateFractionalSubgate" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
      <xs:attribute name="type" type="gateTypes" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## gate




extends *basegate*



Conveniently named baseGate.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Exposures
```Table:

**fcond**$  *(from basegate)* $Dimensionless
**q**$  *(from basegate)* $Dimensionless

```
````

```` Schema
``` xml
<xs:complexType name="GateHHUndetermined">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="forwardRate" type="HHRate" minOccurs="0"/>
        <xs:element name="reverseRate" type="HHRate" minOccurs="0"/>
        <xs:element name="timeCourse" type="HHTime" minOccurs="0"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="0"/>
        <xs:element name="subGate" type="GateFractionalSubgate" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
      <xs:attribute name="type" type="gateTypes" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHUndetermined" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHUndetermined
from neuroml.utils import component_factory

variable = component_factory(
    GateHHUndetermined,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    type: 'a gateTypes (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    forward_rate: 'a HHRate (optional)' = None,
    reverse_rate: 'a HHRate (optional)' = None,
    time_course: 'a HHTime (optional)' = None,
    steady_state: 'a HHVariable (optional)' = None,
    sub_gates: 'list of GateFractionalSubgate(s) (optional)' = None,
)
```
````
`````


## gateHHrates




extends gate



Gate which follows the general Hodgkin Huxley formalism.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Child list
```Table:

**forwardRate**$  $ basevoltagedeprate
**reverseRate**$  $ basevoltagedeprate

```
````

```` Children list
```Table:

**q10Settings**$  $ baseq10settings

```
````

```` Exposures
```Table:

**alpha**$  $dimensions:per_time
**beta**$  $dimensions:per_time
**fcond**$  *(from basegate)* $Dimensionless
**inf**$  $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Dynamics



**State Variables**
: **q**: Dimensionless (exposed as **q**)









**On Start**
: **q** = inf





**Derived Variables**
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    : **alpha** =&nbsp;forwardRate->r(exposed as **alpha**)
    : **beta** =&nbsp;reverseRate->r(exposed as **beta**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    : **inf** =&nbsp;alpha/(alpha+beta)(exposed as **inf**)
    : **tau** =&nbsp;1/((alpha+beta) * rateScale)(exposed as **tau**)
    





**Time Derivatives**
    : d **q** /dt = (inf - q) / tau
    

````

```` Schema
``` xml
<xs:complexType name="GateHHRates">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="forwardRate" type="HHRate" minOccurs="1"/>
        <xs:element name="reverseRate" type="HHRate" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHRates" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHRates
from neuroml.utils import component_factory

variable = component_factory(
    GateHHRates,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    forward_rate: 'a HHRate (required)' = None,
    reverse_rate: 'a HHRate (required)' = None,
)
```
````
```` Usage: XML
``` xml
<gateHHrates id="m" instances="3">
    <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
    <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
</gateHHrates>
```
``` xml
<gateHHrates id="h" instances="1">
    <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65mV" scale="-20mV"/>
    <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="10mV"/>
</gateHHrates>
```
``` xml
<gateHHrates id="m" instances="3">
    <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
    <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
</gateHHrates>
```
````
`````


## gateHHtauInf




extends gate



Gate which follows the general Hodgkin Huxley formalism.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Child list
```Table:

**timeCourse**$  $ basevoltagedeptime
**steadyState**$  $ basevoltagedepvariable

```
````

```` Children list
```Table:

**q10Settings**$  $ baseq10settings

```
````

```` Exposures
```Table:

**fcond**$  *(from basegate)* $Dimensionless
**inf**$  $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Dynamics



**State Variables**
: **q**: Dimensionless (exposed as **q**)









**On Start**
: **q** = inf





**Derived Variables**
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    : **inf** =&nbsp;steadyState->x(exposed as **inf**)
    : **tauUnscaled** =&nbsp;timeCourse->t
    : **tau** =&nbsp;tauUnscaled / rateScale(exposed as **tau**)
    





**Time Derivatives**
    : d **q** /dt = (inf - q) / tau
    

````

```` Schema
``` xml
<xs:complexType name="GateHHTauInf">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="timeCourse" type="HHTime" minOccurs="1"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHTauInf" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHTauInf
from neuroml.utils import component_factory

variable = component_factory(
    GateHHTauInf,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    time_course: 'a HHTime (required)' = None,
    steady_state: 'a HHVariable (required)' = None,
)
```
````
`````


## gateHHInstantaneous




extends gate



Gate which follows the general Hodgkin Huxley formalism but is instantaneous, so tau = 0 and gate follows exactly inf value.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Constants
```Table:

**SEC** = 1 s$  $ dimensions:time

```
````

```` Child list
```Table:

**steadyState**$  $ basevoltagedepvariable

```
````

```` Exposures
```Table:

**fcond**$  *(from basegate)* $Dimensionless
**inf**$  $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**tau**$  $dimensions:time

```
````

```` Dynamics








**Derived Variables**
    : **inf** =&nbsp;steadyState->x(exposed as **inf**)
    : **tau** =&nbsp;0 * SEC(exposed as **tau**)
    : **q** =&nbsp;inf(exposed as **q**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    





````

```` Schema
``` xml
<xs:complexType name="GateHHInstantaneous">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHInstantaneous" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHInstantaneous
from neuroml.utils import component_factory

variable = component_factory(
    GateHHInstantaneous,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    steady_state: 'a HHVariable (required)' = None,
)
```
````
`````


## gateHHratesTau




extends gate



Gate which follows the general Hodgkin Huxley formalism.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Child list
```Table:

**forwardRate**$  $ basevoltagedeprate
**reverseRate**$  $ basevoltagedeprate
**timeCourse**$  $ basevoltagedeptime

```
````

```` Children list
```Table:

**q10Settings**$  $ baseq10settings

```
````

```` Exposures
```Table:

**alpha**$  $dimensions:per_time
**beta**$  $dimensions:per_time
**fcond**$  *(from basegate)* $Dimensionless
**inf**$  $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Dynamics



**State Variables**
: **q**: Dimensionless (exposed as **q**)









**On Start**
: **q** = inf





**Derived Variables**
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    : **alpha** =&nbsp;forwardRate->r(exposed as **alpha**)
    : **beta** =&nbsp;reverseRate->r(exposed as **beta**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    : **inf** =&nbsp;alpha/(alpha+beta)(exposed as **inf**)
    : **tauUnscaled** =&nbsp;timeCourse->t
    : **tau** =&nbsp;tauUnscaled / rateScale(exposed as **tau**)
    





**Time Derivatives**
    : d **q** /dt = (inf - q) / tau
    

````

```` Schema
``` xml
<xs:complexType name="GateHHRatesTau">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="forwardRate" type="HHRate" minOccurs="1"/>
        <xs:element name="reverseRate" type="HHRate" minOccurs="1"/>
        <xs:element name="timeCourse" type="HHTime" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHRatesTau" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHRatesTau
from neuroml.utils import component_factory

variable = component_factory(
    GateHHRatesTau,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    forward_rate: 'a HHRate (required)' = None,
    reverse_rate: 'a HHRate (required)' = None,
    time_course: 'a HHTime (required)' = None,
)
```
````
`````


## gateHHratesInf




extends gate



Gate which follows the general Hodgkin Huxley formalism.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Child list
```Table:

**forwardRate**$  $ basevoltagedeprate
**reverseRate**$  $ basevoltagedeprate
**steadyState**$  $ basevoltagedepvariable

```
````

```` Children list
```Table:

**q10Settings**$  $ baseq10settings

```
````

```` Exposures
```Table:

**alpha**$  $dimensions:per_time
**beta**$  $dimensions:per_time
**fcond**$  *(from basegate)* $Dimensionless
**inf**$  $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Dynamics



**State Variables**
: **q**: Dimensionless (exposed as **q**)









**On Start**
: **q** = inf





**Derived Variables**
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    : **alpha** =&nbsp;forwardRate->r(exposed as **alpha**)
    : **beta** =&nbsp;reverseRate->r(exposed as **beta**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    : **inf** =&nbsp;steadyState->x(exposed as **inf**)
    : **tau** =&nbsp;1/((alpha+beta) * rateScale)(exposed as **tau**)
    





**Time Derivatives**
    : d **q** /dt = (inf - q) / tau
    

````

```` Schema
``` xml
<xs:complexType name="GateHHRatesInf">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="forwardRate" type="HHRate" minOccurs="1"/>
        <xs:element name="reverseRate" type="HHRate" minOccurs="1"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHRatesInf" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHRatesInf
from neuroml.utils import component_factory

variable = component_factory(
    GateHHRatesInf,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    forward_rate: 'a HHRate (required)' = None,
    reverse_rate: 'a HHRate (required)' = None,
    steady_state: 'a HHVariable (required)' = None,
)
```
````
`````


## gateHHratesTauInf




extends gate



Gate which follows the general Hodgkin Huxley formalism.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Child list
```Table:

**forwardRate**$  $ basevoltagedeprate
**reverseRate**$  $ basevoltagedeprate
**timeCourse**$  $ basevoltagedeptime
**steadyState**$  $ basevoltagedepvariable

```
````

```` Children list
```Table:

**q10Settings**$  $ baseq10settings

```
````

```` Exposures
```Table:

**alpha**$  $dimensions:per_time
**beta**$  $dimensions:per_time
**fcond**$  *(from basegate)* $Dimensionless
**inf**$  $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Dynamics



**State Variables**
: **q**: Dimensionless (exposed as **q**)









**On Start**
: **q** = inf





**Derived Variables**
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    : **alpha** =&nbsp;forwardRate->r(exposed as **alpha**)
    : **beta** =&nbsp;reverseRate->r(exposed as **beta**)
    : **inf** =&nbsp;steadyState->x(exposed as **inf**)
    : **tauUnscaled** =&nbsp;timeCourse->t
    : **tau** =&nbsp;tauUnscaled / rateScale(exposed as **tau**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    





**Time Derivatives**
    : d **q** /dt = (inf - q) / tau
    

````

```` Schema
``` xml
<xs:complexType name="GateHHRatesTauInf">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="forwardRate" type="HHRate" minOccurs="1"/>
        <xs:element name="reverseRate" type="HHRate" minOccurs="1"/>
        <xs:element name="timeCourse" type="HHTime" minOccurs="1"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateHHRatesTauInf" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateHHRatesTauInf
from neuroml.utils import component_factory

variable = component_factory(
    GateHHRatesTauInf,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    forward_rate: 'a HHRate (required)' = None,
    reverse_rate: 'a HHRate (required)' = None,
    time_course: 'a HHTime (required)' = None,
    steady_state: 'a HHVariable (required)' = None,
)
```
````
`````


## gateFractional




extends gate



Gate composed of subgates contributing with fractional conductance.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Children list
```Table:

**q10Settings**$  $ baseq10settings
**subGate**$  $ subgate

```
````

```` Exposures
```Table:

**fcond**$  *(from basegate)* $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless

```
````

```` Dynamics








**Derived Variables**
    : **q** =&nbsp;subGate[*]->qfrac(reduce method: add)(exposed as **q**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    





````

```` Schema
``` xml
<xs:complexType name="GateFractional">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="subGate" type="GateFractionalSubgate" minOccurs="1" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateFractional" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateFractional
from neuroml.utils import component_factory

variable = component_factory(
    GateFractional,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    sub_gates: 'list of GateFractionalSubgate(s) (required)' = None,
)
```
````
`````


## subGate




Gate composed of subgates contributing with fractional conductance.


`````
```` Parameters
```Table:

**fractionalConductance**$  $Dimensionless

```
````

```` Child list
```Table:

**notes**$  $ notes
**timeCourse**$  $ basevoltagedeptime
**steadyState**$  $ basevoltagedepvariable

```
````

```` Exposures
```Table:

**inf**$  $Dimensionless
**q**$  $Dimensionless
**qfrac**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Requirements
```Table:

**rateScale**$  $Dimensionless

```
````

```` Dynamics



**State Variables**
: **q**: Dimensionless (exposed as **q**)









**On Start**
: **q** = inf





**Derived Variables**
    : **inf** =&nbsp;steadyState->x(exposed as **inf**)
    : **tauUnscaled** =&nbsp;timeCourse->t
    : **tau** =&nbsp;tauUnscaled / rateScale(exposed as **tau**)
    : **qfrac** =&nbsp;q * fractionalConductance(exposed as **qfrac**)
    





**Time Derivatives**
    : d **q** /dt = (inf - q) / tau
    

````

```` Schema
``` xml
<xs:complexType name="GateFractionalSubgate">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="steadyState" type="HHVariable" minOccurs="1"/>
        <xs:element name="timeCourse" type="HHTime" minOccurs="1"/>
      </xs:all>
      <xs:attribute name="fractionalConductance" type="Nml2Quantity_none" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateFractionalSubgate" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateFractionalSubgate
from neuroml.utils import component_factory

variable = component_factory(
    GateFractionalSubgate,
    id: 'a NmlId (required)' = None,
    fractional_conductance: 'a Nml2Quantity_none (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    steady_state: 'a HHVariable (required)' = None,
    time_course: 'a HHTime (required)' = None,
)
```
````
`````


## *baseIonChannel*




Base for all ion channel ComponentTypes.


`````
```` Parameters
```Table:

**conductance**$  $dimensions:conductance

```
````

```` Text fields
```Table:

**neuroLexId**$ 

````

```` Child list
```Table:

**notes**$  $ notes
**annotation**$  $ annotation

```
````

```` Exposures
```Table:

**fopen**$  $Dimensionless
**g**$  $dimensions:conductance

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````
`````


## ionChannelPassive




extends ionchannel



Simple passive ion channel where the constant conductance through the channel is equal to **conductance**.


`````
```` Parameters
```Table:

**conductance**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Text fields
```Table:

**species**$ 

````

```` Exposures
```Table:

**fopen**$  *(from baseionchannel)* $Dimensionless
**g**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Requirements
```Table:

**v**$  *(from baseionchannel)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **fopen** =&nbsp;1(exposed as **fopen**)
    : **g** =&nbsp;conductance(exposed as **g**)
    





````
`````


## ionChannelHH




extends *baseionchannel*



Note  ionchannel and  ionchannelhh are currently functionally identical. This is needed since many existing examples use ionChannel, some use ionChannelHH. NeuroML v2beta4 should remove one of these, probably ionChannelHH.


`````
```` Parameters
```Table:

**conductance**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Text fields
```Table:

**species**$ 

````

```` Children list
```Table:

**conductanceScaling**$  $ baseconductancescaling
**gates**$  $ gate

```
````

```` Exposures
```Table:

**fopen**$  *(from baseionchannel)* $Dimensionless
**g**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Requirements
```Table:

**v**$  *(from baseionchannel)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **conductanceScale** =&nbsp;conductanceScaling[*]->factor(reduce method: multiply)
    : **fopen0** =&nbsp;gates[*]->fcond(reduce method: multiply)
    : **fopen** =&nbsp;conductanceScale * fopen0(exposed as **fopen**)
    : **g** =&nbsp;conductance * fopen(exposed as **g**)
    





````

```` Schema
``` xml
<xs:complexType name="IonChannelHH">
  <xs:complexContent>
    <xs:extension base="IonChannel"/>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IonChannelHH" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IonChannelHH
from neuroml.utils import component_factory

variable = component_factory(
    IonChannelHH,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    q10_conductance_scalings: 'list of Q10ConductanceScaling(s) (optional)' = None,
    species: 'a NmlId (optional)' = None,
    type: 'a channelTypes (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (optional)' = None,
    gates: 'list of GateHHUndetermined(s) (optional)' = None,
    gate_hh_rates: 'list of GateHHRates(s) (optional)' = None,
    gate_h_hrates_taus: 'list of GateHHRatesTau(s) (optional)' = None,
    gate_hh_tau_infs: 'list of GateHHTauInf(s) (optional)' = None,
    gate_h_hrates_infs: 'list of GateHHRatesInf(s) (optional)' = None,
    gate_h_hrates_tau_infs: 'list of GateHHRatesTauInf(s) (optional)' = None,
    gate_hh_instantaneouses: 'list of GateHHInstantaneous(s) (optional)' = None,
    gate_fractionals: 'list of GateFractional(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<ionChannelHH id="pas" conductance="10pS"/>
```
``` xml
<ionChannelHH id="HH_Na" conductance="10pS" species="na">  
        
    </ionChannelHH>
```
``` xml
<ionChannelHH id="NaConductance" conductance="10pS" species="na">
    <gateHHrates id="m" instances="3">
        <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
        <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
    </gateHHrates>
    <gateHHrates id="h" instances="1">
        <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65mV" scale="-20mV"/>
        <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="10mV"/>
    </gateHHrates>
</ionChannelHH>
```
````
`````


## ionChannel




extends ionchannelhh



Note  ionchannel and  ionchannelhh are currently functionally identical. This is needed since many existing examples use ionChannel, some use ionChannelHH. NeuroML v2beta4 should remove one of these, probably ionChannelHH.


`````
```` Parameters
```Table:

**conductance**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Exposures
```Table:

**fopen**$  *(from baseionchannel)* $Dimensionless
**g**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Requirements
```Table:

**v**$  *(from baseionchannel)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **conductanceScale** =&nbsp;conductanceScaling[*]->factor(reduce method: multiply)
    : **fopen0** =&nbsp;gates[*]->fcond(reduce method: multiply)
    : **fopen** =&nbsp;conductanceScale * fopen0(exposed as **fopen**)
    : **g** =&nbsp;conductance * fopen(exposed as **g**)
    





````

```` Schema
``` xml
<xs:complexType name="IonChannel">
  <xs:complexContent>
    <xs:extension base="IonChannelScalable">
      <xs:choice>
        <xs:element name="gate" type="GateHHUndetermined" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateHHrates" type="GateHHRates" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateHHratesTau" type="GateHHRatesTau" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateHHtauInf" type="GateHHTauInf" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateHHratesInf" type="GateHHRatesInf" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateHHratesTauInf" type="GateHHRatesTauInf" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateHHInstantaneous" type="GateHHInstantaneous" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="gateFractional" type="GateFractional" minOccurs="0" maxOccurs="unbounded"/>
      </xs:choice>
      <xs:attribute name="species" type="NmlId" use="optional"/>
      <xs:attribute name="type" type="channelTypes" use="optional"/>
      <xs:attribute name="conductance" type="Nml2Quantity_conductance" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IonChannel" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IonChannel
from neuroml.utils import component_factory

variable = component_factory(
    IonChannel,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    q10_conductance_scalings: 'list of Q10ConductanceScaling(s) (optional)' = None,
    species: 'a NmlId (optional)' = None,
    type: 'a channelTypes (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (optional)' = None,
    gates: 'list of GateHHUndetermined(s) (optional)' = None,
    gate_hh_rates: 'list of GateHHRates(s) (optional)' = None,
    gate_h_hrates_taus: 'list of GateHHRatesTau(s) (optional)' = None,
    gate_hh_tau_infs: 'list of GateHHTauInf(s) (optional)' = None,
    gate_h_hrates_infs: 'list of GateHHRatesInf(s) (optional)' = None,
    gate_h_hrates_tau_infs: 'list of GateHHRatesTauInf(s) (optional)' = None,
    gate_hh_instantaneouses: 'list of GateHHInstantaneous(s) (optional)' = None,
    gate_fractionals: 'list of GateFractional(s) (optional)' = None,
    extensiontype_=None,
)
```
````
`````


## ionChannelVShift




extends ionchannel



Same as  ionchannel, but with a **vShift** parameter to change voltage activation of gates. The exact usage of **vShift** in expressions for rates is determined by the individual gates.


`````
```` Parameters
```Table:

**conductance**$  *(from baseionchannel)* $dimensions:conductance
**vShift**$  $dimensions:voltage

```
````

```` Text fields
```Table:

**species**$ 

````

```` Exposures
```Table:

**fopen**$  *(from baseionchannel)* $Dimensionless
**g**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Requirements
```Table:

**v**$  *(from baseionchannel)* $dimensions:voltage

```
````

```` Schema
``` xml
<xs:complexType name="IonChannelVShift">
  <xs:complexContent>
    <xs:extension base="IonChannel">
      <xs:attribute name="vShift" type="Nml2Quantity_voltage" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IonChannelVShift" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IonChannelVShift
from neuroml.utils import component_factory

variable = component_factory(
    IonChannelVShift,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    q10_conductance_scalings: 'list of Q10ConductanceScaling(s) (optional)' = None,
    species: 'a NmlId (optional)' = None,
    type: 'a channelTypes (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (optional)' = None,
    gates: 'list of GateHHUndetermined(s) (optional)' = None,
    gate_hh_rates: 'list of GateHHRates(s) (optional)' = None,
    gate_h_hrates_taus: 'list of GateHHRatesTau(s) (optional)' = None,
    gate_hh_tau_infs: 'list of GateHHTauInf(s) (optional)' = None,
    gate_h_hrates_infs: 'list of GateHHRatesInf(s) (optional)' = None,
    gate_h_hrates_tau_infs: 'list of GateHHRatesTauInf(s) (optional)' = None,
    gate_hh_instantaneouses: 'list of GateHHInstantaneous(s) (optional)' = None,
    gate_fractionals: 'list of GateFractional(s) (optional)' = None,
    v_shift: 'a Nml2Quantity_voltage (required)' = None,
)
```
````
`````


## KSState




One of the states in which a  gateks can be. The rates of transitions between these states are given by  kstransitions.


`````
```` Parameters
```Table:

**relativeConductance**$  $Dimensionless

```
````

```` Exposures
```Table:

**occupancy**$  $Dimensionless
**q**$  $Dimensionless

```
````

```` Dynamics



**State Variables**
: **occupancy**: Dimensionless (exposed as **occupancy**)







**Derived Variables**
    : **q** =&nbsp;relativeConductance * occupancy(exposed as **q**)
    





````
`````


## closedState




extends ksstate



A  ksstate with **relativeConductance** of 0.


`````
```` Parameters
```Table:

**relativeConductance**$  *(from ksstate)* $Dimensionless

```
````

```` Exposures
```Table:

**occupancy**$  *(from ksstate)* $Dimensionless
**q**$  *(from ksstate)* $Dimensionless

```
````

```` Schema
``` xml
<xs:complexType name="ClosedState">
  <xs:complexContent>
    <xs:extension base="Base">
      </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ClosedState" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ClosedState
from neuroml.utils import component_factory

variable = component_factory(
    ClosedState,
    id: 'a NmlId (required)' = None,
)
```
````
`````


## openState




extends ksstate



A  ksstate with **relativeConductance** of 1.


`````
```` Parameters
```Table:

**relativeConductance**$  *(from ksstate)* $Dimensionless

```
````

```` Exposures
```Table:

**occupancy**$  *(from ksstate)* $Dimensionless
**q**$  *(from ksstate)* $Dimensionless

```
````

```` Schema
``` xml
<xs:complexType name="OpenState">
  <xs:complexContent>
    <xs:extension base="Base">
      </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=OpenState" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import OpenState
from neuroml.utils import component_factory

variable = component_factory(
    OpenState,
    id: 'a NmlId (required)' = None,
)
```
````
`````


## ionChannelKS




extends *baseionchannel*



A kinetic scheme based ion channel with multiple  gatekss, each of which consists of multiple  ksstates and  kstransitions giving the rates of transition between them.


`````
```` Parameters
```Table:

**conductance**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Text fields
```Table:

**species**$ 

````

```` Children list
```Table:

**conductanceScaling**$  $ baseconductancescaling
**gates**$  $ gateks

```
````

```` Exposures
```Table:

**fopen**$  *(from baseionchannel)* $Dimensionless
**g**$  *(from baseionchannel)* $dimensions:conductance

```
````

```` Requirements
```Table:

**v**$  *(from baseionchannel)* $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **fopen** =&nbsp;gates[*]->fcond(reduce method: multiply)(exposed as **fopen**)
    : **g** =&nbsp;fopen * conductance(exposed as **g**)
    





````

```` Schema
``` xml
<xs:complexType name="IonChannelKS">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="gateKS" type="GateKS" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="species" type="NmlId" use="optional"/>
      <xs:attribute name="conductance" type="Nml2Quantity_conductance" use="optional"/>
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IonChannelKS" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IonChannelKS
from neuroml.utils import component_factory

variable = component_factory(
    IonChannelKS,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    species: 'a NmlId (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gate_kses: 'list of GateKS(s) (optional)' = None,
)
```
````
`````


## KSTransition




Specified the forward and reverse rates of transition between two  ksstates in a  gateks.


`````
```` Exposures
```Table:

**rf**$  $dimensions:per_time
**rr**$  $dimensions:per_time

```
````
`````


## forwardTransition




extends kstransition



A forward only  kstransition for a  gateks which specifies a **rate** ( type  basehhrate ) which follows one of the standard Hodgkin Huxley forms ( e.g.  hhexprate,  hhsigmoidrate,  hhexplinearrate.


`````
```` Constants
```Table:

**SEC** = 1s$  $ dimensions:time

```
````

```` Child list
```Table:

**rate**$  $ basehhrate

```
````

```` Exposures
```Table:

**rf**$  *(from kstransition)* $dimensions:per_time
**rr**$  *(from kstransition)* $dimensions:per_time

```
````

```` Dynamics








**Derived Variables**
    : **rf0** =&nbsp;rate->r
    : **rf** =&nbsp;rf0(exposed as **rf**)
    : **rr** =&nbsp;0/SEC(exposed as **rr**)
    





````

```` Schema
``` xml
<xs:complexType name="ForwardTransition">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="from" type="NmlId" use="required"/>
      <xs:attribute name="to" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ForwardTransition" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ForwardTransition
from neuroml.utils import component_factory

variable = component_factory(
    ForwardTransition,
    id: 'a NmlId (required)' = None,
    from_: 'a NmlId (required)' = None,
    to: 'a NmlId (required)' = None,
    anytypeobjs_=None,
)
```
````
`````


## reverseTransition




extends kstransition



A reverse only  kstransition for a  gateks which specifies a **rate** ( type  basehhrate ) which follows one of the standard Hodgkin Huxley forms ( e.g.  hhexprate,  hhsigmoidrate,  hhexplinearrate.


`````
```` Constants
```Table:

**SEC** = 1s$  $ dimensions:time

```
````

```` Child list
```Table:

**rate**$  $ basehhrate

```
````

```` Exposures
```Table:

**rf**$  *(from kstransition)* $dimensions:per_time
**rr**$  *(from kstransition)* $dimensions:per_time

```
````

```` Dynamics








**Derived Variables**
    : **rr0** =&nbsp;rate->r
    : **rr** =&nbsp;rr0(exposed as **rr**)
    : **rf** =&nbsp;0/SEC(exposed as **rf**)
    





````

```` Schema
``` xml
<xs:complexType name="ReverseTransition">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="from" type="NmlId" use="required"/>
      <xs:attribute name="to" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ReverseTransition" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ReverseTransition
from neuroml.utils import component_factory

variable = component_factory(
    ReverseTransition,
    id: 'a NmlId (required)' = None,
    from_: 'a NmlId (required)' = None,
    to: 'a NmlId (required)' = None,
    anytypeobjs_=None,
)
```
````
`````


## vHalfTransition




extends kstransition



Transition which specifies both the forward and reverse rates of transition.


`````
```` Parameters
```Table:

**gamma**$  $Dimensionless
**tau**$  $dimensions:time
**tauMin**$  $dimensions:time
**vHalf**$  $dimensions:voltage
**z**$  $Dimensionless

```
````

```` Constants
```Table:

**kte** = 25.3mV$  $ dimensions:voltage

```
````

```` Exposures
```Table:

**rf**$  *(from kstransition)* $dimensions:per_time
**rr**$  *(from kstransition)* $dimensions:per_time

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **rf0** =&nbsp;exp(z * gamma * (v - vHalf) / kte) / tau
    : **rr0** =&nbsp;exp(-z * (1 - gamma) * (v - vHalf) / kte) / tau
    : **rf** =&nbsp;1 / (1/rf0 + tauMin)(exposed as **rf**)
    : **rr** =&nbsp;1 / (1/rr0 + tauMin)(exposed as **rr**)
    





````
`````


## tauInfTransition




extends kstransition



KS Transition specified in terms of time constant  tau and steady state  inf.


`````
```` Child list
```Table:

**timeCourse**$  $ basevoltagedeptime
**steadyState**$  $ basevoltagedepvariable

```
````

```` Exposures
```Table:

**rf**$  *(from kstransition)* $dimensions:per_time
**rr**$  *(from kstransition)* $dimensions:per_time

```
````

```` Dynamics








**Derived Variables**
    : **tau** =&nbsp;timeCourse->t
    : **inf** =&nbsp;steadyState->x
    : **rf** =&nbsp;inf/tau(exposed as **rf**)
    : **rr** =&nbsp;(1-inf)/tau(exposed as **rr**)
    





````

```` Schema
``` xml
<xs:complexType name="TauInfTransition">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:all>
        <xs:element name="steadyState" type="HHVariable"/>
        <xs:element name="timeCourse" type="HHTime"/>
      </xs:all>
      <xs:attribute name="from" type="NmlId" use="required"/>
      <xs:attribute name="to" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=TauInfTransition" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import TauInfTransition
from neuroml.utils import component_factory

variable = component_factory(
    TauInfTransition,
    id: 'a NmlId (required)' = None,
    from_: 'a NmlId (required)' = None,
    to: 'a NmlId (required)' = None,
    steady_state: 'a HHVariable (required)' = None,
    time_course: 'a HHTime (required)' = None,
)
```
````
`````


## gateKS




extends *basegate*



A gate which consists of multiple  ksstates and  kstransitions giving the rates of transition between them.


`````
```` Parameters
```Table:

**instances**$  *(from basegate)* $Dimensionless

```
````

```` Children list
```Table:

**states**$  $ ksstate
**transitions**$  $ kstransition
**q10Settings**$  $ baseq10settings

```
````

```` Exposures
```Table:

**fcond**$  *(from basegate)* $Dimensionless
**q**$  *(from basegate)* $Dimensionless
**rateScale**$  $Dimensionless

```
````

```` Dynamics








**Derived Variables**
    : **rateScale** =&nbsp;q10Settings[*]->q10(reduce method: multiply)(exposed as **rateScale**)
    : **q** =&nbsp;states[*]->q(reduce method: add)(exposed as **q**)
    : **fcond** =&nbsp;q^instances(exposed as **fcond**)
    





````

```` Schema
``` xml
<xs:complexType name="GateKS">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="notes" type="Notes" minOccurs="0"/>
        <xs:element name="q10Settings" type="Q10Settings" minOccurs="0"/>
        <xs:element name="closedState" type="ClosedState" minOccurs="1" maxOccurs="unbounded"/>
        <xs:element name="openState" type="OpenState" minOccurs="1" maxOccurs="unbounded"/>
        <xs:choice minOccurs="1" maxOccurs="unbounded">
          <xs:group ref="ForwardReverseTransition"/>
          <xs:element name="tauInfTransition" type="TauInfTransition"/>
        </xs:choice>
      </xs:sequence>
      <xs:attribute name="instances" type="PositiveInteger" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GateKS" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GateKS
from neuroml.utils import component_factory

variable = component_factory(
    GateKS,
    id: 'a NmlId (required)' = None,
    instances: 'a PositiveInteger (required)' = None,
    notes: 'a string (optional)' = None,
    q10_settings: 'a Q10Settings (optional)' = None,
    closed_states: 'list of ClosedState(s) (required)' = None,
    open_states: 'list of OpenState(s) (required)' = None,
    forward_transition: 'list of ForwardTransition(s) (required)' = None,
    reverse_transition: 'list of ReverseTransition(s) (required)' = None,
    tau_inf_transition: 'list of TauInfTransition(s) (required)' = None,
)
```
````
`````

# Synapses

**A number of synaptic ComponentTypes for use in NeuroML 2 documents, e.g.  exponesynapse,  exptwosynapse,  blockingplasticsynapse. These extend the  basesynapse ComponentType. Also defined continuously transmitting synapses, e.g.  gapjunction and  gradedsynapse.**

---


Original ComponentType definitions: [Synapses.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//Synapses.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## *baseSynapse*




extends *basepointcurrent*



Base type for all synapses, i.e. ComponentTypes which produce a current ( dimension current ) and change Dynamics in response to an incoming event.


[Bioportal entry for Computational Neuroscience Ontology related to baseSynapse.](https://bioportal.bioontology.org/ontologies/CNO/?p=classes&conceptid=cno_0000009)

`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$ $Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="BaseSynapse">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseSynapse
from neuroml.utils import component_factory

variable = component_factory(
    BaseSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    extensiontype_=None,
)
```
````
`````


## *baseVoltageDepSynapse*




extends *basesynapse*



Base type for synapses with a dependence on membrane potential.


`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="BaseVoltageDepSynapse">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">

            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseVoltageDepSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseVoltageDepSynapse
from neuroml.utils import component_factory

variable = component_factory(
    BaseVoltageDepSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    extensiontype_=None,
)
```
````
`````


## *baseSynapseDL*




extends *basevoltagedeppointcurrentdl*



Base type for all synapses, i.e. ComponentTypes which produce a dimensionless current and change Dynamics in response to an incoming event.


[Bioportal entry for Computational Neuroscience Ontology related to baseSynapseDL.](https://bioportal.bioontology.org/ontologies/CNO/?p=classes&conceptid=cno_0000009)

`````
```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType *(from basepointcurrentdl)* $Dimensionless

```
````

```` Requirements
```Table:

**V**$ The current may vary with the dimensionless voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrentdl)* $Dimensionless

```
````
`````


## *baseCurrentBasedSynapse*




extends *basesynapse*



Synapse model which produces a synaptic current.


`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="BaseCurrentBasedSynapse">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">

            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseCurrentBasedSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseCurrentBasedSynapse
from neuroml.utils import component_factory

variable = component_factory(
    BaseCurrentBasedSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    extensiontype_=None,
)
```
````
`````


## alphaCurrentSynapse




extends *basecurrentbasedsynapse*



Alpha current synapse: rise time and decay time are both **tau.**.


`````
```` Parameters
```Table:

**ibase**$ Baseline current increase after receiving a spike $dimensions:current
**tau**$ Time course for rise and decay $dimensions:time

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **I**: dimensions:current 
: **J**: dimensions:current 









**On Start**
: **I** = 0
: **J** = 0


**On Events**

: EVENT IN on port: **in**
: **J** = J + weight * ibase





**Derived Variables**
    : **i** =&nbsp;I(exposed as **i**)
    





**Time Derivatives**
    : d **I** /dt = (2.7182818284590451*J - I)/tau
    : d **J** /dt = -J/tau
    

````

```` Schema
``` xml
<xs:complexType name="AlphaCurrentSynapse">
  <xs:complexContent>
    <xs:extension base="BaseCurrentBasedSynapse">
      <xs:attribute name="tau" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="ibase" type="Nml2Quantity_current" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=AlphaCurrentSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import AlphaCurrentSynapse
from neuroml.utils import component_factory

variable = component_factory(
    AlphaCurrentSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    tau: 'a Nml2Quantity_time (required)' = None,
    ibase: 'a Nml2Quantity_current (required)' = None,
)
```
````
`````


## *baseConductanceBasedSynapse*




extends *basevoltagedepsynapse*



Synapse model which exposes a conductance **g** in addition to producing a current. Not necessarily ohmic!!


[Bioportal entry for Computational Neuroscience Ontology related to baseConductanceBasedSynapse.](https://bioportal.bioontology.org/ontologies/CNO/?p=classes&conceptid=cno_0000027)

`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse $dimensions:voltage
**gbase**$ Baseline conductance, generally the maximum conductance following a single spike $dimensions:conductance

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="BaseConductanceBasedSynapse">
  <xs:complexContent>
    <xs:extension base="BaseVoltageDepSynapse">
      <xs:attribute name="gbase" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="erev" type="Nml2Quantity_voltage" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseConductanceBasedSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseConductanceBasedSynapse
from neuroml.utils import component_factory

variable = component_factory(
    BaseConductanceBasedSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    extensiontype_=None,
)
```
````
`````


## *baseConductanceBasedSynapseTwo*




extends *basevoltagedepsynapse*



Synapse model suited for a sum of two expTwoSynapses which exposes a conductance **g** in addition to producing a current. Not necessarily ohmic!!


[Bioportal entry for Computational Neuroscience Ontology related to baseConductanceBasedSynapseTwo.](https://bioportal.bioontology.org/ontologies/CNO/?p=classes&conceptid=cno_0000027)

`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse $dimensions:voltage
**gbase1**$ Baseline conductance 1 $dimensions:conductance
**gbase2**$ Baseline conductance 2 $dimensions:conductance

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="BaseConductanceBasedSynapseTwo">
  <xs:complexContent>
    <xs:extension base="BaseVoltageDepSynapse">
      <xs:attribute name="gbase1" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="gbase2" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="erev" type="Nml2Quantity_voltage" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BaseConductanceBasedSynapseTwo" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BaseConductanceBasedSynapseTwo
from neuroml.utils import component_factory

variable = component_factory(
    BaseConductanceBasedSynapseTwo,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase1: 'a Nml2Quantity_conductance (required)' = None,
    gbase2: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    extensiontype_=None,
)
```
````
`````


## expOneSynapse




extends *baseconductancebasedsynapse*



Ohmic synapse model whose conductance rises instantaneously by ( **gbase** * **weight** ) on receiving an event, and which decays exponentially to zero with time course **tauDecay**.


`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse *(from baseconductancebasedsynapse)* $dimensions:voltage
**gbase**$ Baseline conductance, generally the maximum conductance following a single spike *(from baseconductancebasedsynapse)* $dimensions:conductance
**tauDecay**$ Time course of decay $dimensions:time

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse *(from baseconductancebasedsynapse)* $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **g**: dimensions:conductance (exposed as **g**)









**On Start**
: **g** = 0


**On Events**

: EVENT IN on port: **in**
: **g** = g + (weight * gbase)





**Derived Variables**
    : **i** =&nbsp;g * (erev - v)(exposed as **i**)
    





**Time Derivatives**
    : d **g** /dt = -g / tauDecay
    

````

```` Schema
``` xml
<xs:complexType name="ExpOneSynapse">
  <xs:complexContent>
    <xs:extension base="BaseConductanceBasedSynapse">
      <xs:attribute name="tauDecay" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ExpOneSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ExpOneSynapse
from neuroml.utils import component_factory

variable = component_factory(
    ExpOneSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    tau_decay: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<expOneSynapse id="syn1" gbase="5nS" erev="0mV" tauDecay="3ms"/>
```
``` xml
<expOneSynapse id="syn2" gbase="10nS" erev="0mV" tauDecay="2ms"/>
```
``` xml
<expOneSynapse id="syn1" gbase="5nS" erev="0mV" tauDecay="3ms"/>
```
````
`````


## alphaSynapse




extends *baseconductancebasedsynapse*



Ohmic synapse model where rise time and decay time are both **tau.** Max conductance reached during this time ( assuming zero conductance before ) is **gbase** * **weight.**.


`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse *(from baseconductancebasedsynapse)* $dimensions:voltage
**gbase**$ Baseline conductance, generally the maximum conductance following a single spike *(from baseconductancebasedsynapse)* $dimensions:conductance
**tau**$ Time course of rise/decay $dimensions:time

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse *(from baseconductancebasedsynapse)* $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **g**: dimensions:conductance (exposed as **g**)
: **A**: dimensions:conductance 









**On Start**
: **g** = 0
: **A** = 0


**On Events**

: EVENT IN on port: **in**
: **A** = A + (gbase*weight)





**Derived Variables**
    : **i** =&nbsp;g * (erev - v)(exposed as **i**)
    





**Time Derivatives**
    : d **g** /dt = (2.7182818284590451 * A - g)/tau
    : d **A** /dt = -A / tau
    

````

```` Schema
``` xml
<xs:complexType name="AlphaSynapse">
  <xs:complexContent>
    <xs:extension base="BaseConductanceBasedSynapse">
      <xs:attribute name="tau" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=AlphaSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import AlphaSynapse
from neuroml.utils import component_factory

variable = component_factory(
    AlphaSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    tau: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<alphaSynapse id="synalpha" gbase="0.5nS" erev="0mV" tau="2ms">
    <notes>An alpha synapse with time for rise equal to decay.</notes>
</alphaSynapse>
```
````
`````


## expTwoSynapse




extends *baseconductancebasedsynapse*



Ohmic synapse model whose conductance waveform on receiving an event has a rise time of **tauRise** and a decay time of **tauDecay.** Max conductance reached during this time ( assuming zero conductance before ) is **gbase** * **weight.**.


`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse *(from baseconductancebasedsynapse)* $dimensions:voltage
**gbase**$ Baseline conductance, generally the maximum conductance following a single spike *(from baseconductancebasedsynapse)* $dimensions:conductance
**tauDecay**$  $dimensions:time
**tauRise**$  $dimensions:time

```
````

```` Derived parameters
```Table:

**peakTime**$  $dimensions:time
```
**peakTime** = log(tauDecay / tauRise) * (tauRise * tauDecay)/(tauDecay - tauRise)
```Table:

**waveformFactor**$  $Dimensionless
```
**waveformFactor** = 1 / (-exp(-peakTime / tauRise) + exp(-peakTime / tauDecay))

````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse *(from baseconductancebasedsynapse)* $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **A**: Dimensionless 
: **B**: Dimensionless 









**On Start**
: **A** = 0
: **B** = 0


**On Events**

: EVENT IN on port: **in**
: **A** = A + (weight * waveformFactor)
: **B** = B + (weight * waveformFactor)





**Derived Variables**
    : **g** =&nbsp;gbase * (B - A)(exposed as **g**)
    : **i** =&nbsp;g * (erev - v)(exposed as **i**)
    





**Time Derivatives**
    : d **A** /dt = -A / tauRise
    : d **B** /dt = -B / tauDecay
    

````

```` Schema
``` xml
<xs:complexType name="ExpTwoSynapse">
  <xs:complexContent>
    <xs:extension base="BaseConductanceBasedSynapse">
      <xs:attribute name="tauDecay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="tauRise" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ExpTwoSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ExpTwoSynapse
from neuroml.utils import component_factory

variable = component_factory(
    ExpTwoSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    tau_decay: 'a Nml2Quantity_time (required)' = None,
    tau_rise: 'a Nml2Quantity_time (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<expTwoSynapse id="AMPA" gbase="0.5nS" erev="0mV" tauRise="1ms" tauDecay="2ms"/>
```
``` xml
<expTwoSynapse id="synInput" gbase="8nS" erev="20mV" tauRise="1ms" tauDecay="5ms"/>
```
``` xml
<expTwoSynapse id="synInputFast" gbase="1nS" erev="20mV" tauRise="0.2ms" tauDecay="1ms"/>
```
````
`````


## expThreeSynapse




extends *baseconductancebasedsynapsetwo*



Ohmic synapse similar to expTwoSynapse but consisting of two components that can differ in decay times and max conductances but share the same rise time.


`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse *(from baseconductancebasedsynapsetwo)* $dimensions:voltage
**gbase1**$ Baseline conductance 1 *(from baseconductancebasedsynapsetwo)* $dimensions:conductance
**gbase2**$ Baseline conductance 2 *(from baseconductancebasedsynapsetwo)* $dimensions:conductance
**tauDecay1**$  $dimensions:time
**tauDecay2**$  $dimensions:time
**tauRise**$  $dimensions:time

```
````

```` Derived parameters
```Table:

**peakTime1**$  $dimensions:time
```
**peakTime1** = log(tauDecay1 / tauRise) * (tauRise * tauDecay1)/(tauDecay1 - tauRise)
```Table:

**peakTime2**$  $dimensions:time
```
**peakTime2** = log(tauDecay2 / tauRise) * (tauRise * tauDecay2)/(tauDecay2 - tauRise)
```Table:

**waveformFactor1**$  $Dimensionless
```
**waveformFactor1** = 1 / (-exp(-peakTime1 / tauRise) + exp(-peakTime1 / tauDecay1))
```Table:

**waveformFactor2**$  $Dimensionless
```
**waveformFactor2** = 1 / (-exp(-peakTime2 / tauRise) + exp(-peakTime2 / tauDecay2))

````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse *(from baseconductancebasedsynapsetwo)* $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **A**: Dimensionless 
: **B**: Dimensionless 
: **C**: Dimensionless 









**On Start**
: **A** = 0
: **B** = 0
: **C** = 0


**On Events**

: EVENT IN on port: **in**
: **A** = A + (gbase1*weight * waveformFactor1 + gbase2*weight*waveformFactor2 )/(gbase1+gbase2)
: **B** = B + (weight * waveformFactor1)
: **C** = C + (weight * waveformFactor2)





**Derived Variables**
    : **g** =&nbsp;gbase1*(B - A) + gbase2*(C-A)(exposed as **g**)
    : **i** =&nbsp;g * (erev - v)(exposed as **i**)
    





**Time Derivatives**
    : d **A** /dt = -A / tauRise
    : d **B** /dt = -B / tauDecay1
    : d **C** /dt = -C / tauDecay2
    

````

```` Schema
``` xml
<xs:complexType name="ExpThreeSynapse">
  <xs:complexContent>
    <xs:extension base="BaseConductanceBasedSynapseTwo">
      <xs:attribute name="tauDecay1" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="tauDecay2" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="tauRise" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ExpThreeSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ExpThreeSynapse
from neuroml.utils import component_factory

variable = component_factory(
    ExpThreeSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase1: 'a Nml2Quantity_conductance (required)' = None,
    gbase2: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    tau_decay1: 'a Nml2Quantity_time (required)' = None,
    tau_decay2: 'a Nml2Quantity_time (required)' = None,
    tau_rise: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<expThreeSynapse id="synInputFastTwo" gbase1="1.5nS" tauRise="0.1ms" tauDecay1="0.7ms" gbase2="0.5nS" tauDecay2="2.5ms" erev="0mV"/>
```
``` xml
<expThreeSynapse id="AMPA" gbase1="1.5nS" tauRise="0.1ms" tauDecay1="0.7ms" gbase2="0.5nS" tauDecay2="2.5ms" erev="0mV">
    <notes>A synapse consisting of one rise and two decay time courses.</notes>
</expThreeSynapse>
```
````
`````


## *baseBlockMechanism*




Base of any ComponentType which produces a varying scaling ( or blockage ) of synaptic strength of magnitude **scaling**.


`````
```` Exposures
```Table:

**blockFactor**$  $Dimensionless

```
````
`````


## voltageConcDepBlockMechanism




extends *baseblockmechanism*



Synaptic blocking mechanism which varys with membrane potential across the synapse, e.g. in NMDA receptor mediated synapses.


`````
```` Parameters
```Table:

**blockConcentration**$  $dimensions:concentration
**scalingConc**$  $dimensions:concentration
**scalingVolt**$  $dimensions:voltage

```
````

```` Text fields
```Table:

**species**$ 

````

```` Exposures
```Table:

**blockFactor**$  *(from baseblockmechanism)* $Dimensionless

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````

```` Dynamics








**Derived Variables**
    : **blockFactor** =&nbsp;1/(1 + (blockConcentration / scalingConc)* exp(-1 * (v / scalingVolt)))(exposed as **blockFactor**)
    





````
`````


## *basePlasticityMechanism*




Base plasticity mechanism.


`````
```` Exposures
```Table:

**plasticityFactor**$  $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ This is where the plasticity mechanism receives spike events from the parent synapse.$Direction: in

```
````
`````


## tsodyksMarkramDepMechanism




extends *baseplasticitymechanism*



Depression-only Tsodyks-Markram model, as in Tsodyks and Markram 1997.


`````
```` Parameters
```Table:

**initReleaseProb**$  $Dimensionless
**tauRec**$  $dimensions:time

```
````

```` Exposures
```Table:

**plasticityFactor**$  *(from baseplasticitymechanism)* $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ This is where the plasticity mechanism receives spike events from the parent synapse. *(from baseplasticitymechanism)*$Direction: in

```
````

```` Dynamics

**Structure**
: WITH **parent** AS **a**
: WITH **this** AS **b**
: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **R**: Dimensionless 









**On Start**
: **R** = 1


**On Events**

: EVENT IN on port: **in**
: **R** = R * (1 - U)





**Derived Variables**
    : **U** =&nbsp;initReleaseProb
    : **plasticityFactor** =&nbsp;R * U(exposed as **plasticityFactor**)
    





**Time Derivatives**
    : d **R** /dt = (1 - R) / tauRec
    

````
`````


## tsodyksMarkramDepFacMechanism




extends *baseplasticitymechanism*



Full Tsodyks-Markram STP model with both depression and facilitation, as in Tsodyks, Pawelzik and Markram 1998.


`````
```` Parameters
```Table:

**initReleaseProb**$  $Dimensionless
**tauFac**$  $dimensions:time
**tauRec**$  $dimensions:time

```
````

```` Exposures
```Table:

**plasticityFactor**$  *(from baseplasticitymechanism)* $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ This is where the plasticity mechanism receives spike events from the parent synapse. *(from baseplasticitymechanism)*$Direction: in

```
````

```` Dynamics

**Structure**
: WITH **parent** AS **a**
: WITH **this** AS **b**
: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **R**: Dimensionless 
: **U**: Dimensionless 









**On Start**
: **R** = 1
: **U** = initReleaseProb


**On Events**

: EVENT IN on port: **in**
: **R** = R * (1 - U)
: **U** = U + initReleaseProb * (1 - U)





**Derived Variables**
    : **plasticityFactor** =&nbsp;R * U(exposed as **plasticityFactor**)
    





**Time Derivatives**
    : d **R** /dt = (1 - R) / tauRec
    : d **U** /dt = (initReleaseProb - U) / tauFac
    

````
`````


## blockingPlasticSynapse




extends exptwosynapse



Biexponential synapse that allows for optional block and plasticity mechanisms, which can be expressed as child elements.


`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse *(from baseconductancebasedsynapse)* $dimensions:voltage
**gbase**$ Baseline conductance, generally the maximum conductance following a single spike *(from baseconductancebasedsynapse)* $dimensions:conductance
**tauDecay**$  *(from exptwosynapse)* $dimensions:time
**tauRise**$  *(from exptwosynapse)* $dimensions:time

```
````

```` Derived parameters
```Table:

**peakTime**$  *(from exptwosynapse)* $dimensions:time
```
**peakTime** = log(tauDecay / tauRise) * (tauRise * tauDecay)/(tauDecay - tauRise)
```Table:

**waveformFactor**$  *(from exptwosynapse)* $Dimensionless
```
**waveformFactor** = 1 / (-exp(-peakTime / tauRise) + exp(-peakTime / tauDecay))

````

```` Children list
```Table:

**plasticityMechanisms**$  $ baseplasticitymechanism
**blockMechanisms**$  $ baseblockmechanism

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**g**$ Time varying conductance through the synapse *(from baseconductancebasedsynapse)* $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in
**relay**$ Used to relay incoming spikes to child plasticity mechanism$Direction: out

```
````

```` Dynamics



**State Variables**
: **A**: Dimensionless 
: **B**: Dimensionless 









**On Start**
: **A** = 0
: **B** = 0


**On Events**

: EVENT IN on port: **in**
: **A** = A + (weight * plasticityFactor * waveformFactor)
: **B** = B + (weight * plasticityFactor * waveformFactor)
: EVENT OUT on port: **relay**





**Derived Variables**
    : **plasticityFactor** =&nbsp;plasticityMechanisms[*]->plasticityFactor(reduce method: multiply)
    : **blockFactor** =&nbsp;blockMechanisms[*]->blockFactor(reduce method: multiply)
    : **g** =&nbsp;blockFactor * gbase * (B - A)(exposed as **g**)
    : **i** =&nbsp;g * (erev - v)(exposed as **i**)
    





**Time Derivatives**
    : d **A** /dt = -A / tauRise
    : d **B** /dt = -B / tauDecay
    

````

```` Schema
``` xml
<xs:complexType name="BlockingPlasticSynapse">
  <xs:complexContent>
    <xs:extension base="ExpTwoSynapse">
      <xs:sequence>
        <xs:element name="plasticityMechanism" type="PlasticityMechanism" minOccurs="0"/>
        <xs:element name="blockMechanism" type="BlockMechanism" minOccurs="0"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BlockingPlasticSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BlockingPlasticSynapse
from neuroml.utils import component_factory

variable = component_factory(
    BlockingPlasticSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    gbase: 'a Nml2Quantity_conductance (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
    tau_decay: 'a Nml2Quantity_time (required)' = None,
    tau_rise: 'a Nml2Quantity_time (required)' = None,
    plasticity_mechanism: 'a PlasticityMechanism (optional)' = None,
    block_mechanism: 'a BlockMechanism (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<blockingPlasticSynapse id="NMDA" gbase=".8nS" tauRise="1e-3s" tauDecay="13.3333e-3s" erev="0V">
    <blockMechanism type="voltageConcDepBlockMechanism" species="mg" blockConcentration="1.2mM" scalingConc="1.9205441817997078mM" scalingVolt="0.016129032258064516V"/>
</blockingPlasticSynapse>
```
``` xml
<blockingPlasticSynapse id="blockStpSynDep" gbase="1nS" erev="0mV" tauRise="0.1ms" tauDecay="2ms">
    <notes>A biexponential blocking synapse, with STD.</notes>
    <plasticityMechanism type="tsodyksMarkramDepMechanism" initReleaseProb="0.5" tauRec="120 ms"/>
    <blockMechanism type="voltageConcDepBlockMechanism" species="mg" blockConcentration="1.2 mM" scalingConc="1.920544 mM" scalingVolt="16.129 mV"/>
</blockingPlasticSynapse>
```
``` xml
<blockingPlasticSynapse id="blockStpSynDepFac" gbase="1nS" erev="0mV" tauRise="0.1ms" tauDecay="2ms">
    <notes>A biexponential blocking synapse with short term
            depression and facilitation.</notes>
    <plasticityMechanism type="tsodyksMarkramDepFacMechanism" initReleaseProb="0.5" tauRec="120 ms" tauFac="10 ms"/>
    <blockMechanism type="voltageConcDepBlockMechanism" species="mg" blockConcentration="1.2 mM" scalingConc="1.920544 mM" scalingVolt="16.129 mV"/>
</blockingPlasticSynapse>
```
````
`````


## doubleSynapse




extends *basevoltagedepsynapse*



Synapse consisting of two independent synaptic mechanisms ( e.g. AMPA-R and NMDA-R ), which can be easily colocated in connections.


`````
```` Paths
```Table:

**synapse1Path**$ 
**synapse2Path**$ 

````

```` Component References
```Table:

**synapse1**$  $ basesynapse
**synapse2**$  $ basesynapse

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in
**relay**$ Used to relay incoming spikes to child mechanisms$Direction: out

```
````

```` Dynamics

**Structure**
: WITH **this** AS **a**
: WITH **synapse1Path** AS **b**
: WITH **synapse2Path** AS **c**
: CHILD INSTANCE: **synapse1**
: CHILD INSTANCE: **synapse2**
: EVENT CONNECTION from **a** TO  **c**   

: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **weightFactor**: Dimensionless 











**On Events**

: EVENT IN on port: **in**
: **weightFactor** = weight
: EVENT OUT on port: **relay**





**Derived Variables**
    : **i1** =&nbsp;synapse1->i
    : **i2** =&nbsp;synapse2->i
    : **i** =&nbsp;weightFactor * (i1 + i2)(exposed as **i**)
    





````

```` Schema
``` xml
<xs:complexType name="DoubleSynapse">
  <xs:complexContent>
    <xs:extension base="BaseVoltageDepSynapse">
      <xs:attribute name="synapse1" type="NmlId" use="required"/>
      <xs:attribute name="synapse2" type="NmlId" use="required"/>
      <xs:attribute name="synapse1Path" type="xs:string" use="required"/>
      <xs:attribute name="synapse2Path" type="xs:string" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=DoubleSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import DoubleSynapse
from neuroml.utils import component_factory

variable = component_factory(
    DoubleSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    synapse1: 'a NmlId (required)' = None,
    synapse2: 'a NmlId (required)' = None,
    synapse1_path: 'a string (required)' = None,
    synapse2_path: 'a string (required)' = None,
)
```
````
```` Usage: XML
``` xml
<doubleSynapse id="AMPA_NMDA" synapse1="AMPA" synapse1Path="./AMPA" synapse2="NMDA" synapse2Path="./NMDA">
    <notes>A single "synapse" which contains both AMPA and NMDA. It is planned that the need for extra synapse1Path/synapse2Path attributes can be removed in later versions.</notes>
</doubleSynapse>
```
````
`````


## stdpSynapse




extends exptwosynapse



Spike timing dependent plasticity mechanism, NOTE: EXAMPLE NOT YET WORKING!!!!


[Bioportal entry for Computational Neuroscience Ontology related to stdpSynapse.](https://bioportal.bioontology.org/ontologies/CNO/?p=classes&conceptid=cno_0000034)

`````
```` Parameters
```Table:

**erev**$ Reversal potential of the synapse *(from baseconductancebasedsynapse)* $dimensions:voltage
**gbase**$ Baseline conductance, generally the maximum conductance following a single spike *(from baseconductancebasedsynapse)* $dimensions:conductance
**tauDecay**$  *(from exptwosynapse)* $dimensions:time
**tauRise**$  *(from exptwosynapse)* $dimensions:time

```
````

```` Constants
```Table:

**tsinceRate** = 1$  $ Dimensionless
**longTime** = 1000s$  $ dimensions:time

```
````

```` Derived parameters
```Table:

**peakTime**$  *(from exptwosynapse)* $dimensions:time
```
**peakTime** = log(tauDecay / tauRise) * (tauRise * tauDecay)/(tauDecay - tauRise)
```Table:

**waveformFactor**$  *(from exptwosynapse)* $Dimensionless
```
**waveformFactor** = 1 / (-exp(-peakTime / tauRise) + exp(-peakTime / tauDecay))

````

```` Exposures
```Table:

**M**$  $Dimensionless
**P**$  $Dimensionless
**g**$ Time varying conductance through the synapse *(from baseconductancebasedsynapse)* $dimensions:conductance
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current
**tsince**$  $dimensions:time

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **A**: Dimensionless 
: **B**: Dimensionless 
: **M**: Dimensionless (exposed as **M**)
: **P**: Dimensionless (exposed as **P**)
: **tsince**: dimensions:time (exposed as **tsince**)









**On Start**
: **A** = 0
: **B** = 0
: **M** = 1
: **P** = 1
: **tsince** = longTime


**On Events**

: EVENT IN on port: **in**
: **A** = A + waveformFactor
: **B** = B + waveformFactor
: **tsince** = 0





**Derived Variables**
    : **g** =&nbsp;gbase * (B - A)(exposed as **g**)
    : **i** =&nbsp;g * (erev - v)(exposed as **i**)
    





**Time Derivatives**
    : d **A** /dt = -A / tauRise
    : d **B** /dt = -B / tauDecay
    : d **tsince** /dt = tsinceRate
    

````
`````


## gapJunction




extends *basesynapse*



Gap junction/single electrical connection.


`````
```` Parameters
```Table:

**conductance**$  $dimensions:conductance

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics








**Derived Variables**
    : **vpeer** =&nbsp;peer->v
    : **i** =&nbsp;weight * conductance * (vpeer - v)(exposed as **i**)
    





````

```` Schema
``` xml
<xs:complexType name="GapJunction">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">
      <xs:attribute name="conductance" type="Nml2Quantity_conductance" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GapJunction" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GapJunction
from neuroml.utils import component_factory

variable = component_factory(
    GapJunction,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (required)' = None,
)
```
````
```` Usage: XML
``` xml
<gapJunction id="gj1" conductance="10pS"/>
```
``` xml
<gapJunction id="gj1" conductance="10pS"/>
```
````
`````


## *baseGradedSynapse*




extends *basesynapse*



Base type for graded synapses.


`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````
`````


## silentSynapse




extends *basegradedsynapse*



Dummy synapse which emits no current. Used as presynaptic endpoint for analog synaptic connection.


`````
```` Constants
```Table:

**AMP** = 1A$  $ dimensions:current

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics








**Derived Variables**
    : **vpeer** =&nbsp;peer->v
    : **i** =&nbsp;0 * AMP(exposed as **i**)
    





````

```` Schema
``` xml
<xs:complexType name="SilentSynapse">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">

            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SilentSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SilentSynapse
from neuroml.utils import component_factory

variable = component_factory(
    SilentSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<silentSynapse id="silent1"/>
```
``` xml
<silentSynapse id="silent2"/>
```
``` xml
<silentSynapse id="silent1"/>
```
````
`````


## linearGradedSynapse




extends *basegradedsynapse*



Behaves just like a one way gap junction.


`````
```` Parameters
```Table:

**conductance**$  $dimensions:conductance

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics








**Derived Variables**
    : **vpeer** =&nbsp;peer->v
    : **i** =&nbsp;weight * conductance * (vpeer - v)(exposed as **i**)
    





````

```` Schema
``` xml
<xs:complexType name="LinearGradedSynapse">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">
      <xs:attribute name="conductance" type="Nml2Quantity_conductance" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=LinearGradedSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import LinearGradedSynapse
from neuroml.utils import component_factory

variable = component_factory(
    LinearGradedSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (required)' = None,
)
```
````
```` Usage: XML
``` xml
<linearGradedSynapse id="gs1" conductance="5pS"/>
```
````
`````


## gradedSynapse




extends *basegradedsynapse*



Graded/analog synapse. Based on synapse in Methods of http://www.nature.com/neuro/journal/v7/n12/abs/nn1352.html.


`````
```` Parameters
```Table:

**Vth**$ The half-activation voltage of the synapse $dimensions:voltage
**conductance**$  $dimensions:conductance
**delta**$ Slope of the activation curve $dimensions:voltage
**erev**$ The reversal potential of the synapse $dimensions:voltage
**k**$ Rate constant for transmitter-receptor dissociation rate $dimensions:per_time

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current
**inf**$  $Dimensionless
**tau**$  $dimensions:time

```
````

```` Requirements
```Table:

**v**$  $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **s**: Dimensionless 












**On Conditions**

: IF (1-inf) &lt; 1e-4 THEN
: **s** = inf





**Derived Variables**
    : **vpeer** =&nbsp;peer->v
    : **inf** =&nbsp;1/(1 + exp((Vth - vpeer)/delta))(exposed as **inf**)
    : **tau** =&nbsp;(1-inf)/k(exposed as **tau**)
    : **i** =&nbsp;weight * conductance * s * (erev-v)(exposed as **i**)
    



**Conditional Derived Variables**
    
: IF (1-inf) &gt; 1e-4 THEN
:  **s_rate** = (inf - s)/tau 
: OTHERWISE
:  **s_rate** = 0 


**Time Derivatives**
    : d **s** /dt = s_rate
    

````

```` Schema
``` xml
<xs:complexType name="GradedSynapse">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">
      <xs:attribute name="conductance" type="Nml2Quantity_conductance" use="required"/>
      <xs:attribute name="delta" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="Vth" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="k" type="Nml2Quantity_pertime" use="required"/>
      <xs:attribute name="erev" type="Nml2Quantity_voltage" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=GradedSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import GradedSynapse
from neuroml.utils import component_factory

variable = component_factory(
    GradedSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    conductance: 'a Nml2Quantity_conductance (required)' = None,
    delta: 'a Nml2Quantity_voltage (required)' = None,
    Vth: 'a Nml2Quantity_voltage (required)' = None,
    k: 'a Nml2Quantity_pertime (required)' = None,
    erev: 'a Nml2Quantity_voltage (required)' = None,
)
```
````
```` Usage: XML
``` xml
<gradedSynapse id="gs2" conductance="5pS" delta="5mV" Vth="-55mV" k="0.025per_ms" erev="0mV"/>
```
``` xml
<gradedSynapse id="gs1" conductance="0.1nS" delta="5mV" Vth="-35mV" k="0.025per_ms" erev="0mV"/>
```
````
`````

# Inputs

**A number of ComponentTypes for providing spiking ( e.g.  spikegeneratorpoisson,  spikearray ) and current inputs ( e.g.  pulsegenerator,  voltageclamp,  timedsynapticinput,  poissonfiringsynapse ) to other ComponentTypes**

---


Original ComponentType definitions: [Inputs.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//Inputs.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## *basePointCurrent*




extends *basestandalone*



Base type for all ComponentTypes which produce a current **i** ( with dimension current ).


`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType $dimensions:current

```
````
`````


## *baseVoltageDepPointCurrent*




extends *basepointcurrent*



Base type for all ComponentTypes which produce a current **i** ( with dimension current ) and require a voltage **v** exposed on the parent Component, which would often be the membrane potential of a Component extending  basecellmembpot.


`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed $dimensions:voltage

```
````
`````


## *baseVoltageDepPointCurrentSpiking*




extends *basevoltagedeppointcurrent*



Base type for all ComponentTypes which produce a current **i,** require a membrane potential **v** exposed on the parent and emit spikes ( on a port **spike** ). The exposed variable **tsince** can be used for plotting the time since the Component has spiked last.


`````
```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current
**tsince**$ Time since the last spike was emitted $dimensions:time

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted$Direction: out

```
````
`````


## *basePointCurrentDL*




Base type for all ComponentTypes which produce a dimensionless current **I.** There are many dimensionless equivalents of all the core current producing ComponentTypes such as  pulsegenerator /  pulsegeneratordl,  sinegenerator /  sinegeneratordl and  rampgenerator /  rampgeneratordl.


`````
```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType $Dimensionless

```
````
`````


## *baseVoltageDepPointCurrentDL*




extends *basepointcurrentdl*



Base type for all ComponentTypes which produce a dimensionless current **I** and require a dimensionless membrane potential **V** exposed on the parent Component.


`````
```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType *(from basepointcurrentdl)* $Dimensionless

```
````

```` Requirements
```Table:

**V**$ The current may vary with the dimensionless voltage exposed by the ComponentType on which this is placed $Dimensionless

```
````
`````


## *baseSpikeSource*




Base for any ComponentType whose main purpose is to emit spikes ( on a port **spike** ). The exposed variable **tsince** can be used for plotting the time since the Component has spiked last.


`````
```` Exposures
```Table:

**tsince**$ Time since the last spike was emitted $dimensions:time

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted$Direction: out

```
````
`````


## spikeGenerator




extends *basespikesource*



Simple generator of spikes at a regular interval set by **period**.


`````
```` Parameters
```Table:

**period**$ Time between spikes. The first spike will be emitted after this time. $dimensions:time

```
````

```` Constants
```Table:

**SMALL_TIME** = 1e-9ms$ A useful constant for use as a non zero time increment $ dimensions:time

```
````

```` Exposures
```Table:

**tnext**$ When the next spike should ideally be emitted (dt permitting) $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnext**: dimensions:time (exposed as **tnext**)









**On Start**
: **tsince** = 0
: **tnext** = period



**On Conditions**

: IF tnext - t &lt; SMALL_TIME THEN
: **tsince** = 0
: **tnext** = tnext+period
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnext** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="SpikeGenerator">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="period" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeGenerator" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeGenerator
from neuroml.utils import component_factory

variable = component_factory(
    SpikeGenerator,
    id: 'a NonNegativeInteger (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    period: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<spikeGenerator id="spikeGenRegular" period="20 ms"/>
```
````
`````


## spikeGeneratorRandom




extends *basespikesource*



Generator of spikes with a random interspike interval of at least **minISI** and at most **maxISI**.


`````
```` Parameters
```Table:

**maxISI**$ Maximum interspike interval $dimensions:time
**minISI**$ Minimum interspike interval $dimensions:time

```
````

```` Constants
```Table:

**MSEC** = 1ms$ Required for converting time values to/from dimensionless quantities $ dimensions:time

```
````

```` Exposures
```Table:

**isi**$ The interval until the next spike $dimensions:time
**tnext**$ When the next spike should ideally be emitted (dt permitting) $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnext**: dimensions:time (exposed as **tnext**)
: **isi**: dimensions:time (exposed as **isi**)









**On Start**
: **tsince** = 0
: **isi** = minISI + MSEC * random((maxISI - minISI) / MSEC)
: **tnext** = isi



**On Conditions**

: IF t &gt; tnext THEN
: **isi** = minISI + MSEC * random((maxISI - minISI) / MSEC)
: **tsince** = 0
: **tnext** = tnext+isi
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnext** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="SpikeGeneratorRandom">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="maxISI" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="minISI" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeGeneratorRandom" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeGeneratorRandom
from neuroml.utils import component_factory

variable = component_factory(
    SpikeGeneratorRandom,
    id: 'a NonNegativeInteger (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    max_isi: 'a Nml2Quantity_time (required)' = None,
    min_isi: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<spikeGeneratorRandom id="spikeGenRandom" minISI="10 ms" maxISI="30 ms"/>
```
````
`````


## spikeGeneratorPoisson




extends *basespikesource*



Generator of spikes whose ISI is distributed according to an exponential PDF with scale: 1 / **averageRate**.


`````
```` Parameters
```Table:

**averageRate**$ The average rate at which spikes are emitted $dimensions:per_time

```
````

```` Constants
```Table:

**SMALL_TIME** = 1e-9ms$  $ dimensions:time

```
````

```` Exposures
```Table:

**isi**$ The interval until the next spike $dimensions:time
**tnextIdeal**$ This is the ideal/perfect next spike time, based on a newly generated isi, but dt precision will mean that it's usually slightly later than this $dimensions:time
**tnextUsed**$ This is the next spike time for practical purposes, ensuring that it's later than the current time $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnextIdeal**: dimensions:time (exposed as **tnextIdeal**)
: **tnextUsed**: dimensions:time (exposed as **tnextUsed**)
: **isi**: dimensions:time (exposed as **isi**)









**On Start**
: **tsince** = 0
: **isi** = -1 * log(random(1)) / averageRate
: **tnextIdeal** = isi
: **tnextUsed** = isi



**On Conditions**

: IF t &gt; tnextUsed THEN
: **tsince** = 0
: **isi** = -1 * log(random(1)) / averageRate
: **tnextIdeal** = (tnextIdeal+isi)
: **tnextUsed** = tnextIdeal*H( (tnextIdeal-t)/t ) + (t+SMALL_TIME)*H( (t-tnextIdeal)/t )
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnextUsed** /dt = 0
    : d **tnextIdeal** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="SpikeGeneratorPoisson">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="averageRate" type="Nml2Quantity_pertime" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeGeneratorPoisson" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeGeneratorPoisson
from neuroml.utils import component_factory

variable = component_factory(
    SpikeGeneratorPoisson,
    id: 'a NonNegativeInteger (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    average_rate: 'a Nml2Quantity_pertime (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<spikeGeneratorPoisson id="spikeGenPoisson" averageRate="50 Hz"/>
```
````
`````


## spikeGeneratorRefPoisson




extends spikegeneratorpoisson



Generator of spikes whose ISI distribution is the maximum entropy distribution over [ **minimumISI,** +infinity ) with mean: 1 / **averageRate**.


`````
```` Parameters
```Table:

**averageRate**$ The average rate at which spikes are emitted *(from spikegeneratorpoisson)* $dimensions:per_time
**minimumISI**$ The minimum interspike interval $dimensions:time

```
````

```` Derived parameters
```Table:

**averageIsi**$ The average interspike interval $dimensions:time
```
**averageIsi** = 1 / averageRate

````

```` Exposures
```Table:

**isi**$ The interval until the next spike *(from spikegeneratorpoisson)* $dimensions:time
**tnextIdeal**$ This is the ideal/perfect next spike time, based on a newly generated isi, but dt precision will mean that it's usually slightly later than this *(from spikegeneratorpoisson)* $dimensions:time
**tnextUsed**$ This is the next spike time for practical purposes, ensuring that it's later than the current time *(from spikegeneratorpoisson)* $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnextIdeal**: dimensions:time (exposed as **tnextIdeal**)
: **tnextUsed**: dimensions:time (exposed as **tnextUsed**)
: **isi**: dimensions:time (exposed as **isi**)









**On Start**
: **tsince** = 0
: **isi** = minimumISI - (averageIsi-minimumISI) * log(random(1))
: **tnextIdeal** = isi
: **tnextUsed** = isi



**On Conditions**

: IF t &gt; tnextUsed THEN
: **tsince** = 0
: **isi** = minimumISI - (averageIsi-minimumISI) * log(random(1))
: **tnextIdeal** = (tnextIdeal+isi)
: **tnextUsed** = tnextIdeal*H( (tnextIdeal-t)/t ) + (t+SMALL_TIME)*H( (t-tnextIdeal)/t )
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnextUsed** /dt = 0
    : d **tnextIdeal** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="SpikeGeneratorRefPoisson">
  <xs:complexContent>
    <xs:extension base="SpikeGeneratorPoisson">
      <xs:attribute name="minimumISI" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeGeneratorRefPoisson" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeGeneratorRefPoisson
from neuroml.utils import component_factory

variable = component_factory(
    SpikeGeneratorRefPoisson,
    id: 'a NonNegativeInteger (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    average_rate: 'a Nml2Quantity_pertime (required)' = None,
    minimum_isi: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<spikeGeneratorRefPoisson id="spikeGenRefPoisson" averageRate="50 Hz" minimumISI="10 ms"/>
```
````
`````


## poissonFiringSynapse




extends *basevoltagedeppointcurrentspiking*



Poisson spike generator firing at **averageRate,** which is connected to single **synapse** that is triggered every time a spike is generated, producing an input current. See also  transientpoissonfiringsynapse.


`````
```` Parameters
```Table:

**averageRate**$ The average rate at which spikes are emitted $dimensions:per_time

```
````

```` Constants
```Table:

**SMALL_TIME** = 1e-9ms$  $ dimensions:time

```
````

```` Derived parameters
```Table:

**averageIsi**$ The average interspike interval $dimensions:time
```
**averageIsi** = 1 / averageRate

````

```` Paths
```Table:

**spikeTarget**$ The target of the spikes, i.e. the synapse

````

```` Component References
```Table:

**synapse**$  $ basesynapse

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current
**isi**$ The interval until the next spike $dimensions:time
**tnextIdeal**$  $dimensions:time
**tnextUsed**$  $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basevoltagedeppointcurrentspiking)* $dimensions:time

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in
**spike**$ Port on which spikes are emitted$Direction: out
**spike**$ Port on which spikes are emitted *(from basevoltagedeppointcurrentspiking)*$Direction: out

```
````

```` Dynamics

**Structure**
: WITH **this** AS **a**
: WITH **spikeTarget** AS **b**
: CHILD INSTANCE: **synapse**
: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnextIdeal**: dimensions:time (exposed as **tnextIdeal**)
: **tnextUsed**: dimensions:time (exposed as **tnextUsed**)
: **isi**: dimensions:time (exposed as **isi**)









**On Start**
: **tsince** = 0
: **isi** = - averageIsi * log(random(1))
: **tnextIdeal** = isi
: **tnextUsed** = isi



**On Conditions**

: IF t &gt; tnextUsed THEN
: **tsince** = 0
: **isi** = - averageIsi * log(1 - random(1))
: **tnextIdeal** = (tnextIdeal+isi)
: **tnextUsed** = tnextIdeal*H( (tnextIdeal-t)/t ) + (t+SMALL_TIME)*H( (t-tnextIdeal)/t )
: EVENT OUT on port: **spike**





**Derived Variables**
    : **iSyn** =&nbsp;synapse->i
    : **i** =&nbsp;weight * iSyn(exposed as **i**)
    





**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnextUsed** /dt = 0
    : d **tnextIdeal** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="PoissonFiringSynapse">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="averageRate" type="Nml2Quantity_pertime" use="required"/>
      <xs:attribute name="synapse" type="xs:string" use="required"/>
      <xs:attribute name="spikeTarget" type="xs:string" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=PoissonFiringSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import PoissonFiringSynapse
from neuroml.utils import component_factory

variable = component_factory(
    PoissonFiringSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    average_rate: 'a Nml2Quantity_pertime (required)' = None,
    synapse: 'a string (required)' = None,
    spike_target: 'a string (required)' = None,
)
```
````
```` Usage: XML
``` xml
<poissonFiringSynapse id="poissonFiringSyn" averageRate="10 Hz" synapse="synInput" spikeTarget="./synInput"/>
```
````
`````


## transientPoissonFiringSynapse




extends *basevoltagedeppointcurrentspiking*



Poisson spike generator firing at **averageRate** after a **delay** and for a **duration,** connected to single **synapse** that is triggered every time a spike is generated, providing an input current. Similar to ComponentType  poissonfiringsynapse.


`````
```` Parameters
```Table:

**averageRate**$  $dimensions:per_time
**delay**$  $dimensions:time
**duration**$  $dimensions:time

```
````

```` Constants
```Table:

**SMALL_TIME** = 1e-9ms$  $ dimensions:time
**LONG_TIME** = 1e9hour$  $ dimensions:time

```
````

```` Derived parameters
```Table:

**averageIsi**$  $dimensions:time
```
**averageIsi** = 1 / averageRate

````

```` Paths
```Table:

**spikeTarget**$ 

````

```` Component References
```Table:

**synapse**$  $ basesynapse

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current
**isi**$  $dimensions:time
**tnextIdeal**$  $dimensions:time
**tnextUsed**$  $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basevoltagedeppointcurrentspiking)* $dimensions:time

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in
**spike**$ Port on which spikes are emitted$Direction: out
**spike**$ Port on which spikes are emitted *(from basevoltagedeppointcurrentspiking)*$Direction: out

```
````

```` Dynamics

**Structure**
: WITH **this** AS **a**
: WITH **spikeTarget** AS **b**
: CHILD INSTANCE: **synapse**
: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnextIdeal**: dimensions:time (exposed as **tnextIdeal**)
: **tnextUsed**: dimensions:time (exposed as **tnextUsed**)
: **isi**: dimensions:time (exposed as **isi**)









**On Start**
: **tsince** = 0
: **isi** = - averageIsi * log(1 - random(1))  +delay
: **tnextIdeal** = isi
: **tnextUsed** = isi



**On Conditions**

: IF t &gt; tnextUsed THEN
: **tsince** = 0
: **isi** = - averageIsi * log(1 - random(1))
: **tnextIdeal** = (tnextIdeal+isi) + H(((t+isi) - (delay+duration))/duration)*LONG_TIME
: **tnextUsed** = tnextIdeal*H( (tnextIdeal-t)/t ) + (t+SMALL_TIME)*H( (t-tnextIdeal)/t )
: EVENT OUT on port: **spike**





**Derived Variables**
    : **iSyn** =&nbsp;synapse->i
    : **i** =&nbsp;weight * iSyn(exposed as **i**)
    





**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnextUsed** /dt = 0
    : d **tnextIdeal** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="TransientPoissonFiringSynapse">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="averageRate" type="Nml2Quantity_pertime" use="required"/>
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="synapse" type="xs:string" use="required"/>
      <xs:attribute name="spikeTarget" type="xs:string" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=TransientPoissonFiringSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import TransientPoissonFiringSynapse
from neuroml.utils import component_factory

variable = component_factory(
    TransientPoissonFiringSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    average_rate: 'a Nml2Quantity_pertime (required)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    synapse: 'a string (required)' = None,
    spike_target: 'a string (required)' = None,
)
```
````
```` Usage: XML
``` xml
<transientPoissonFiringSynapse id="transPoissonFiringSyn" delay="50ms" duration="50ms" averageRate="300 Hz" synapse="synInputFast" spikeTarget="./synInputFast"/>
```
``` xml
<transientPoissonFiringSynapse id="transPoissonFiringSyn2" delay="50ms" duration="500ms" averageRate="10 Hz" synapse="synInputFastTwo" spikeTarget="./synInputFastTwo"/>
```
````
`````


## timedSynapticInput




extends *basevoltagedeppointcurrentspiking*



Spike array connected to a single **synapse,** producing a current triggered by each  spike in the array.


`````
```` Paths
```Table:

**spikeTarget**$ 

````

```` Component References
```Table:

**synapse**$  $ basesynapse

```
````

```` Children list
```Table:

**spikes**$  $ spike

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current
**tsince**$ Time since the last spike was emitted *(from basevoltagedeppointcurrentspiking)* $dimensions:time

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$ This will receive events from the children$Direction: in
**spike**$ Port on which spikes are emitted *(from basevoltagedeppointcurrentspiking)*$Direction: out

```
````

```` Dynamics

**Structure**
: WITH **this** AS **a**
: WITH **spikeTarget** AS **b**
: CHILD INSTANCE: **synapse**
: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)











**On Events**

: EVENT IN on port: **in**
: **tsince** = 0
: EVENT OUT on port: **spike**





**Derived Variables**
    : **iSyn** =&nbsp;synapse->i
    : **i** =&nbsp;weight * iSyn(exposed as **i**)
    





**Time Derivatives**
    : d **tsince** /dt = 1
    

````

```` Schema
``` xml
<xs:complexType name="TimedSynapticInput">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="spike" type="Spike" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="synapse" type="NmlId" use="required"/>
      <xs:attribute name="spikeTarget" type="xs:string" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=TimedSynapticInput" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import TimedSynapticInput
from neuroml.utils import component_factory

variable = component_factory(
    TimedSynapticInput,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    synapse: 'a NmlId (required)' = None,
    spike_target: 'a string (required)' = None,
    spikes: 'list of Spike(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<timedSynapticInput id="synTrain" synapse="synInputFastTwo" spikeTarget="./synInputFastTwo">
    <spike id="0" time="2 ms"/>
    <spike id="1" time="15 ms"/>
    <spike id="2" time="27 ms"/>
    <spike id="3" time="40 ms"/>
    <spike id="4" time="45 ms"/>
    <spike id="5" time="50 ms"/>
    <spike id="6" time="52 ms"/>
    <spike id="7" time="54 ms"/>
    <spike id="8" time="54.5 ms"/>
    <spike id="9" time="54.6 ms"/>
    <spike id="10" time="54.7 ms"/>
    <spike id="11" time="54.8 ms"/>
    <spike id="12" time="54.9 ms"/>
    <spike id="13" time="55 ms"/>
    <spike id="14" time="55.1 ms"/>
    <spike id="15" time="55.2 ms"/>
</timedSynapticInput>
```
````
`````


## spikeArray




extends *basespikesource*



Set of spike ComponentTypes, each emitting one spike at a certain time. Can be used to feed a predetermined spike train into a cell.


`````
```` Children list
```Table:

**spikes**$  $ spike

```
````

```` Exposures
```Table:

**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**in**$ This will receive events from the children$Direction: in
**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)









**On Start**
: **tsince** = 0


**On Events**

: EVENT IN on port: **in**
: **tsince** = 0
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    

````

```` Schema
``` xml
<xs:complexType name="SpikeArray">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="spike" type="Spike" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeArray" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeArray
from neuroml.utils import component_factory

variable = component_factory(
    SpikeArray,
    id: 'a NonNegativeInteger (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    spikes: 'list of Spike(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<spikeArray id="spkArr">
    <spike id="0" time="50 ms"/>
    <spike id="1" time="100 ms"/>
    <spike id="2" time="150 ms"/>
    <spike id="3" time="155 ms"/>
    <spike id="4" time="250 ms"/>
</spikeArray>
```
````
`````


## spike




extends *basespikesource*



Emits a single spike at the specified **time**.


`````
```` Parameters
```Table:

**time**$ Time at which to emit one spike event $dimensions:time

```
````

```` Exposures
```Table:

**spiked**$ 0 signals not yet spiked, 1 signals has spiked $Dimensionless
**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics

**Structure**
: WITH **this** AS **a**
: WITH **parent** AS **b**
: EVENT CONNECTION from **a** TO  **b**   





**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **spiked**: Dimensionless (exposed as **spiked**)









**On Start**
: **tsince** = 0



**On Conditions**

: IF (t &gt;= time) AND (spiked = 0) THEN
: **spiked** = 1
: **tsince** = 0
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    

````

```` Schema
``` xml
<xs:complexType name="Spike">
  <xs:complexContent>
    <xs:extension base="BaseNonNegativeIntegerId">
      <xs:attribute name="time" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Spike" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Spike
from neuroml.utils import component_factory

variable = component_factory(
    Spike,
    id: 'a NonNegativeInteger (required)' = None,
    time: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<spike id="0" time="50 ms"/>
```
``` xml
<spike id="1" time="100 ms"/>
```
``` xml
<spike id="2" time="150 ms"/>
```
````
`````


## pulseGenerator




extends *basepointcurrent*



Generates a constant current pulse of a certain **amplitude** for a specified **duration** after a **delay.** Scaled by **weight,** if set.


`````
```` Parameters
```Table:

**amplitude**$ Amplitude of current pulse $dimensions:current
**delay**$ Delay before change in current. Current is zero  prior to this. $dimensions:time
**duration**$ Duration for holding current at amplitude. Current is zero after delay + duration. $dimensions:time

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$ Note: this is not used here. Will be removed in future$Direction: in

```
````

```` Dynamics



**State Variables**
: **i**: dimensions:current (exposed as **i**)











**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **i** = 0

: IF t &gt;= delay AND t &lt; duration + delay THEN
: **i** = weight * amplitude

: IF t &gt;= duration + delay THEN
: **i** = 0








````

```` Schema
``` xml
<xs:complexType name="PulseGenerator">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="amplitude" type="Nml2Quantity_current" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=PulseGenerator" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import PulseGenerator
from neuroml.utils import component_factory

variable = component_factory(
    PulseGenerator,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    amplitude: 'a Nml2Quantity_current (required)' = None,
)
```
````
```` Usage: XML
``` xml
<pulseGenerator id="pulseGen1" delay="50ms" duration="200ms" amplitude="0.0032nA"/>
```
``` xml
<pulseGenerator id="pulseGen2" delay="400ms" duration="200ms" amplitude="0.0020nA"/>
```
``` xml
<pulseGenerator id="pulseGen3" delay="700ms" duration="200ms" amplitude="0.0010nA"/>
```
````
`````


## compoundInput




extends *basepointcurrent*



Generates a current which is the sum of all its child  basepointcurrent element, e.g. can be a combination of  pulsegenerator,  sinegenerator elements producing a single **i.** Scaled by **weight,** if set.


`````
```` Children list
```Table:

**currents**$  $ basepointcurrent

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in

```
````

```` Dynamics












**On Events**

: EVENT IN on port: **in**





**Derived Variables**
    : **i_total** =&nbsp;currents[*]->i(reduce method: add)
    : **i** =&nbsp;weight * i_total(exposed as **i**)
    





````

```` Schema
``` xml
<xs:complexType name="CompoundInput">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="pulseGenerator" type="PulseGenerator" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="sineGenerator" type="SineGenerator" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="rampGenerator" type="RampGenerator" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=CompoundInput" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import CompoundInput
from neuroml.utils import component_factory

variable = component_factory(
    CompoundInput,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    pulse_generators: 'list of PulseGenerator(s) (optional)' = None,
    sine_generators: 'list of SineGenerator(s) (optional)' = None,
    ramp_generators: 'list of RampGenerator(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<compoundInput id="ci0">
    <pulseGenerator id="pg1" delay="50ms" duration="200ms" amplitude=".8 nA"/>
    <pulseGenerator id="pg2" delay="100ms" duration="100ms" amplitude=".4 nA"/>
    <sineGenerator id="sg0" phase="0" delay="125ms" duration="50ms" amplitude=".4nA" period="25ms"/>
</compoundInput>
```
````
`````


## compoundInputDL




extends *basepointcurrentdl*



Generates a current which is the sum of all its child  basepointcurrentdl elements, e.g. can be a combination of  pulsegeneratordl,  sinegeneratordl elements producing a single **i.** Scaled by **weight,** if set.


`````
```` Children list
```Table:

**currents**$  $ basepointcurrentdl

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType *(from basepointcurrentdl)* $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in

```
````

```` Dynamics












**On Events**

: EVENT IN on port: **in**





**Derived Variables**
    : **I_total** =&nbsp;currents[*]->I(reduce method: add)
    : **I** =&nbsp;weight * I_total(exposed as **I**)
    





````

```` Schema
``` xml
<xs:complexType name="CompoundInputDL">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="pulseGeneratorDL" type="PulseGeneratorDL" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="sineGeneratorDL" type="SineGeneratorDL" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="rampGeneratorDL" type="RampGeneratorDL" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=CompoundInputDL" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import CompoundInputDL
from neuroml.utils import component_factory

variable = component_factory(
    CompoundInputDL,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    pulse_generator_dls: 'list of PulseGeneratorDL(s) (optional)' = None,
    sine_generator_dls: 'list of SineGeneratorDL(s) (optional)' = None,
    ramp_generator_dls: 'list of RampGeneratorDL(s) (optional)' = None,
)
```
````
`````


## pulseGeneratorDL




extends *basepointcurrentdl*



Dimensionless equivalent of  pulsegenerator. Generates a constant current pulse of a certain **amplitude** for a specified **duration** after a **delay.** Scaled by **weight,** if set.


`````
```` Parameters
```Table:

**amplitude**$ Amplitude of current pulse $Dimensionless
**delay**$ Delay before change in current. Current is zero  prior to this. $dimensions:time
**duration**$ Duration for holding current at amplitude. Current is zero after delay + duration. $dimensions:time

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType *(from basepointcurrentdl)* $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in

```
````

```` Dynamics



**State Variables**
: **I**: Dimensionless (exposed as **I**)











**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **I** = 0

: IF t &gt;= delay AND t &lt; duration + delay THEN
: **I** = weight * amplitude

: IF t &gt;= duration + delay THEN
: **I** = 0








````

```` Schema
``` xml
<xs:complexType name="PulseGeneratorDL">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="amplitude" type="Nml2Quantity_none" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=PulseGeneratorDL" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import PulseGeneratorDL
from neuroml.utils import component_factory

variable = component_factory(
    PulseGeneratorDL,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    amplitude: 'a Nml2Quantity_current (required)' = None,
)
```
````
`````


## sineGenerator




extends *basepointcurrent*



Generates a sinusoidally varying current after a time **delay,** for a fixed **duration.** The **period** and maximum **amplitude** of the current can be set as well as the **phase** at which to start. Scaled by **weight,** if set.


`````
```` Parameters
```Table:

**amplitude**$ Maximum amplitude of current $dimensions:current
**delay**$ Delay before change in current. Current is zero  prior to this. $dimensions:time
**duration**$ Duration for holding current at amplitude. Current is zero after delay + duration. $dimensions:time
**period**$ Time period of oscillation $dimensions:time
**phase**$ Phase (between 0 and 2*pi) at which to start the varying current (i.e. at time given by delay) $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$ $Direction: in

```
````

```` Dynamics



**State Variables**
: **i**: dimensions:current (exposed as **i**)











**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **i** = 0

: IF t &gt;= delay AND t &lt; duration+delay THEN
: **i** = weight * amplitude * sin(phase + (2 * 3.14159265 * (t-delay)/period) )

: IF t &gt;= duration+delay THEN
: **i** = 0








````

```` Schema
``` xml
<xs:complexType name="SineGenerator">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="phase" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="amplitude" type="Nml2Quantity_current" use="required"/>
      <xs:attribute name="period" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SineGenerator" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SineGenerator
from neuroml.utils import component_factory

variable = component_factory(
    SineGenerator,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    phase: 'a Nml2Quantity_none (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    amplitude: 'a Nml2Quantity_current (required)' = None,
    period: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<sineGenerator id="sg0" phase="0" delay="50ms" duration="200ms" amplitude="1.4nA" period="50ms"/>
```
``` xml
<sineGenerator id="sg0" phase="0" delay="125ms" duration="50ms" amplitude=".4nA" period="25ms"/>
```
````
`````


## sineGeneratorDL




extends *basepointcurrentdl*



Dimensionless equivalent of  sinegenerator. Generates a sinusoidally varying current after a time **delay,** for a fixed **duration.** The **period** and maximum **amplitude** of the current can be set as well as the **phase** at which to start. Scaled by **weight,** if set.


`````
```` Parameters
```Table:

**amplitude**$ Maximum amplitude of current $Dimensionless
**delay**$ Delay before change in current. Current is zero  prior to this. $dimensions:time
**duration**$ Duration for holding current at amplitude. Current is zero after delay + duration. $dimensions:time
**period**$ Time period of oscillation $dimensions:time
**phase**$ Phase (between 0 and 2*pi) at which to start the varying current (i.e. at time given by delay) $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType *(from basepointcurrentdl)* $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ $Direction: in

```
````

```` Dynamics



**State Variables**
: **I**: Dimensionless (exposed as **I**)











**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **I** = 0

: IF t &gt;= delay AND t &lt; duration+delay THEN
: **I** = weight * amplitude * sin(phase + (2 * 3.14159265 * (t-delay)/period) )

: IF t &gt;= duration+delay THEN
: **I** = 0








````

```` Schema
``` xml
<xs:complexType name="SineGeneratorDL">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="phase" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="amplitude" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="period" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SineGeneratorDL" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SineGeneratorDL
from neuroml.utils import component_factory

variable = component_factory(
    SineGeneratorDL,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    phase: 'a Nml2Quantity_none (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    amplitude: 'a Nml2Quantity_current (required)' = None,
    period: 'a Nml2Quantity_time (required)' = None,
)
```
````
`````


## rampGenerator




extends *basepointcurrent*



Generates a ramping current after a time **delay,** for a fixed **duration.** During this time the current steadily changes from **startAmplitude** to **finishAmplitude.** Scaled by **weight,** if set.


`````
```` Parameters
```Table:

**baselineAmplitude**$ Amplitude of current before time delay, and after time delay + duration $dimensions:current
**delay**$ Delay before change in current. Current is baselineAmplitude prior to this. $dimensions:time
**duration**$ Duration for holding current at amplitude. Current is baselineAmplitude after delay + duration. $dimensions:time
**finishAmplitude**$ Amplitude of linearly varying current at time delay + duration $dimensions:current
**startAmplitude**$ Amplitude of linearly varying current at time delay $dimensions:current

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Event Ports
```Table:

**in**$ $Direction: in

```
````

```` Dynamics



**State Variables**
: **i**: dimensions:current (exposed as **i**)









**On Start**
: **i** = baselineAmplitude


**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **i** = weight * baselineAmplitude

: IF t &gt;= delay AND t &lt; duration+delay THEN
: **i** = weight * (startAmplitude + (finishAmplitude - startAmplitude) * (t - delay) / (duration))

: IF t &gt;= duration+delay THEN
: **i** = weight * baselineAmplitude








````

```` Schema
``` xml
<xs:complexType name="RampGenerator">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="startAmplitude" type="Nml2Quantity_current" use="required"/>
      <xs:attribute name="finishAmplitude" type="Nml2Quantity_current" use="required"/>
      <xs:attribute name="baselineAmplitude" type="Nml2Quantity_current" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=RampGenerator" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import RampGenerator
from neuroml.utils import component_factory

variable = component_factory(
    RampGenerator,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    start_amplitude: 'a Nml2Quantity_current (required)' = None,
    finish_amplitude: 'a Nml2Quantity_current (required)' = None,
    baseline_amplitude: 'a Nml2Quantity_current (required)' = None,
)
```
````
```` Usage: XML
``` xml
<rampGenerator id="rg0" delay="50ms" duration="200ms" startAmplitude="0.5nA" finishAmplitude="4nA" baselineAmplitude="0nA"/>
```
````
`````


## rampGeneratorDL




extends *basepointcurrentdl*



Dimensionless equivalent of  rampgenerator. Generates a ramping current after a time **delay,** for a fixed **duration.** During this time the dimensionless current steadily changes from **startAmplitude** to **finishAmplitude.** Scaled by **weight,** if set.


`````
```` Parameters
```Table:

**baselineAmplitude**$ Amplitude of current before time delay, and after time delay + duration $Dimensionless
**delay**$ Delay before change in current. Current is baselineAmplitude prior to this. $dimensions:time
**duration**$ Duration for holding current at amplitude. Current is baselineAmplitude after delay + duration. $dimensions:time
**finishAmplitude**$ Amplitude of linearly varying current at time delay + duration $Dimensionless
**startAmplitude**$ Amplitude of linearly varying current at time delay $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**I**$ The total (time varying) current produced by this ComponentType *(from basepointcurrentdl)* $Dimensionless

```
````

```` Event Ports
```Table:

**in**$ $Direction: in

```
````

```` Dynamics



**State Variables**
: **I**: Dimensionless (exposed as **I**)









**On Start**
: **I** = baselineAmplitude


**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **I** = weight * baselineAmplitude

: IF t &gt;= delay AND t &lt; duration+delay THEN
: **I** = weight * (startAmplitude + (finishAmplitude - startAmplitude) * (t - delay) / (duration))

: IF t &gt;= duration+delay THEN
: **I** = weight * baselineAmplitude








````

```` Schema
``` xml
<xs:complexType name="RampGeneratorDL">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="startAmplitude" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="finishAmplitude" type="Nml2Quantity_none" use="required"/>
      <xs:attribute name="baselineAmplitude" type="Nml2Quantity_none" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=RampGeneratorDL" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import RampGeneratorDL
from neuroml.utils import component_factory

variable = component_factory(
    RampGeneratorDL,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    start_amplitude: 'a Nml2Quantity_current (required)' = None,
    finish_amplitude: 'a Nml2Quantity_current (required)' = None,
    baseline_amplitude: 'a Nml2Quantity_current (required)' = None,
)
```
````
`````


## voltageClamp




extends *basevoltagedeppointcurrent*



Voltage clamp. Applies a variable current **i** to try to keep parent at **targetVoltage.** Not yet fully tested!!! Consider using voltageClampTriple!!


`````
```` Parameters
```Table:

**delay**$ Delay before change in current. Current is zero prior to this. $dimensions:time
**duration**$ Duration for attempting to keep parent at targetVoltage. Current is zero after delay + duration. $dimensions:time
**simpleSeriesResistance**$ Current will be calculated by the difference in voltage between the target and parent, divided by this value $dimensions:resistance
**targetVoltage**$ Current will be applied to try to get parent to this target voltage $dimensions:voltage

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in

```
````

```` Dynamics



**State Variables**
: **i**: dimensions:current (exposed as **i**)











**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF t &lt; delay THEN
: **i** = 0

: IF t &gt;= delay THEN
: **i** = weight * (targetVoltage - v) / simpleSeriesResistance

: IF t &gt; duration + delay THEN
: **i** = 0








````

```` Schema
``` xml
<xs:complexType name="VoltageClamp">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="targetVoltage" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="simpleSeriesResistance" type="Nml2Quantity_resistance" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=VoltageClamp" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import VoltageClamp
from neuroml.utils import component_factory

variable = component_factory(
    VoltageClamp,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    target_voltage: 'a Nml2Quantity_voltage (required)' = None,
    simple_series_resistance: 'a Nml2Quantity_resistance (required)' = None,
)
```
````
`````


## voltageClampTriple




extends *basevoltagedeppointcurrent*



Voltage clamp with 3 clamp levels. Applies a variable current **i** ( through **simpleSeriesResistance** ) to try to keep parent cell at **conditioningVoltage** until time **delay,** **testingVoltage** until **delay** + **duration,** and **returnVoltage** afterwards. Only enabled if **active** = 1.


`````
```` Parameters
```Table:

**active**$ Whether the voltage clamp is active (1) or inactive (0). $Dimensionless
**conditioningVoltage**$ Target voltage before time delay $dimensions:voltage
**delay**$ Delay before switching from conditioningVoltage to testingVoltage. $dimensions:time
**duration**$ Duration to hold at testingVoltage. $dimensions:time
**returnVoltage**$ Target voltage after time duration $dimensions:voltage
**simpleSeriesResistance**$ Current will be calculated by the difference in voltage between the target and parent, divided by this value $dimensions:resistance
**testingVoltage**$ Target voltage between times delay and delay + duration $dimensions:voltage

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedeppointcurrent)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$ Note this is not used here. Will be removed in future$Direction: in

```
````

```` Dynamics



**State Variables**
: **i**: dimensions:current (exposed as **i**)











**On Events**

: EVENT IN on port: **in**



**On Conditions**

: IF active = 1 AND t &lt; delay THEN
: **i** = weight * (conditioningVoltage - v) / simpleSeriesResistance

: IF active = 1 AND t &gt;= delay THEN
: **i** = weight * (testingVoltage - v) / simpleSeriesResistance

: IF active = 1 AND t &gt; duration + delay THEN
: **i** = weight * (returnVoltage - v) / simpleSeriesResistance








````

```` Schema
``` xml
<xs:complexType name="VoltageClampTriple">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="active" type="ZeroOrOne" use="required"/>
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="conditioningVoltage" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="testingVoltage" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="returnVoltage" type="Nml2Quantity_voltage" use="required"/>
      <xs:attribute name="simpleSeriesResistance" type="Nml2Quantity_resistance" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=VoltageClampTriple" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import VoltageClampTriple
from neuroml.utils import component_factory

variable = component_factory(
    VoltageClampTriple,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    active: 'a ZeroOrOne (required)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    conditioning_voltage: 'a Nml2Quantity_voltage (required)' = None,
    testing_voltage: 'a Nml2Quantity_voltage (required)' = None,
    return_voltage: 'a Nml2Quantity_voltage (required)' = None,
    simple_series_resistance: 'a Nml2Quantity_resistance (required)' = None,
)
```
````
```` Usage: XML
``` xml
<voltageClampTriple id="vClamp0" active="1" delay="50ms" duration="200ms" conditioningVoltage="-70mV" testingVoltage="-50mV" returnVoltage="-70mV" simpleSeriesResistance="1e6ohm"/>
```
````
`````

# Networks

**Network descriptions for NeuroML 2. Describes  network elements containing  populations ( potentially of type  populationlist, and so specifying a list of cell  locations ),  projections ( i.e. lists of  connections ) and  inputs.**

---


Original ComponentType definitions: [Networks.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//Networks.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## network




extends *basestandalone*



Network containing:  populations ( potentially of type  populationlist, and so specifying a list of cell  locations );  projections ( with lists of  connections ) and/or  explicitconnections; and  inputlists ( with lists of  inputs ) and/or  explicitinputs. Note: often in NeuroML this will be of type  networkwithtemperature if there are temperature dependent elements ( e.g. ion channels ).


`````
```` Children list
```Table:

**regions**$  $ region
**populations**$  $ basepopulation
**projections**$  $ projection
**synapticConnections**$  $ explicitconnection
**electricalProjection**$  $ electricalprojection
**continuousProjection**$  $ continuousprojection
**explicitInputs**$  $ explicitinput
**inputs**$  $ inputlist

```
````

```` Schema
``` xml
<xs:complexType name="Network">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:sequence>
        <xs:element name="space" type="Space" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="region" type="Region" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="extracellularProperties" type="ExtracellularPropertiesLocal" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="population" type="Population" maxOccurs="unbounded"/>
        <xs:element name="cellSet" type="CellSet" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="synapticConnection" type="SynapticConnection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="projection" type="Projection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="electricalProjection" type="ElectricalProjection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="continuousProjection" type="ContinuousProjection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="explicitInput" type="ExplicitInput" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="inputList" type="InputList" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="type" type="networkTypes" use="optional"/>
      <xs:attribute name="temperature" type="Nml2Quantity_temperature" use="optional"/>
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Network" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Network
from neuroml.utils import component_factory

variable = component_factory(
    Network,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    type: 'a networkTypes (optional)' = None,
    temperature: 'a Nml2Quantity_temperature (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    spaces: 'list of Space(s) (optional)' = None,
    regions: 'list of Region(s) (optional)' = None,
    extracellular_properties: 'list of ExtracellularPropertiesLocal(s) (optional)' = None,
    populations: 'list of Population(s) (required)' = None,
    cell_sets: 'list of CellSet(s) (optional)' = None,
    synaptic_connections: 'list of SynapticConnection(s) (optional)' = None,
    projections: 'list of Projection(s) (optional)' = None,
    electrical_projections: 'list of ElectricalProjection(s) (optional)' = None,
    continuous_projections: 'list of ContinuousProjection(s) (optional)' = None,
    explicit_inputs: 'list of ExplicitInput(s) (optional)' = None,
    input_lists: 'list of InputList(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<network id="net1">
    <population id="iafPop1" component="iaf" size="1"/>
    <population id="iafPop2" component="iaf" size="1"/>
    <population id="iafPop3" component="iaf" size="1"/>
    <continuousProjection id="testLinearGradedConn" presynapticPopulation="iafPop1" postsynapticPopulation="iafPop2">
        <continuousConnection id="0" preCell="0" postCell="0" preComponent="silent1" postComponent="gs1"/>
    </continuousProjection>
    <continuousProjection id="testGradedConn" presynapticPopulation="iafPop1" postsynapticPopulation="iafPop3">
        <continuousConnection id="0" preCell="0" postCell="0" preComponent="silent2" postComponent="gs2"/>
    </continuousProjection>
    <explicitInput target="iafPop1[0]" input="pulseGen1" destination="synapses"/>
    <explicitInput target="iafPop1[0]" input="pulseGen2" destination="synapses"/>
    <explicitInput target="iafPop1[0]" input="pulseGen3" destination="synapses"/>
</network>
```
``` xml
<network id="net2">
    <population id="hhPop1" component="hhcell" size="1" type="populationList">
        <instance id="0">
            <location x="0" y="0" z="0"/>
        </instance>
    </population>
    <population id="hhPop2" component="hhcell" size="1" type="populationList">
        <instance id="0">
            <location x="100" y="0" z="0"/>
        </instance>
    </population>
    <continuousProjection id="testGradedConn" presynapticPopulation="hhPop1" postsynapticPopulation="hhPop2">
        <continuousConnectionInstanceW id="0" preCell="../hhPop1/0/hhcell" postCell="../hhPop2/0/hhcell" preComponent="silent1" postComponent="gs1" weight="1"/>
    </continuousProjection>
    <inputList id="i1" component="pulseGen1" population="hhPop1">
        <input id="0" target="../hhPop1/0/hhcell" destination="synapses"/>
    </inputList>
</network>
```
``` xml
<network id="PyrCellNet">
    <population id="Population1" component="PyrCell" extracellularProperties="extracellular" size="9"> 
        </population>
    <projection id="Proj1" presynapticPopulation="Population1" postsynapticPopulation="Population1" synapse="AMPA">
           
        </projection>
</network>
```
````
`````


## networkWithTemperature




extends network



Same as  network, but with an explicit **temperature** for temperature dependent elements ( e.g. ion channels ).


`````
```` Parameters
```Table:

**temperature**$  $dimensions:temperature

```
````
`````


## *basePopulation*




extends *basestandalone*



A population of multiple instances of a specific **component,** which anything which extends  basecell.


`````
```` Component References
```Table:

**component**$  $ basecell

```
````

```` Child list
```Table:

**notes**$  $ notes
**annotation**$  $ annotation

```
````

```` Children list
```Table:

**property**$  $ property

```
````
`````


## population




extends *basepopulation*



A population of components, with just one parameter for the **size,** i.e. number of components to create. Note: quite often this is used with type= populationlist which means the size is determined by the number of  instances ( with  locations ) in the list. The **size** attribute is still set, and there will be a validation error if this does not match the number in the list.


`````
```` Parameters
```Table:

**size**$ Number of instances of this Component to create when the population is instantiated $Dimensionless

```
````

```` Schema
``` xml
<xs:complexType name="Population">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:choice>
        <xs:element name="layout" type="Layout" minOccurs="0"/>
        <xs:element name="instance" type="Instance" maxOccurs="unbounded"/>
      </xs:choice>
      <xs:attribute name="component" type="NmlId" use="required"/>
      <xs:attribute name="size" type="NonNegativeInteger" use="optional"/>
      <xs:attribute name="type" type="populationTypes" use="optional"/>
      <xs:attribute name="extracellularProperties" type="NmlId" use="optional"/>
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Population" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Population
from neuroml.utils import component_factory

variable = component_factory(
    Population,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    component: 'a NmlId (required)' = None,
    size: 'a NonNegativeInteger (optional)' = None,
    type: 'a populationTypes (optional)' = None,
    extracellular_properties: 'a NmlId (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    layout: 'a Layout (optional)' = None,
    instances: 'list of Instance(s) (required)' = None,
)
```
````
```` Usage: XML
``` xml
<population id="iafPop1" component="iaf" size="1"/>
```
``` xml
<population id="iafPop2" component="iaf" size="1"/>
```
``` xml
<population id="iafPop3" component="iaf" size="1"/>
```
````
`````


## populationList




extends *basepopulation*



An explicit list of  instances ( with  locations ) of components in the population.


`````
```` Text fields
```Table:

**size**$ Note: the size of the populationList to create is set by the number of explicitly defined instances. The size attribute is still set, and there will be a validation error if this does not match the number in the list.

````

```` Children list
```Table:

**instances**$  $ instance

```
````
`````


## instance




Specifies a single instance of a component in a  population ( placed at  location ).


`````
```` Child list
```Table:

**location**$  $ location

```
````

```` Schema
``` xml
<xs:complexType name="Instance">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:sequence>
        <xs:element name="location" type="Location"/>
      </xs:sequence>
      <xs:attribute name="id" type="xs:nonNegativeInteger"/>
      <xs:attribute name="i" type="xs:nonNegativeInteger"/>
      <xs:attribute name="j" type="xs:nonNegativeInteger"/>
      <xs:attribute name="k" type="xs:nonNegativeInteger"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Instance" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Instance
from neuroml.utils import component_factory

variable = component_factory(
    Instance,
    id: 'a nonNegativeInteger (optional)' = None,
    i: 'a nonNegativeInteger (optional)' = None,
    j: 'a nonNegativeInteger (optional)' = None,
    k: 'a nonNegativeInteger (optional)' = None,
    location: 'a Location (required)' = None,
)
```
````
```` Usage: XML
``` xml
<instance id="0">
    <location x="0" y="0" z="0"/>
</instance>
```
``` xml
<instance id="0">
    <location x="100" y="0" z="0"/>
</instance>
```
``` xml
<instance id="0">
    <location x="0" y="0" z="0"/>
</instance>
```
````
`````


## location




Specifies the ( x, y, z ) location of a single  instance of a component in a  population.


`````
```` Parameters
```Table:

**x**$  $Dimensionless
**y**$  $Dimensionless
**z**$  $Dimensionless

```
````

```` Schema
``` xml
<xs:complexType name="Location">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="x" type="xs:float" use="required"/>
      <xs:attribute name="y" type="xs:float" use="required"/>
      <xs:attribute name="z" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Location" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Location
from neuroml.utils import component_factory

variable = component_factory(
    Location,
    x: 'a float (required)' = None,
    y: 'a float (required)' = None,
    z: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<location x="0" y="0" z="0"/>
```
``` xml
<location x="100" y="0" z="0"/>
```
``` xml
<location x="0" y="0" z="0"/>
```
````
`````


## region




Initial attempt to specify 3D region for placing cells. Work in progress...


`````
```` Child list
```Table:

**rectangularExtent**$  $ rectangularextent

```
````

```` Schema
``` xml
<xs:complexType name="Region">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:any processContents="skip" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="space" type="NmlId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Region" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Region
from neuroml.utils import component_factory

variable = component_factory(
    Region,
    id: 'a NmlId (required)' = None,
    spaces: 'a NmlId (optional)' = None,
    anytypeobjs_=None,
)
```
````
`````


## rectangularExtent




For defining a 3D rectangular box.


`````
```` Parameters
```Table:

**xLength**$  $Dimensionless
**xStart**$  $Dimensionless
**yLength**$  $Dimensionless
**yStart**$  $Dimensionless
**zLength**$  $Dimensionless
**zStart**$  $Dimensionless

```
````
`````


## projection




Projection from one population, **presynapticPopulation** to another, **postsynapticPopulation,** through **synapse.** Contains lists of  connection or  connectionwd elements.


`````
```` Paths
```Table:

**presynapticPopulation**$ 
**postsynapticPopulation**$ 

````

```` Component References
```Table:

**synapse**$  $ basesynapse

```
````

```` Children list
```Table:

**connections**$  $ connection
**connectionsWD**$  $ connectionwd

```
````

```` Schema
``` xml
<xs:complexType name="Projection">
  <xs:complexContent>
    <xs:extension base="BaseProjection">
      <xs:sequence>
        <xs:element name="connection" type="Connection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="connectionWD" type="ConnectionWD" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="synapse" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Projection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Projection
from neuroml.utils import component_factory

variable = component_factory(
    Projection,
    id: 'a NmlId (required)' = None,
    presynaptic_population: 'a NmlId (required)' = None,
    postsynaptic_population: 'a NmlId (required)' = None,
    synapse: 'a NmlId (required)' = None,
    connections: 'list of Connection(s) (optional)' = None,
    connection_wds: 'list of ConnectionWD(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<projection id="Proj1" presynapticPopulation="Population1" postsynapticPopulation="Population1" synapse="AMPA">
           
        </projection>
```
``` xml
<projection id="internal1" presynapticPopulation="iafCells" postsynapticPopulation="iafCells" synapse="syn1">
            <synapseComponent component="syn1"/>-->
    <connection id="0" preCellId="../iafCells/0/iaf" postCellId="../iafCells/1/iaf"/>
</projection>
```
``` xml
<projection id="internal2" presynapticPopulation="iafCells" postsynapticPopulation="iafCells" synapse="syn2">
    <connection id="0" preCellId="../iafCells/0/iaf" postCellId="../iafCells/2/iaf"/>
</projection>
```
````
`````


## explicitConnection




Explicit event connection between components.


`````
```` Text fields
```Table:

**targetPort**$ 

````

```` Paths
```Table:

**from**$ 
**to**$ 

````
`````


## connection




Event connection directly between named components, which gets processed via a new instance of a **synapse** component which is created on the target component. Normally contained inside a  projection element.


`````
```` Text fields
```Table:

**destination**$ 
**preFractionAlong**$ 
**postFractionAlong**$ 
**preSegmentId**$ 
**postSegmentId**$ 

````

```` Paths
```Table:

**preCellId**$ 
**postCellId**$ 

````

```` Schema
``` xml
<xs:complexType name="Connection">
  <xs:complexContent>
    <xs:extension base="BaseConnectionOldFormat">
            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Connection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Connection
from neuroml.utils import component_factory

variable = component_factory(
    Connection,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell_id: 'a Nml2PopulationReferencePath (required)' = None,
    pre_segment_id: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell_id: 'a Nml2PopulationReferencePath (required)' = None,
    post_segment_id: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
)
```
````
```` Usage: XML
``` xml
<connection id="0" preCellId="../iafCells/0/iaf" postCellId="../iafCells/1/iaf"/>
```
``` xml
<connection id="0" preCellId="../iafCells/0/iaf" postCellId="../iafCells/2/iaf"/>
```
``` xml
<connection id="0" preCellId="../pop0/0/MultiCompCell" postCellId="../pop0/1/MultiCompCell" preSegmentId="0" preFractionAlong="0.5" postSegmentId="0" postFractionAlong="0.5"/>
```
````
`````


## synapticConnection




extends explicitconnection



Explicit event connection between named components, which gets processed via a new instance of a **synapse** component which is created on the target component.


`````
```` Text fields
```Table:

**destination**$ 

````

```` Paths
```Table:

**from**$ 
**to**$ 

````

```` Component References
```Table:

**synapse**$  $ basesynapse

```
````

```` Schema
``` xml
<xs:complexType name="SynapticConnection">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="neuroLexId" type="NeuroLexId" use="optional"/>
      <xs:attribute name="from" type="Nml2PopulationReferencePath" use="required"/>
      <xs:attribute name="to" type="Nml2PopulationReferencePath" use="required"/>
      <xs:attribute name="synapse" type="NmlId" use="required"/>
      <xs:attribute name="destination" type="NmlId" use="optional"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SynapticConnection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SynapticConnection
from neuroml.utils import component_factory

variable = component_factory(
    SynapticConnection,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    from_: 'a Nml2PopulationReferencePath (required)' = None,
    to: 'a Nml2PopulationReferencePath (required)' = None,
    synapse: 'a NmlId (required)' = None,
    destination: 'a NmlId (optional)' = None,
)
```
````
`````


## synapticConnectionWD




extends synapticconnection



Explicit event connection between named components, which gets processed via a new instance of a **synapse** component which is created on the target component, includes setting of **weight** and **delay** for the synaptic connection.


`````
```` Parameters
```Table:

**delay**$  $dimensions:time
**weight**$  $Dimensionless

```
````

```` Paths
```Table:

**from**$ 
**to**$ 

````
`````


## connectionWD




extends connection



Event connection between named components, which gets processed via a new instance of a synapse component which is created on the target component, includes setting of **weight** and **delay** for the synaptic connection.


`````
```` Parameters
```Table:

**delay**$  $dimensions:time
**weight**$  $Dimensionless

```
````

```` Text fields
```Table:

**destination**$ 
**preFractionAlong**$ 
**postFractionAlong**$ 
**preSegmentId**$ 
**postSegmentId**$ 

````

```` Paths
```Table:

**preCellId**$ 
**postCellId**$ 

````

```` Schema
``` xml
<xs:complexType name="ConnectionWD">
  <xs:complexContent>
    <xs:extension base="BaseConnectionOldFormat">
      <xs:attribute name="weight" type="xs:float" use="required"/>
      <xs:attribute name="delay" type="Nml2Quantity_time" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ConnectionWD" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ConnectionWD
from neuroml.utils import component_factory

variable = component_factory(
    ConnectionWD,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell_id: 'a Nml2PopulationReferencePath (required)' = None,
    pre_segment_id: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell_id: 'a Nml2PopulationReferencePath (required)' = None,
    post_segment_id: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    weight: 'a float (required)' = None,
    delay: 'a Nml2Quantity_time (required)' = None,
)
```
````
```` Usage: XML
``` xml
<connectionWD id="0" preCellId="../pop_EIF_cond_exp_isfa_ista[0]" postCellId="../pop_target[0]" weight="0.01" delay="10ms"/>
```
``` xml
<connectionWD id="0" preCellId="../pop_EIF_cond_alpha_isfa_ista[0]" postCellId="../pop_target[1]" weight="0.005" delay="20ms"/>
```
``` xml
<connectionWD id="0" preCellId="../pop_IF_curr_alpha[0]" postCellId="../pop_target[2]" weight="1" delay="30ms"/>
```
````
`````


## electricalConnection




To enable connections between populations through gap junctions.


`````
```` Component References
```Table:

**synapse**$  $ gapjunction

```
````

```` Schema
``` xml
<xs:complexType name="ElectricalConnection">
  <xs:complexContent>
    <xs:extension base="BaseConnectionNewFormat">
      <xs:attribute name="synapse" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ElectricalConnection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ElectricalConnection
from neuroml.utils import component_factory

variable = component_factory(
    ElectricalConnection,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell: 'a string (required)' = None,
    pre_segment: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell: 'a string (required)' = None,
    post_segment: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    synapse: 'a NmlId (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<electricalConnection id="0" preCell="0" postCell="0" synapse="gj1"/>
```
````
`````


## electricalConnectionInstance




To enable connections between populations through gap junctions. Populations need to be of type  populationlist and contain  instance and  location elements.


`````
```` Text fields
```Table:

**preFractionAlong**$ 
**postFractionAlong**$ 
**preSegment**$ 
**postSegment**$ 

````

```` Paths
```Table:

**preCell**$ 
**postCell**$ 

````

```` Component References
```Table:

**synapse**$  $ gapjunction

```
````

```` Schema
``` xml
<xs:complexType name="ElectricalConnectionInstance">
  <xs:complexContent>
    <xs:extension base="ElectricalConnection"/>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ElectricalConnectionInstance" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ElectricalConnectionInstance
from neuroml.utils import component_factory

variable = component_factory(
    ElectricalConnectionInstance,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell: 'a string (required)' = None,
    pre_segment: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell: 'a string (required)' = None,
    post_segment: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    synapse: 'a NmlId (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<electricalConnectionInstance id="0" preCell="../iafPop1/0/iaf" postCell="../iafPop2/0/iaf" preSegment="0" preFractionAlong="0.5" postSegment="0" postFractionAlong="0.5" synapse="gj1"/>
```
````
`````


## electricalConnectionInstanceW




extends electricalconnectioninstance



To enable connections between populations through gap junctions. Populations need to be of type  populationlist and contain  instance and  location elements. Includes setting of **weight** for the connection.


`````
```` Parameters
```Table:

**weight**$  $Dimensionless

```
````

```` Text fields
```Table:

**preFractionAlong**$ 
**postFractionAlong**$ 
**preSegment**$ 
**postSegment**$ 

````

```` Paths
```Table:

**preCell**$ 
**postCell**$ 

````

```` Schema
``` xml
<xs:complexType name="ElectricalConnectionInstanceW">
  <xs:complexContent>
    <xs:extension base="ElectricalConnectionInstance">
      <xs:attribute name="weight" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ElectricalConnectionInstanceW" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ElectricalConnectionInstanceW
from neuroml.utils import component_factory

variable = component_factory(
    ElectricalConnectionInstanceW,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell: 'a string (required)' = None,
    pre_segment: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell: 'a string (required)' = None,
    post_segment: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    synapse: 'a NmlId (required)' = None,
    weight: 'a float (required)' = None,
)
```
````
`````


## electricalProjection




A projection between **presynapticPopulation** to another **postsynapticPopulation** through gap junctions.


`````
```` Component References
```Table:

**presynapticPopulation**$  $ population
**postsynapticPopulation**$  $ population

```
````

```` Children list
```Table:

**connections**$  $ electricalconnection
**connectionInstances**$  $ electricalconnectioninstance

```
````

```` Schema
``` xml
<xs:complexType name="ElectricalProjection">
  <xs:complexContent>
    <xs:extension base="BaseProjection">
      <xs:sequence>
        <xs:element name="electricalConnection" type="ElectricalConnection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="electricalConnectionInstance" type="ElectricalConnectionInstance" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="electricalConnectionInstanceW" type="ElectricalConnectionInstanceW" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ElectricalProjection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ElectricalProjection
from neuroml.utils import component_factory

variable = component_factory(
    ElectricalProjection,
    id: 'a NmlId (required)' = None,
    presynaptic_population: 'a NmlId (required)' = None,
    postsynaptic_population: 'a NmlId (required)' = None,
    electrical_connections: 'list of ElectricalConnection(s) (optional)' = None,
    electrical_connection_instances: 'list of ElectricalConnectionInstance(s) (optional)' = None,
    electrical_connection_instance_ws: 'list of ElectricalConnectionInstanceW(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<electricalProjection id="testGJconn" presynapticPopulation="iafPop1" postsynapticPopulation="iafPop2">
    <electricalConnectionInstance id="0" preCell="../iafPop1/0/iaf" postCell="../iafPop2/0/iaf" preSegment="0" preFractionAlong="0.5" postSegment="0" postFractionAlong="0.5" synapse="gj1"/>
</electricalProjection>
```
``` xml
<electricalProjection id="testGJconn" presynapticPopulation="iafPop1" postsynapticPopulation="iafPop2">
    <electricalConnection id="0" preCell="0" postCell="0" synapse="gj1"/>
</electricalProjection>
```
````
`````


## continuousConnection




An instance of a connection in a  continuousprojection between **presynapticPopulation** to another **postsynapticPopulation** through a **preComponent** at the start and **postComponent** at the end. Can be used for analog synapses.


`````
```` Component References
```Table:

**preComponent**$  $ basegradedsynapse
**postComponent**$  $ basegradedsynapse

```
````

```` Schema
``` xml
<xs:complexType name="ContinuousConnection">
  <xs:complexContent>
    <xs:extension base="BaseConnectionNewFormat">
      <xs:attribute name="preComponent" type="NmlId" use="required"/>
      <xs:attribute name="postComponent" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ContinuousConnection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ContinuousConnection
from neuroml.utils import component_factory

variable = component_factory(
    ContinuousConnection,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell: 'a string (required)' = None,
    pre_segment: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell: 'a string (required)' = None,
    post_segment: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    pre_component: 'a NmlId (required)' = None,
    post_component: 'a NmlId (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<continuousConnection id="0" preCell="0" postCell="0" preComponent="silent1" postComponent="gs1"/>
```
``` xml
<continuousConnection id="0" preCell="0" postCell="0" preComponent="silent2" postComponent="gs2"/>
```
````
`````


## continuousConnectionInstance




An instance of a connection in a  continuousprojection between **presynapticPopulation** to another **postsynapticPopulation** through a **preComponent** at the start and **postComponent** at the end. Populations need to be of type  populationlist and contain  instance and  location elements. Can be used for analog synapses.


`````
```` Text fields
```Table:

**preFractionAlong**$ 
**postFractionAlong**$ 
**preSegment**$ 
**postSegment**$ 

````

```` Paths
```Table:

**preCell**$ 
**postCell**$ 

````

```` Component References
```Table:

**preComponent**$  $ basegradedsynapse
**postComponent**$  $ basegradedsynapse

```
````

```` Schema
``` xml
<xs:complexType name="ContinuousConnectionInstance">
  <xs:complexContent>
    <xs:extension base="ContinuousConnection"/>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ContinuousConnectionInstance" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ContinuousConnectionInstance
from neuroml.utils import component_factory

variable = component_factory(
    ContinuousConnectionInstance,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell: 'a string (required)' = None,
    pre_segment: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell: 'a string (required)' = None,
    post_segment: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    pre_component: 'a NmlId (required)' = None,
    post_component: 'a NmlId (required)' = None,
    extensiontype_=None,
)
```
````
`````


## continuousConnectionInstanceW




extends continuousconnectioninstance



An instance of a connection in a  continuousprojection between **presynapticPopulation** to another **postsynapticPopulation** through a **preComponent** at the start and **postComponent** at the end. Populations need to be of type  populationlist and contain  instance and  location elements. Can be used for analog synapses. Includes setting of **weight** for the connection.


`````
```` Parameters
```Table:

**weight**$  $Dimensionless

```
````

```` Text fields
```Table:

**preFractionAlong**$ 
**postFractionAlong**$ 
**preSegment**$ 
**postSegment**$ 

````

```` Paths
```Table:

**preCell**$ 
**postCell**$ 

````

```` Schema
``` xml
<xs:complexType name="ContinuousConnectionInstanceW">
  <xs:complexContent>
    <xs:extension base="ContinuousConnectionInstance">
      <xs:attribute name="weight" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ContinuousConnectionInstanceW" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ContinuousConnectionInstanceW
from neuroml.utils import component_factory

variable = component_factory(
    ContinuousConnectionInstanceW,
    id: 'a NonNegativeInteger (required)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    pre_cell: 'a string (required)' = None,
    pre_segment: 'a NonNegativeInteger (optional)' = '0',
    pre_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    post_cell: 'a string (required)' = None,
    post_segment: 'a NonNegativeInteger (optional)' = '0',
    post_fraction_along: 'a ZeroToOne (optional)' = '0.5',
    pre_component: 'a NmlId (required)' = None,
    post_component: 'a NmlId (required)' = None,
    weight: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<continuousConnectionInstanceW id="0" preCell="../hhPop1/0/hhcell" postCell="../hhPop2/0/hhcell" preComponent="silent1" postComponent="gs1" weight="1"/>
```
````
`````


## continuousProjection




A projection between **presynapticPopulation** and **postsynapticPopulation** through components **preComponent** at the start and **postComponent** at the end of a  continuousconnection or  continuousconnectioninstance. Can be used for analog synapses.


`````
```` Component References
```Table:

**presynapticPopulation**$  $ population
**postsynapticPopulation**$  $ population

```
````

```` Children list
```Table:

**connections**$  $ continuousconnection
**connectionInstances**$  $ continuousconnectioninstance

```
````

```` Schema
``` xml
<xs:complexType name="ContinuousProjection">
  <xs:complexContent>
    <xs:extension base="BaseProjection">
      <xs:sequence>
        <xs:element name="continuousConnection" type="ContinuousConnection" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="continuousConnectionInstance" type="ContinuousConnectionInstance" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="continuousConnectionInstanceW" type="ContinuousConnectionInstanceW" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ContinuousProjection" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ContinuousProjection
from neuroml.utils import component_factory

variable = component_factory(
    ContinuousProjection,
    id: 'a NmlId (required)' = None,
    presynaptic_population: 'a NmlId (required)' = None,
    postsynaptic_population: 'a NmlId (required)' = None,
    continuous_connections: 'list of ContinuousConnection(s) (optional)' = None,
    continuous_connection_instances: 'list of ContinuousConnectionInstance(s) (optional)' = None,
    continuous_connection_instance_ws: 'list of ContinuousConnectionInstanceW(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<continuousProjection id="testLinearGradedConn" presynapticPopulation="iafPop1" postsynapticPopulation="iafPop2">
    <continuousConnection id="0" preCell="0" postCell="0" preComponent="silent1" postComponent="gs1"/>
</continuousProjection>
```
``` xml
<continuousProjection id="testGradedConn" presynapticPopulation="iafPop1" postsynapticPopulation="iafPop3">
    <continuousConnection id="0" preCell="0" postCell="0" preComponent="silent2" postComponent="gs2"/>
</continuousProjection>
```
``` xml
<continuousProjection id="testGradedConn" presynapticPopulation="hhPop1" postsynapticPopulation="hhPop2">
    <continuousConnectionInstanceW id="0" preCell="../hhPop1/0/hhcell" postCell="../hhPop2/0/hhcell" preComponent="silent1" postComponent="gs1" weight="1"/>
</continuousProjection>
```
````
`````


## explicitInput




An explicit input ( anything which extends  basepointcurrent ) to a target cell in a population.


`````
```` Text fields
```Table:

**destination**$ 
**sourcePort**$ 
**targetPort**$ 

````

```` Paths
```Table:

**target**$ 

````

```` Component References
```Table:

**input**$  $ basepointcurrent

```
````

```` Schema
``` xml
<xs:complexType name="ExplicitInput">
  <xs:complexContent>
    <xs:extension base="BaseWithoutId">
      <xs:attribute name="target" type="Nml2PopulationReferencePath" use="required"/>
      <xs:attribute name="input" type="NmlId" use="required"/>
      <xs:attribute name="destination" type="NmlId"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ExplicitInput" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ExplicitInput
from neuroml.utils import component_factory

variable = component_factory(
    ExplicitInput,
    target: 'a Nml2PopulationReferencePath (required)' = None,
    input: 'a NmlId (required)' = None,
    destination: 'a NmlId (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<explicitInput target="iafPop1[0]" input="pulseGen1" destination="synapses"/>
```
``` xml
<explicitInput target="iafPop1[0]" input="pulseGen2" destination="synapses"/>
```
``` xml
<explicitInput target="iafPop1[0]" input="pulseGen3" destination="synapses"/>
```
````
`````


## inputList




An explicit list of  inputs to a **population.**.


`````
```` Text fields
```Table:

**population**$ 

````

```` Component References
```Table:

**component**$  $ basepointcurrent

```
````

```` Children list
```Table:

**inputs**$  $ input

```
````

```` Schema
``` xml
<xs:complexType name="InputList">
  <xs:complexContent>
    <xs:extension base="Base">
      <xs:sequence>
        <xs:element name="input" type="Input" minOccurs="0" maxOccurs="unbounded"/>
        <xs:element name="inputW" type="InputW" minOccurs="0" maxOccurs="unbounded"/>
      </xs:sequence>
      <xs:attribute name="population" type="NmlId" use="required"/>
      <xs:attribute name="component" type="NmlId" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=InputList" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import InputList
from neuroml.utils import component_factory

variable = component_factory(
    InputList,
    id: 'a NonNegativeInteger (required)' = None,
    populations: 'a NmlId (required)' = None,
    component: 'a NmlId (required)' = None,
    input: 'list of Input(s) (optional)' = None,
    input_ws: 'list of InputW(s) (optional)' = None,
)
```
````
```` Usage: XML
``` xml
<inputList id="i1" component="pulseGen1" population="hhPop1">
    <input id="0" target="../hhPop1/0/hhcell" destination="synapses"/>
</inputList>
```
``` xml
<inputList id="i1" component="pulseGen1" population="iafPop1">
    <input id="0" target="../iafPop1/0/iaf" destination="synapses"/>
</inputList>
```
``` xml
<inputList id="i2" component="pulseGen2" population="iafPop2">
    <input id="0" target="../iafPop2/0/iaf" destination="synapses"/>
</inputList>
```
````
`````


## input




Specifies a single input to a **target,** optionally giving the **segmentId** ( default 0 ) and **fractionAlong** the segment ( default 0.5 ).


`````
```` Text fields
```Table:

**segmentId**$ Optional specification of the segment to target, default 0
**fractionAlong**$ Optional specification of the fraction along the specified segment, default 0.5
**destination**$ 

````

```` Paths
```Table:

**target**$ 

````

```` Schema
``` xml
<xs:complexType name="Input">
  <xs:complexContent>
    <xs:extension base="BaseNonNegativeIntegerId">
      <xs:attribute name="target" type="Nml2PopulationReferencePath" use="required"/>
      <xs:attribute name="destination" type="NmlId" use="required"/>
      <xs:attribute name="segmentId" type="NonNegativeInteger"/>
      <xs:attribute name="fractionAlong" type="ZeroToOne"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=Input" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import Input
from neuroml.utils import component_factory

variable = component_factory(
    Input,
    id: 'a NonNegativeInteger (required)' = None,
    target: 'a Nml2PopulationReferencePath (required)' = None,
    destination: 'a NmlId (required)' = None,
    segment_id: 'a NonNegativeInteger (optional)' = None,
    fraction_along: 'a ZeroToOne (optional)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<input id="0" target="../hhPop1/0/hhcell" destination="synapses"/>
```
``` xml
<input id="0" target="../iafPop1/0/iaf" destination="synapses"/>
```
``` xml
<input id="0" target="../iafPop2/0/iaf" destination="synapses"/>
```
````
`````


## inputW




extends input



Specifies input lists. Can set **weight** to scale individual inputs.


`````
```` Parameters
```Table:

**weight**$  $Dimensionless

```
````

```` Text fields
```Table:

**destination**$ 

````

```` Paths
```Table:

**target**$ 

````

```` Schema
``` xml
<xs:complexType name="InputW">
  <xs:complexContent>
    <xs:extension base="Input">
      <xs:attribute name="weight" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=InputW" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import InputW
from neuroml.utils import component_factory

variable = component_factory(
    InputW,
    id: 'a NonNegativeInteger (required)' = None,
    target: 'a Nml2PopulationReferencePath (required)' = None,
    destination: 'a NmlId (required)' = None,
    segment_id: 'a NonNegativeInteger (optional)' = None,
    fraction_along: 'a ZeroToOne (optional)' = None,
    weight: 'a float (required)' = None,
)
```
````
`````

# PyNN

**A number of ComponentType description of PyNN standard cells. All of the cells extend  basepynncell, and the synapses extend  basepynnsynapse.**

---


Original ComponentType definitions: [PyNN.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//PyNN.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## *basePyNNCell*




extends *basecellmembpot*



Base type of any PyNN standard cell model. Note: membrane potential **v** has dimensions voltage, but all other parameters are dimensionless. This is to facilitate translation to and from PyNN scripts in Python, where these parameters have implicit units, see http://neuralensemble.org/trac/PyNN/wiki/StandardModels.


`````
```` Parameters
```Table:

**cm**$  $Dimensionless
**i_offset**$  $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell $Dimensionless
**v_init**$  $Dimensionless

```
````

```` Constants
```Table:

**MSEC** = 1ms$  $ dimensions:time
**MVOLT** = 1mV$  $ dimensions:voltage
**NFARAD** = 1nF$  $ dimensions:capacitance

```
````

```` Exposures
```Table:

**iSyn**$  $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$ $Direction: in
**spike_in_I**$ $Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="basePyNNCell">
  <xs:complexContent>
    <xs:extension base="BaseCell">
      <xs:attribute name="cm" type="xs:float" use="required"/>
      <xs:attribute name="i_offset" type="xs:float" use="required"/>
      <xs:attribute name="tau_syn_E" type="xs:float" use="required"/>
      <xs:attribute name="tau_syn_I" type="xs:float" use="required"/>
      <xs:attribute name="v_init" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## *basePyNNIaFCell*




extends *basepynncell*



Base type of any PyNN standard integrate and fire model.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  $Dimensionless
**tau_refrac**$  $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  $Dimensionless
**v_rest**$  $Dimensionless
**v_thresh**$  $Dimensionless

```
````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="basePyNNIaFCell">
  <xs:complexContent>
    <xs:extension base="basePyNNCell">
      <xs:attribute name="tau_m" type="xs:float" use="required"/>
      <xs:attribute name="tau_refrac" type="xs:float" use="required"/>
      <xs:attribute name="v_reset" type="xs:float" use="required"/>
      <xs:attribute name="v_rest" type="xs:float" use="required"/>
      <xs:attribute name="v_thresh" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## *basePyNNIaFCondCell*




extends *basepynniafcell*



Base type of conductance based PyNN IaF cell models.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**e_rev_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell $Dimensionless
**e_rev_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="basePyNNIaFCondCell">
  <xs:complexContent>
    <xs:extension base="basePyNNIaFCell">
      <xs:attribute name="e_rev_E" type="xs:float" use="required"/>
      <xs:attribute name="e_rev_I" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````
`````


## IF_curr_alpha




extends *basepynniafcell*



Leaky integrate and fire model with fixed threshold and alpha-function-shaped post-synaptic current.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = v_init * MVOLT





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    






**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = v_reset \* MVOLT
: **On Conditions**
:  IF t &gt; lastSpikeTime + (tau_refrac*MSEC) THEN
: TRANSITION to REGIME **integrating**

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; v_thresh * MVOLT THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (MVOLT * ((i_offset/cm) +  ((v_rest - (v/MVOLT)) / tau_m))/MSEC) + (iSyn / (cm * NFARAD))
````

```` Schema
``` xml
<xs:complexType name="IF_curr_alpha">
  <xs:complexContent>
    <xs:extension base="basePyNNIaFCell">
            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IF_curr_alpha" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IF_curr_alpha
from neuroml.utils import component_factory

variable = component_factory(
    IF_curr_alpha,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    tau_m: 'a float (required)' = None,
    tau_refrac: 'a float (required)' = None,
    v_reset: 'a float (required)' = None,
    v_rest: 'a float (required)' = None,
    v_thresh: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<IF_curr_alpha id="IF_curr_alpha" cm="1.0" i_offset="0.9" tau_m="20.0" tau_refrac="10.0" tau_syn_E="0.5" tau_syn_I="0.5" v_init="-65" v_reset="-62.0" v_rest="-65.0" v_thresh="-52.0"/>
```
````
`````


## IF_curr_exp




extends *basepynniafcell*



Leaky integrate and fire model with fixed threshold and decaying-exponential post-synaptic current.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = v_init * MVOLT





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    






**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = v_reset \* MVOLT
: **On Conditions**
:  IF t &gt; lastSpikeTime + (tau_refrac*MSEC) THEN
: TRANSITION to REGIME **integrating**

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; v_thresh * MVOLT THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (MVOLT * (((i_offset)/cm) +  ((v_rest - (v/MVOLT)) / tau_m))/MSEC) + (iSyn / (cm * NFARAD))
````

```` Schema
``` xml
<xs:complexType name="IF_curr_exp">
  <xs:complexContent>
    <xs:extension base="basePyNNIaFCell">
            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IF_curr_exp" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IF_curr_exp
from neuroml.utils import component_factory

variable = component_factory(
    IF_curr_exp,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    tau_m: 'a float (required)' = None,
    tau_refrac: 'a float (required)' = None,
    v_reset: 'a float (required)' = None,
    v_rest: 'a float (required)' = None,
    v_thresh: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<IF_curr_exp id="IF_curr_exp" cm="1.0" i_offset="1.0" tau_m="20.0" tau_refrac="8.0" tau_syn_E="5.0" tau_syn_I="5.0" v_init="-65" v_reset="-70.0" v_rest="-65.0" v_thresh="-50.0"/>
```
````
`````


## IF_cond_alpha




extends *basepynniafcondcell*



Leaky integrate and fire model with fixed threshold and alpha-function-shaped post-synaptic conductance.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**e_rev_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**e_rev_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = v_init * MVOLT





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    






**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = v_reset \* MVOLT
: **On Conditions**
:  IF t &gt; lastSpikeTime + (tau_refrac*MSEC) THEN
: TRANSITION to REGIME **integrating**

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; v_thresh * MVOLT THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (MVOLT * (((i_offset) / cm) +  ((v_rest - (v / MVOLT)) / tau_m)) / MSEC) + (iSyn / (cm * NFARAD))
````

```` Schema
``` xml
<xs:complexType name="IF_cond_alpha">
  <xs:complexContent>
    <xs:extension base="basePyNNIaFCondCell">
            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IF_cond_alpha" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IF_cond_alpha
from neuroml.utils import component_factory

variable = component_factory(
    IF_cond_alpha,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    tau_m: 'a float (required)' = None,
    tau_refrac: 'a float (required)' = None,
    v_reset: 'a float (required)' = None,
    v_rest: 'a float (required)' = None,
    v_thresh: 'a float (required)' = None,
    e_rev_E: 'a float (required)' = None,
    e_rev_I: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<IF_cond_alpha id="IF_cond_alpha" cm="1.0" e_rev_E="0.0" e_rev_I="-70.0" i_offset="0.9" tau_m="20.0" tau_refrac="5.0" tau_syn_E="0.3" tau_syn_I="0.5" v_init="-65" v_reset="-65.0" v_rest="-65.0" v_thresh="-50.0"/>
```
``` xml
<IF_cond_alpha id="silent_cell" cm="1.0" e_rev_E="0.0" e_rev_I="-70.0" i_offset="0" tau_m="20.0" tau_refrac="5.0" tau_syn_E="5" tau_syn_I="10" v_init="-65" v_reset="-65.0" v_rest="-65.0" v_thresh="-50.0"/>
```
````
`````


## IF_cond_exp




extends *basepynniafcondcell*



Leaky integrate and fire model with fixed threshold and exponentially-decaying post-synaptic conductance.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**e_rev_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**e_rev_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = v_init * MVOLT





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    






**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = v_reset \* MVOLT
: **On Conditions**
:  IF t &gt; lastSpikeTime + (tau_refrac*MSEC) THEN
: TRANSITION to REGIME **integrating**

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; v_thresh * MVOLT THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (MVOLT * (((i_offset)/cm) +  ((v_rest - (v / MVOLT)) / tau_m)) / MSEC) + (iSyn / (cm * NFARAD))
````

```` Schema
``` xml
<xs:complexType name="IF_cond_exp">
  <xs:complexContent>
    <xs:extension base="basePyNNIaFCondCell">
            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=IF_cond_exp" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import IF_cond_exp
from neuroml.utils import component_factory

variable = component_factory(
    IF_cond_exp,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    tau_m: 'a float (required)' = None,
    tau_refrac: 'a float (required)' = None,
    v_reset: 'a float (required)' = None,
    v_rest: 'a float (required)' = None,
    v_thresh: 'a float (required)' = None,
    e_rev_E: 'a float (required)' = None,
    e_rev_I: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<IF_cond_exp id="IF_cond_exp" cm="1.0" e_rev_E="0.0" e_rev_I="-70.0" i_offset="1.0" tau_m="20.0" tau_refrac="5.0" tau_syn_E="5.0" tau_syn_I="5.0" v_init="-65" v_reset="-68.0" v_rest="-65.0" v_thresh="-52.0"/>
```
````
`````


## EIF_cond_exp_isfa_ista




extends *basepynniafcondcell*



Adaptive exponential integrate and fire neuron according to Brette R and Gerstner W ( 2005 ) with exponentially-decaying post-synaptic conductance.


`````
```` Parameters
```Table:

**a**$  $Dimensionless
**b**$  $Dimensionless
**cm**$  *(from basepynncell)* $Dimensionless
**delta_T**$  $Dimensionless
**e_rev_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**e_rev_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_w**$  $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_spike**$  $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Derived parameters
```Table:

**eif_threshold**$  $Dimensionless
```
**eif_threshold** = v_spike * H(delta_T-1e-12) + v_thresh * H(-1*delta_T+1e-9)

````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage
**w**$  $Dimensionless

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **w**: Dimensionless (exposed as **w**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = v_init * MVOLT
: **w** = 0





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    



**Conditional Derived Variables**
    
: IF delta_T &gt; 0 THEN
:  **delta_I** = delta_T \* exp(((v / MVOLT) - v_thresh) / delta_T) 
: IF delta_T = 0 THEN
:  **delta_I** = 0 



**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = v_reset \* MVOLT
:  **w** = w+b
: **On Conditions**
:  IF t &gt; lastSpikeTime + (tau_refrac*MSEC) THEN
: TRANSITION to REGIME **integrating**
: **Time Derivatives**
:  d **w** /dt = (1 / tau_w) * (a * ((v / MVOLT) - v_rest) - w) / MSEC

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; eif_threshold * MVOLT THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (MVOLT * ((-1 * ((v / MVOLT) - v_rest) + delta_I) / tau_m + (i_offset - w) / cm) / MSEC) + (iSyn / (cm * NFARAD))
:  d **w** /dt = (1 / tau_w) * (a * ((v / MVOLT) - v_rest) - w) / MSEC
````

```` Schema
``` xml
<xs:complexType name="EIF_cond_exp_isfa_ista">
  <xs:complexContent>
    <xs:extension base="basePyNNIaFCondCell">
      <xs:attribute name="a" type="xs:float" use="required"/>
      <xs:attribute name="b" type="xs:float" use="required"/>
      <xs:attribute name="delta_T" type="xs:float" use="required"/>
      <xs:attribute name="tau_w" type="xs:float" use="required"/>
      <xs:attribute name="v_spike" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=EIF_cond_exp_isfa_ista" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import EIF_cond_exp_isfa_ista
from neuroml.utils import component_factory

variable = component_factory(
    EIF_cond_exp_isfa_ista,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    tau_m: 'a float (required)' = None,
    tau_refrac: 'a float (required)' = None,
    v_reset: 'a float (required)' = None,
    v_rest: 'a float (required)' = None,
    v_thresh: 'a float (required)' = None,
    e_rev_E: 'a float (required)' = None,
    e_rev_I: 'a float (required)' = None,
    a: 'a float (required)' = None,
    b: 'a float (required)' = None,
    delta_T: 'a float (required)' = None,
    tau_w: 'a float (required)' = None,
    v_spike: 'a float (required)' = None,
    extensiontype_=None,
)
```
````
```` Usage: XML
``` xml
<EIF_cond_exp_isfa_ista id="EIF_cond_exp_isfa_ista" a="0.0" b="0.0805" cm="0.281" delta_T="2.0" e_rev_E="0.0" e_rev_I="-80.0" i_offset="0.6" tau_m="9.3667" tau_refrac="5" tau_syn_E="5.0" tau_syn_I="5.0" tau_w="144.0" v_init="-65" v_reset="-68.0" v_rest="-70.6" v_spike="-40.0" v_thresh="-52.0"/>
```
````
`````


## EIF_cond_alpha_isfa_ista




extends *basepynniafcondcell*



Adaptive exponential integrate and fire neuron according to Brette R and Gerstner W ( 2005 ) with alpha-function-shaped post-synaptic conductance.


`````
```` Parameters
```Table:

**a**$  $Dimensionless
**b**$  $Dimensionless
**cm**$  *(from basepynncell)* $Dimensionless
**delta_T**$  $Dimensionless
**e_rev_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**e_rev_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynniafcondcell)* $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_m**$  *(from basepynniafcell)* $Dimensionless
**tau_refrac**$  *(from basepynniafcell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_w**$  $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_reset**$  *(from basepynniafcell)* $Dimensionless
**v_rest**$  *(from basepynniafcell)* $Dimensionless
**v_spike**$  $Dimensionless
**v_thresh**$  *(from basepynniafcell)* $Dimensionless

```
````

```` Derived parameters
```Table:

**eif_threshold**$  $Dimensionless
```
**eif_threshold** = v_spike * H(delta_T-1e-12) + v_thresh * H(-1*delta_T+1e-9)

````

```` Exposures
```Table:

**iSyn**$  *(from basepynncell)* $dimensions:current
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage
**w**$  $Dimensionless

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **w**: Dimensionless (exposed as **w**)
: **lastSpikeTime**: dimensions:time 









**On Start**
: **v** = v_init * MVOLT
: **w** = 0





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    



**Conditional Derived Variables**
    
: IF delta_T &gt; 0 THEN
:  **delta_I** = delta_T \* exp(((v / MVOLT) - v_thresh) / delta_T) 
: IF delta_T = 0 THEN
:  **delta_I** = 0 



**Regime: refractory (initial)**
: **On Entry**
:  **lastSpikeTime** = t
:  **v** = v_reset \* MVOLT
:  **w** = w + b
: **On Conditions**
:  IF t &gt; lastSpikeTime + (tau_refrac * MSEC) THEN
: TRANSITION to REGIME **integrating**
: **Time Derivatives**
:  d **w** /dt = (1 / tau_w) * (a * ((v / MVOLT) - v_rest) - w) / MSEC

**Regime: integrating (initial)**
: **On Conditions**
:  IF v &gt; eif_threshold * MVOLT THEN
: EVENT OUT on port: **spike**
: TRANSITION to REGIME **refractory**
: **Time Derivatives**
:  d **v** /dt = (MVOLT * ((-1 * ( (v / MVOLT) - v_rest) + delta_I) / tau_m + (i_offset - w) / cm) / MSEC) + (iSyn / (cm * NFARAD))
:  d **w** /dt = (1/ tau_w) * (a*((v/MVOLT)-v_rest) - w) /MSEC
````

```` Schema
``` xml
<xs:complexType name="EIF_cond_alpha_isfa_ista">
  <xs:complexContent>
    <xs:extension base="EIF_cond_exp_isfa_ista">
            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=EIF_cond_alpha_isfa_ista" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import EIF_cond_alpha_isfa_ista
from neuroml.utils import component_factory

variable = component_factory(
    EIF_cond_alpha_isfa_ista,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    tau_m: 'a float (required)' = None,
    tau_refrac: 'a float (required)' = None,
    v_reset: 'a float (required)' = None,
    v_rest: 'a float (required)' = None,
    v_thresh: 'a float (required)' = None,
    e_rev_E: 'a float (required)' = None,
    e_rev_I: 'a float (required)' = None,
    a: 'a float (required)' = None,
    b: 'a float (required)' = None,
    delta_T: 'a float (required)' = None,
    tau_w: 'a float (required)' = None,
    v_spike: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<EIF_cond_alpha_isfa_ista id="EIF_cond_alpha_isfa_ista" a="0.0" b="0.0805" cm="0.281" delta_T="0" e_rev_E="0.0" e_rev_I="-80.0" i_offset="0.6" tau_m="9.3667" tau_refrac="5" tau_syn_E="5.0" tau_syn_I="5.0" tau_w="144.0" v_init="-65" v_reset="-68.0" v_rest="-70.6" v_spike="-40.0" v_thresh="-52.0"/>
```
````
`````


## HH_cond_exp




extends *basepynncell*



Single-compartment Hodgkin-Huxley-type neuron with transient sodium and delayed-rectifier potassium currents using the ion channel models from Traub.


`````
```` Parameters
```Table:

**cm**$  *(from basepynncell)* $Dimensionless
**e_rev_E**$  $Dimensionless
**e_rev_I**$  $Dimensionless
**e_rev_K**$  $Dimensionless
**e_rev_Na**$  $Dimensionless
**e_rev_leak**$  $Dimensionless
**g_leak**$  $Dimensionless
**gbar_K**$  $Dimensionless
**gbar_Na**$  $Dimensionless
**i_offset**$  *(from basepynncell)* $Dimensionless
**tau_syn_E**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**tau_syn_I**$ This parameter is never used in the NeuroML2 description of this cell! Any synapse producing a current can be placed on this cell *(from basepynncell)* $Dimensionless
**v_init**$  *(from basepynncell)* $Dimensionless
**v_offset**$  $Dimensionless

```
````

```` Exposures
```Table:

**h**$  $Dimensionless
**iSyn**$  *(from basepynncell)* $dimensions:current
**m**$  $Dimensionless
**n**$  $Dimensionless
**v**$ Membrane potential *(from basecellmembpot)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**spike**$ Spike event *(from basespikingcell)*$Direction: out
**spike_in_E**$  *(from basepynncell)*$Direction: in
**spike_in_I**$  *(from basepynncell)*$Direction: in

```
````

```` Attachments
```Table:

**synapses**$  $ basesynapse

```
````

```` Dynamics



**State Variables**
: **v**: dimensions:voltage (exposed as **v**)
: **m**: Dimensionless (exposed as **m**)
: **h**: Dimensionless (exposed as **h**)
: **n**: Dimensionless (exposed as **n**)









**On Start**
: **v** = v_init * MVOLT





**Derived Variables**
    : **iSyn** =&nbsp;synapses[*]->i(reduce method: add)(exposed as **iSyn**)
    : **iLeak** =&nbsp;g_leak * (e_rev_leak - (v / MVOLT))
    : **iNa** =&nbsp;gbar_Na * (m * m * m) * h * (e_rev_Na - (v / MVOLT))
    : **iK** =&nbsp;gbar_K * (n * n * n * n) * (e_rev_K - (v / MVOLT))
    : **iMemb** =&nbsp;iLeak + iNa + iK + i_offset
    : **alpham** =&nbsp;0.32 * (13 - (v / MVOLT) + v_offset) / (exp((13 - (v / MVOLT) + v_offset) / 4.0) - 1)
    : **betam** =&nbsp;0.28 * ((v / MVOLT) - v_offset - 40) / (exp(((v / MVOLT) - v_offset - 40) / 5.0) - 1)
    : **alphah** =&nbsp;0.128 * exp((17 - (v / MVOLT) + v_offset) / 18.0)
    : **betah** =&nbsp;4.0 / (1 + exp((40 - (v / MVOLT) + v_offset) / 5))
    : **alphan** =&nbsp;0.032 * (15 - (v / MVOLT) + v_offset) / (exp((15 - (v / MVOLT) + v_offset) / 5) - 1)
    : **betan** =&nbsp;0.5 * exp((10 - (v / MVOLT) + v_offset) / 40)
    





**Time Derivatives**
    : d **v** /dt = (MVOLT * (iMemb / cm) / MSEC) + (iSyn / (cm * NFARAD))
    : d **m** /dt = (alpham * (1 - m) - betam * m) / MSEC
    : d **h** /dt = (alphah * (1 - h) - betah * h) / MSEC
    : d **n** /dt = (alphan * (1 - n) - betan * n) / MSEC
    

````

```` Schema
``` xml
<xs:complexType name="HH_cond_exp">
  <xs:complexContent>
    <xs:extension base="basePyNNCell">
      <xs:attribute name="v_offset" type="xs:float" use="required"/>
      <xs:attribute name="e_rev_E" type="xs:float" use="required"/>
      <xs:attribute name="e_rev_I" type="xs:float" use="required"/>
      <xs:attribute name="e_rev_K" type="xs:float" use="required"/>
      <xs:attribute name="e_rev_Na" type="xs:float" use="required"/>
      <xs:attribute name="e_rev_leak" type="xs:float" use="required"/>
      <xs:attribute name="g_leak" type="xs:float" use="required"/>
      <xs:attribute name="gbar_K" type="xs:float" use="required"/>
      <xs:attribute name="gbar_Na" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=HH_cond_exp" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import HH_cond_exp
from neuroml.utils import component_factory

variable = component_factory(
    HH_cond_exp,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    cm: 'a float (required)' = None,
    i_offset: 'a float (required)' = None,
    tau_syn_E: 'a float (required)' = None,
    tau_syn_I: 'a float (required)' = None,
    v_init: 'a float (required)' = None,
    v_offset: 'a float (required)' = None,
    e_rev_E: 'a float (required)' = None,
    e_rev_I: 'a float (required)' = None,
    e_rev_K: 'a float (required)' = None,
    e_rev_Na: 'a float (required)' = None,
    e_rev_leak: 'a float (required)' = None,
    g_leak: 'a float (required)' = None,
    gbar_K: 'a float (required)' = None,
    gbar_Na: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<HH_cond_exp id="HH_cond_exp" cm="0.2" e_rev_E="0.0" e_rev_I="-80.0" e_rev_K="-90.0" e_rev_Na="50.0" e_rev_leak="-65.0" g_leak="0.01" gbar_K="6.0" gbar_Na="20.0" i_offset="0.2" tau_syn_E="0.2" tau_syn_I="2.0" v_init="-65" v_offset="-63.0"/>
```
````
`````


## *basePynnSynapse*




extends *basevoltagedepsynapse*



Base type for all PyNN synapses. Note, the current **I** produced is dimensionless, but it requires a membrane potential **v** with dimension voltage.


`````
```` Parameters
```Table:

**tau_syn**$  $Dimensionless

```
````

```` Constants
```Table:

**MSEC** = 1ms$  $ dimensions:time
**MVOLT** = 1mV$  $ dimensions:voltage
**NAMP** = 1nA$  $ dimensions:current

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Schema
``` xml
<xs:complexType name="BasePynnSynapse">
  <xs:complexContent>
    <xs:extension base="BaseSynapse">
      <xs:attribute name="tau_syn" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=BasePynnSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import BasePynnSynapse
from neuroml.utils import component_factory

variable = component_factory(
    BasePynnSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    tau_syn: 'a float (required)' = None,
    extensiontype_=None,
)
```
````
`````


## expCondSynapse




extends *basepynnsynapse*



Conductance based synapse with instantaneous rise and single exponential decay ( with time constant tau_syn ).


`````
```` Parameters
```Table:

**e_rev**$  $Dimensionless
**tau_syn**$  *(from basepynnsynapse)* $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**g**$  $Dimensionless
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **g**: Dimensionless (exposed as **g**)











**On Events**

: EVENT IN on port: **in**
: **g** = g+weight





**Derived Variables**
    : **i** =&nbsp;g * (e_rev - (v/MVOLT)) * NAMP(exposed as **i**)
    





**Time Derivatives**
    : d **g** /dt = -g / (tau_syn*MSEC)
    

````

```` Schema
``` xml
<xs:complexType name="ExpCondSynapse">
  <xs:complexContent>
    <xs:extension base="BasePynnSynapse">
      <xs:attribute name="e_rev" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ExpCondSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ExpCondSynapse
from neuroml.utils import component_factory

variable = component_factory(
    ExpCondSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    tau_syn: 'a float (required)' = None,
    e_rev: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<expCondSynapse id="syn1" tau_syn="5" e_rev="0"/>
```
````
`````


## expCurrSynapse




extends *basepynnsynapse*



Current based synapse with instantaneous rise and single exponential decay ( with time constant tau_syn ).


`````
```` Parameters
```Table:

**tau_syn**$  *(from basepynnsynapse)* $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **I**: Dimensionless 











**On Events**

: EVENT IN on port: **in**
: **I** = I + weight





**Derived Variables**
    : **i** =&nbsp;I * NAMP(exposed as **i**)
    





**Time Derivatives**
    : d **I** /dt = -I / (tau_syn*MSEC)
    

````

```` Schema
``` xml
<xs:complexType name="ExpCurrSynapse">
  <xs:complexContent>
    <xs:extension base="BasePynnSynapse">

            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=ExpCurrSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import ExpCurrSynapse
from neuroml.utils import component_factory

variable = component_factory(
    ExpCurrSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    tau_syn: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<expCurrSynapse id="syn3" tau_syn="5"/>
```
````
`````


## alphaCondSynapse




extends *basepynnsynapse*



Alpha synapse: rise time and decay time are both tau_syn. Conductance based synapse.


`````
```` Parameters
```Table:

**e_rev**$  $Dimensionless
**tau_syn**$  *(from basepynnsynapse)* $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**A**$  $Dimensionless
**g**$  $Dimensionless
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **g**: Dimensionless (exposed as **g**)
: **A**: Dimensionless (exposed as **A**)











**On Events**

: EVENT IN on port: **in**
: **A** = A + weight





**Derived Variables**
    : **i** =&nbsp;g * (e_rev - (v/MVOLT)) * NAMP(exposed as **i**)
    





**Time Derivatives**
    : d **g** /dt = (2.7182818*A - g)/(tau_syn*MSEC)
    : d **A** /dt = -A /(tau_syn*MSEC)
    

````

```` Schema
``` xml
<xs:complexType name="AlphaCondSynapse">
  <xs:complexContent>
    <xs:extension base="BasePynnSynapse">
      <xs:attribute name="e_rev" type="xs:float" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=AlphaCondSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import AlphaCondSynapse
from neuroml.utils import component_factory

variable = component_factory(
    AlphaCondSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    tau_syn: 'a float (required)' = None,
    e_rev: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<alphaCondSynapse id="syn2" tau_syn="5" e_rev="0"/>
```
````
`````


## alphaCurrSynapse




extends *basepynnsynapse*



Alpha synapse: rise time and decay time are both tau_syn. Current based synapse.


`````
```` Parameters
```Table:

**tau_syn**$  *(from basepynnsynapse)* $Dimensionless

```
````

```` Properties
```Table:

**weight** (default: 1)$  $ Dimensionless

```
````

```` Exposures
```Table:

**A**$  $dimensions:current
**i**$ The total (usually time varying) current produced by this ComponentType *(from basepointcurrent)* $dimensions:current

```
````

```` Requirements
```Table:

**v**$ The current may vary with the voltage exposed by the ComponentType on which this is placed *(from basevoltagedepsynapse)* $dimensions:voltage

```
````

```` Event Ports
```Table:

**in**$  *(from basesynapse)*$Direction: in

```
````

```` Dynamics



**State Variables**
: **I**: Dimensionless 
: **A**: Dimensionless (exposed as **A**)











**On Events**

: EVENT IN on port: **in**
: **A** = A + weight





**Derived Variables**
    : **i** =&nbsp;I * NAMP(exposed as **i**)
    





**Time Derivatives**
    : d **I** /dt = (2.7182818*A - I)/(tau_syn*MSEC)
    : d **A** /dt = -A /(tau_syn*MSEC)
    

````

```` Schema
``` xml
<xs:complexType name="AlphaCurrSynapse">
  <xs:complexContent>
    <xs:extension base="BasePynnSynapse">

            </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=AlphaCurrSynapse" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import AlphaCurrSynapse
from neuroml.utils import component_factory

variable = component_factory(
    AlphaCurrSynapse,
    id: 'a NmlId (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    neuro_lex_id: 'a NeuroLexId (optional)' = None,
    tau_syn: 'a float (required)' = None,
)
```
````
```` Usage: XML
``` xml
<alphaCurrSynapse id="syn4" tau_syn="5"/>
```
````
`````


## SpikeSourcePoisson




extends *basespikesource*



Spike source, generating spikes according to a Poisson process.


`````
```` Parameters
```Table:

**duration**$  $dimensions:time
**rate**$  $dimensions:per_time
**start**$  $dimensions:time

```
````

```` Constants
```Table:

**LONG_TIME** = 1e9hour$  $ dimensions:time
**SMALL_TIME** = 1e-9ms$  $ dimensions:time

```
````

```` Derived parameters
```Table:

**end**$  $dimensions:time
```
**end** = start + duration

````

```` Exposures
```Table:

**isi**$  $dimensions:time
**tnextIdeal**$  $dimensions:time
**tnextUsed**$  $dimensions:time
**tsince**$ Time since the last spike was emitted *(from basespikesource)* $dimensions:time

```
````

```` Event Ports
```Table:

**in**$ $Direction: in
**spike**$ Port on which spikes are emitted *(from basespikesource)*$Direction: out

```
````

```` Dynamics



**State Variables**
: **tsince**: dimensions:time (exposed as **tsince**)
: **tnextIdeal**: dimensions:time (exposed as **tnextIdeal**)
: **tnextUsed**: dimensions:time (exposed as **tnextUsed**)
: **isi**: dimensions:time (exposed as **isi**)









**On Start**
: **isi** = start - log(random(1))/rate
: **tsince** = 0
: **tnextIdeal** = isi + H(((isi) - (start+duration))/duration)*LONG_TIME
: **tnextUsed** = tnextIdeal



**On Conditions**

: IF t &gt; tnextUsed THEN
: **isi** = -1 * log(random(1))/rate
: **tnextIdeal** = (tnextIdeal+isi) + H(((tnextIdeal+isi) - (start+duration))/duration)*LONG_TIME
: **tnextUsed** = tnextIdeal*H( (tnextIdeal-t)/t ) + (t+SMALL_TIME)*H( (t-tnextIdeal)/t )
: **tsince** = 0
: EVENT OUT on port: **spike**








**Time Derivatives**
    : d **tsince** /dt = 1
    : d **tnextUsed** /dt = 0
    : d **tnextIdeal** /dt = 0
    

````

```` Schema
``` xml
<xs:complexType name="SpikeSourcePoisson">
  <xs:complexContent>
    <xs:extension base="Standalone">
      <xs:attribute name="start" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="duration" type="Nml2Quantity_time" use="required"/>
      <xs:attribute name="rate" type="Nml2Quantity_pertime" use="required"/>
    </xs:extension>
  </xs:complexContent>
</xs:complexType>

```
````

```` Usage: Python
*<a href="https://libneuroml.readthedocs.io/en/latest/search.html?q=SpikeSourcePoisson" target="_blank">Go to the libNeuroML documentation</a>*
``` python
from neuroml import SpikeSourcePoisson
from neuroml.utils import component_factory

variable = component_factory(
    SpikeSourcePoisson,
    id: 'a NonNegativeInteger (required)' = None,
    metaid: 'a MetaId (optional)' = None,
    notes: 'a string (optional)' = None,
    properties: 'list of Property(s) (optional)' = None,
    annotation: 'a Annotation (optional)' = None,
    start: 'a Nml2Quantity_time (required)' = None,
    duration: 'a Nml2Quantity_time (required)' = None,
    rate: 'a Nml2Quantity_pertime (required)' = None,
)
```
````
```` Usage: XML
``` xml
<SpikeSourcePoisson id="spikes1" start="50ms" duration="400ms" rate="50Hz"/>
```
``` xml
<SpikeSourcePoisson id="spikes2" start="50ms" duration="300ms" rate="80Hz"/>
```
````
`````

# Simulation

**Specification of the LEMS Simulation element which is normally used to define simulations of NeuroML2 files. Note: not actually part of NeuroML v2, but this is required by much of the NeuroML toolchain for defining Simulations ( which NeuroML model to use and how long to run for ), as well as what to  display and what to save in  outputfiles.**

---


Original ComponentType definitions: [Simulation.xml](https://github.com/NeuroML/NeuroML2/blob/master/NeuroML2CoreTypes//Simulation.xml).
Schema against which NeuroML based on these should be valid: [NeuroML_v2.3.xsd](https://github.com/NeuroML/NeuroML2/tree/master/Schemas/NeuroML2/NeuroML_v2.3.xsd).
Generated on 14/08/24 from [this](https://github.com/NeuroML/NeuroML2/commit/352244cff605cb1ba24fa7c11757dc818fe90fd2) commit.
Please file any issues or questions at the [issue tracker here](https://github.com/NeuroML/NeuroML2/issues).

---


## Simulation




The main element in a LEMS Simulation file. Defines the **length** of simulation, the timestep ( dt ) **step** and an optional **seed** to use for stochastic elements, as well as  displays,  outputfiles and  eventoutputfiles to record. Specifies a **target** component to run, usually the id of a  network.


`````
```` Parameters
```Table:

**length**$ Duration of the simulation run $dimensions:time
**step**$ Time step (dt) to use in the simulation $dimensions:time

```
````

```` Text fields
```Table:

**seed**$ The seed to use in the random number generator for stochastic entities

````

```` Component References
```Table:

**target**$  $ component

```
````

```` Children list
```Table:

**metas**$  $ meta
**displays**$  $ display
**outputs**$  $ outputfile
**events**$  $ eventoutputfile

```
````

```` Dynamics



**State Variables**
: **t**: dimensions:time 










````
`````


## Display




Details of a display to generate ( usually a set of traces given by  lines in a newly opened window ) on completion of the simulation.


`````
```` Parameters
```Table:

**timeScale**$ A scaling of the time axis, e.g. 1ms means display in milliseconds. Note: all quantities are recorded in SI units $dimensions:time
**xmax**$ The maximum value on the x axis (i.e time variable) of the display $Dimensionless
**xmin**$ The minimum value on the x axis (i.e time variable) of the display $Dimensionless
**ymax**$ The maximum value on the x axis of the display $Dimensionless
**ymin**$ The minimum value on the y axis of the display $Dimensionless

```
````

```` Text fields
```Table:

**title**$ The title of the display, e.g. to use for the window

````

```` Children list
```Table:

**lines**$  $ line

```
````
`````


## Line




Specification of a single time varying **quantity** to plot on the  display. Note that all quantities are handled internally in LEMS in SI units, and so a **scale** should be used if it is to be displayed in other units.


`````
```` Parameters
```Table:

**scale**$ A scaling factor to DIVIDE the quantity by. Can be dimensional, so using scale=1mV means a value of -0.07V is displayed as -70. Alternatively, scale=0.001 would achieve the same thing. $dimensions:*
**timeScale**$ An optional scaling of the time axis, e.g. 1ms means display in milliseconds. Note: if present, this overrides timeScale from _Display_ $dimensions:*

```
````

```` Text fields
```Table:

**color**$ A hex string for the color to display the trace for this quantity, e.g. #aa33ff

````

```` Paths
```Table:

**quantity**$ Path to the quantity to display, see see https://docs.neuroml.org/Userdocs/Paths.html.

````
`````


## OutputFile




A file in which to save recorded values from the simulation.


`````
```` Text fields
```Table:

**path**$ Optional path to the directory in which to store the file
**fileName**$ Name of the file to generate. Can include a relative path (from the LEMS Simulation file location).

````

```` Children list
```Table:

**outputColumn**$  $ outputcolumn

```
````
`````


## OutputColumn




Specification of a single time varying **quantity** to record during the simulation. Note that all quantities are handled internally in LEMS in SI units, and so the value for the quantity in the file ( as well as time ) will be in SI units.


`````
```` Paths
```Table:

**quantity**$ Path to the quantity to save, see see https://docs.neuroml.org/Userdocs/Paths.html. Note that all quantities are saved in SI units.

````
`````


## EventOutputFile




A file in which to save event information ( e.g. spikes from cells in a population ) in a specified **format**.


`````
```` Text fields
```Table:

**path**$ Optional path to the directory in which to store the file
**fileName**$ Name of the file to generate. Can include a relative path (from the LEMS Simulation file location).
**format**$ Takes values TIME_ID or ID_TIME, depending on the preferred order of the time or event id (from _EventSelection_) in each row of the file

````

```` Children list
```Table:

**eventSelection**$  $ eventselection

```
````
`````


## EventSelection




A specific source of events with an associated **id,** which will be recorded inside the file specified in the parent  eventoutputfile. The attribute **select** should point to a cell inside a  population ( e.g. hhpop[0], see https://docs.neuroml.org/Userdocs/Paths.html ), and the **eventPort** specifies the port for the emitted events, which usually has id: spike. Note: the **id** used on this element ( and appearing in the file alongside the event time ) can be different from the id/index of the cell in the population.


`````
```` Text fields
```Table:

**eventPort**$ The port on the cell which generates the events, usually: spike

````

```` Paths
```Table:

**select**$ The cell which will be emitting the events

````
`````


## Meta




Metadata to add to simulation.


`````
```` Text fields
```Table:

**for**$ Simulator name
**method**$ Integration method to use
**abs_tolerance**$ Absolute tolerance for NEURON's cvode method
**rel_tolerance**$ Relative tolerance for NEURON's cvode method

````
`````
# Index

<a name="adexiafcell"/>

- adExIaFCell <adexiafcell>
<a name="alphacondsynapse"/>

- alphaCondSynapse <alphacondsynapse>
<a name="alphacurrentsynapse"/>

- alphaCurrentSynapse <alphacurrentsynapse>
<a name="alphacurrsynapse"/>

- alphaCurrSynapse <alphacurrsynapse>
<a name="alphasynapse"/>

- alphaSynapse <alphasynapse>
<a name="annotation"/>

- annotation <annotation>
<a name="area"/>

- area <dimensions:area>
<a name="baseblockmechanism"/>

- baseBlockMechanism <baseblockmechanism>
<a name="basebqbiol"/>

- baseBqbiol <basebqbiol>
<a name="basecell"/>

- baseCell <basecell>
<a name="basecellmembpot"/>

- baseCellMembPot <basecellmembpot>
<a name="basecellmembpotcap"/>

- baseCellMembPotCap <basecellmembpotcap>
<a name="basecellmembpotdl"/>

- baseCellMembPotDL <basecellmembpotdl>
<a name="basechanneldensity"/>

- baseChannelDensity <basechanneldensity>
<a name="basechanneldensitycond"/>

- baseChannelDensityCond <basechanneldensitycond>
<a name="basechannelpopulation"/>

- baseChannelPopulation <basechannelpopulation>
<a name="baseconductancebasedsynapse"/>

- baseConductanceBasedSynapse <baseconductancebasedsynapse>
<a name="baseconductancebasedsynapsetwo"/>

- baseConductanceBasedSynapseTwo <baseconductancebasedsynapsetwo>
<a name="baseconductancescaling"/>

- baseConductanceScaling <baseconductancescaling>
<a name="baseconductancescalingcadependent"/>

- baseConductanceScalingCaDependent <baseconductancescalingcadependent>
<a name="basecurrentbasedsynapse"/>

- baseCurrentBasedSynapse <basecurrentbasedsynapse>
<a name="basegate"/>

- baseGate <basegate>
<a name="basegradedsynapse"/>

- baseGradedSynapse <basegradedsynapse>
<a name="basehhrate"/>

- baseHHRate <basehhrate>
<a name="basehhvariable"/>

- baseHHVariable <basehhvariable>
<a name="baseiaf"/>

- baseIaf <baseiaf>
<a name="baseiafcapcell"/>

- baseIafCapCell <baseiafcapcell>
<a name="baseionchannel"/>

- baseIonChannel <baseionchannel>
<a name="baseplasticitymechanism"/>

- basePlasticityMechanism <baseplasticitymechanism>
<a name="basepointcurrent"/>

- basePointCurrent <basepointcurrent>
<a name="basepointcurrentdl"/>

- basePointCurrentDL <basepointcurrentdl>
<a name="basepopulation"/>

- basePopulation <basepopulation>
<a name="basepynncell"/>

- basePyNNCell <basepynncell>
<a name="basepynniafcell"/>

- basePyNNIaFCell <basepynniafcell>
<a name="basepynniafcondcell"/>

- basePyNNIaFCondCell <basepynniafcondcell>
<a name="basepynnsynapse"/>

- basePynnSynapse <basepynnsynapse>
<a name="baseq10settings"/>

- baseQ10Settings <baseq10settings>
<a name="basespikesource"/>

- baseSpikeSource <basespikesource>
<a name="basespikingcell"/>

- baseSpikingCell <basespikingcell>
<a name="basestandalone"/>

- baseStandalone <basestandalone>
<a name="basesynapse"/>

- baseSynapse <basesynapse>
<a name="basesynapsedl"/>

- baseSynapseDL <basesynapsedl>
<a name="basevoltageconcdeprate"/>

- baseVoltageConcDepRate <basevoltageconcdeprate>
<a name="basevoltageconcdeptime"/>

- baseVoltageConcDepTime <basevoltageconcdeptime>
<a name="basevoltageconcdepvariable"/>

- baseVoltageConcDepVariable <basevoltageconcdepvariable>
<a name="basevoltagedeppointcurrent"/>

- baseVoltageDepPointCurrent <basevoltagedeppointcurrent>
<a name="basevoltagedeppointcurrentdl"/>

- baseVoltageDepPointCurrentDL <basevoltagedeppointcurrentdl>
<a name="basevoltagedeppointcurrentspiking"/>

- baseVoltageDepPointCurrentSpiking <basevoltagedeppointcurrentspiking>
<a name="basevoltagedeprate"/>

- baseVoltageDepRate <basevoltagedeprate>
<a name="basevoltagedepsynapse"/>

- baseVoltageDepSynapse <basevoltagedepsynapse>
<a name="basevoltagedeptime"/>

- baseVoltageDepTime <basevoltagedeptime>
<a name="basevoltagedepvariable"/>

- baseVoltageDepVariable <basevoltagedepvariable>
<a name="biophysicalproperties"/>

- biophysicalProperties <biophysicalproperties>
<a name="biophysicalproperties2capools"/>

- biophysicalProperties2CaPools <biophysicalproperties2capools>
<a name="blockingplasticsynapse"/>

- blockingPlasticSynapse <blockingplasticsynapse>
<a name="bqbiol_encodes"/>

- bqbiol_encodes <bqbiol_encodes>
<a name="bqbiol_haspart"/>

- bqbiol_hasPart <bqbiol_haspart>
<a name="bqbiol_hasproperty"/>

- bqbiol_hasProperty <bqbiol_hasproperty>
<a name="bqbiol_hastaxon"/>

- bqbiol_hasTaxon <bqbiol_hastaxon>
<a name="bqbiol_hasversion"/>

- bqbiol_hasVersion <bqbiol_hasversion>
<a name="bqbiol_is"/>

- bqbiol_is <bqbiol_is>
<a name="bqbiol_isdescribedby"/>

- bqbiol_isDescribedBy <bqbiol_isdescribedby>
<a name="bqbiol_isencodedby"/>

- bqbiol_isEncodedBy <bqbiol_isencodedby>
<a name="bqbiol_ishomologto"/>

- bqbiol_isHomologTo <bqbiol_ishomologto>
<a name="bqbiol_ispartof"/>

- bqbiol_isPartOf <bqbiol_ispartof>
<a name="bqbiol_ispropertyof"/>

- bqbiol_isPropertyOf <bqbiol_ispropertyof>
<a name="bqbiol_isversionof"/>

- bqbiol_isVersionOf <bqbiol_isversionof>
<a name="bqbiol_occursin"/>

- bqbiol_occursIn <bqbiol_occursin>
<a name="bqmodel_is"/>

- bqmodel_is <bqmodel_is>
<a name="bqmodel_isderivedfrom"/>

- bqmodel_isDerivedFrom <bqmodel_isderivedfrom>
<a name="bqmodel_isdescribedby"/>

- bqmodel_isDescribedBy <bqmodel_isdescribedby>
<a name="capacitance"/>

- capacitance <dimensions:capacitance>
<a name="cell"/>

- cell <cell>
<a name="cell2capools"/>

- cell2CaPools <cell2capools>
<a name="channeldensity"/>

- channelDensity <channeldensity>
<a name="channeldensityghk"/>

- channelDensityGHK <channeldensityghk>
<a name="channeldensityghk2"/>

- channelDensityGHK2 <channeldensityghk2>
<a name="channeldensitynernst"/>

- channelDensityNernst <channeldensitynernst>
<a name="channeldensitynernstca2"/>

- channelDensityNernstCa2 <channeldensitynernstca2>
<a name="channeldensitynonuniform"/>

- channelDensityNonUniform <channeldensitynonuniform>
<a name="channeldensitynonuniformghk"/>

- channelDensityNonUniformGHK <channeldensitynonuniformghk>
<a name="channeldensitynonuniformnernst"/>

- channelDensityNonUniformNernst <channeldensitynonuniformnernst>
<a name="channeldensityvshift"/>

- channelDensityVShift <channeldensityvshift>
<a name="channelpopulation"/>

- channelPopulation <channelpopulation>
<a name="channelpopulationnernst"/>

- channelPopulationNernst <channelpopulationnernst>
<a name="charge"/>

- charge <dimensions:charge>
<a name="charge_per_mole"/>

- charge_per_mole <dimensions:charge_per_mole>
<a name="closedstate"/>

- closedState <closedstate>
<a name="compoundinput"/>

- compoundInput <compoundinput>
<a name="compoundinputdl"/>

- compoundInputDL <compoundinputdl>
<a name="concentration"/>

- concentration <dimensions:concentration>
<a name="concentrationmodel"/>

- concentrationModel <concentrationmodel>
<a name="conductance"/>

- conductance <dimensions:conductance>
<a name="conductance_per_voltage"/>

- conductance_per_voltage <dimensions:conductance_per_voltage>
<a name="conductanceDensity"/>

- conductanceDensity <dimensions:conductanceDensity>
<a name="connection"/>

- connection <connection>
<a name="connectionwd"/>

- connectionWD <connectionwd>
<a name="continuousconnection"/>

- continuousConnection <continuousconnection>
<a name="continuousconnectioninstance"/>

- continuousConnectionInstance <continuousconnectioninstance>
<a name="continuousconnectioninstancew"/>

- continuousConnectionInstanceW <continuousconnectioninstancew>
<a name="continuousprojection"/>

- continuousProjection <continuousprojection>
<a name="current"/>

- current <dimensions:current>
<a name="currentDensity"/>

- currentDensity <dimensions:currentDensity>
<a name="decayingpoolconcentrationmodel"/>

- decayingPoolConcentrationModel <decayingpoolconcentrationmodel>
<a name="display"/>

- Display <display>
<a name="distal"/>

- distal <distal>
<a name="distaldetails"/>

- distalDetails <distaldetails>
<a name="doublesynapse"/>

- doubleSynapse <doublesynapse>
<a name="eif_cond_alpha_isfa_ista"/>

- EIF_cond_alpha_isfa_ista <eif_cond_alpha_isfa_ista>
<a name="eif_cond_exp_isfa_ista"/>

- EIF_cond_exp_isfa_ista <eif_cond_exp_isfa_ista>
<a name="electricalconnection"/>

- electricalConnection <electricalconnection>
<a name="electricalconnectioninstance"/>

- electricalConnectionInstance <electricalconnectioninstance>
<a name="electricalconnectioninstancew"/>

- electricalConnectionInstanceW <electricalconnectioninstancew>
<a name="electricalprojection"/>

- electricalProjection <electricalprojection>
<a name="eventoutputfile"/>

- EventOutputFile <eventoutputfile>
<a name="eventselection"/>

- EventSelection <eventselection>
<a name="expcondsynapse"/>

- expCondSynapse <expcondsynapse>
<a name="expcurrsynapse"/>

- expCurrSynapse <expcurrsynapse>
<a name="explicitconnection"/>

- explicitConnection <explicitconnection>
<a name="explicitinput"/>

- explicitInput <explicitinput>
<a name="exponesynapse"/>

- expOneSynapse <exponesynapse>
<a name="expthreesynapse"/>

- expThreeSynapse <expthreesynapse>
<a name="exptwosynapse"/>

- expTwoSynapse <exptwosynapse>
<a name="fitzhughnagumocell"/>

- fitzHughNagumoCell <fitzhughnagumocell>
<a name="fixedfactorconcentrationmodel"/>

- fixedFactorConcentrationModel <fixedfactorconcentrationmodel>
<a name="fixedfactorconcentrationmodeltraub"/>

- fixedFactorConcentrationModelTraub <fixedfactorconcentrationmodeltraub>
<a name="fixedtimecourse"/>

- fixedTimeCourse <fixedtimecourse>
<a name="forwardtransition"/>

- forwardTransition <forwardtransition>
<a name="from"/>

- from <from>
<a name="gapjunction"/>

- gapJunction <gapjunction>
<a name="gate"/>

- gate <gate>
<a name="gatefractional"/>

- gateFractional <gatefractional>
<a name="gatehhinstantaneous"/>

- gateHHInstantaneous <gatehhinstantaneous>
<a name="gatehhrates"/>

- gateHHrates <gatehhrates>
<a name="gatehhratesinf"/>

- gateHHratesInf <gatehhratesinf>
<a name="gatehhratestau"/>

- gateHHratesTau <gatehhratestau>
<a name="gatehhratestauinf"/>

- gateHHratesTauInf <gatehhratestauinf>
<a name="gatehhtauinf"/>

- gateHHtauInf <gatehhtauinf>
<a name="gateks"/>

- gateKS <gateks>
<a name="gradedsynapse"/>

- gradedSynapse <gradedsynapse>
<a name="hh_cond_exp"/>

- HH_cond_exp <hh_cond_exp>
<a name="hhexplinearrate"/>

- HHExpLinearRate <hhexplinearrate>
<a name="hhexplinearvariable"/>

- HHExpLinearVariable <hhexplinearvariable>
<a name="hhexprate"/>

- HHExpRate <hhexprate>
<a name="hhexpvariable"/>

- HHExpVariable <hhexpvariable>
<a name="hhsigmoidrate"/>

- HHSigmoidRate <hhsigmoidrate>
<a name="hhsigmoidvariable"/>

- HHSigmoidVariable <hhsigmoidvariable>
<a name="hindmarshrose1984cell"/>

- hindmarshRose1984Cell <hindmarshrose1984cell>
<a name="iafcell"/>

- iafCell <iafcell>
<a name="iafrefcell"/>

- iafRefCell <iafrefcell>
<a name="iaftaucell"/>

- iafTauCell <iaftaucell>
<a name="iaftaurefcell"/>

- iafTauRefCell <iaftaurefcell>
<a name="idealGasConstantDims"/>

- idealGasConstantDims <dimensions:idealGasConstantDims>
<a name="if_cond_alpha"/>

- IF_cond_alpha <if_cond_alpha>
<a name="if_cond_exp"/>

- IF_cond_exp <if_cond_exp>
<a name="if_curr_alpha"/>

- IF_curr_alpha <if_curr_alpha>
<a name="if_curr_exp"/>

- IF_curr_exp <if_curr_exp>
<a name="include"/>

- include <include>
<a name="includetype"/>

- IncludeType <includetype>
<a name="inhomogeneousparameter"/>

- inhomogeneousParameter <inhomogeneousparameter>
<a name="inhomogeneousvalue"/>

- inhomogeneousValue <inhomogeneousvalue>
<a name="initmembpotential"/>

- initMembPotential <initmembpotential>
<a name="input"/>

- input <input>
<a name="inputlist"/>

- inputList <inputlist>
<a name="inputw"/>

- inputW <inputw>
<a name="instance"/>

- instance <instance>
<a name="intracellularproperties"/>

- intracellularProperties <intracellularproperties>
<a name="intracellularproperties2capools"/>

- intracellularProperties2CaPools <intracellularproperties2capools>
<a name="ionchannel"/>

- ionChannel <ionchannel>
<a name="ionchannelhh"/>

- ionChannelHH <ionchannelhh>
<a name="ionchannelks"/>

- ionChannelKS <ionchannelks>
<a name="ionchannelpassive"/>

- ionChannelPassive <ionchannelpassive>
<a name="ionchannelvshift"/>

- ionChannelVShift <ionchannelvshift>
<a name="izhikevich2007cell"/>

- izhikevich2007Cell <izhikevich2007cell>
<a name="izhikevichcell"/>

- izhikevichCell <izhikevichcell>
<a name="ksstate"/>

- KSState <ksstate>
<a name="kstransition"/>

- KSTransition <kstransition>
<a name="length"/>

- length <dimensions:length>
<a name="line"/>

- Line <line>
<a name="lineargradedsynapse"/>

- linearGradedSynapse <lineargradedsynapse>
<a name="location"/>

- location <location>
<a name="member"/>

- member <member>
<a name="membraneproperties"/>

- membraneProperties <membraneproperties>
<a name="membraneproperties2capools"/>

- membraneProperties2CaPools <membraneproperties2capools>
<a name="meta"/>

- Meta <meta>
<a name="morphology"/>

- morphology <morphology>
<a name="network"/>

- network <network>
<a name="networkwithtemperature"/>

- networkWithTemperature <networkwithtemperature>
<a name="neuromldocument"/>

- NeuroMLDocument <neuromldocument>
<a name="notes"/>

- notes <notes>
<a name="openstate"/>

- openState <openstate>
<a name="outputcolumn"/>

- OutputColumn <outputcolumn>
<a name="outputfile"/>

- OutputFile <outputfile>
<a name="parent"/>

- parent <parent>
<a name="path"/>

- path <path>
<a name="per_time"/>

- per_time <dimensions:per_time>
<a name="per_voltage"/>

- per_voltage <dimensions:per_voltage>
<a name="permeability"/>

- permeability <dimensions:permeability>
<a name="pinskyrinzelca3cell"/>

- pinskyRinzelCA3Cell <pinskyrinzelca3cell>
<a name="point3dwithdiam"/>

- point3DWithDiam <point3dwithdiam>
<a name="pointcellcondbased"/>

- pointCellCondBased <pointcellcondbased>
<a name="pointcellcondbasedca"/>

- pointCellCondBasedCa <pointcellcondbasedca>
<a name="poissonfiringsynapse"/>

- poissonFiringSynapse <poissonfiringsynapse>
<a name="population"/>

- population <population>
<a name="populationlist"/>

- populationList <populationlist>
<a name="projection"/>

- projection <projection>
<a name="property"/>

- property <property>
<a name="proximal"/>

- proximal <proximal>
<a name="proximaldetails"/>

- proximalDetails <proximaldetails>
<a name="pulsegenerator"/>

- pulseGenerator <pulsegenerator>
<a name="pulsegeneratordl"/>

- pulseGeneratorDL <pulsegeneratordl>
<a name="q10conductancescaling"/>

- q10ConductanceScaling <q10conductancescaling>
<a name="q10exptemp"/>

- q10ExpTemp <q10exptemp>
<a name="q10fixed"/>

- q10Fixed <q10fixed>
<a name="rampgenerator"/>

- rampGenerator <rampgenerator>
<a name="rampgeneratordl"/>

- rampGeneratorDL <rampgeneratordl>
<a name="rdf_bag"/>

- rdf_Bag <rdf_bag>
<a name="rdf_description"/>

- rdf_Description <rdf_description>
<a name="rdf_li"/>

- rdf_li <rdf_li>
<a name="rdf_rdf"/>

- rdf_RDF <rdf_rdf>
<a name="rectangularextent"/>

- rectangularExtent <rectangularextent>
<a name="region"/>

- region <region>
<a name="resistance"/>

- resistance <dimensions:resistance>
<a name="resistivity"/>

- resistivity <dimensions:resistivity>
<a name="reversetransition"/>

- reverseTransition <reversetransition>
<a name="rho_factor"/>

- rho_factor <dimensions:rho_factor>
<a name="segment"/>

- segment <segment>
<a name="segmentgroup"/>

- segmentGroup <segmentgroup>
<a name="silentsynapse"/>

- silentSynapse <silentsynapse>
<a name="simulation"/>

- Simulation <simulation>
<a name="sinegenerator"/>

- sineGenerator <sinegenerator>
<a name="sinegeneratordl"/>

- sineGeneratorDL <sinegeneratordl>
<a name="species"/>

- species <species>
<a name="specificCapacitance"/>

- specificCapacitance <dimensions:specificCapacitance>
<a name="spike"/>

- spike <spike>
<a name="spikearray"/>

- spikeArray <spikearray>
<a name="spikegenerator"/>

- spikeGenerator <spikegenerator>
<a name="spikegeneratorpoisson"/>

- spikeGeneratorPoisson <spikegeneratorpoisson>
<a name="spikegeneratorrandom"/>

- spikeGeneratorRandom <spikegeneratorrandom>
<a name="spikegeneratorrefpoisson"/>

- spikeGeneratorRefPoisson <spikegeneratorrefpoisson>
<a name="spikesourcepoisson"/>

- SpikeSourcePoisson <spikesourcepoisson>
<a name="spikethresh"/>

- spikeThresh <spikethresh>
<a name="stdpsynapse"/>

- stdpSynapse <stdpsynapse>
<a name="subgate"/>

- subGate <subgate>
<a name="substance"/>

- substance <dimensions:substance>
<a name="subtree"/>

- subTree <subtree>
<a name="synapticconnection"/>

- synapticConnection <synapticconnection>
<a name="synapticconnectionwd"/>

- synapticConnectionWD <synapticconnectionwd>
<a name="tauinftransition"/>

- tauInfTransition <tauinftransition>
<a name="temperature"/>

- temperature <dimensions:temperature>
<a name="time"/>

- time <dimensions:time>
<a name="timedsynapticinput"/>

- timedSynapticInput <timedsynapticinput>
<a name="to"/>

- to <to>
<a name="transientpoissonfiringsynapse"/>

- transientPoissonFiringSynapse <transientpoissonfiringsynapse>
<a name="tsodyksmarkramdepfacmechanism"/>

- tsodyksMarkramDepFacMechanism <tsodyksmarkramdepfacmechanism>
<a name="tsodyksmarkramdepmechanism"/>

- tsodyksMarkramDepMechanism <tsodyksmarkramdepmechanism>
<a name="variableparameter"/>

- variableParameter <variableparameter>
<a name="vhalftransition"/>

- vHalfTransition <vhalftransition>
<a name="voltage"/>

- voltage <dimensions:voltage>
<a name="voltageclamp"/>

- voltageClamp <voltageclamp>
<a name="voltageclamptriple"/>

- voltageClampTriple <voltageclamptriple>
<a name="voltageconcdepblockmechanism"/>

- voltageConcDepBlockMechanism <voltageconcdepblockmechanism>
<a name="volume"/>

- volume <dimensions:volume>

