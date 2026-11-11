"""
thermo.py
=========
Modelo de Combustão de Turbina a Gás — Parte 2: Funções Termodinâmicas

Contém:
  - Coeficientes polinomiais de Cp por espécie    (GAS_COEFFS)
  - Cp(T, species)                               [kJ/(mol·K)]
  - Cp_fuel(T, Fmolfrac, MMFuel)                 [kJ/(kg·K)]
  - Cp_air(T)                                    [kJ/(kg·K)]
  - Cp_comb(T, Combmolfrac, MMComb)              [kJ/(kg·K)]

  - Entalpias de formação de referência           (HF_REF)   [kJ/mol]
  - H(T, species)                                [kJ/mol]
  - H_fuel(T, Fmolfrac, MMFuel)                  [kJ/kg]
  - H_air(T)                                     [kJ/kg]
  - H_comb(T, Combmolfrac, MMComb)               [kJ/kg]

  - Entropias de referência                      (SF_REF)   [kJ/(mol·K)]
  - S(T, species, P)                             [kJ/(mol·K)]
  - S_fuel(T, P, Fmolfrac, MMFuel)               [kJ/(kg·K)]
  - S_air(T, P)                                  [kJ/(kg·K)]
  - S_comb(T, P, Combmolfrac, MMComb)            [kJ/(kg·K)]

Fórmula de Cp (Yaws, Carl L):
    Cp = a + b·T + c·T² + d·T³ + e·T⁴  [kJ/(mol·K)]

Todas as fórmulas são implementadas usando modelos polinomiais termodinâmicos.
Temperatura de referência: T_ref = 298 K
Pressão de referência    : P0    = 1 bar (importada de properties)
"""
import math
try:
    from properties import R, P0, MM_AIR, SPECIES, molar_mass
except ImportError:
    from .properties import R, P0, MM_AIR, SPECIES, molar_mass

# ---------------------------------------------------------------------------
# Coeficientes polinomiais de Cp
# Fonte: Yaws, Carl L.
# Formato por espécie: {"a": ..., "b": ..., "c": ..., "d": ..., "e": ...}
# Unidades: coeficientes do polinômio de capacidade calorífica absoluta em kJ/(mol·K).
# ---------------------------------------------------------------------------


GAS_COEFFS = {
    "H2": {
        "a":  25.399,
        "b":  0.020178,
        "c":  -0.038549,
        "d":  3.188e-08,
        "e":  -8.7585e-12,
    },
    "CO": {
        "a":  29.556,
        "b":  -0.0065807,
        "c":  2.013e-05,
        "d":  -1.2227e-08,
        "e":  2.2617e-12,
    },
    "CO2": {
        "a":  27.437,
        "b":  0.042315,
        "c":  -1.9555e-05,
        "d":  3.9968e-09,
        "e":  -2.9872e-13,
    },
    "N2": {
        "a":  29.342,
        "b":  -0.0035395,
        "c":  1.0076e-05,
        "d":  -4.3116e-09,
        "e":  2.5935e-13,
    },
    "SO2": {
        "a":  29.637,
        "b":  0.034735,
        "c":  9.2903e-06,
        "d":  -2.9885e-08,
        "e":  1.0937e-11,
    },
    "CH4": {
        "a":  34.942,
        "b":  -0.039957,
        "c":  0.00019184,
        "d":  -1.5303e-07,
        "e":  3.9321e-11,
    },
    "H2O": {
        "a":  33.933,
        "b":  -0.0084186,
        "c":  2.9906e-05,
        "d":  -1.7825e-08,
        "e":  3.6934e-12,
    },
    "SO3": {
        "a":  22.466,
        "b":  0.11981,
        "c":  -9.0842e-05,
        "d":  2.5503e-08,
        "e":  -7.9208e-13,
    },
    "O2": {
        "a":  29.526,
        "b":  -0.0088999,
        "c":  3.8083e-05,
        "d":  -3.2629e-08,
        "e":  8.8607e-12,
    },
    "Air": {
        "a":  29.381,
        "b":  -0.0046652,
        "c":  1.5957e-05,
        "d":  -1.0258e-08,
        "e":  2.0656e-12,
    },
    "C2H6O": {
        "a":  27.091,
        "b":  0.11055,
        "c":  0.00010957,
        "d":  -1.5046e-07,
        "e":  4.6601e-11,
    },
    "C19H36O2": {
        "a":  -44.058,
        "b":  2.0299,
        "c":  -1.5136e-03,
        "d":  6.1226e-07,
        "e":  -1.0656e-10,
    },
}


# ---------------------------------------------------------------------------
# Entalpias padrão de formação H_f° [kJ/mol]
# ---------------------------------------------------------------------------
HF_REF = {
    "H2":    0.0,
    "CO":  -110.54,
    "CH4":  -74.85,
    "CO2": -393.51,
    "SO2": -296.8,
    "N2":    0.0,
    "H2O": -241.8,
    "SO3": -395.7,
    "O2":    0.0,
    "Air":    0.0,
    "C2H6O": -234.81,
    "C19H36O2": -626.00,  
}

# ---------------------------------------------------------------------------
# Entropias padrão S°(298 K, 1 bar) [kJ/(mol·K)]
# ---------------------------------------------------------------------------
SF_REF = {
    "H2":        0.13057,
    "CO":        0.19754,
    "CH4":       0.18627,
    "CO2":       0.21369,
    "SO2":       0.24811,
    "N2":        0.19150,
    "H2O":       0.18872,
    "SO3":       0.25651,
    "O2":        0.20504,
    "Air":       0.19434, 
    "C2H6O":     0.28259,
    "C19H36O2":  0.95600,
}


# Temperatura de referência [K]
T_REF = 298.0


# ===========================================================================
# Funções de Cp
# ===========================================================================

def cp_species(T: float, species: str) -> float:
    """
    Capacidade calorífica molar de uma espécie pura à temperatura T.

    Cp = [a + b·T + c·T² + d·T³ + e·T⁴] / 1000 [kJ/(mol·K)]

    Parâmetros
    ----------
    T       : float  temperatura [K]
    species : str    nome da espécie (deve estar em GAS_COEFFS)

    Retorna
    -------
    float  Cp  [kJ/(mol·K)]
    """
    g = GAS_COEFFS[species]
    a = g["a"]
    b = g["b"]
    c = g["c"]
    d = g["d"]
    e = g["e"]
    return (
        a
        + b * T
        + c * T**2
        + d * T**3
        + e * T**4
    ) / 1000.0


def cp_fuel(T: float, Fmolfrac: dict, MMFuel: float) -> float:
    """
    Capacidade calorífica específica da mistura de combustível [kJ/(kg·K)].

    Tradução de: CpFuelfunc[T_]

    Parâmetros
    ----------
    T        : float  temperatura [K]
    Fmolfrac : dict   {species: mole_fraction} (mistura de combustível)
    MMFuel   : float  massa molar média do combustível [kg/mol]

    Retorna
    -------
    float  Cp_fuel  [kJ/(kg·K)]
    """
    total = sum(cp_species(T, sp) * Fmolfrac[sp] for sp in SPECIES)
    return total / MMFuel


def cp_air(T: float) -> float:
    """
    Capacidade calorífica específica do ar (representado por "Air") [kJ/(kg·K)].

    Tradução de: CpArfunc[T_]

    Parâmetros
    ----------
    T : float  temperatura [K]

    Retorna
    -------
    float  Cp_air  [kJ/(kg·K)]
    """
    return cp_species(T, "Air") / MM_AIR


def cp_comb(T: float, Combmolfrac: dict, MMComb: float) -> float:
    """
    Capacidade calorífica específica da mistura de produtos da combustão [kJ/(kg·K)].

    Tradução de: CpCombfunc[T_]

    Parâmetros
    ----------
    T           : float  temperatura [K]
    Combmolfrac : dict   {species: mole_fraction} (produtos da combustão)
    MMComb      : float  massa molar média dos produtos [kg/mol]

    Retorna
    -------
    float  Cp_comb  [kJ/(kg·K)]
    """
    total = sum(cp_species(T, sp) * Combmolfrac[sp] for sp in SPECIES)
    return total / MMComb


# ===========================================================================
# Funções de H (entalpia)
# ===========================================================================

def h_species(T: float, species: str) -> float:
    """
    Entalpia molar de uma espécie pura à temperatura T [kJ/mol].

    H(T) = Hf°/1000
           + [a·(T-298)
              + b/2·(T²-298²)
              + c/3·(T³-298³)
              + d/4·(T⁴-298⁴)
              + e/5·(T⁵-298⁵)

    Observação: o /1000 aplicado a HF_REF é mantido por consistência com a
    formulação de entalpia adotada.

    Tradução de: Hfunc[Th_, comp_]

    Parâmetros
    ----------
    T       : float  temperatura [K]
    species : str    nome da espécie

    Retorna
    -------
    float  H  [kJ/mol]
    """
    g = GAS_COEFFS[species]
    a, b, c, d, e = (
        g["a"], g["b"], g["c"], g["d"], g["e"]
    )
    Tref = T_REF

    integral = (
        a * (T - Tref)
        + (b / 2) * (T**2 - Tref**2)
        + (c / 3) * (T**3 - Tref**3)
        + (d / 4) * (T**4 - Tref**4)
        + (e / 5) * (T**5 - Tref**5)
    )
    return HF_REF[species] / 1000.0 + integral / 1000.0


def h_fuel(T: float, Fmolfrac: dict, MMFuel: float) -> float:
    """
    Entalpia específica da mistura de combustível [kJ/kg].

    Tradução de: HFuelfunc[Th_]

    Parâmetros
    ----------
    T        : float  temperatura [K]
    Fmolfrac : dict   {species: mole_fraction}
    MMFuel   : float  massa molar média do combustível [kg/mol]

    Retorna
    -------
    float  H_fuel  [kJ/kg]
    """
    total = sum(h_species(T, sp) * Fmolfrac[sp] for sp in SPECIES)
    return total / MMFuel


def h_air(T: float) -> float:
    """
    Entalpia específica do ar [kJ/kg].

    Tradução de: HArfunc[T_]

    Parâmetros
    ----------
    T : float  temperatura [K]

    Retorna
    -------
    float  H_air  [kJ/kg]
    """
    return h_species(T, "Air") / MM_AIR


def h_comb(T: float, Combmolfrac: dict, MMComb: float) -> float:
    """
    Entalpia específica da mistura de produtos da combustão [kJ/kg].

    Tradução de: HCombfunc[Th_]

    Parâmetros
    ----------
    T           : float  temperatura [K]
    Combmolfrac : dict   {species: mole_fraction}
    MMComb      : float  massa molar média dos produtos [kg/mol]

    Retorna
    -------
    float  H_comb  [kJ/kg]
    """
    total = sum(h_species(T, sp) * Combmolfrac[sp] for sp in SPECIES)
    return total / MMComb


# ===========================================================================
# Funções de S (entropia)
# ===========================================================================

def s_species(T: float, species: str, P: float) -> float:
    """
    Entropia molar de uma espécie pura em (T, P) [kJ/(mol·K)].

    S(T,P) = S°_ref
             + [a·ln(T/298)
                + b·(T-298)
                + c/2·(T²-298²)
                + d/3·(T³-298³)
                + e/4·(T⁴-298⁴)
             − R·ln(P/P0)

    Tradução de: Sfunc[Th_, comp_, P_]

    Parâmetros
    ----------
    T       : float  temperatura [K]
    species : str    nome da espécie
    P       : float  pressão [bar]

    Retorna
    -------
    float  S  [kJ/(mol·K)]
    """
    g = GAS_COEFFS[species]
    a, b, c, d, e = (
        g["a"], g["b"], g["c"], g["d"], g["e"]
    )
    Tref = T_REF

    integral = (
        a * math.log(T / Tref)
        + b * (T - Tref)
        + (c / 2) * (T**2 - Tref**2)
        + (d / 3) * (T**3 - Tref**3)
        + (e / 4) * (T**4 - Tref**4)
    )
    return SF_REF[species] + integral / 1000.0 - R * math.log(P / P0)


def s_fuel(T: float, P: float, Fmolfrac: dict, MMFuel: float) -> float:
    """
    Entropia específica da mistura de combustível [kJ/(kg·K)].

    Tradução de: SFuelfunc[Th_, P_]

    Parâmetros
    ----------
    T        : float  temperatura [K]
    P        : float  pressão [bar]
    Fmolfrac : dict   {species: mole_fraction}
    MMFuel   : float  massa molar média do combustível [kg/mol]

    Retorna
    -------
    float  S_fuel  [kJ/(kg·K)]
    """
    total = sum(s_species(T, sp, P) * Fmolfrac[sp] for sp in SPECIES)
    return total / MMFuel


def s_air(T: float, P: float) -> float:
    """
    Entropia específica do ar [kJ/(kg·K)].

    Tradução de: SArfunc[T_, P_]

    Parâmetros
    ----------
    T : float  temperatura [K]
    P : float  pressão [bar]

    Retorna
    -------
    float  S_air  [kJ/(kg·K)]
    """
    return s_species(T, "Air", P) / MM_AIR


def s_comb(T: float, P: float, Combmolfrac: dict, MMComb: float) -> float:
    """
    Entropia específica da mistura de produtos da combustão [kJ/(kg·K)].

    Tradução de: SCombfunc[Th_, P_]

    Parâmetros
    ----------
    T           : float  temperatura [K]
    P           : float  pressão [bar]
    Combmolfrac : dict   {species: mole_fraction}
    MMComb      : float  massa molar média dos produtos [kg/mol]

    Retorna
    -------
    float  S_comb  [kJ/(kg·K)]
    """
    total = sum(s_species(T, sp, P) * Combmolfrac[sp] for sp in SPECIES)
    return total / MMComb