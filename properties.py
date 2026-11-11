"""
properties.py
=============
Modelo de Combustão de Turbina a Gás — Parte 1: Propriedades

Contém:
  - Bancos de dados de frações mássicas de combustível (FmassfracCases)
  - Lista de índices das espécies                      (ILIST)
  - Tabela de massas molares                            (molar_mass)
  - Constantes do ar e dos gases universais             (MM_AIR, CP_AIR, R, P0)

As unidades são consistentes em todo o modelo:
  - Massas molares : kg/mol
  - R              : kJ/(mol·K)
  - CP_AIR         : kJ/(kg·K)
  - P0             : bar
"""
# ---------------------------------------------------------------------------
# Constantes universais
# ---------------------------------------------------------------------------
R = 8.314e-3   # kJ/(mol·K) — constante universal dos gases
P0 = 1.0        # bar         — pressão de referência

# ---------------------------------------------------------------------------
# Propriedades do ar
# ---------------------------------------------------------------------------
MM_AIR = 0.02896  # kg/mol
CP_AIR = 1.0061   # kJ/(kg·K) (constante, usada como alternativa)

# ---------------------------------------------------------------------------
# Composição atômica por espécie (para a estequiometria da combustão)
# Dicionário {species: {"C": int, "H": int, "O": int, "S": int, "N": int}}
# Usado para generalizar os cálculos de combustão para qualquer composição de combustível.
# ---------------------------------------------------------------------------
ATOMIC_COMPOSITION = {
    "CH4": {"C": 1, "H": 4, "O": 0, "S": 0, "N": 0},
    "CO":  {"C": 1, "H": 0, "O": 1, "S": 0, "N": 0},
    "CO2": {"C": 1, "H": 0, "O": 2, "S": 0, "N": 0},
    "H2":  {"C": 0, "H": 2, "O": 0, "S": 0, "N": 0},
    "H2O": {"C": 0, "H": 2, "O": 1, "S": 0, "N": 0},
    "N2":  {"C": 0, "H": 0, "O": 0, "S": 0, "N": 2},
    "SO2": {"C": 0, "H": 0, "O": 2, "S": 1, "N": 0},
    "O2":  {"C": 0, "H": 0, "O": 2, "S": 0, "N": 0},
    "Air": {"C": 0, "H": 0, "O": 0, "S": 0, "N": 0}, 
    "SO3": {"C": 0, "H": 0, "O": 3, "S": 1, "N": 0},
    "C2H6O": {"C": 2, "H": 6, "O": 1, "S": 0, "N": 0}, 
    "C19H36O2": {"C": 19, "H": 36, "O": 2, "S": 0, "N": 0},  
}
ILIST = {
    "CH4":1,
    "CO":2,
    "CO2":3,
    "H2":4,
    "H2O":5,
    "N2":6,
    "SO2":7,
    "O2":8,
    "Air":9,
    "SO3":10,
    "C2H6O":11,
    "C19H36O2":12
}

SPECIES = list(ILIST.keys())   # lista ordenada com os nomes de todas as espécies

# ---------------------------------------------------------------------------
# Massas molares [kg/mol]
# "Air" recebe o valor MM_AIR (= 0.02896) para representar o ar.
# ---------------------------------------------------------------------------
molar_mass = {
    "CH4":16.04e-3,
    "CO":28.01e-3,
    "CO2":44.01e-3,
    "H2":2.016e-3,
    "H2O":18.00e-3,
    "N2":28.014e-3,
    "SO2":64.07e-3,
    "O2":32.00e-3,
    "Air":MM_AIR,
    "SO3":80.00e-3,
    "C2H6O":46.07e-3,
    "C19H36O2":296.5e-3
}

# ---------------------------------------------------------------------------
# Bancos de dados de frações mássicas dos combustíveis (FmassfracCases)
# Cada entrada é um dicionário {species: mass_fraction}.
# ---------------------------------------------------------------------------
FMASSFRAC_CASES = {
    "Ethanol": {
        "CH4":0.0,
        "CO":0.0,
        "CO2":0.0,
        "H2":0.0,
        "H2O":0.07,
        "N2":0.0,
        "SO2":0.0,
        "O2":0.0,
        "Air":0.0,
        "SO3":0.0,
        "C2H6O":0.93,
        "C19H36O2":0.0
    },
    "Biomethane": {
        "CH4":0.95,
        "CO":0.0,
        "CO2":0.03,
        "H2":0.0,
        "H2O":0.0,
        "N2":0.02,
        "SO2":0.0,
        "O2":0.0,
        "Air":0.0,
        "SO3":0.0,
        "C2H6O":0.0,
        "C19H36O2":0.0
    },
    "Biodiesel": {
        "CH4":0.0,
        "CO":0.0,
        "CO2":0.0,
        "H2":0.0,
        "H2O":0.0,
        "N2":0.0,
        "SO2":0.0,
        "O2":0.0,
        "Air":0.0,
        "SO3":0.0,
        "C2H6O":0,
        "C19H36O2":1.0
    }
}


# ---------------------------------------------------------------------------
# Funções auxiliares derivadas
# ---------------------------------------------------------------------------

def get_mole_fractions(mass_fracs: dict) -> dict:
    """
    Converte frações mássicas em frações molares para uma dada composição de combustível.

    Parâmetros
    ----------
    mass_fracs : dict  {species: mass_fraction}

    Retorna
    -------
    dict {species: mole_fraction}
    """
    # O número de mols_i é proporcional a (w_i / MM_i)
    moles = {sp: mass_fracs[sp] / molar_mass[sp] for sp in SPECIES}
    total = sum(moles.values())
    if total == 0:
        return {sp: 0.0 for sp in SPECIES}
    return {sp: moles[sp] / total for sp in SPECIES}


def get_mean_molar_mass(mole_fracs: dict) -> float:
    """
    Massa molar média de uma mistura gasosa [kg/mol].

    Parâmetros
    ----------
    mole_fracs : dict  {species: mole_fraction}

    Retorna
    -------
    float  MM_mix  [kg/mol]
    """
    return sum(mole_fracs[sp] * molar_mass[sp] for sp in SPECIES)