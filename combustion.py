"""
combustion.py
=============
Modelo de Combustão de Turbina a Gás — Parte 3: Modelo de Combustão (COMB MODEL)

Implementa o balanço de combustão e os cálculos da composição dos produtos.

O modelo abrange os dois regimes de combustão:

  α ≥ 1  —  excesso de ar → combustão completa (linear, solução analítica)
  α < 1  —  deficiência de oxigênio → combustão parcial com equilíbrio
             de reforma a seco (Kp = 1 × 10⁻⁴, valor simplificado fixo)

Retorna
-------
Combmol  : dict {species: mole_fraction}  — composição dos produtos da combustão
alpha    : float  — razão de excesso de ar (nAr / nAr_stoich)
nArstq   : float  — vazão molar estequiométrica de ar [mol/s ou mol/kg,
                    dependendo de como mFuel é informado]
"""
import math
from typing import Tuple, Dict
from properties import (
    SPECIES,
    molar_mass,
    MM_AIR,
    get_mole_fractions,
    get_mean_molar_mass,
    FMASSFRAC_CASES,
    ATOMIC_COMPOSITION,
    R,
)

# Razão N₂/O₂ no ar seco (λ = 0.79 / 0.21)
_LAM = 0.79 / 0.21

# Constante de equilíbrio fixa para a reforma a seco do metano (simplificada)
# CH₄ + CO₂ → 2 CO + 2 H₂   Kp ≈ 1 × 10⁻⁴ (superestimada para 1 000–1 800 K)
_KP_DRY_REFORM = 1e-4


# ---------------------------------------------------------------------------

def balance_reaction(
    mAr: float,
    mFuel: float,
    Fmolfrac: Dict[str, float],
    MMFuel: float,
) -> Tuple[Dict[str, float], float, float]:
    """
    Calcula a composição dos produtos da combustão a partir das vazões mássicas
    de ar e combustível.

    A função recebe ``Fmolfrac`` e ``MMFuel`` como parâmetros explícitos,
    para maior clareza e reprodutibilidade.

    Parâmetros
    ----------
    mAr     : float  vazão mássica de ar [kg/s]
    mFuel   : float  vazão mássica de combustível [kg/s]
    Fmolfrac: dict   {species: mole_fraction} da mistura de combustível
    MMFuel  : float  massa molar média do combustível [kg/mol]

    Retorna
    -------
    Combmol : dict {species: mole_fraction}
    alpha   : float  razão de excesso de ar (ar real / ar estequiométrico)
    nArstq  : float  vazão molar estequiométrica de ar [mol/s]
    """
    # ── Vazões molares e quantidades totais de átomos no combustível [mol/s] ──
    # Calcula a quantidade total de mols de cada átomo no combustível
    total_C = sum(Fmolfrac[sp] * ATOMIC_COMPOSITION[sp]["C"] for sp in SPECIES) * mFuel / MMFuel
    total_H = sum(Fmolfrac[sp] * ATOMIC_COMPOSITION[sp]["H"] for sp in SPECIES) * mFuel / MMFuel
    total_O = sum(Fmolfrac[sp] * ATOMIC_COMPOSITION[sp]["O"] for sp in SPECIES) * mFuel / MMFuel
    total_S = sum(Fmolfrac[sp] * ATOMIC_COMPOSITION[sp]["S"] for sp in SPECIES) * mFuel / MMFuel
    total_N = sum(Fmolfrac[sp] * ATOMIC_COMPOSITION[sp]["N"] for sp in SPECIES) * mFuel / MMFuel

    # ── Vazões molares do ar ───────────────────────────────────────────────
    nAr = mAr / MM_AIR          # ar total [mol/s]
    nO2 = nAr * 0.21            # O₂ no ar [mol/s]

    # ── O₂ estequiométrico (generalizado a partir das equações algébricas de balanço) ──
    #
    # Reação completa: combustível + O₂ → CO₂ + H₂O + SO₃ + N₂
    # Balanços atômicos: C → CO₂, H → H₂O, S → SO₃, N → N₂
    # Balanço de O: 2*nO2stq = 2*total_C + total_H/2 + 3*total_S - total_O
    nO2stq = (2 * total_C + total_H / 2 + 3 * total_S - total_O) / 2.0
    nArstq = nO2stq / 0.21      # ar estequiométrico [mol/s]
    alpha  = nAr / nArstq       # razão de excesso de ar

    # ── Escolha do regime conforme α ──────────────────────────────────────
    if alpha >= 1.0:
        raw = _complete_combustion(
            nO2, total_C, total_H, total_O, total_S, total_N
        )
    else:
        raw = _partial_combustion(
            nO2, total_C, total_H, total_O, total_S, total_N
        )

    # ── Normaliza para obter frações molares ──────────────────────────────
    total = sum(raw.values())
    Combmol = {sp: raw.get(sp, 0.0) / total for sp in SPECIES}

    return Combmol, alpha, nArstq


# ---------------------------------------------------------------------------
# Funções auxiliares internas
# ---------------------------------------------------------------------------

def _complete_combustion(
    nO2, total_C, total_H, total_O, total_S, total_N
) -> Dict[str, float]:
    """
    α ≥ 1 — combustão completa com excesso de ar.

    Generalizada para qualquer composição de combustível usando as quantidades
    totais de átomos.

    Reação: combustível + (O₂ + λ N₂)·nO2 → CO₂·xCO2 + H₂O·xH2O + N₂·xN2 + SO₃·xSO3 + O₂·xO2
    """
    xCO2 = total_C                    # Balanço de C
    xH2O = total_H / 2.0              # Balanço de H
    xSO3 = total_S                    # Balanço de S
    xN2  = nO2 * _LAM + total_N / 2.0 # Balanço de N
    # Balanço de O: 2*nO2 + total_O = 2*xCO2 + xH2O + 3*xSO3 + 2*xO2
    xO2  = (2*nO2 + total_O - 2*xCO2 - xH2O - 3*xSO3) / 2.0

    return {
        "H2":  0.0,
        "CO":  0.0,
        "CH4": 0.0,
        "CO2": xCO2,
        "SO2": 0.0,
        "N2":  xN2,
        "H2O": xH2O,
        "SO3": xSO3,
        "O2":  max(xO2, 0.0),
        "Air":  0.0,
    }


def _partial_combustion(
    nO2, total_C, total_H, total_O, total_S, total_N
) -> Dict[str, float]:
    """
    α < 1 — combustão com deficiência de oxigênio.

    Aproximação generalizada: supõe que o combustível possa ser representado por
    quantidades equivalentes de CH4 e CO2 no equilíbrio da reforma a seco.
    Essa é uma simplificação para novos combustíveis.
    """
    # Aproxima o combustível como se fosse composto por CH4 e CO2
    # Assume-se que total_C vem de CH4 e CO2; como não temos as quantidades individuais,
    # usamos total_C para nCH4 + nCO2, entre outros termos.
    # É uma aproximação; para combustão parcial precisa, é necessário um modelo
    # completo de equilíbrio.
    nCH4_equiv = total_C * 0.5  # divisão aproximada
    nCO2_equiv = total_C * 0.5
    nH2_equiv = total_H / 2
    nSO2_equiv = total_S
    nN2_equiv = total_N / 2
    nH2O_equiv = 0  # assume ausência de H2O no combustível para combustão parcial

    # Usa a lógica original com as quantidades equivalentes
    A = nCO2_equiv + nCH4_equiv
    B = nH2_equiv + 3*nCH4_equiv - 2*nO2 - nCO2_equiv + nSO2_equiv
    C = 2*nO2 + nCO2_equiv - nSO2_equiv + nH2O_equiv - nCH4_equiv

    if nCO2_equiv > 0 and nCH4_equiv > 0:
        rhs = (_KP_DRY_REFORM * nCO2_equiv * nCH4_equiv) ** 0.5
    else:
        rhs = 0.0

    half_b = -(A - B) / 2.0
    disc = half_b**2 - (rhs - A * B)

    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
        xCO2 = half_b - sqrt_disc
    else:
        xCO2 = half_b

    xCO2 = max(xCO2, 0.0)
    xCO = max(A - xCO2, 0.0)
    xH2O = max(C - xCO2, 0.0)
    xH2 = max(B + xCO2, 0.0)
    xSO3 = nSO2_equiv
    xN2 = nO2 * _LAM + nN2_equiv

    return {
        "H2":  xH2,
        "CO":  xCO,
        "CH4": 0.0,
        "CO2": xCO2,
        "SO2": 0.0,
        "N2":  xN2,
        "H2O": xH2O,
        "SO3": xSO3,
        "O2":  0.0,
        "Air":  0.0,
    }


# ---------------------------------------------------------------------------
# Função auxiliar para obter a massa de ar estequiométrica
# ---------------------------------------------------------------------------

def stoichiometric_air_mass(mFuel: float, Fmolfrac: dict, MMFuel: float) -> float:
    """
    Massa de ar estequiométrica para uma dada vazão de combustível.

    Calcula a massa de ar estequiométrica a partir do resultado do balanço
    de combustão.

    Parâmetros
    ----------
    mFuel   : float  vazão mássica de combustível [kg/s]
    Fmolfrac: dict   frações molares do combustível
    MMFuel  : float  massa molar média do combustível [kg/mol]

    Retorna
    -------
    float  vazão mássica estequiométrica de ar [kg/s]
    """
    # Usa um valor fictício de mAir suficientemente alto para garantir α >= 1
    # (e executar corretamente o cálculo estequiométrico); depois, extrai nArstq.
    _, _, nArstq = balance_reaction(1e6, mFuel, Fmolfrac, MMFuel)
    return nArstq * MM_AIR


# ---------------------------------------------------------------------------
# Função auxiliar: massa molar média e constante específica dos gases dos produtos
# da combustão (RComb = R / MMComb — usada em todo o solucionador do ciclo Brayton)
# ---------------------------------------------------------------------------

def combustion_properties(Combmol: dict) -> Tuple[float, float]:
    """
    Retorna (MMComb, RComb) para uma composição de produtos.

    Equação: RComb = R / MMComb

    Parâmetros
    ----------
    Combmol : dict {species: mole_fraction}

    Retorna
    -------
    MMComb  [kg/mol]
    RComb   [kJ/(kg·K)]
    """
    from properties import R
    MMComb = sum(Combmol[sp] * molar_mass[sp] for sp in SPECIES)
    RComb  = R / MMComb
    return MMComb, RComb