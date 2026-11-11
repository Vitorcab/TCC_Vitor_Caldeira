"""
brayton.py
==========
Modelo de Combustão de Turbina a Gás — Parte 4: Solucionador do Ciclo Brayton

Implementa o solucionador do ciclo Brayton para o modelo de combustão de turbina a gás.

Unidades utilizadas:
  Temperaturas  [K]
  Pressões      [bar]
  Vazões mássicas [kg/s]
  Entalpias     [kJ/kg]
  Entropias     [kJ/(kg·K)]
  Trabalho/calor [kW]
  LHV           [kJ/kg de combustível]
"""

import math
from typing import Optional, Union
import scipy.optimize as opt
import maps
from properties import (
    P0, R,
    FMASSFRAC_CASES, SPECIES,
    molar_mass, MM_AIR,
    get_mole_fractions, get_mean_molar_mass,
    )
from thermo import h_air, s_air, h_fuel, s_fuel, h_comb, s_comb
from combustion import balance_reaction, combustion_properties

# ---------------------------------------------------------------------------
# Banco de dados de propriedades críticas para o cálculo do fator de compressibilidade
# Formato: {"species": {"Tc": K, "Pc": bar, "omega": fator acêntrico}}
# ---------------------------------------------------------------------------
CRITICAL_PROPERTIES = {
    "H2":    {"Tc": 33.18,    "Pc": 13.13,   "omega": -0.220},
    "CO":    {"Tc": 132.92,   "Pc": 34.99,   "omega": 0.066},  
    "CH4":   {"Tc": 190.58,   "Pc": 46.04,   "omega": 0.011},  
    "CO2":   {"Tc": 304.19,   "Pc": 73.82,   "omega": 0.228},  
    "SO2":   {"Tc": 430.75,   "Pc": 78.84,   "omega": 0.245},  
    "N2":    {"Tc": 126.10,   "Pc": 33.94,   "omega": 0.040}, 
    "H2O":   {"Tc": 647.13,   "Pc": 220.55,  "omega": 0.345}, 
    "SO3":   {"Tc": 490.85,   "Pc": 82.07,   "omega": 0.422},  
    "O2":    {"Tc": 154.58,   "Pc": 50.43,   "omega": 0.022},  
    "C2H6O": {"Tc": 516.25,   "Pc": 63.84,   "omega": 0.637}, 
    "C19H36O2": {"Tc": 758.0, "Pc": 12.18,   "omega": 1.049},  
}


def _z_pitzer(T: float, P: float, Tc: float, Pc: float, omega: float) -> float:
    """
    Calcula o fator de compressibilidade usando a correlação de Pitzer.
    
    Z = Z^(0) + omega * Z^(1)
    
    em que Z^(0) e Z^(1) são funções da temperatura e da pressão reduzidas.
    
    Parâmetros
    ----------
    T     : float  temperatura [K]
    P     : float  pressão [bar]
    Tc    : float  temperatura crítica [K]
    Pc    : float  pressão crítica [bar]
    omega : float  fator acêntrico [-]
    
    Retorna
    -------
    float  fator de compressibilidade Z [-]
    """
    # Propriedades reduzidas
    Tr = T / Tc
    Pr = P / Pc
    
    # Coeficientes da correlação de Pitzer
    B0 = 0.083 - 0.422 / (Tr ** 1.6)
    B1 = 0.139 - 0.172 / (Tr ** 4.2)
    
    # Fator de compressibilidade
    Z_0 = 1.0 + (B0 * Pr) / Tr
    Z_1 = (B1 * Pr) / Tr
    
    Z = Z_0 + omega * Z_1
    
    return max(Z, 0.1)  # Garante valores fisicamente plausíveis


def calc_compressibility_factor(T: float, P: float, species: str) -> float:
    """
    Calcula o fator de compressibilidade de uma espécie pura usando a correlação de Pitzer.
    
    Parâmetros
    ----------
    T      : float  temperatura [K]
    P      : float  pressão [bar]
    species: str    nome da espécie
    
    Retorna
    -------
    float  fator de compressibilidade Z [-]
    """
    if species not in CRITICAL_PROPERTIES:
        # Assume comportamento de gás ideal para espécies desconhecidas
        return 1.0
    
    props = CRITICAL_PROPERTIES[species]
    return _z_pitzer(T, P, props["Tc"], props["Pc"], props["omega"])


def calc_mixture_compressibility(T: float, P: float, molfrac: dict) -> float:
    """
    Calcula o fator de compressibilidade de uma mistura gasosa.
    
    Usa regras de mistura:
    - Temperatura pseudocrítica: Tpc = Σ(yi * Tc,i)
    - Pressão pseudocrítica: Ppc = Σ(yi * Pc,i)
    - Fator pseudoacêntrico: ωpc = Σ(yi * ωi)
    
    Em seguida, calcula Z usando a correlação de Pitzer com as propriedades pseudocríticas.
    
    Parâmetros
    ----------
    T       : float        temperatura [K]
    P       : float        pressão [bar]
    molfrac : dict {str: float}  frações molares das espécies
    
    Retorna
    -------
    float  fator de compressibilidade Z [-]
    """
    # Propriedades pseudocríticas
    Tpc = 0.0
    Ppc = 0.0
    omega_pc = 0.0
    
    for sp, yi in molfrac.items():
        if sp in CRITICAL_PROPERTIES:
            props = CRITICAL_PROPERTIES[sp]
            Tpc += yi * props["Tc"]
            Ppc += yi * props["Pc"]
            omega_pc += yi * props["omega"]
    
    if Tpc == 0 or Ppc == 0:
        return 1.0  # Usa gás ideal como alternativa
    
    return _z_pitzer(T, P, Tpc, Ppc, omega_pc)


# ---------------------------------------------------------------------------
# Parâmetros padrão de operação
# ---------------------------------------------------------------------------
DEFAULT_mAir   = 19.5    # kg/s  vazão mássica de ar
DEFAULT_MFUEL =  0.5    # kg/s  vazão mássica de combustível
DEFAULT_T1    = 288.2   # K     temperatura de entrada do compressor
DEFAULT_TFUEL = 288.2   # K     temperatura do combustível
NCOMB         = 1.0     # eficiência de combustão (ncomb = 1)
T0            = 288.15  # K     temperatura ambiente

# Valores de LHV por espécie [MJ/kg] — usados para calcular a LHV com base na composição.
#   LHV = Σ_i (w_i × LHV_i)
LHV_SPECIES = {
    "CH4":50.0, "CO":10.0, "CO2":0.0, "H2":120.0,
    "H2O":0.0, "N2":0.0, "SO2":0.0, "O2":0.0,
    "Air":0.0, "SO3":0.0, "C2H6O":26.7, "C19H36O2":37.0
}

def compute_lhv(case_name: str) -> float:
    """
    Calcula a LHV [kJ/kg] a partir da composição mássica do combustível.

    LHV = Σ_i (w_i × LHV_i), em que LHV_i são os poderes caloríficos inferiores específicos
    de cada combustível.
    """
    Fmassfrac = FMASSFRAC_CASES[case_name]
    return sum(LHV_SPECIES.get(sp, 0.0) * Fmassfrac.get(sp, 0.0)
               for sp in LHV_SPECIES) * 1000.0   # MJ/kg → kJ/kg

# ---------------------------------------------------------------------------
# Ajustes polinomiais de eficiência e razão de pressão baseados em mapas
# (coeficientes obtidos dos mapas ajustados do compressor e da turbina)
# ---------------------------------------------------------------------------

def calc_comp_eff(mfr: float, rp: float) -> float:
    """
    Eficiência isentrópica do compressor η_c(mfr, rp).
    """
    return maps.calc_comp_eff(mfr, rp)


def calc_turb_eff(mfr: float, rp: float) -> float:
    """
    Eficiência isentrópica da turbina η_t(mfr, rp).
    """
    return maps.calc_turb_eff(mfr, rp)


# ---------------------------------------------------------------------------
# Buscador de raízes (equivalente a SafeFindRoot)
# ---------------------------------------------------------------------------

_SENTINEL = -1e4   # valor retornado por SafeFindRoot em caso de falha


def safe_find_root(func, x0: float, bracket: tuple = None,
                   method: str = "secant") -> float:
    """
    Buscador robusto de raízes escalares. Retorna _SENTINEL em caso de falha.

    Primeiro tenta encontrar a raiz em um intervalo delimitado; se necessário,
    recorre a um método baseado na secante.
    """
    try:
        if bracket is not None:
            a, b = bracket
            fa, fb = func(a), func(b)
            if fa * fb < 0:
                return opt.brentq(func, a, b, xtol=1e-8, maxiter=200)
        result = opt.root_scalar(func, x0=x0, method="secant",
                                 x1=x0 * 1.001 + 0.1,
                                 xtol=1e-8, maxiter=200)
        if result.converged:
            return result.root
    except Exception:
        pass
    return _SENTINEL


# ---------------------------------------------------------------------------
# Funções de análise termoeconômica (abordagem funcional)
# ---------------------------------------------------------------------------

def calc_thermoeconomic_params(
    C_m: float,
    m_dot_f: float,
    LHV: float,
    gamma: float,
    Ex_dest_dict: dict,
    Etaex: float,
) -> dict:
    """
    Calcula parâmetros termoeconômicos para um ciclo Brayton alimentado por biocombustível.
    
    Implementa as relações solicitadas:
      exch   = gamma * LHV  [kJ/kg]
      cfuel  = C_m / exch  [monetary units per kJ]
      C_dot_D_k = cfuel * Ex_dest_k  [monetary units per second]
      cW     = Etaex * cfuel  [monetary units per kJ]
    
    Parâmetros
    ----------
    C_m          : float  custo de mercado do biocombustível [unidades monetárias por kg]
    m_dot_f      : float  vazão mássica de combustível [kg/s]
    LHV          : float  poder calorífico inferior [kJ/kg]
    gamma        : float  fator de exergia (padrão ~1.06 para etanol/biodiesel)
    Ex_dest_dict : dict   {"compressor": kW, "combustor": kW, "turbine": kW}
    Etaex        : float  eficiência de segunda lei do ciclo [-]
    
    Retorna
    -------
    dict com as chaves: exch, cfuel, destruction_costs (dict),
    total_destruction_cost e cW
    """
    # Exergia química específica
    exch = gamma * LHV  # [kJ/kg]
    
    # Custo unitário da exergia do combustível
    cfuel = C_m / exch  # [monetary units per kJ]
    
    # Taxa de custo da exergia do combustível
    fuel_cost_rate = C_m * m_dot_f  # [monetary units per second]
    exergy_input_rate = m_dot_f * exch  # [kW]
    
    # Custos de destruição para cada componente
    destruction_costs = {
        component: cfuel * ex_dest
        for component, ex_dest in Ex_dest_dict.items()
    }
    
    # Custo total de destruição
    total_destruction_cost = sum(destruction_costs.values())
    
    # Custo unitário do trabalho útil
    cW = cfuel / Etaex  # [monetary units per kJ]
    
    return {
        "exch": exch,
        "cfuel": cfuel,
        "fuel_cost_rate": fuel_cost_rate,
        "exergy_input_rate": exergy_input_rate,
        "destruction_costs": destruction_costs,
        "total_destruction_cost": total_destruction_cost,
        "cW": cW,
    }


# ---------------------------------------------------------------------------
# Solucionador principal do ciclo Brayton
# ---------------------------------------------------------------------------

def brayton(
    T1: float = 298.0,
    Tfuel: float = 298.0,
    LHV: Optional[float] = None,
    mFuel: float = None,
    mAir: float = None,
    case_name: str = None,
    ncomb: float = NCOMB,
    C_m: Optional[float] = None,
    gamma: float = 1.06,
    rp: float = None,
    eta_t: float = None,
    eta_c: float = None,
) -> dict:
    """
    Solucionador completo do ciclo Brayton com análise termoeconômica opcional.

    Tradução de: brayton[T1_, Tfuel_, rp_, LHV_, mFuel_, mAir_, param_, caseName_]
    Retorna um dicionário com todas as saídas do ciclo Brayton e, opcionalmente,
    parâmetros termoeconômicos, caso C_m seja informado.

    Parâmetros
    ----------
    T1        : float         temperatura de entrada do compressor [K]
    Tfuel     : float         temperatura de fornecimento do combustível [K]
    rp        : float         razão de pressão (P2/P1)
    LHV       : float | None  poder calorífico inferior [kJ/kg].
                              Se None (recomendado para os casos padrão), LHV é
                              calculada a partir da composição do combustível,
                              usando os valores definidos para cada combustível.
    mFuel     : float         vazão mássica de combustível [kg/s]
    mAir      : float         vazão mássica de ar [kg/s]
    case_name : str           nome do caso de combustível (para o modelo de combustão)
    ncomb     : float         eficiência de combustão (padrão 1.0)
    C_m       : float | None  custo de mercado do combustível [unidades monetárias por kg].
                              Se informado, inclui a análise termoeconômica.
    gamma     : float         fator de exergia do combustível (padrão 1.06).

    Retorna
    -------
    dict com todas as saídas do ciclo (consulte OUTPUT_KEYS para a lista completa),
    além das chaves termoeconômicas opcionais, caso C_m seja informado:
      - "exch": exergia química específica [kJ/kg]
      - "cfuel": custo unitário da exergia do combustível [unidades monetárias por kJ]
      - "fuel_cost_rate": taxa de custo do combustível [unidades monetárias por segundo]
      - "exergy_input_rate": taxa de entrada de exergia do combustível [kW]
      - "destruction_costs": taxa de custo por componente [unidades monetárias por segundo]
      - "total_destruction_cost": soma de todos os custos [unidades monetárias por segundo]
      - "cW": custo unitário do trabalho útil [unidades monetárias por kJ]
    """
    # ── Propriedades do combustível / combustão ───────────────────────────
    Fmassfrac = FMASSFRAC_CASES[case_name]
    Fmolfrac  = get_mole_fractions(Fmassfrac)
    MMFuel    = get_mean_molar_mass(Fmolfrac)

    Combmolfrac, _alpha, _nArstq = balance_reaction(mAir, mFuel, Fmolfrac, MMFuel)
    MMComb, _RComb = combustion_properties(Combmolfrac)

    # LHV calculada a partir da composição do combustível para o balanço de energia interna.
    # Se o chamador informar uma LHV explícita, ela será usada para manter
    # a compatibilidade com versões anteriores.
    LHVfunc = sum(LHV_SPECIES.get(sp, 0.0) * Fmassfrac.get(sp, 0.0)
                  for sp in LHV_SPECIES) * 1000.0   # MJ/kg → kJ/kg
    LHV_used = LHVfunc if LHV is None else LHV


    # ── Pressões ──────────────────────────────────────────────────────────
    P1 = P0          # entrada [bar]
    
    mcorr_comp = mAir*(((T1/T0)**(1/2))/(P1/P0))
    if rp == None:
        rp = maps.pressure_ratio_from_mass_flow(mcorr_comp)

    P2 = P1 * rp     # saída do compressor
    P3 = P2          # saída da câmara de combustão (sem queda de pressão)
    P4 = P1          # saída da turbina

    # ── Temperatura isentrópica do compressor T2s ─────────────────────────
    # Resolver: s_air(T2s, P2) − s_air(T1, P1) = 0
    s1_air = s_air(T1, P1)

    def eq_t2s(T2s):
        return s_air(T2s, P2) - s1_air

    T2s = safe_find_root(eq_t2s, x0=600.0, bracket=(T1, 2000.0))

    WcompressorS = mAir * (h_air(T2s) - h_air(T1))

    # ── Temperatura real do compressor T2 ─────────────────────────────────
    if eta_c == None:
        eta_c = calc_comp_eff(mcorr_comp, rp)

    T2 = T1 + (T2s - T1) / eta_c
    Wcompressor = mAir * (h_air(T2) - h_air(T1))

    # ── Temperatura de saída da câmara de combustão T3 ────────────────────
    # Balanço de energia (temperatura de referência da entalpia = T1):
    #   (mAir+mFuel) * (hComb(T3) − hComb(T1))
    #   = mAir*(hAir(T2)−hAir(T1)) + ncomb*mFuel*LHV + mFuel*(hFuel(Tfuel)−hFuel(T1))
    h1_comb = h_comb(T1, Combmolfrac, MMComb)
    h2_air  = h_air(T2)
    h1_air  = h_air(T1)
    h_fuel_in  = h_fuel(Tfuel, Fmolfrac, MMFuel)
    h_fuel_ref = h_fuel(T1,    Fmolfrac, MMFuel)

    rhs_comb = (mAir  * (h2_air - h1_air)
                + ncomb * mFuel * LHV_used
                + mFuel * (h_fuel_in - h_fuel_ref))

    def eq_t3(t3):
        return ((mAir + mFuel) * (h_comb(t3, Combmolfrac, MMComb) - h1_comb)
                - rhs_comb)

    T3 = safe_find_root(eq_t3, x0=1200.0, bracket=(T1, 3000.0))

    # ── Temperatura isentrópica da turbina T4s ────────────────────────────
    # Resolver: s_comb(T4s, P4) − s_comb(T3, P3) = 0
    s3_comb = s_comb(T3, P3, Combmolfrac, MMComb)

    def eq_t4s(T4s):
        return s_comb(T4s, P4, Combmolfrac, MMComb) - s3_comb

    T4s = safe_find_root(eq_t4s, x0=800.0, bracket=(200.0, T3))

    WturbinaS = (mAir + mFuel) * (h_comb(T3, Combmolfrac, MMComb)
                                  - h_comb(T4s, Combmolfrac, MMComb))

    # ── Temperatura real da turbina T4 ────────────────────────────────────
    mcorr_turb = (mAir + mFuel)*(((T3/T0)**(1/2))/(P3/P0))
    if eta_t == None:
        eta_t = calc_turb_eff(mcorr_turb, rp)
        
    T4 = T3 + eta_t * (T4s - T3)
    Wturbina = (mAir + mFuel) * (h_comb(T3, Combmolfrac, MMComb)
                                 - h_comb(T4, Combmolfrac, MMComb))

    # ── Geração de entropia [kJ/(K·s)] ────────────────────────────────────
    Sgcomp_raw = (s_air(T2, P2) - s_air(T1, P1)) * mAir
    Sgcomb_raw = (s_comb(T3, P3, Combmolfrac, MMComb) * (mAir + mFuel)
                  - s_air(T2, P2) * mAir
                  - s_fuel(Tfuel, P1, Fmolfrac, MMFuel) * mFuel)
    Sgturb_raw = (s_comb(T4, P4, Combmolfrac, MMComb)
                  - s_comb(T3, P3, Combmolfrac, MMComb)) * (mAir + mFuel)

    # ── Desempenho ────────────────────────────────────────────────────────
    Wl       = Wturbina - Wcompressor
    Qh       = ncomb * LHV_used * mFuel
    EtaTh    = Wl / Qh
    EtaCarnot = 1.0 - T1 / T3
    Etaex    =  EtaTh/gamma

    # ── Fatores de compressibilidade em cada estado ───────────────────────
    # Verifica a hipótese de gás ideal: para comportamento ideal, Z deve ser ≈ 1.0
    # Ar (as frações molares são majoritariamente N2 e O2)
    air_molfrac = {"N2": 0.79, "O2": 0.21}
    Z1 = calc_mixture_compressibility(T1, P1, air_molfrac)
    Z2 = calc_mixture_compressibility(T2, P2, air_molfrac)
    Z3 = calc_mixture_compressibility(T3, P3, Combmolfrac)
    Z4 = calc_mixture_compressibility(T4, P4, Combmolfrac)

    # ── Análise termoeconômica (se solicitada) ─────────────────────────────
    thermoeconomic_results = {}
    if C_m is not None:
        # Taxas de destruição de exergia de cada componente [kW]
        Ex_dest_dict = {
            "compressor": Sgcomp_raw * T0,
            "combustor": Sgcomb_raw * T0,
            "turbine": Sgturb_raw * T0,
        }
        
        # Usa sempre a eficiência exergética calculada (Etaex)
        thermoeconomic_results = calc_thermoeconomic_params(
            C_m=C_m,
            m_dot_f=mFuel,
            LHV=LHV_used,
            gamma=gamma,
            Ex_dest_dict=Ex_dest_dict,
            Etaex=Etaex,
        )

    # ── Agrupa os resultados ──────────────────────────────────────────────
    # Sgcomp/Sgcomb/Sgturb/SgMax são armazenados como ×T1/1000 (como na saída do Switch)
    result = {
        "T1":          T1,
        "T2":          T2,
        "T3":          T3,
        "T4":          T4,
        "P1":          P1,
        "P2":          P2,
        "P3":          P3,
        "P4":          P4,
        "s1":          s1_air,
        "s2":          s_air(T2, P2),
        "s3":          s3_comb,
        "s4":          s_comb(T4, P4, Combmolfrac, MMComb),
        "Z1":          Z1,
        "Z2":          Z2,
        "Z3":          Z3,
        "Z4":          Z4,
        "Wcompressor": Wcompressor,
        "Wturbina":    Wturbina,
        "Wl":          Wl,
        "Qh":          Qh,
        "EtaTh":       EtaTh,
        "EtaCarnot":   EtaCarnot,
        "Etaex":       Etaex,
        "Exdcomp":      Sgcomp_raw * T0,
        "Exdcomb":      Sgcomb_raw * T0,
        "Exdturb":      Sgturb_raw * T0,
        "mcorr_comp":  mcorr_comp,
        "mcorr_turb":  mcorr_turb,
        "LHVfunc":     LHVfunc,
        "Combmolfrac": Combmolfrac,
        # Variáveis internas (úteis para análise)
        "_T2s":        T2s,
        "_T4s":        T4s,
        "_eta_c":      eta_c,
        "_eta_t":      eta_t,
        "_alpha":      _alpha,
        "_MMComb":     MMComb,
        "_WcompressorS": WcompressorS,
        "_WturbinaS":    WturbinaS,
        "_Sgcomp_raw":   Sgcomp_raw,
        "_Sgcomb_raw":   Sgcomb_raw,
        "_Sgturb_raw":   Sgturb_raw,
        "_fuel":         case_name,
        "_rp":           rp
    }
    
    # Adiciona os resultados termoeconômicos se C_m tiver sido informado
    if thermoeconomic_results:
        result.update(thermoeconomic_results)
    
    return result
