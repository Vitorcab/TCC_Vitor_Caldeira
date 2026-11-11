"""
thermoeconomics.py
==================
Modelo de Combustão de Turbina a Gás — Parte 6: Análise Termoeconômica

Disponibiliza ferramentas de análise termoeconômica para turbinas a gás de
ciclo Brayton que utilizam biocombustíveis.

Implementa as relações de custo exergético e retorna um relatório compacto,
adequado para integração com simuladores que já calculam propriedades
termodinâmicas e balanços de entropia.

Unidades:
  C_m          : unidades monetárias por kg de combustível (por exemplo, R$/kg)
  m_dot_f      : kg/s
  LHV          : kJ/kg
  gamma        : fator exergético adimensional
  exch         : kJ/kg de combustível
  cfuel        : unidades monetárias por kJ de exergia do combustível
  E˙xdest,k     : kW (kJ/s)
  C˙D,k        : unidades monetárias por segundo (taxa de custo)
  cW           : unidades monetárias por kJ de trabalho útil
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional


class ThermoeconomicAnalyzer:
    """
    Realiza uma análise termoeconômica simplificada de um ciclo Brayton
    alimentado por biocombustível.

    A classe implementa as relações:
      exch   = gamma * LHV
      cfuel  = C_m / exch
      CdotDk = cfuel * Ex_dest_k
      cW     = Etaex * cfuel

    Observações
    -----------
    - C_m representa o custo de mercado do combustível em unidades monetárias
      por kg.
    - A exergia específica do combustível, exch, é calculada em kJ/kg.
    - Portanto, o custo unitário da exergia do combustível, cfuel, é retornado
      em unidades monetárias por kJ.
    """

    def __init__(
        self,
        C_m: float,
        m_dot_f: float,
        LHV: float,
        gamma: float = 1.06,
        Ex_dest_dict: Optional[Mapping[str, float]] = None,
        Etaex: float = 0.0,
    ) -> None:
        self.C_m = float(C_m)
        self.m_dot_f = float(m_dot_f)
        self.LHV = float(LHV)
        self.gamma = float(gamma)
        self.Etaex = float(Etaex)
        self.Ex_dest_dict = dict(Ex_dest_dict) if Ex_dest_dict is not None else {}

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.C_m <= 0.0:
            raise ValueError(
                "C_m deve ser positivo (custo de mercado do combustível em "
                "unidades monetárias por kg)."
            )
        if self.m_dot_f < 0.0:
            raise ValueError("m_dot_f não pode ser negativo.")
        if self.LHV <= 0.0:
            raise ValueError("LHV deve ser positiva em kJ/kg.")
        if self.gamma <= 0.0:
            raise ValueError("gamma deve ser positivo.")
        if not 0.0 <= self.Etaex <= 1.0:
            raise ValueError("Etaex deve estar entre 0 e 1.")
        for component, value in self.Ex_dest_dict.items():
            if value < 0.0:
                raise ValueError(
                    f"A taxa de destruição de exergia de '{component}' "
                    "não pode ser negativa."
                )

    @property
    def exch(self) -> float:
        """Exergia química específica do biocombustível [kJ/kg]."""
        return self.gamma * self.LHV

    @property
    def cfuel(self) -> float:
        """Custo unitário da exergia do combustível [unidades monetárias por kJ]."""
        return self.C_m / self.exch

    @property
    def fuel_cost_rate(self) -> float:
        """Taxa de custo do combustível [unidades monetárias por segundo]."""
        return self.C_m * self.m_dot_f

    @property
    def exergy_input_rate(self) -> float:
        """Taxa de entrada de exergia do combustível [kJ/s = kW]."""
        return self.m_dot_f * self.exch

    def destruction_costs(self) -> Dict[str, float]:
        """Calcula a taxa de custo da destruição de exergia em cada componente."""
        return {
            component: self.cfuel * ex_dest
            for component, ex_dest in self.Ex_dest_dict.items()
        }

    @property
    def total_destruction_cost(self) -> float:
        """Soma das taxas de custo da destruição de exergia [unidades monetárias por segundo]."""
        return sum(self.destruction_costs().values())

    @property
    def cW(self) -> float:
        """Custo unitário do trabalho líquido [unidades monetárias por kJ]."""
        return  self.cfuel / self.Etaex

    def report(self) -> Dict[str, float]:
        """Retorna um relatório consolidado dos resultados termoeconômicos."""
        destructions = self.destruction_costs()
        return {
            "C_m": self.C_m,
            "m_dot_f": self.m_dot_f,
            "LHV": self.LHV,
            "gamma": self.gamma,
            "Etaex": self.Etaex,
            "exch": self.exch,
            "cfuel": self.cfuel,
            "fuel_cost_rate": self.fuel_cost_rate,
            "exergy_input_rate": self.exergy_input_rate,
            "exergy_destruction_costs": destructions,
            "total_destruction_cost": self.total_destruction_cost,
            "cW": self.cW,
        }


def compute_thermoeconomic_report(
    C_m: float,
    m_dot_f: float,
    LHV: float,
    gamma: float,
    Ex_dest_dict: Mapping[str, float],
    Etaex: float,
) -> Dict[str, float]:
    """Função auxiliar que retorna o relatório termoeconômico."""
    analyzer = ThermoeconomicAnalyzer(
        C_m=C_m,
        m_dot_f=m_dot_f,
        LHV=LHV,
        gamma=gamma,
        Ex_dest_dict=Ex_dest_dict,
        Etaex=Etaex,
    )
    return analyzer.report()
