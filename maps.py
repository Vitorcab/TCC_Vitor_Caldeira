"""
maps_new.py
===========
Modelo de Combustão de Turbina a Gás — Parte 5: Mapas do Compressor e da Turbina

Módulo compatível com o programa principal. Calcula as correlações dos mapas
usando coeficientes gerados a partir dos CSVs em ``Maps/``. Ao executar este
módulo diretamente, ``main()`` ajusta os modelos, imprime os coeficientes e os
salva em ``maps_coefficients.txt``.
"""

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import ConvexHull

Points = List[Tuple[float, float]]

DATAWORK_COMP: Points = []

TURBEFFVALS: List[float] = [0.80, 0.84, 0.86, 0.87, 0.88, 0.89]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPRESSOR_DATA_DIR = PROJECT_ROOT / "Maps" / "csv_comp"
TURBINE_DATA_DIR = PROJECT_ROOT / "Maps" / "csv_turb"
COEFFICIENTS_FILE = Path(__file__).with_name("maps_coefficients.txt")

# Fração mínima de folga, em razão de pressão, que a linha de máxima eficiência
# deve manter abaixo da linha de surge durante o ajuste: rp <= surge * (1 - margem).
WORK_SURGE_MARGIN = 0.05

# Tolerância (RMSE, na unidade da razão de pressão) usada para escolher o menor
# grau, entre 2 e 8, do polinômio da linha de máxima eficiência.
WORK_FIT_TOLERANCE = 0.05

# O que pressure_ratio_from_mass_flow faz fora da faixa em que a linha foi ajustada:
# "raise" levanta ValueError; "clip" usa o valor da borda da faixa.
WORK_OUT_OF_RANGE = "raise"
_EFFICIENCY_TERMS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (0, 2),
    (1, 1),
    (3, 0),
    (0, 3),
    (2, 1),
    (1, 2),
)
def _load_coefficients():
    global _COEFFICIENTS
    if _COEFFICIENTS is not None:
        return _COEFFICIENTS
    try:
        lines = COEFFICIENTS_FILE.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RuntimeError(
            "Os coeficientes ainda não foram gerados. Execute "
            f"'{Path(__file__).name}' diretamente para ajustar os mapas."
        ) from error

    values = {}
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            assignment = ast.parse(line, mode="exec").body[0]
            if (
                not isinstance(assignment, ast.Assign)
                or len(assignment.targets) != 1
                or not isinstance(assignment.targets[0], ast.Name)
            ):
                raise ValueError("esperada uma atribuição simples")
            values[assignment.targets[0].id] = ast.literal_eval(assignment.value)
        except (SyntaxError, ValueError, IndexError) as error:
            raise ValueError(
                f"Coeficiente inválido em {COEFFICIENTS_FILE}:{line_number}."
            ) from error

    required_names = {
        "WORK_CENTER",
        "WORK_SCALE",
        "WORK_COEFFICIENTS",
        "SURGE_COEFFICIENTS",
        "COMP_EFFICIENCY_COEFFICIENTS",
        "TURB_EFFICIENCY_COEFFICIENTS",
    }
    missing_names = required_names - values.keys()
    if missing_names:
        raise ValueError(
            f"Faltam coeficientes em {COEFFICIENTS_FILE}: "
            f"{', '.join(sorted(missing_names))}."
        )

    _COEFFICIENTS = SimpleNamespace(**values)
    return _COEFFICIENTS


def _read_points(path: Path) -> np.ndarray:
    points = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split(";")
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: esperado um par separado por ';'.")
        try:
            points.append(tuple(float(field.strip().replace(",", ".")) for field in fields))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: ponto numérico inválido.") from error

    if not points:
        raise ValueError(f"Nenhum ponto válido foi encontrado em {path}.")
    return np.asarray(points, dtype=float)


def _load_efficiency_data(
    directory: Path,
    filename_pattern: str,
    value_pattern: str,
    value_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(directory.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo '{filename_pattern}' foi encontrado em {directory}."
        )

    mass_flows: List[float] = []
    pressure_ratios: List[float] = []
    efficiencies: List[float] = []
    for path in files:
        match = re.fullmatch(value_pattern, path.name)
        if match is None:
            raise ValueError(f"Nome de arquivo de mapa inválido: {path.name}")
        efficiency = float(match.group(1)) * value_scale
        for mass_flow, pressure_ratio in _read_points(path):
            mass_flows.append(float(mass_flow))
            pressure_ratios.append(float(pressure_ratio))
            efficiencies.append(efficiency)

    return (
        np.asarray(mass_flows),
        np.asarray(pressure_ratios),
        np.asarray(efficiencies),
    )


def _design_matrix(
    mass_flow: np.ndarray,
    pressure_ratio: np.ndarray,
    terms: Sequence[Tuple[int, int]],
) -> np.ndarray:
    return np.column_stack(
        [
            mass_flow**mass_power * pressure_ratio**pressure_power
            for mass_power, pressure_power in terms
        ]
    )


def _fit_linear_model(
    design: np.ndarray,
    values: np.ndarray,
    description: str,
) -> Tuple[np.ndarray, float, float]:
    column_scales = np.linalg.norm(design, axis=0)
    if np.any(column_scales == 0):
        raise ValueError(f"Há termos sem variação nos dados de {description}.")

    scaled_design = design / column_scales
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(scaled_design, values, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(
            f"Dados insuficientes para ajustar todos os termos de {description} "
            f"(posto {rank}, esperado {design.shape[1]})."
        )

    coefficients = scaled_coefficients / column_scales
    residuals = values - design @ coefficients
    rmse = float(np.sqrt(np.mean(residuals**2)))
    total_variance = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total_variance
    return coefficients, rmse, r_squared


def _efficiency_values(
    coefficients: np.ndarray,
    mass_flow: Union[float, np.ndarray],
    pressure_ratio: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    mass_flow_array, pressure_ratio_array = np.broadcast_arrays(
        np.asarray(mass_flow, dtype=float),
        np.asarray(pressure_ratio, dtype=float),
    )
    result = np.zeros_like(mass_flow_array)
    for coefficient, (mass_power, pressure_power) in zip(
        coefficients, _EFFICIENCY_TERMS
    ):
        result += (
            coefficient
            * mass_flow_array**mass_power
            * pressure_ratio_array**pressure_power
        )
    if result.ndim == 0:
        return float(result)
    return result


def _fit_surge_line() -> Tuple[np.ndarray, float, float]:
    points = _read_points(COMPRESSOR_DATA_DIR / "comp_surge_line.csv")
    design = np.column_stack(
        [
            np.ones(len(points)),
            *[points[:, 0] ** power for power in range(4, 9)],
        ]
    )
    coefficients, rmse, r_squared = _fit_linear_model(
        design, points[:, 1], "linha de surge do compressor"
    )
    return coefficients, rmse, r_squared


def _surge_values(
    coefficients: Sequence[float],
    mass_flow: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    flow = np.asarray(mass_flow, dtype=float)
    result = coefficients[0] + sum(
        coefficient * flow**power
        for coefficient, power in zip(coefficients[1:], range(4, 9))
    )
    return float(result) if np.ndim(result) == 0 else result


def _vertical_bounds(hull_vertices: np.ndarray, mass_flow: float) -> Tuple[float, float]:
    intersections = []
    for start, end in zip(hull_vertices, np.roll(hull_vertices, -1, axis=0)):
        x1, y1 = start
        x2, y2 = end
        if min(x1, x2) - 1e-10 <= mass_flow <= max(x1, x2) + 1e-10:
            if abs(x2 - x1) < 1e-12:
                intersections.extend((float(y1), float(y2)))
            else:
                fraction = (mass_flow - x1) / (x2 - x1)
                intersections.append(float(y1 + fraction * (y2 - y1)))

    if len(intersections) < 2:
        raise ValueError(
            f"Não foi possível delimitar a região do mapa para mfr={mass_flow:.6g}."
        )
    return min(intersections), max(intersections)


def _maximum_efficiency_line(
    mass_flows: np.ndarray,
    pressure_ratios: np.ndarray,
    efficiency_coefficients: np.ndarray,
    surge_coefficients: Sequence[float],
    surge_margin: float = WORK_SURGE_MARGIN,
    sample_count: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Máximo da eficiência em cada vazão, dentro do casco convexo e abaixo do surge.

    Retorna (vazões, razões de pressão, máscara dos pontos cujo máximo ficou na
    borda do domínio de busca, ou seja, onde não há máximo interior).
    """
    points = np.column_stack((mass_flows, pressure_ratios))
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    flow_samples = np.linspace(float(mass_flows.min()), float(mass_flows.max()), sample_count)
    pressure_samples = np.empty_like(flow_samples)
    on_boundary = np.zeros(sample_count, dtype=bool)

    for index, flow in enumerate(flow_samples):
        lower, upper = _vertical_bounds(hull_vertices, float(flow))
        surge_cap = float(_surge_values(surge_coefficients, float(flow))) * (1.0 - surge_margin)
        upper = min(upper, surge_cap)
        if upper - lower < 1e-12:
            pressure_samples[index] = min(lower, upper)
            on_boundary[index] = True
            continue

        optimum = minimize_scalar(
            lambda pressure: -_efficiency_values(
                efficiency_coefficients, float(flow), pressure
            ),
            bounds=(lower, upper),
            method="bounded",
        )
        candidates = (lower, upper, float(optimum.x))
        pressure_samples[index] = max(
            candidates,
            key=lambda pressure: _efficiency_values(
                efficiency_coefficients, float(flow), pressure
            ),
        )
        on_boundary[index] = (
            abs(pressure_samples[index] - lower) < 1e-4
            or abs(pressure_samples[index] - upper) < 1e-4
        )

    return flow_samples, pressure_samples, on_boundary


def _fit_work_line(
    mass_flows: np.ndarray,
    pressure_ratios: np.ndarray,
    degree: int = 8,
) -> Tuple[float, float, np.ndarray, float, float]:
    center = float((mass_flows.min() + mass_flows.max()) / 2.0)
    scale = float((mass_flows.max() - mass_flows.min()) / 2.0)
    if scale <= 0:
        raise ValueError("A linha de máxima eficiência exige mais de uma vazão.")

    normalized_flow = (mass_flows - center) / scale
    polynomial = np.polynomial.Polynomial.fit(
        normalized_flow, pressure_ratios, deg=degree, domain=[-1.0, 1.0]
    )
    coefficients = polynomial.coef
    predicted = polynomial(normalized_flow)
    residuals = pressure_ratios - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    total_variance = float(np.sum((pressure_ratios - np.mean(pressure_ratios)) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total_variance
    return center, scale, coefficients, rmse, r_squared


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Índices da maior sequência contínua de valores verdadeiros em ``mask``."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Nenhuma vazão da linha de máxima eficiência tem máximo interior.")
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return max(runs, key=len)


def _select_work_fit(
    mass_flows: np.ndarray,
    pressure_ratios: np.ndarray,
    tolerance: float = WORK_FIT_TOLERANCE,
    degrees: Sequence[int] = range(2, 9),
):
    """Escolhe o menor grau cujo RMSE fica dentro da tolerância.

    Se nenhum atender, escolhe o de menor RMSE. Retorna (grau, ajuste, tabela),
    com a tabela formada por tuplas (grau, RMSE, R²).
    """
    fits = {
        degree: _fit_work_line(mass_flows, pressure_ratios, degree) for degree in degrees
    }
    table = [(degree, fit[3], fit[4]) for degree, fit in fits.items()]
    acceptable = [degree for degree, rmse, _ in table if rmse <= tolerance]
    chosen = min(acceptable) if acceptable else min(table, key=lambda row: row[1])[0]
    return chosen, fits[chosen], table


def _check_work_line(
    center: float,
    scale: float,
    work_coefficients: np.ndarray,
    surge_coefficients: Sequence[float],
    flow_min: float,
    flow_max: float,
) -> None:
    """Falha se a linha ajustada cruzar o surge; avisa se ela não for monotônica."""
    flow = np.linspace(flow_min, flow_max, 400)
    work = np.polynomial.polynomial.polyval((flow - center) / scale, work_coefficients)
    surge = _surge_values(surge_coefficients, flow)

    violations = flow[work >= surge]
    if violations.size:
        raise ValueError(
            "A linha de máxima eficiência ajustada atinge a linha de surge entre "
            f"mfr={violations.min():.3g} e mfr={violations.max():.3g}; "
            "os coeficientes não foram gravados."
        )

    slope_changes = int(np.sum(np.diff(np.sign(np.diff(work))) != 0))
    margin = (surge - work) / surge
    print(f"  Menor folga em relação ao surge: {100 * margin.min():.1f}% "
          f"(em mfr={flow[np.argmin(margin)]:.3g})")
    if slope_changes:
        print(f"  AVISO: a linha de máxima eficiência muda de sentido {slope_changes}x na faixa.")


def _format_coefficients(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.17g}" for value in values) + ",)"


def _write_coefficients(
    work_center: float,
    work_scale: float,
    work_coefficients: np.ndarray,
    surge_coefficients: np.ndarray,
    compressor_efficiency_coefficients: np.ndarray,
    turbine_efficiency_coefficients: np.ndarray,
) -> None:
    contents = f'''# Coeficientes ajustados por maps_new.py.
# Copie estas atribuicoes para o modulo maps.py para atualizar as correlacoes.

WORK_CENTER = {work_center:.17g}
WORK_SCALE = {work_scale:.17g}
WORK_COEFFICIENTS = {_format_coefficients(work_coefficients)}
SURGE_COEFFICIENTS = {_format_coefficients(surge_coefficients)}
COMP_EFFICIENCY_COEFFICIENTS = {_format_coefficients(compressor_efficiency_coefficients)}
TURB_EFFICIENCY_COEFFICIENTS = {_format_coefficients(turbine_efficiency_coefficients)}
'''
    COEFFICIENTS_FILE.write_text(contents, encoding="utf-8")


def _print_coefficients(name: str, coefficients: Sequence[float]) -> None:
    print(f"\n{name}:")
    for index, value in enumerate(coefficients):
        print(f"  c{index} = {float(value):+.16e}")


def main() -> None:
    """Ajusta as correlações dos mapas e grava os coeficientes resultantes."""
    comp_mass_flow, comp_pressure_ratio, comp_efficiency = _load_efficiency_data(
        COMPRESSOR_DATA_DIR,
        "comp_*_eff.csv",
        r"comp_(\d+)_eff\.csv",
        value_scale=0.01,
    )
    turb_mass_flow, turb_pressure_ratio, turb_efficiency = _load_efficiency_data(
        TURBINE_DATA_DIR,
        "turb_*_eff.csv",
        r"turb_([0-9]+(?:\.[0-9]+)?)_eff\.csv",
    )

    comp_efficiency_coefficients, comp_efficiency_rmse, comp_efficiency_r2 = (
        _fit_linear_model(
            _design_matrix(comp_mass_flow, comp_pressure_ratio, _EFFICIENCY_TERMS),
            comp_efficiency,
            "eficiência do compressor",
        )
    )
    turb_efficiency_coefficients, turb_efficiency_rmse, turb_efficiency_r2 = (
        _fit_linear_model(
            _design_matrix(turb_mass_flow, turb_pressure_ratio, _EFFICIENCY_TERMS),
            turb_efficiency,
            "eficiência da turbina",
        )
    )

    surge_coefficients, surge_rmse, surge_r2 = _fit_surge_line()
    work_mass_flow, work_pressure_ratio, work_on_boundary = _maximum_efficiency_line(
        comp_mass_flow,
        comp_pressure_ratio,
        comp_efficiency_coefficients,
        surge_coefficients,
    )
    interior = _longest_run(~work_on_boundary)
    fit_flow = work_mass_flow[interior]
    fit_pressure = work_pressure_ratio[interior]
    work_degree, work_fit, work_candidates = _select_work_fit(fit_flow, fit_pressure)
    work_center, work_scale, work_coefficients, work_rmse, work_r2 = work_fit
    print("Ajuste das correlações dos mapas")
    print(f"  Eficiência do compressor: {len(comp_mass_flow)} pontos")
    print(f"    RMSE={comp_efficiency_rmse:.6g}; R²={comp_efficiency_r2:.6g}")
    print(f"  Eficiência da turbina: {len(turb_mass_flow)} pontos")
    print(f"    RMSE={turb_efficiency_rmse:.6g}; R²={turb_efficiency_r2:.6g}")
    print(f"  Linha de máxima eficiência: {len(work_mass_flow)} pontos")
    print(f"    Faixa ajustada (máximo interior): mfr = {fit_flow.min():.2f} a "
          f"{fit_flow.max():.2f} ({len(fit_flow)} pontos)")
    for degree, rmse, r2 in work_candidates:
        mark = "  <- escolhido" if degree == work_degree else ""
        print(f"    grau {degree}: RMSE={rmse:.4g} bar; R²={r2:.6g}{mark}")
    if work_rmse > WORK_FIT_TOLERANCE:
        print(f"    AVISO: nenhum grau atingiu RMSE <= {WORK_FIT_TOLERANCE} bar.")
    print(f"    Pontos no limite do domínio de busca: "
          f"{int(work_on_boundary.sum())} de {len(work_on_boundary)}")
    surge_cap = _surge_values(surge_coefficients, work_mass_flow) * (1.0 - WORK_SURGE_MARGIN)
    at_cap = work_on_boundary & (np.abs(work_pressure_ratio - surge_cap) < 1e-4)
    for label, mask in (
        ("no teto do surge", at_cap),
        ("no limite inferior do casco", work_on_boundary & ~at_cap),
    ):
        if mask.any():
            indices = np.flatnonzero(mask)
            runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            ranges = ", ".join(
                f"{work_mass_flow[run[0]]:.2f}-{work_mass_flow[run[-1]]:.2f}" for run in runs
            )
            print(f"      {label}: {int(mask.sum())} pontos, mfr em {ranges}")
    print("  Linha de surge do compressor")
    print(f"    RMSE={surge_rmse:.6g} bar; R²={surge_r2:.6g}")
    _print_coefficients("WORK_COEFFICIENTS (variável normalizada)", work_coefficients)
    print(f"  WORK_CENTER = {work_center:.16e}")
    print(f"  WORK_SCALE  = {work_scale:.16e}")
    _print_coefficients("SURGE_COEFFICIENTS (a0, a4, a5, a6, a7, a8)", surge_coefficients)
    _print_coefficients("COMP_EFFICIENCY_COEFFICIENTS", comp_efficiency_coefficients)
    _print_coefficients("TURB_EFFICIENCY_COEFFICIENTS", turb_efficiency_coefficients)
    _check_work_line(
        work_center,
        work_scale,
        work_coefficients,
        surge_coefficients,
        float(fit_flow.min()),
        float(fit_flow.max()),
    )
    _write_coefficients(
        work_center,
        work_scale,
        work_coefficients,
        surge_coefficients,
        comp_efficiency_coefficients,
        turb_efficiency_coefficients,
    )
    global _COEFFICIENTS
    _COEFFICIENTS = None
    _load_coefficients()
    print(f"\nCoeficientes salvos em: {COEFFICIENTS_FILE}")


_COEFFICIENTS = None


def pressure_ratio_from_mass_flow(mfr: float) -> float:
    """Calcula a razão de pressão na linha de máxima eficiência do compressor.

    Válida apenas na faixa WORK_CENTER ± WORK_SCALE em que a linha foi ajustada.
    Fora dela, o comportamento é definido por WORK_OUT_OF_RANGE.
    """
    coefficients = _load_coefficients()
    lower = coefficients.WORK_CENTER - coefficients.WORK_SCALE
    upper = coefficients.WORK_CENTER + coefficients.WORK_SCALE
    if mfr < lower - 1e-9 or mfr > upper + 1e-9:
        if WORK_OUT_OF_RANGE == "raise":
            raise ValueError(
                f"mfr={mfr:.4g} fora da faixa de validade da linha de máxima "
                f"eficiência [{lower:.4g}, {upper:.4g}]."
            )
        mfr = min(max(mfr, lower), upper)
    normalized_flow = (mfr - coefficients.WORK_CENTER) / coefficients.WORK_SCALE
    return float(np.polynomial.polynomial.polyval(
        normalized_flow, coefficients.WORK_COEFFICIENTS
    ))


def mass_flow_from_pressure_ratio(rp_target: float) -> float:
    """Calcula a vazão mássica correspondente à razão de pressão em ``pressure_ratio_from_mass_flow``."""
    if not np.isfinite(rp_target):
        raise ValueError("rp_target deve ser um número finito.")

    coefficients = _load_coefficients()
    lower = coefficients.WORK_CENTER - coefficients.WORK_SCALE
    upper = coefficients.WORK_CENTER + coefficients.WORK_SCALE
    rp_lower = pressure_ratio_from_mass_flow(lower)
    rp_upper = pressure_ratio_from_mass_flow(upper)
    rp_min, rp_max = sorted((rp_lower, rp_upper))
    if rp_target < rp_min or rp_target > rp_max:
        raise ValueError(
            f"rp_target={rp_target:.4g} fora da faixa de razão de pressão "
            f"da linha de máxima eficiência [{rp_min:.4g}, {rp_max:.4g}]."
        )

    return float(brentq(lambda mfr: pressure_ratio_from_mass_flow(mfr) - rp_target, lower, upper))


def funcsurge(mfr: float) -> float:
    """Calcula a razão de pressão na linha de surge do compressor."""
    coefficients = _load_coefficients().SURGE_COEFFICIENTS
    return float(
        coefficients[0]
        + sum(
            coefficient * mfr**power
            for coefficient, power in zip(coefficients[1:], range(4, 9))
        )
    )


def calc_comp_eff(mfr: float, rp: float) -> float:
    """Calcula a eficiência isentrópica do compressor."""
    coefficients = _load_coefficients().COMP_EFFICIENCY_COEFFICIENTS
    return float(_efficiency_values(coefficients, mfr, rp))


def calc_turb_eff(mfr: float, rp: float) -> float:
    """Calcula a eficiência isentrópica da turbina."""
    coefficients = _load_coefficients().TURB_EFFICIENCY_COEFFICIENTS
    return float(_efficiency_values(coefficients, mfr, rp))


def is_surging(mfr: float, rp: float) -> bool:
    """Indica se a razão de pressão atingiu ou ultrapassou a linha de surge."""
    return rp >= funcsurge(mfr)


def operating_point(mAr: float) -> Tuple[float, float]:
    """Retorna o ponto ``(mfr, rp)`` da linha de máxima eficiência."""
    return mAr, pressure_ratio_from_mass_flow(mAr)


def comp_map_summary(mfr: float, rp: float) -> Dict[str, Union[float, bool]]:
    """Retorna as grandezas calculadas do compressor no ponto informado."""
    return {
        "mfr": mfr,
        "rp": rp,
        "eta_c": calc_comp_eff(mfr, rp),
        "rp_surge": funcsurge(mfr),
        "rp_design": pressure_ratio_from_mass_flow(mfr),
        "is_surging": is_surging(mfr, rp),
    }


def turb_map_summary(mfr: float, rp: float) -> Dict[str, float]:
    """Retorna as grandezas calculadas da turbina no ponto informado."""
    return {
        "mfr": mfr,
        "rp": rp,
        "eta_t": calc_turb_eff(mfr, rp),
    }


__all__ = [
    "Points",
    "DATAWORK_COMP",
    "TURBEFFVALS",
    "pressure_ratio_from_mass_flow",
    "mass_flow_from_pressure_ratio",
    "funcsurge",
    "calc_comp_eff",
    "calc_turb_eff",
    "is_surging",
    "operating_point",
    "comp_map_summary",
    "turb_map_summary",
    "main",
]


if __name__ == "__main__":
    main()