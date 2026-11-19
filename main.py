"""
main.py
=======
Modelo de Combustão de Turbina a Gás — Execução e Análises

Orquestra as simulações, análises de eficiência e avaliações termoeconômicas
dos ciclos Brayton com biocombustíveis.
"""

import brayton
import maps
import combustion
import properties
import thermo
import thermoeconomics
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import optimization


def plot_gasturb_performance_validation():
    """Compara potência líquida e eficiência térmica com os resultados do GasTurb."""
    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")

    biocombustiveis = ["Biometano", "Etanol", "Biodiesel"]
    x = np.arange(len(biocombustiveis))
    largura = 0.32
    wnet_python = [4487.52, 3935.11, 3866.75]
    wnet_gasturb = [4385.10, 3836.50, 3772.70]
    eta_python = [34.61, 33.86, 33.50]
    eta_gasturb = [33.82, 33.01, 32.68]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Validação Multicombustível: Modelo Python vs. GasTurb 15.0",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    rects1 = ax1.bar(
        x - largura / 2, wnet_python, largura, label="Modelo Python",
        color="#1f77b4", edgecolor="black", linewidth=0.5,
    )
    rects2 = ax1.bar(
        x + largura / 2, wnet_gasturb, largura, label="GasTurb 15.0",
        color="#ff7f0e", edgecolor="black", linewidth=0.5,
    )
    ax1.set_ylabel(r"Potência Líquida Útil $\dot{W}_{net}$ (kW)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Comparação de Potência Útil", fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(biocombustiveis, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 5200)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")

    for rect in rects1:
        height = rect.get_height()
        label = f"{height:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        ax1.annotate(
            label, xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    for rect in rects2:
        height = rect.get_height()
        label = f"{height:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        ax1.annotate(
            label, xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=9,
        )

    rects3 = ax2.bar(
        x - largura / 2, eta_python, largura, label="Modelo Python",
        color="#1f77b4", edgecolor="black", linewidth=0.5,
    )
    rects4 = ax2.bar(
        x + largura / 2, eta_gasturb, largura, label="GasTurb 15.0",
        color="#ff7f0e", edgecolor="black", linewidth=0.5,
    )
    ax2.set_ylabel(r"Eficiência Térmica Global $\eta_{th}$ (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Comparação de Rendimento Térmico", fontsize=12, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(biocombustiveis, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 42)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none")

    for rects in (rects3, rects4):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(
                f"{height:.2f}%".replace(".", ","),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if rects is rects3 else "normal",
            )

    sns.despine()
    plt.tight_layout(pad=2.0)
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/validacao_desempenho_gasturb.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("-> Gráfico de validação de desempenho do GasTurb salvo!")


def plot_gasturb_temperature_validation():
    """Compara temperaturas do ciclo Brayton com os resultados do GasTurb."""
    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")

    metricas = [r"$T_2$ (Compressor)", r"$TIT$ (Combustor)", r"$T_4$ (Exaustão)"]
    x = np.arange(len(metricas))
    largura = 0.12
    temperaturas = [
        (x - 2.5 * largura, [630.98, 1191.59, 693.73], "Biometano (Python)", "#2ca02c", None),
        (x - 1.5 * largura, [630.37, 1184.71, 697.00], "Biometano (GasTurb)", "#98df8a", "//"),
        (x - 0.5 * largura, [630.98, 1129.12, 655.18], "Etanol (Python)", "#1f77b4", None),
        (x + 0.5 * largura, [630.37, 1122.63, 657.60], "Etanol (GasTurb)", "#aec7e8", "//"),
        (x + 1.5 * largura, [630.98, 1139.18, 660.56], "Biodiesel (Python)", "#ff7f0e", None),
        (x + 2.5 * largura, [630.37, 1130.63, 662.06], "Biodiesel (GasTurb)", "#ffbb78", "//"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    for positions, values, label, color, hatch in temperaturas:
        ax.bar(positions, values, largura, label=label, color=color, hatch=hatch)

    ax.set_ylabel("Temperatura Absoluta (K)", fontsize=11, fontweight="bold")
    ax.set_title("Validação de Temperaturas dos Estados do Ciclo", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1400)
    ax.legend(ncol=3, loc="upper center", frameon=True, facecolor="white", fontsize=9)

    sns.despine()
    plt.tight_layout(pad=1.5)
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/validacao_temperaturas_gasturb.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("-> Gráfico de validação de temperaturas do GasTurb salvo!")


def run_additional_reference_simulations():
    """Executa as simulações de referência de etanol e biometano do notebook."""
    results = {}
    for fuel, mFuel in (("Ethanol", 19.5 * 2.4 / 100), ("Biomethane", 19.5 * 1.4 / 100)):
        result = brayton.brayton(
            T1=298.0,
            Tfuel=390.0,
            LHV=None,
            mFuel=mFuel,
            mAir=19.5,
            case_name=fuel,
            ncomb=1,
            C_m=7.98,
            gamma=1.06,
            rp=12,
            eta_c=0.85,
            eta_t=0.89,
        )
        results[fuel] = result
        print(f"Resultados da simulação de referência: {fuel}")
        print(result)
    return results


def run_initial_simulation():
    """
    Executa a simulação inicial do ciclo Brayton com parâmetros de referência do biodiesel.
    """
    print("=" * 80)
    print(" SIMULAÇÃO INICIAL DO CICLO BRAYTON (REFERÊNCIA: BIODIESEL) ")
    print("=" * 80)
    
    result = brayton.brayton(
        T1=298.0,
        Tfuel=390.0,
        LHV=None,
        mFuel=19.5 * 1.6 / 100,
        mAir=19.5,
        case_name="Biodiesel",
        ncomb=1,
        C_m=7.98,
        gamma=1.06,
        rp=12,
        eta_c=0.85,
        eta_t=0.89,
    )
    
    print(result)
    print("=" * 80)
    return result


def validate_compressibility_factor():
    """
    Valida a hipótese de gás ideal analisando o fator de compressibilidade (Z)
    em diferentes estados do ciclo Brayton para vários biocombustíveis.
    """
    print("\n" + "=" * 80)
    print(" VALIDAÇÃO DO FATOR DE COMPRESSIBILIDADE (Z) NOS ESTADOS DO CICLO BRAYTON ")
    print("=" * 80)
    
    # 1. Define a variação da razão de pressão
    rp_values = np.linspace(7.5, 12.5, 11)
    
    # 2. Dicionário de parâmetros dos combustíveis
    combustiveis = {
        "Biomethane": {
            "label": "Biometano",
            "f": 1.465201,
            "gamma": 1.04,
            "color": "#35d073",
            "Z1": [], "Z2": [], "Z3": [], "Z4": []
        },
        "Ethanol": {
            "label": "Etanol",
            "f": 1.941392,
            "gamma": 1.06,
            "color": "#48a4e0",
            "Z1": [], "Z2": [], "Z3": [], "Z4": []
        },
        "Biodiesel": {
            "label": "Biodiesel",
            "f": 1.520147,
            "gamma": 1.06,
            "color": "#eb8c34",
            "Z1": [], "Z2": [], "Z3": [], "Z4": []
        }
    }
    
    # 3. Executa as simulações e coleta os valores de Z
    for fuel, data in combustiveis.items():
        for rp in rp_values:
            mAir = maps.mass_flow_from_pressure_ratio(rp)
            res = brayton.brayton(
                T1=298.0,
                Tfuel=390.0,
                LHV=None,
                mFuel=mAir * data["f"] / 100,
                mAir=mAir,
                case_name=fuel,
                ncomb=1,
                C_m=7.98,
                gamma=data["gamma"],
                rp=rp
            )
            
            data["Z1"].append(res.get("Z1"))
            data["Z2"].append(res.get("Z2"))
            data["Z3"].append(res.get("Z3"))
            data["Z4"].append(res.get("Z4"))
    
    # 4. Cria uma figura com subgráficos para os quatro estados do ciclo
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    fig.suptitle("Fator de Compressibilidade ($Z$) nos Estados do Ciclo Brayton", 
                 fontsize=13, fontweight='bold', y=0.98)
    
    estados = [
        ("Z1", "Estado 1: Entrada do Compressor ($P_1, T_1$)", axs[0, 0]),
        ("Z2", "Estado 2: Saída do Compressor ($P_2, T_2$)", axs[0, 1]),
        ("Z3", "Estado 3: Saída do Combustor ($P_3, T_3$)", axs[1, 0]),
        ("Z4", "Estado 4: Saída da Turbina ($P_4, T_4$)", axs[1, 1]),
    ]
    
    for key, title, ax in estados:
        for fuel, data in combustiveis.items():
            ax.plot(rp_values, data[key], label=data["label"], color=data["color"], 
                   linewidth=2, marker='o', markersize=4)
        
        # Linha de idealidade e faixa de aceitação (±2%)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.axhspan(0.98, 1.02, color='green', alpha=0.06)
        
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, linestyle=':', alpha=0.4, color='#d0d0d0')
    
    # Configuração dos eixos
    for ax in axs[1, :]:
        ax.set_xlabel(r"Razão de Pressão ($r_p = P_2/P_1$)", fontsize=10, fontweight='bold')
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Fator de Compressibilidade ($Z$)", fontsize=10, fontweight='bold')
    
    plt.ylim(0.95, 1.05)
    plt.xlim(7, 14)
    
    # Legenda única na parte inferior
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Idealidade ($Z=1$)'),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.15, label=f'Tolerância ($\\pm 2\\%$)')
    ])
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, 
              bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.10, top=0.92, wspace=0.15, hspace=0.20)
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/compressibility_factor_vs_rp.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("-> Gráfico do fator de compressibilidade salvo!")


def plot_ts_diagram():
    """
    Plota o diagrama T-s para diferentes biocombustíveis em condições
    específicas de operação.
    """
    print("\n" + "=" * 80)
    print(" DIAGRAMA T-s PARA COMPARAÇÃO DO CICLO BRAYTON ")
    print("=" * 80)
    
    # Simula diferentes cenários de combustível
    result_biomethane = brayton.brayton(
        T1=298.0, Tfuel=390.0, LHV=None,
        mFuel=1.465201/10, mAir=10,
        case_name="Biomethane", ncomb=1, C_m=7.98, gamma=1.04, rp=8.667,
    )
    
    result_ethanol = brayton.brayton(
        T1=298.0, Tfuel=390.0, LHV=None,
        mFuel=2.11/10, mAir=10,
        case_name="Ethanol", ncomb=1, C_m=7.98, gamma=1.06, rp=8.0,
    )
    
    result_biodiesel = brayton.brayton(
        T1=298.0, Tfuel=390.0, LHV=None,
        mFuel=1.70/10, mAir=10,
        case_name="Biodiesel", ncomb=1, C_m=7.98, gamma=1.06, rp=8.33,
    )
    
    # Organização dos dados
    resultados = {
        "Biometano": {
            "s": [result_biomethane["s1"], result_biomethane["s2"], 
                  result_biomethane["s3"], result_biomethane["s4"], result_biomethane["s1"]], 
            "T": [result_biomethane["T1"], result_biomethane["T2"], 
                  result_biomethane["T3"], result_biomethane["T4"], result_biomethane["T1"]], 
            "cor": "green"
        },
        "Etanol": {
            "s": [result_ethanol["s1"], result_ethanol["s2"], 
                  result_ethanol["s3"], result_ethanol["s4"], result_ethanol["s1"]], 
            "T": [result_ethanol["T1"], result_ethanol["T2"], 
                  result_ethanol["T3"], result_ethanol["T4"], result_ethanol["T1"]], 
            "cor": "blue"
        },
        "Biodiesel": {
            "s": [result_biodiesel["s1"], result_biodiesel["s2"], 
                  result_biodiesel["s3"], result_biodiesel["s4"], result_biodiesel["s1"]], 
            "T": [result_biodiesel["T1"], result_biodiesel["T2"], 
                  result_biodiesel["T3"], result_biodiesel["T4"], result_biodiesel["T1"]], 
            "cor": "orange"
        }
    }
    
    plt.figure(figsize=(10, 7))
    
    for fuel, data in resultados.items():
        plt.plot(data["s"], data["T"], label=fuel, color=data["cor"], linewidth=2, marker='o')
        
        # Adiciona rótulos aos estados do ciclo
        for i, (s, t) in enumerate(zip(data["s"][:4], data["T"][:4])):
            plt.text(s, t, f' {i+1}', fontsize=10, fontweight='bold', color=data["cor"])
    
    plt.title("Ciclo Brayton em Diagrama T-s: Comparação de Biocombustíveis", fontsize=14)
    plt.xlabel("Entropia Específica (kJ/(kg·K))", fontsize=12)
    plt.ylabel("Temperatura (K)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/ts_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("-> Diagrama T-s salvo!")


def plot_efficiency_power_vs_rp():
    """
    Plota a eficiência térmica e a potência líquida em função da razão de pressão
    para diferentes biocombustíveis.
    """
    print("\n" + "=" * 80)
    print(" EFICIÊNCIA TÉRMICA E POTÊNCIA LÍQUIDA vs. RAZÃO DE PRESSÃO ")
    print("=" * 80)
    
    # 1. Define a variação da razão de pressão
    rp_values = np.linspace(7.5, 12.5, 11)
    
    # 2. Dicionário de parâmetros dos combustíveis
    combustiveis = {
        "Biomethane": {
            "label": "Biometano",
            "f": 1.465201,
            "gamma": 1.04,
            "color": "#35d073",
            "eta": [],
            "W_net": []
        },
        "Ethanol": {
            "label": "Etanol",
            "f": 1.941392,
            "gamma": 1.06,
            "color": "#48a4e0",
            "eta": [],
            "W_net": []
        },
        "Biodiesel": {
            "label": "Biodiesel",
            "f": 1.520147,
            "gamma": 1.06,
            "color": "#eb8c34",
            "eta": [],
            "W_net": []
        }
    }
    
    for fuel, data in combustiveis.items():
        for rp in rp_values:
            mAir = maps.mass_flow_from_pressure_ratio(rp)
            res = brayton.brayton(
                T1=298.0, Tfuel=390.0, LHV=None,
                mFuel=mAir * data["f"] / 100,
                mAir=mAir,
                case_name=fuel, ncomb=1, C_m=7.98, gamma=data["gamma"], rp=rp
            )
            
            eta_val = res.get("EtaTh")
            if eta_val <= 1.0:
                eta_val *= 100
            wnet_val = res.get("Wl")
            
            data["eta"].append(eta_val)
            data["W_net"].append(wnet_val)
    
    # 4. Cria uma figura com dois subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    
    fig.suptitle("Desempenho Termodinâmico vs. Razão de Pressão", 
                 fontsize=13, fontweight='bold', y=0.98)
    
    # Plota as curvas
    for fuel, data in combustiveis.items():
        ax1.plot(rp_values, data["eta"], label=data["label"], color=data["color"], 
                linewidth=2.5, marker='o', markersize=4.5)
        ax2.plot(rp_values, data["W_net"], label=data["label"], color=data["color"], 
                linewidth=2.5, marker='s', markersize=4.5)
    
    # Configuração do subgráfico 1
    ax1.set_title("Eficiência Térmica vs. Razão de Pressão", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel(f"Razão de Pressão ($r_p = P_2/P_1$)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Eficiência Térmica (%)", fontsize=10, fontweight='bold')
    ax1.set_ylim(20, 45)
    ax1.set_xlim(6.5, 14)
    ax1.grid(True, linestyle=':', alpha=0.4, color='#d0d0d0')
    
    # Configuração do subgráfico 2
    ax2.set_title("Potência Líquida vs. Razão de Pressão", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel(f"Razão de Pressão ($r_p = P_2/P_1$)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Potência Líquida (kW)", fontsize=10, fontweight='bold')
    ax2.set_xlim(6.5, 14)
    ax2.grid(True, linestyle=':', alpha=0.4, color='#d0d0d0')
    
    # Legenda compartilhada na parte inferior
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, 
              bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14, top=0.88, wspace=0.22)
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/efficiency_power_vs_rp.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("-> Gráfico de eficiência e potência salvo!")


def analyze_exergy_destruction():
    """
    Analisa a distribuição da destruição de exergia nos componentes do ciclo.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE DA DISTRIBUIÇÃO DA DESTRUIÇÃO DE EXERGIA ")
    print("=" * 80)
    
    fuels = ["Biodiesel", "Ethanol", "Biomethane"]
    labels_map = {"Biomethane": "Biometano", "Ethanol": "Etanol", "Biodiesel": "Biodiesel"}
    
    PARAMS_BASE = {
        'T1': 298.0,
        'Tfuel': 298.0,
        'LHV': None,
        'mFuel': 0.3,
        'ncomb': 1,
        'C_m': 7.98,
        'gamma': 1.06,
    }
    
    exergy = {}
    
    for fuel in fuels:
        mAir = 19.5
        params = PARAMS_BASE.copy()
        params['mAir'] = mAir
        params['case_name'] = fuel
        params['rp'] = 8.7
        
        results = brayton.brayton(**params)
        total = (results["Exdcomp"] + results["Exdcomb"] + results["Exdturb"])
        exergy[fuel] = {
            "comp": ((results["Exdcomp"] / total) * 100),
            "comb": ((results["Exdcomb"] / total) * 100),
            "turb": ((results["Exdturb"] / total) * 100)
        }
    
    # Plota os resultados
    fig, ax = plt.subplots(figsize=(11, 6))
    
    combustiveis_list = list(exergy.keys())
    labels = [labels_map.get(c, c) for c in combustiveis_list]
    
    comp_values = [exergy[c]['comp'] for c in combustiveis_list]
    comb_values = [exergy[c]['comb'] for c in combustiveis_list]
    turb_values = [exergy[c]['turb'] for c in combustiveis_list]
    
    x = np.arange(len(combustiveis_list))
    width = 0.6
    
    ax.bar(x, comp_values, width, label='Compressor', color='#4a4a4a', alpha=0.8)
    ax.bar(x, comb_values, width, bottom=comp_values, label='Câmara de Combustão', color='#ff6b35', alpha=0.85)
    ax.bar(x, turb_values, width, bottom=np.array(comp_values)+np.array(comb_values), 
           label='Turbina', color='#004e89', alpha=0.8)
    
    ax.set_ylabel('Destruição de Exergia (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Combustível', fontsize=12, fontweight='bold')
    ax.set_title('Localização de Irreversibilidades por Componente\n(% da destruição total)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.legend(loc='right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/exergy_dest_component_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("-> Gráfico da análise de destruição de exergia salvo!")


def plot_exergetic_efficiency_vs_rp():
    """
    Plota a eficiência de segunda lei (exergética) em função da razão de pressão.
    """
    print("\n" + "=" * 80)
    print(" EFICIÊNCIA EXERGÉTICA vs. RAZÃO DE PRESSÃO ")
    print("=" * 80)
    
    # Faixa da razão de pressão para a análise paramétrica
    rp_valores = np.linspace(8.0, 12.0, 50)
    
    # Listas para armazenar os resultados da eficiência exergética
    eta_ex_Biomethane = []
    eta_ex_etanol = []
    eta_ex_biodiesel = []
    
    # Laço da análise paramétrica de sensibilidade
    for rp in rp_valores:
        # 1. Simulação com biometano
        res_bg = brayton.brayton(T1=298.0, Tfuel=298.0, LHV=None, mFuel=0.55, mAir=19.5,
                        case_name="Biomethane", ncomb=1, C_m=0.29, gamma=1.05, rp=rp)
        eta_ex_Biomethane.append(res_bg["Etaex"] * 100)
        
        # 2. Simulação com etanol
        res_et = brayton.brayton(T1=298.0, Tfuel=298.0, LHV=None, mFuel=0.37, mAir=19.5,
                        case_name="Ethanol", ncomb=1, C_m=4.69, gamma=1.06, rp=rp)
        eta_ex_etanol.append(res_et["Etaex"] * 100)
        
        # 3. Simulação com biodiesel
        res_bd = brayton.brayton(T1=298.0, Tfuel=298.0, LHV=None, mFuel=0.27, mAir=19.5,
                        case_name="Biodiesel", ncomb=1, C_m=7.94, gamma=1.06, rp=rp)
        eta_ex_biodiesel.append(res_bd["Etaex"] * 100)
    
    # Geração do gráfico
    plt.figure(figsize=(9, 6))
    
    plt.plot(rp_valores, eta_ex_Biomethane, color='green', linewidth=2, label='Biometano')
    plt.plot(rp_valores, eta_ex_etanol, color='blue', linewidth=2, label='Etanol')
    plt.plot(rp_valores, eta_ex_biodiesel, color='orange', linewidth=2, label='Biodiesel')
    
    plt.title("Eficiência de Segunda Lei ($\\eta_{II}$) vs. Razão de Pressão ($r_p$)", 
             fontsize=13, fontweight='bold')
    plt.xlabel("Razão de Pressão ($r_p = P_2 / P_1$)", fontsize=11)
    plt.ylabel("Eficiência Exergética Global $\\eta_{II}$ (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.ylim(0, 60)
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/second_law_vs_rp.png', dpi=150)
    plt.show()
    plt.close()
    print("-> Gráfico da eficiência exergética salvo!")


def analyze_destruction_costs():
    """
    Analisa o custo da destruição de exergia por componente.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE DO CUSTO DA DESTRUIÇÃO DE EXERGIA ")
    print("=" * 80)
    
    # Parâmetros de custo dos combustíveis
    C_m_fuel_base = {
        "Biodiesel": 7.94,
        "Ethanol": 4.69,
        "Biomethane": 4.05
    }
    
    fuel_density = {
        "Biodiesel": 0.880,
        "Ethanol": 0.807,
        "Biomethane": 0.764
    }
    
    C_m_fuel = {
        "Biodiesel": C_m_fuel_base["Biodiesel"] / fuel_density["Biodiesel"],
        "Ethanol": C_m_fuel_base["Ethanol"] / fuel_density["Ethanol"],
        "Biomethane": C_m_fuel_base["Biomethane"] / fuel_density["Biomethane"]
    }
    
    fuel_gamma = {
        "Biodiesel": 1.06,
        "Ethanol": 1.06,
        "Biomethane": 1.05
    }
    
    # Executa as simulações para cada combustível
    result_Biomethane = brayton.brayton(
        T1=298.0, Tfuel=390.0, LHV=None,
        mFuel=0.3, mAir=19.5,
        case_name="Biomethane", ncomb=1,
        C_m=C_m_fuel["Biomethane"], gamma=fuel_gamma["Biomethane"], rp=8.7,
    )
    
    result_ethanol = brayton.brayton(
        T1=298.0, Tfuel=390.0, LHV=None,
        mFuel=0.3, mAir=19.5,
        case_name="Ethanol", ncomb=1,
        C_m=C_m_fuel["Ethanol"], gamma=fuel_gamma["Ethanol"], rp=8.7,
    )
    
    result_biodiesel = brayton.brayton(
        T1=298.0, Tfuel=390.0, LHV=None,
        mFuel=0.3, mAir=19.5,
        case_name="Biodiesel", ncomb=1,
        C_m=C_m_fuel["Biodiesel"], gamma=fuel_gamma["Biodiesel"], rp=8.7,
    )
    
    # Organiza os custos de destruição
    destruction_cost = {
        "Biomethane": {
            "Compressor": result_Biomethane["destruction_costs"]["compressor"] * 3600,
            "Câmara de Combustão": result_Biomethane["destruction_costs"]["combustor"] * 3600,
            "Turbina": result_Biomethane["destruction_costs"]["turbine"] * 3600
        },
        "Ethanol": {
            "Compressor": result_ethanol["destruction_costs"]["compressor"] * 3600,
            "Câmara de Combustão": result_ethanol["destruction_costs"]["combustor"] * 3600,
            "Turbina": result_ethanol["destruction_costs"]["turbine"] * 3600
        },
        "Biodiesel": {
            "Compressor": result_biodiesel["destruction_costs"]["compressor"] * 3600,
            "Câmara de Combustão": result_biodiesel["destruction_costs"]["combustor"] * 3600,
            "Turbina": result_biodiesel["destruction_costs"]["turbine"] * 3600
        },
    }
    
    # Plota os resultados
    componentes = ["Compressor", "Câmara de Combustão", "Turbina"]
    x = np.arange(len(componentes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    rects1 = ax.bar(x - width, [destruction_cost["Biomethane"][c] for c in componentes], 
                   width, label='Biometano', color='#2ca02c')
    rects2 = ax.bar(x, [destruction_cost["Ethanol"][c] for c in componentes], 
                   width, label='Etanol', color='#1f77b4')
    rects3 = ax.bar(x + width, [destruction_cost["Biodiesel"][c] for c in componentes], 
                   width, label='Biodiesel', color='#ff7f0e')
    
    ax.set_ylabel(rf"Taxa de Custo de Destruição de Exergia ($\dot{{C}}_{{D,k}}$ [R\$/h])", 
                 fontsize=12, fontweight="bold")
    ax.set_title("Distribuição da Taxa de Custo de Destruição de Exergia por Componente", 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(componentes, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Adiciona rótulos com os valores sobre as barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'R$ {height:.2f}/h',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    max_val = max([destruction_cost[f][c] for c in componentes for f in destruction_cost])
    ax.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/destruction_cost_components.png', dpi=150)
    plt.show()
    plt.close()
    print("-> Gráfico da análise de custo de destruição salvo!")


def plot_unit_work_cost(C_m_fuel, fuel_gamma):
    """
    Plota o custo unitário do trabalho produzido em função da razão de pressão.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE DE SENSIBILIDADE DO CUSTO UNITÁRIO DO TRABALHO ")
    print("=" * 80)
    
    rp_array = np.linspace(8.0, 12.0, 50)
    
    fuels = ["Biomethane", "Ethanol", "Biodiesel"]
    labels_map = {"Biomethane": "Biometano", "Ethanol": "Etanol", "Biodiesel": "Biodiesel"}
    colors_map = {"Biomethane": "#2ca02c", "Ethanol": "#1f77b4", "Biodiesel": "#ff7f0e"}
    
    # Armazena os resultados de c_W
    cw_results = {f: [] for f in fuels}
    
    # Varredura paramétrica usando o solucionador Brayton
    for fuel in fuels:
        for rp_val in rp_array:
            res = brayton.brayton(
                T1=298.0, Tfuel=298.0, LHV=None,
                mFuel=0.3, mAir=19.5,
                case_name=fuel, ncomb=1,
                C_m=C_m_fuel[fuel],
                gamma=fuel_gamma[fuel],
                rp=rp_val,
            )
            
            cw_val = res["cW"] * 3600000.0  # R$/MWh
            cw_results[fuel].append(cw_val)
    
    # Converte os resultados em matrizes NumPy
    for fuel in fuels:
        cw_results[fuel] = np.array(cw_results[fuel])
    
    # Gera a figura para apresentação acadêmica
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    for fuel in fuels:
        ax.plot(
            rp_array,
            cw_results[fuel],
            color=colors_map[fuel],
            linewidth=2.5,
            label=labels_map[fuel]
        )
    
    ax.set_title(r'Sensibilidade do Custo Unitário do Trabalho ($c_W$) vs. Razão de Pressão ($r_p$)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(r'Razão de Pressão do Compressor ($r_p$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Custo Unitário do Trabalho ($c_W$ [R\$/MWh])', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/unit_work_cost.png', dpi=150)
    plt.show()
    plt.close()
    print("-> Gráfico do custo unitário do trabalho salvo!")


def main():
    """Fluxo principal de execução."""
    print("\n")
    print("  ANÁLISE TERMOECONÔMICA DE CICLOS BRAYTON COM BIOCOMBUSTÍVEIS".center(78))
    print("\n")
    
    # Executa a sequência de análises
    plot_gasturb_performance_validation()
    plot_gasturb_temperature_validation()
    run_initial_simulation()
    run_additional_reference_simulations()
    validate_compressibility_factor()
    plot_ts_diagram()
    plot_efficiency_power_vs_rp()
    analyze_exergy_destruction()
    plot_exergetic_efficiency_vs_rp()
    analyze_destruction_costs()
    
    # Parâmetros de custo dos combustíveis para análise adicional
    C_m_fuel_base = {
        "Biodiesel": 7.94,
        "Ethanol": 4.69,
        "Biomethane": 4.05
    }
    
    fuel_density = {
        "Biodiesel": 0.880,
        "Ethanol": 0.807,
        "Biomethane": 0.764
    }
    
    C_m_fuel = {
        "Biodiesel": C_m_fuel_base["Biodiesel"] / fuel_density["Biodiesel"],
        "Ethanol": C_m_fuel_base["Ethanol"] / fuel_density["Ethanol"],
        "Biomethane": C_m_fuel_base["Biomethane"] / fuel_density["Biomethane"]
    }
    
    fuel_gamma = {
        "Biodiesel": 1.06,
        "Ethanol": 1.06,
        "Biomethane": 1.05
    }
    
    plot_unit_work_cost(C_m_fuel, fuel_gamma)
    optimization.main()
    
    print("\n" + "=" * 80)
    print(" TODOS OS GRÁFICOS DE ANÁLISE FORAM GERADOS E SALVOS NO DIRETÓRIO './plots/' ")
    print("=" * 80)
    print("A tabela de parâmetros ótimos também foi salva no diretório './plots/'.\n")


if __name__ == "__main__":
    main()
