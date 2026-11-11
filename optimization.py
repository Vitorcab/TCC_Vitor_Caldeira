"""
optimization.py
===============
Modelo de Combustão de Turbina a Gás — Parte 7: Otimização Termoeconômica

Realiza análises paramétricas de sensibilidade, identifica a fronteira de
Pareto e avalia correlações e importância de atributos com aprendizado de máquina.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import brayton
import maps


# Parâmetros de custo dos combustíveis (R$/kg)
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

# Configuração da apresentação dos gráficos
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def run_parametric_sensitivity():
    """
    Executa uma análise paramétrica de sensibilidade da razão de pressão e da
    fração de combustível para todos os biocombustíveis. Gera gráficos da
    eficiência exergética e dos custos de destruição.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE PARAMÉTRICA DE SENSIBILIDADE (TIT versus Etaex e CD_comb) ")
    print("=" * 80)
    
    fuels = ["Biomethane", "Ethanol", "Biodiesel"]
    labels_map = {"Biomethane": "Biometano", "Ethanol": "Etanol", "Biodiesel": "Biodiesel"}
    
    os.makedirs('./plots', exist_ok=True)
    
    # Define as faixas dos parâmetros
    rp_range = np.linspace(8.0, 12.0, 25)
    
    # Faixa da razão combustível/ar (f = mFuel / mAir)
    f_Fuel_dict = {
        "Biomethane": np.linspace(0.20 / 19.5, 0.35 / 19.5, 20),
        "Ethanol": np.linspace(0.20 / 19.5, 0.70 / 19.5, 20),
        "Biodiesel": np.linspace(0.20 / 19.5, 0.45 / 19.5, 20)
    }
    
    simulation_data = []
    
    print("Executando simulações para a varredura paramétrica...")
    for fuel in fuels:
        print(f"  Processando {labels_map[fuel]}...", end=" ")
        count = 0
        for rp in rp_range:
            mAir = maps.mass_flow_from_pressure_ratio(rp)
            
            for f in f_Fuel_dict[fuel]:
                mf = f * mAir
                
                res = brayton.brayton(
                    T1=298.0,
                    Tfuel=390.0,
                    LHV=None,
                    mFuel=mf,
                    mAir=mAir,
                    case_name=fuel,
                    ncomb=1,
                    C_m=C_m_fuel[fuel],
                    gamma=fuel_gamma[fuel],
                    rp=rp
                )
                
                # Filtra as soluções viáveis
                eta_ex = res.get("Etaex", 0.0)
                cW_raw = res.get("cW", 0.0)
                rp_ret = res.get("_rp", rp)
                
                if eta_ex > 0.05 and cW_raw > 0:
                    cw_mwh = cW_raw * 3_600_000.0  # Converte R$/kJ para R$/MWh
                    cd_comb = res.get("destruction_costs", {}).get("combustor", 0.0) * 3600.0
                    
                    simulation_data.append({
                        "fuel": fuel,
                        "rp": rp_ret,
                        "f_ratio": f,
                        "mFuel": mf,
                        "mAir": mAir,
                        "TIT": res["T3"],
                        "Wnet": res["Wl"],
                        "Etaex": eta_ex * 100.0,
                        "cW": cw_mwh,
                        "CD_comb": cd_comb,
                        "Exdest_total": res["Exdcomp"] + res["Exdcomb"] + res["Exdturb"]
                    })
                    count += 1
        
        print(f"{count} pontos válidos")
    
    df = pd.DataFrame(simulation_data)
    print(f"Total de simulações: {len(df)}")
    
    # Gera gráficos de sensibilidade para cada combustível
    rp_samples = [8.0, 9.5, 11.0, 12.0]
    
    for fuel in fuels:
        print(f"Gerando gráficos de sensibilidade para {labels_map[fuel]}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        df_fuel = df[df["fuel"] == fuel]
        
        for rp_val in rp_samples:
            # Filtra pela razão de pressão e ordena pela TIT
            sub = df_fuel[np.isclose(df_fuel["rp"], rp_val, atol=0.08)].sort_values("TIT")
            
            if len(sub) >= 2:
                ax1.plot(sub["TIT"], sub["Etaex"], marker='o', markersize=4, label=f'$r_p = {rp_val:.1f}$')
                ax2.plot(sub["TIT"], sub["CD_comb"], marker='s', markersize=4, label=f'$r_p = {rp_val:.1f}$')
        
        # Configuração do primeiro gráfico
        ax1.set_title(f'(a) Eficiência Exergética ($\\eta_{{II}}$) vs. TIT - {labels_map[fuel]}', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Temperatura de Entrada na Turbina - TIT [K]', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'Eficiência Exergética Global ($\\eta_{{II}}$) [%]', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(title='Razão de Pressão', fontsize=9)
        
        # Configuração do segundo gráfico
        ax2.set_title(f'(b) Custo de Destruição ($\\dot{{C}}_{{D,comb}}$) vs. TIT - {labels_map[fuel]}', 
                     fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Temperatura de Entrada na Turbina - TIT [K]', fontsize=11, fontweight='bold')
        ax2.set_ylabel(f'Custo de Destruição na Câmara ($\\dot{{C}}_{{D,comb}}$) [R\\$/h]', 
                      fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(title='Razão de Pressão', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'./plots/sensibilidade_TIT_rp_{fuel}.png', dpi=200)
        plt.show()
        plt.close()
    
    print("-> Gráficos da análise de sensibilidade gerados com sucesso!")
    
    return df


def find_pareto_front(costs):
    """
    Identifica os pontos não dominados (fronteira de Pareto).
    
    Parâmetros:
    -----------
    costs : ndarray
        Matriz Nx2 em que a coluna 0 representa uma grandeza a MAXIMIZAR
        (-Etaex) e a coluna 1 representa uma grandeza a MINIMIZAR (CD_total).
    
    Retorna:
    --------
    is_efficient : ndarray
        Índices dos pontos eficientes de Pareto.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0
    
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    
    return is_efficient


def generate_pareto_frontier(df):
    """
    Identifica e visualiza a fronteira de Pareto para todos os combustíveis.
    Encontra os parâmetros ótimos de projeto (rp* e f*) no ponto de melhor
    compromisso da fronteira.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE DA FRONTEIRA DE PARETO E IDENTIFICAÇÃO DOS PARÂMETROS ÓTIMOS ")
    print("=" * 80)
    
    fuels = ["Biomethane", "Ethanol", "Biodiesel"]
    labels_map = {"Biomethane": "Biometano", "Ethanol": "Etanol", "Biodiesel": "Biodiesel"}
    colors_map = {"Biomethane": "#2ca02c", "Ethanol": "#1f77b4", "Biodiesel": "#ff7f0e"}
    
    # Calcula o custo total de destruição
    df["cfuel_kJ"] = (df["cW"] / 3_600_000.0) * (df["Etaex"] / 100.0)
    df["CD_total"] = df["Exdest_total"] * df["cfuel_kJ"] * 3600.0  # R$/h
    
    # Gera o gráfico da fronteira de Pareto
    fig, ax = plt.subplots(figsize=(9, 6))
    
    optimal_results = []
    
    for fuel in fuels:
        sub = df[df["fuel"] == fuel].copy()
        
        # Converte o problema em minimização: (-Etaex, CD_total)
        points_to_min = np.column_stack((-sub["Etaex"].values, sub["CD_total"].values))
        pareto_indices = find_pareto_front(points_to_min)
        
        pareto_df = sub.iloc[pareto_indices].sort_values("Etaex")
        
        # Plota a nuvem de pontos
        ax.scatter(sub["Etaex"], sub["CD_total"], color=colors_map[fuel], alpha=0.15, s=20)
        
        # Plota a linha da fronteira de Pareto
        ax.plot(pareto_df["Etaex"], pareto_df["CD_total"], color=colors_map[fuel], 
               linewidth=2.5, label=f'Fronteira Pareto: {labels_map[fuel]}')
        
        # Seleciona o ponto de utopia (minimiza a distância até a utopia)
        if len(pareto_df) > 1:
            eta_range = pareto_df["Etaex"].max() - pareto_df["Etaex"].min()
            cd_range = pareto_df["CD_total"].max() - pareto_df["CD_total"].min()
            
            eta_norm = (pareto_df["Etaex"] - pareto_df["Etaex"].min()) / (eta_range if eta_range > 0 else 1.0)
            cd_norm = (pareto_df["CD_total"] - pareto_df["CD_total"].min()) / (cd_range if cd_range > 0 else 1.0)
            
            # Distância à utopia: o alvo é (Eta_norm=1, CD_norm=0)
            dist_utopia = np.sqrt((1.0 - eta_norm)**2 + (cd_norm)**2)
            knee_idx = dist_utopia.idxmin()
            knee_row = pareto_df.loc[knee_idx]
            
            # Destaca o ponto ótimo com uma estrela vermelha
            ax.scatter(knee_row["Etaex"], knee_row["CD_total"], color='red', marker='.', s=250, zorder=6)
            
            optimal_results.append({
                "Biocombustível": labels_map[fuel],
                "rp*": knee_row["rp"],
                "f* (%)": knee_row["f_ratio"] * 100.0,
                "TIT* [K]": knee_row["TIT"],
                "Wnet* [kW]": knee_row["Wnet"],
                "Etaex* [%]": knee_row["Etaex"],
                "CD_total* [R$/h]": knee_row["CD_total"]
            })
    
    # Adiciona à legenda o marcador do ponto ótimo
    ax.scatter([], [], color='red', marker='.', s=180, label='Ponto Ótimo ($r_p^*, f^*$)')
    
    ax.set_title(f'Fronteira de Pareto: Taxa de Custo de Destruição ($\\dot{{C}}_{{D,total}}$) vs. Eficiência Exergética ($\\eta_{{II}}$)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(f'Eficiência Exergética Global ($\\eta_{{II}}$) [%])', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Taxa Total de Custo de Destruição ($\\dot{{C}}_{{D,total}}$) [R\\$/h])', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('./plots/fronteira_pareto_termoeconomica.png', dpi=300)
    plt.show()
    plt.close()
    
    print("-> Gráfico da fronteira de Pareto gerado com sucesso!")
    
    return pd.DataFrame(optimal_results), df


def generate_correlation_heatmaps(df):
    """
    Gera matrizes de correlação de Pearson para cada combustível.
    Mostra as relações entre os parâmetros de projeto e as métricas de desempenho.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE DE CORRELAÇÃO PARAMÉTRICA (MATRIZES DE PEARSON) ")
    print("=" * 80)
    
    fuels = ["Biomethane", "Ethanol", "Biodiesel"]
    labels_map = {"Biomethane": "Biometano", "Ethanol": "Etanol", "Biodiesel": "Biodiesel"}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    corr_cols = ["rp", "f_ratio", "TIT", "Wnet", "Etaex", "CD_total", "Exdest_total"]
    corr_labels = [f'$r_p$', f'$f$', 'TIT', f'$\\dot{{W}}_{{liq}}$', f'$\\eta_{{II}}$', 
                   f'$\\dot{{C}}_{{D,total}}$', f'$\\dot{{E}}x_{{dest}}$']
    
    for idx, fuel in enumerate(fuels):
        ax = axes[idx]
        df_fuel = df[df["fuel"] == fuel]
        corr_matrix = df_fuel[corr_cols].corr()
        
        show_cbar = (idx == 2)
        cbar_kw = {'label': 'Coeficiente de Correlação de Pearson'} if show_cbar else {}
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmax=1.0,
            vmin=-1.0,
            linewidths=0.8,
            ax=ax,
            xticklabels=corr_labels,
            yticklabels=corr_labels if idx == 0 else [],
            cbar=show_cbar,
            cbar_kws=cbar_kw
        )
        ax.set_title(f'({chr(97+idx)}) {labels_map[fuel]}', fontsize=12, fontweight='bold', pad=12)
    
    plt.suptitle('Matrizes de Correlação Paramétrica de Pearson por Biocombustível', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./plots/heatmap_correlacao_pearson.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("-> Painel de mapas de calor das correlações gerado com sucesso!")


def generate_feature_importance(df):
    """
    Analisa a importância das variáveis usando regressão por floresta aleatória.
    Quantifica o impacto relativo de rp e f na eficiência exergética e nos
    custos de destruição.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISE DA IMPORTÂNCIA DAS VARIÁVEIS (FLORESTA ALEATÓRIA) ")
    print("=" * 80)
    
    fuels = ["Biomethane", "Ethanol", "Biodiesel"]
    labels_map = {"Biomethane": "Biometano", "Ethanol": "Etanol", "Biodiesel": "Biodiesel"}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, fuel in enumerate(fuels):
        ax = axes[idx]
        df_fuel = df[df["fuel"] == fuel]
        
        X = df_fuel[["rp", "f_ratio"]]
        y_eta = df_fuel["Etaex"]
        y_cd = df_fuel["CD_total"]
        
        # Treina os modelos de floresta aleatória
        rf_eta = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_eta)
        rf_cd = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_cd)
        
        # Prepara a tabela de importância das variáveis
        imp_df = pd.DataFrame({
            'Atributo': [f'Razão de Pressão ($r_p$)', 'Fração Combustível-Ar (f)'],
            f'Impacto em ($\\eta_{{II}}$)': rf_eta.feature_importances_ * 100,
            f'Impacto em ($\\dot{{C}}_{{D,total}}$)': rf_cd.feature_importances_ * 100
        })
        
        x_pos = np.arange(len(imp_df['Atributo']))
        width = 0.35
        
        rects1 = ax.bar(x_pos - width/2, imp_df[f'Impacto em ($\\eta_{{II}}$)'], width, 
                       label=f'Eficiência Exergética ($\\eta_{{II}}$)', color='#1f77b4')
        rects2 = ax.bar(x_pos + width/2, imp_df[f'Impacto em ($\\dot{{C}}_{{D,total}}$)'], width, 
                       label=f'Custo de Destruição ($\\dot{{C}}_{{D,total}}$', color='#ff7f0e')
        
        if idx == 0:
            ax.set_ylabel('Importância Relativa [%]', fontsize=11, fontweight='bold')
        
        ax.set_title(f'({chr(97+idx)}) {labels_map[fuel]}', fontsize=12, fontweight='bold', pad=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(imp_df['Atributo'], fontsize=10, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        if idx == 2:
            ax.legend(fontsize=9, loc='upper right')
        
        # Adiciona os valores sobre as barras
        for rect in list(rects1) + list(rects2):
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Importância Relativa das Variáveis de Decisão por Biocombustível', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./plots/feature_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("-> Painel de importância das variáveis gerado com sucesso!")


def display_optimal_parameters(df_optimal):
    """
    Exibe e exporta os parâmetros ótimos de projeto nos pontos de melhor
    compromisso da fronteira de Pareto.
    """
    print("\n" + "=" * 80)
    print(" PARÂMETROS ÓTIMOS DE PROJETO (rp* e f*) NA FRONTEIRA DE PARETO ")
    print("=" * 80)
    
    df_optimal.to_csv('./plots/parametros_otimos_projetos.csv', index=False, encoding='utf-8-sig')
    
    print(df_optimal.to_string(index=False))
    print("=" * 80)
    print("-> Resultados exportados para './plots/parametros_otimos_projetos.csv'")
    
    return df_optimal


def main():
    """Executa o fluxo principal da análise de otimização."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + " OTIMIZAÇÃO TERMOECONÔMICA DE CICLOS BRAYTON - ANÁLISE AVANÇADA".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Cria o diretório de saída
    os.makedirs('./plots', exist_ok=True)
    
    # Executa a sequência de análises
    print("Etapa 1: executando a análise paramétrica de sensibilidade...")
    df = run_parametric_sensitivity()
    
    print("\nEtapa 2: identificando a fronteira de Pareto e os parâmetros ótimos...")
    df_optimal, df = generate_pareto_frontier(df)
    
    print("\nEtapa 3: analisando as correlações paramétricas...")
    generate_correlation_heatmaps(df)
    
    print("\nEtapa 4: calculando a importância das variáveis...")
    generate_feature_importance(df)
    
    print("\nEtapa 5: exibindo e exportando os parâmetros ótimos...")
    display_optimal_parameters(df_optimal)
    
    print("\n" + "=" * 80)
    print(" ANÁLISE DE OTIMIZAÇÃO CONCLUÍDA - GRÁFICOS SALVOS EM './plots/' ")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  - sensibilidade_TIT_rp_*.png          : gráficos de sensibilidade por combustível")
    print("  - fronteira_pareto_termoeconomica.png : visualização da fronteira de Pareto")
    print("  - heatmap_correlacao_pearson.png      : matrizes de correlação")
    print("  - feature_importance_bar.png          : análise da importância das variáveis")
    print("  - parametros_otimos_projetos.csv      : tabela de parâmetros ótimos\n")


if __name__ == "__main__":
    main()
