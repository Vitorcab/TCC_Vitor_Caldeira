# Otimização e Análise Termoeconômica de Turbinas a Gás com Biocombustíveis

Simulador em Python 3 desenvolvido no Trabalho de Conclusão de Curso de Engenharia Mecânica da Universidade Federal Fluminense (UFF), referente ao Apêndice A do trabalho:

> **BASTOS, Vítor Caldeira de Andrada.** *Otimização e análise termoeconômica de turbinas a gás com biocombustíveis.* Trabalho de Conclusão de Curso (Graduação em Engenharia Mecânica) – Universidade Federal Fluminense, Niterói, RJ. Orientadora: Isabela Florindo Pinheiro.

O código simula um ciclo Brayton simples operando com **biometano**, **etanol hidratado** e **biodiesel** (oleato de metila), calculando desempenho termodinâmico (1ª Lei), destruição de exergia (2ª Lei), custos termoeconômicos e a solução ótima de compromisso via fronteira de Pareto.

---

## Principais características

- **Gás ideal com calores específicos variáveis**: polinômios de $c_p(T)$ de Yaws, com entalpia e entropia obtidas por integração.
- **Combustão por frações de espécies**: estequiometria generalizada por balanço atômico (C, H, O, S, N), com tratamento de combustíveis oxigenados e diluentes inertes.
- **Mapas de desempenho**: eficiências isentrópicas do compressor e da turbina por polinômios cúbicos bivariados ajustados a mapas digitalizados do GasTurb; linha de surge e linha de máxima eficiência para definir a vazão de ar a partir de $r_p$.
- **Verificação de gás ideal**: fator de compressibilidade $Z$ nos quatro estados do ciclo (correlação de Pitzer com regras de mistura de Kay).
- **Análise exergética**: destruição de exergia no compressor, câmara de combustão e turbina; eficiência de 2ª Lei.
- **Termoeconomia**: custo unitário da exergia do combustível, taxa de custo de destruição ($\dot{C}_{D,k}$) e custo unitário do trabalho líquido ($c_W$).
- **Otimização**: varredura paramétrica em $(r_p, f)$, fronteira de Pareto, ponto utópico, correlações de Pearson e importância de atributos (Random Forest).
- **Validação**: comparação com o GasTurb 15 (erros < 0,8% nas temperaturas e ~2,5% na potência/eficiência térmica).

---

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── Maps/
│   ├── csv_comp/               # mapa do compressor digitalizado
│   │   ├── comp_<eff>_eff.csv  # ilhas de isoeficiência (eff em %)
│   │   └── comp_surge_line.csv # linha de surge
│   └── csv_turb/               # mapa da turbina digitalizado
│       └── turb_<eff>_eff.csv  # ilhas de isoeficiência (eff em fração)
└── src/
    ├── main.py                 # execução das análises e geração dos gráficos
    ├── properties.py           # constantes, composições e massas molares
    ├── thermo.py               # cp, h e s por espécie e por mistura
    ├── combustion.py           # estequiometria e composição dos produtos
    ├── brayton.py              # solucionador do ciclo + exergia + custos
    ├── maps.py                 # ajuste e consulta dos mapas de desempenho
    ├── thermoeconomics.py      # classe de análise termoeconômica
    └── optimization.py         # varredura, Pareto, Pearson, Random Forest


Conteúdo do `requirements.txt`:

```
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Como executar

Os módulos usam importações diretas (`import brayton`, `import maps`, ...) e os gráficos são salvos em `./plots`, então execute **de dentro da pasta `src/`**.

### 1. Ajustar os mapas de desempenho (apenas na primeira vez)


Esse passo lê os `.csv` em `Maps/`, ajusta as correlações por mínimos quadrados e grava os coeficientes em `src/maps_coefficients.txt`, necessário para as demais rotinas. Também imprime RMSE e R² de cada ajuste (Tabela 5 do TCC).

> O módulo `maps.py` localiza a pasta `Maps/` por `Path(__file__).parents[1]`, ou seja, `Maps/` deve ficar na raiz do repositório e `maps.py` dentro de `src/`.

### 2. Rodar todas as análises

```bash
python main.py
```

Executa, em sequência:

1. Gráficos de validação contra o GasTurb 15 (potência, eficiência térmica e temperaturas)
2. Simulações de referência
3. Verificação do fator de compressibilidade $Z$
4. Diagramas $T$-$s$
5. Eficiência térmica e potência líquida vs. $r_p$
6. Distribuição da destruição de exergia
7. Eficiência de 2ª Lei vs. $r_p$
8. Custo de destruição por componente
9. Custo unitário do trabalho ($c_W$)
10. Otimização completa (`optimization.main()`): sensibilidade, Pareto, correlações e importância de atributos

Os resultados são salvos em `./plots/`, incluindo `parametros_otimos_projetos.csv` com os pontos ótimos $(r_p^*, f^*)$.

### Uso direto do solucionador

```python
import brayton

res = brayton.brayton(
    T1=298.0,            # temperatura de admissão [K]
    Tfuel=390.0,         # temperatura do combustível [K]
    LHV=None,            # None: calculado pela composição
    mFuel=19.5 * 1.6 / 100,
    mAir=19.5,           # [kg/s]
    case_name="Biodiesel",   # "Biomethane" | "Ethanol" | "Biodiesel"
    ncomb=1,
    C_m=7.94 / 0.880,    # custo [R$/kg] (opcional, ativa a termoeconomia)
    gamma=1.06,          # fator exergético
    rp=12,
    eta_c=0.85,
    eta_t=0.89,
)

print(res["T3"], res["Wl"], res["EtaTh"], res["Etaex"])
```

---

## Hipóteses e limitações

- Ciclo Brayton **simples** (sem regeneração, interresfriamento ou ciclo de fundo), sem queda de pressão na câmara de combustão e com eficiência de combustão unitária.
- Gás ideal com $c_p(T)$ polinomial; a hipótese é verificada via $Z$ (desvios < 0,5%).
- Exergia química do combustível estimada por $ex_{ch} = \gamma \cdot LHV$, com $\gamma \approx 1{,}04$–$1{,}06$.
- A vazão de ar é determinada pela **linha de máxima eficiência** do compressor, restringindo a varredura a $3{,}36 \le r_p \le 12{,}77$. Fora desse intervalo as funções retornam erro, sem extrapolação.
- A superfície de eficiência é ajustada apenas a pontos de contornos de isoeficiência; regiões sem contornos digitalizados não foram validadas.
- Combustão com deficiência de oxigênio ($\alpha < 1$) usa um modelo simplificado e não é o foco do estudo.
- A análise de custos considera somente parcelas operacionais (preço do combustível); CAPEX/OPEX não estão incluídos.
- Preços dos combustíveis: ANP (2025, 2026) e FIESP (2025).

---

## Trabalhos futuros

- Modelagem de emissões (NOx, CO, hidrocarbonetos não queimados)
- Regeneração, interresfriamento e ciclos combinados (Rankine/ORC, CHP)
- Análise econômica completa (CAPEX/OPEX, TIR, VPL)

---

## Referências principais

- ÇENGEL, Y.; BOLES, M. *Termodinâmica*. 7ª ed. Bookman, 2013.
- MORAN, M. et al. *Fundamentals of Engineering Thermodynamics*. Wiley, 2014.
- KOTAS, T. *The Exergy Method of Thermal Plant Analysis*. Butterworth-Heinemann, 1985.
- KAVIRI, A. G.; JAAFAR, M. N. M.; LAZIM, T. M. *Energy Conversion and Management*, v. 58, 2012.
- SHAMOUSHAKI, M.; ALIEHYAEI, M. *Journal of Thermal Engineering*, v. 6, n. 2, 2020.
- KURZKE, J. *How to create a performance model of a gas turbine from a limited amount of information*, 2005.
- YAWS, C. *Yaws Handbook of Thermodynamic Properties*. McGraw Hill, 1998.

---


## Licença

Defina a licença do projeto (por exemplo, MIT) e adicione o arquivo `LICENSE`.

## Contato

Vítor Caldeira de Andrada Bastos – Engenharia Mecânica, UFF
