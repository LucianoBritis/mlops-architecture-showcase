# 🔬 Especificação Técnica: Modulador Automático de Regime de Mercado (HMM + GARCH)

## Conceito & Algoritmos Principais

O `HMMRegimeDetector` decodifica e rastreia o regime do mercado em tempo real para adaptar dynamicamente os thresholds de convicção dos motores de negociação.

### 1. Algoritmo Baum-Welch (EM - Expectation-Maximization)
- **Função**: Treina a matriz de transição de estados de Markov ($P_{ij}$) e as distribuições gaussianas de emissão sem supervisão.
- **Processo**: Estima recursivamente os parâmetros do modelo a partir dos retornos logarítmicos e volatilidade histórica.

### 2. Algoritmo de Viterbi
- **Função**: Rastreia a sequência de estados ocultos mais provável em tempo real.
- **Processo**: Decodifica instantaneamente o regime do mercado a cada novo tick/bar.

---

## 🎭 Os 4 Regimes de Mercado

1. **Low Volatility Trending (Tendência Calma)**: Tendência estável, baixo ruído.
2. **High Volatility Trending (Tendência Violenta)**: Forte aceleração institucional com alta variância.
3. **Consolidação / Range (Mercado Lateral de Ruído)**: Excesso de fips, ruído e arbitragem estocástica.
4. **Crise / Flash Crash (Surto de Agressão)**: Surto de volume e volatilidade extrema.

---

## 📐 Emissão Gaussiana
- **Métrica**: Mede médias e covariâncias por regime combinando Retornos Logarítmicos e Volatilidade (GARCH).

---

## ⚙️ Regra de Adaptação Dinâmica de Execução

- **Em Consolidação / Range**:
  - Desativa automaticamente a flag `--high-frequency`.
  - Exige convicção ultra-cirúrgica: $|\Phi| > 0.35$.
- **Em Tendência Forte (Calma ou Violenta)**:
  - Libera a flag `--high-frequency` ($|\Phi| > 0.15$) para surfar a pernada de volume.
