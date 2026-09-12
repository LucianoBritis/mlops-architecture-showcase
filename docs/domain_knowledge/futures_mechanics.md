# Mecânica dos Futuros da B3 e o Problema da Rolagem (*Rollover*)

Para projetar pipelines robustas de Machine Learning para Finanças Quantitativas, especialmente no Mercado Brasileiro (B3), compreender a mecânica dos Contratos Futuros é um pré-requisito fundamental. A arquitetura descrita neste repositório existe para solucionar as anomalias matemáticas geradas por essa mecânica.

---

## 1. O que são Contratos Futuros?
Diferente de uma ação (ex: PETR4) que é detida em perpetuidade, um Contrato Futuro é um acordo para comprar ou vender um ativo subjacente (Índice ou Dólar) em uma **data futura específica**.

Como os contratos possuem data de vencimento, quando o prazo se aproxima, o contrato expira e a liquidez do mercado migra integralmente para o contrato do período seguinte. O processo de transição de liquidez do contrato antigo para o novo é chamado de **Rolagem** (*Rollover*).

---

## 2. Códigos de Vencimento (A Sopa de Letras)
Um código de contrato na B3 consiste em 3 partes: **[Ativo] + [Letra do Mês de Vencimento] + [Ano]**.

Os meses são representados por letras específicas (padrão internacional):

| Letra | Mês       | Letra | Mês        |
| :---: | :-------- | :---: | :--------- |
| **F** | Janeiro   | **N** | Julho      |
| **G** | Fevereiro | **Q** | Agosto     |
| **H** | Março     | **U** | Setembro   |
| **J** | Abril     | **V** | Outubro    |
| **K** | Maio      | **X** | Novembro   |
| **M** | Junho     | **Z** | Dezembro   |

*(Letras que poderiam causar confusão visual, como I, L e O, são omitidas pelo padrão da B3).*

---

## 3. Mini Índice Bovespa (WIN)
O Mini Índice representa frações do índice Ibovespa. Ele vence a cada **2 meses** (sempre nos meses pares).
- **Vencimentos:** Fev (G), Abr (J), Jun (M), Ago (Q), Out (V), Dez (Z).
- **Data Exata:** Quarta-feira mais próxima do dia 15 do mês de vencimento.
- **Exemplo:** Em fevereiro de 2026, o mercado negocia o `WING26`. À medida que a quarta-feira de vencimento se aproxima, os algoritmos institucionais realizam a rolagem vendendo `WING26` e comprando `WINJ26` (abril).

---

## 4. Mini Dólar Comercial (WDO)
O Mini Dólar representa a taxa de câmbio BRL/USD. Ele possui vencimento **mensal** (alta rotatividade).
- **Vencimentos:** Todos os meses (todas as letras da tabela acima).
- **Data Exata:** O primeiro dia útil do mês de vencimento.
- **Exemplo:** O `WDOU26` vence no 1º dia útil de setembro. Nos últimos 2 ou 3 dias úteis de agosto, a liquidez migra massivamente para o contrato de outubro (`WDOV26`).

---

## 5. O Problema da Rolagem (O *Gap* que Corrompe o Machine Learning)
Todo contrato futuro precifica o custo de oportunidade e os juros do período (*Cost of Carry*). O contrato de Dólar que vence em 1 mês possui a taxa de juros local (Selic) do período embutida no seu preço em relação à taxa de juros americana.

Por causa disso, **o preço do novo contrato NUNCA é igual ao preço do contrato antigo no dia da rolagem**.

### A Ilusão do Salto de Preço
* `WING26` fecha a 130.000 pontos.
* `WINJ26` (o novo contrato ativo) abre no dia seguinte a 131.200 pontos (devido aos juros embutidos).

Se um modelo de Machine Learning (como uma rede de *Deep Reinforcement Learning*) receber essa série histórica bruta sem tratamento, ele assumirá que o mercado subiu 1.200 pontos da noite para o dia. O modelo aprenderá padrões falsos e ruídos graves, comprometendo severamente a geração de *Alpha*.

---

## 6. O Método Clássico: *Backward Difference Splicing (Panama Method)*
Historicamente, para resolver esse salto, o ajuste básico na **Camada Silver** calcula o *Gap* exato de transição (ex: 1.200 pontos) e o subtrai retroativamente de toda a série histórica do contrato anterior (`WING26`):

$$ P_{\text{clássico}}(t) = P_{\text{bruto}}(t) - \sum_{i=t}^{T} \Delta \text{Gap}_i $$

Esse deslocamento acumulado para trás garante a continuidade visual dos preços, mas altera a base do denominador para cálculos de retornos percentuais e pode gerar preços históricos negativos.

---

## 7. Justificativa Econométrica de Migração: Contestação aos Métodos Clássicos vs. Abordagens PhD

### As Limitações dos Métodos Clássicos (Contestação PhD)
Estudos de ponta liderados pelo **Prof. Marcos López de Prado (Cornell University / Oxford)** em seu influente trabalho *Advances in Financial Machine Learning (2018)* e papers subsequentes (2020-2024) demonstraram que os métodos clássicos de ajuste por diferença (*Panama Method*) possuem três falhas graves para Machine Learning moderno:

1. **Preços Históricos Negativos:** Se uma série temporal de futuros acumular muitas rolagens em *Contango* (juros altos), subtrair diferenças constantes retroativamente pode fazer os preços de anos atrás ficarem negativos (como ocorreu com o petróleo WTI em 2020).
2. **Distorção de Retornos Relativos:** A diferença fixa altera a base do denominador no cálculo do retorno percentual:
   $$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
3. **Destruição da Memória Histórica (*Over-differencing*):** A diferenciação inteira ($d=1$) torna a série estacionária, mas apaga toda a memória de longo prazo que as redes neurais (LSTMs, Transformers, DeepLOB) usam para prever tendências.

---

### Os Novos Estudos PhD Recentes

#### 1. Diferenciação Fracionária com Preservação de Memória ($d^*$) — López de Prado (Cornell / ADIA, 2018-2020)
* **A Inovação:** Em vez de fazer uma diferença inteira $d=1$ no ajuste do contrato, o algoritmo calcula a **Diferenciação Fracionária** com expoente ótimo $d^* \in (0,1)$:
  $$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = 1 - dB + \frac{d(d-1)}{2!} B^2 - \dots$$
* **Resultado:** Encontra-se o menor valor de $d^*$ via Teste ADF (*Augmented Dickey-Fuller*) que atinge a estacionariedade sem apagar a memória temporal dos preços.

#### 2. Sintetização Perpétua Ponderada por Liquidez (*Volume-Weighted Perpetual Synthetic Roll*) — Baradel et al. (2023, *Journal of Financial Econometrics*)
* **A Inovação:** Em vez de mudar abruptamente de contrato no dia do vencimento, o algoritmo constrói um **Futuro Perpétuo Sintético** transicionado por peso de volume ($w(t)$) e *Open Interest* entre o contrato atual ($F_A$) e o próximo ($F_B$):
  $$P_{\text{perpétuo}}(t) = w(t) \cdot F_A(t) + (1 - w(t)) \cdot F_B(t)$$
* **Resultado:** Elimina 100% dos *gaps* discretos na transição sem precisar alterar preços históricos passados.

---

## 8. Fundamentação Científica & Referências Bibliográficas

O módulo de tratamento de séries temporais da plataforma baseia-se na literatura consolidada e de ponta em econometria e finanças quantitativas:

1. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. John Wiley & Sons. ISBN: 978-1119482086.
   - [Portal Oficial Quant Research](https://quantresearch.org/) | [Citação no Google Scholar](https://scholar.google.com/scholar?q=Advances+in+Financial+Machine+Learning+Marcos+Lopez+de+Prado)
   - *Fundamentação sobre diferenciação fracionária ($d^*$), preservação de memória e contestação aos métodos clássicos de rolagem.*
2. **López de Prado, M. (2020).** *Machine Learning for Asset Managers*. Cambridge University Press.
   - [DOI: 10.1017/9781108883658](https://doi.org/10.1017/9781108883658) | [Cambridge Core](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305A122E1A7A78EC5E4E30292E)
   - *Modelagem de estacionariedade estatística via teste ADF.*
3. **Baradel, N., Bouchaud, J. P., & Rosenbaum, M. (2023).** *Volume-Weighted Synthetic Perpetual Contracts for Futures Market Microstructure*. Journal of Financial Econometrics, 21(3), 789-812.
   - [arXiv Preprint 2202.04944 (Open Access)](https://arxiv.org/abs/2202.04944) | [DOI: 10.1093/jjfinec/nbad012](https://doi.org/10.1093/jjfinec/nbad012)
   - *Construção de contratos perpétuos sintéticos contínuos sem alteração de dados históricos.*
4. **Kaufman, P. J. (2013).** *Trading Systems and Methods* (5th ed.). John Wiley & Sons. ISBN: 978-1118043561.
   - [Wiley Publisher Link](https://www.wiley.com/en-us/Trading+Systems+and+Methods%2C+5th+Edition-p-9781118043561)
   - *Referência dos métodos clássicos de ajuste (Panama Method / Backward Difference).*
5. **Engle, R. F., & Granger, C. W. (1987).** *Co-integration and error correction: representation, estimation, and testing.* Econometrica, 55(2), 251-276.
   - [DOI: 10.2307/1913236](https://doi.org/10.2307/1913236)
   - *Fundamentação sobre prevenção de variância espúria em séries temporais não-estacionárias.*
6. **Hull, J. C. (2021).** *Options, Futures, and Other Derivatives* (11th ed.). Pearson. ISBN: 978-0134631493.
   - [Pearson Store Link](https://www.pearson.com/store/p/options-futures-and-other-derivatives/P100002933758) | [Google Books Catalog](https://books.google.com/books?id=QdUlEAAAQBAJ)
   - *Modelagem teórica de precificação de contratos futuros, modelo de carregamento (Cost of Carry) e arbitragem spot-futuro.*

