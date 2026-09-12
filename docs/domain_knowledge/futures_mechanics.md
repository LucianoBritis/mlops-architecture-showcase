# Mecânica dos Futuros da B3 e o Problema da Rolagem (*Rollover*)

Para projetar pipelines robustas de Machine Learning para Finanças Quantitativas, especialmente no Mercado Brasileiro (B3), compreender a mecânica dos Contratos Futuros é um pré-requisito fundamental. A arquitetura descrita neste repositório (especificamente o algoritmo de *Backward Difference Splicing* na Camada Silver) existe para solucionar as anomalias matemáticas geradas por essa mecânica.

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

## 6. A Solução MLOps: *Backward Difference Splicing*
Para resolver isso, a pipeline de dados na **Camada Silver** não alimenta dados brutos da B3 para os modelos. Nós calculamos o *Gap* exato de transição (ex: 1.200 pontos) e o subtraímos retroativamente de toda a série histórica do contrato anterior (`WING26`).

Esse deslocamento acumulado para trás garante que a série temporal se torne perfeitamente contínua. Os retornos percentuais e as volatilidades históricas são matematicamente preservados, permitindo que a rede neural seja treinada sobre a real variância do mercado sem ser poluída por rolagens artificiais.
