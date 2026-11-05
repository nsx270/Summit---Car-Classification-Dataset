# Classificação da aceitabilidade de automóveis com SVM

Projeto Integrador de Inteligência Artificial · Engenharia de Software · Universidade de Mogi das Cruzes (UMC)

Trabalho apresentado na 4ª edição do UMC Summit.

![Pôster apresentado no UMC Summit](poster_summit_final.png)

## Sobre o projeto

A ideia era simples: dadas as características de um carro (preço, manutenção, portas, passageiros, porta-malas e segurança), o quanto um classificador SVM consegue acertar se ele é *inaceitável*, *aceitável*, *bom* ou *muito bom*?

Para responder, usamos o Car Evaluation Dataset, passamos os dados por um pré-processamento cuidadoso, procuramos os melhores hiperparâmetros com validação cruzada e avaliamos o modelo final num conjunto de teste que ele nunca viu.

## Os dados

O dataset tem 1.728 carros e 7 colunas: seis atributos categóricos ordinais e a classe final.

| Atributo | Valores |
|---|---|
| `buying` | low, med, high, vhigh |
| `maint` | low, med, high, vhigh |
| `doors` | 2, 3, 4, 5more |
| `persons` | 2, 4, more |
| `lug_boot` | small, med, big |
| `safety` | low, med, high |
| `class` | unacc, acc, good, vgood |

Duas coisas que vale saber antes de olhar os resultados:

- A base contém **cada combinação possível de atributos uma única vez** (4 × 4 × 4 × 3 × 3 × 3 = 1.728). Por isso os atributos são distribuídos de forma uniforme e não têm correlação entre si.
- A classe é bem **desbalanceada**: 1.210 *unacc* (70%), 384 *acc* (22,2%), 69 *good* (4%) e 65 *vgood* (3,8%). Um modelo que respondesse sempre "inaceitável" já teria 70% de acurácia, então acurácia sozinha não diz muita coisa.

Não há valores ausentes. Na análise exploratória, o que mais chama atenção é a segurança: todo carro com segurança baixa é *unacc*, não importa o resto.

## Como foi feito

1. **Codificação:** `OrdinalEncoder` com a ordem de cada atributo definida à mão (`low` < `med` < `high` < `vhigh`, por exemplo).
2. **Divisão:** 80% treino (1.382) e 20% teste (346), estratificada pela classe.
3. **Escalonamento:** `StandardScaler` ajustado só no treino e apenas aplicado no teste.
4. **Busca de hiperparâmetros:** `GridSearchCV` com `StratifiedKFold` de 5 dobras, testando
   - `C`: 0,1 · 1 · 10 · 100
   - `gamma`: scale · auto · 0,01 · 0,1 · 1
   - `kernel`: linear · poly · rbf

   São 60 combinações, ou 300 treinamentos.
5. **Avaliação:** o melhor modelo é aplicado ao conjunto de teste.

## Resultados

Melhor configuração: **kernel RBF, C = 10, gamma = scale**.

| Métrica | Valor |
|---|---|
| Acurácia na validação cruzada | 98,34% |
| Acurácia no teste | 99,13% (343 de 346) |
| F1-score ponderado | 0,99 |

Melhor acurácia de validação por kernel:

| Kernel | Acurácia |
|---|---|
| linear | 84,66% |
| poly | 90,88% |
| rbf | 98,34% |

A diferença entre os kernels sugere que a fronteira entre as classes é bem pouco linear.

Matriz de confusão no teste (linhas = classe real, colunas = predição):

| | unacc | acc | good | vgood |
|---|---|---|---|---|
| **unacc** | 240 | 2 | 0 | 0 |
| **acc** | 1 | 76 | 0 | 0 |
| **good** | 0 | 0 | 14 | 0 |
| **vgood** | 0 | 0 | 0 | 13 |

Os três erros ficam na fronteira entre *unacc* e *acc*. As classes pequenas foram todas acertadas, mas com só 14 e 13 exemplos no teste, esse 100% deve ser lido com cuidado.

## Limitações

- O dataset é pequeno e artificial: foi gerado a partir de um modelo de decisão com regras bem definidas e cobre todo o espaço de atributos. Alta acurácia aqui mostra que o SVM aprende bem essa estrutura, não que ele funcionaria igual em dados reais e ruidosos.
- O escalonamento foi feito antes do `GridSearchCV`. Não há vazamento para o teste, mas dentro da validação cruzada o ideal seria um `Pipeline` com o scaler dentro de cada dobra.
- Não usamos `class_weight` nem reamostragem para tratar o desbalanceamento.
- Só o SVM foi avaliado. Comparar com árvores de decisão ou Random Forest seria o próximo passo natural.

## Como rodar

O código está no notebook `atividade-modelo-summit.ipynb`.

```bash
git clone https://github.com/nsx270/Summit---Car-Classification-Dataset.git
cd Summit---Car-Classification-Dataset
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook atividade-modelo-summit.ipynb
```

O notebook foi feito no Kaggle. Se o caminho do arquivo de dados der erro na sua máquina, baixe o CSV pelo link abaixo e ajuste o caminho na célula de leitura.

## Fonte dos dados

- Dataset no Kaggle: <https://www.kaggle.com/datasets/stealthtechnologies/car-evaluation-classification>
- Origem: Car Evaluation Database, UCI Machine Learning Repository. Criado por Marko Bohanec e Vladislav Rajkovic.

## Autores

- [Nome da Pessoa 1]
- [Nome da Pessoa 2]

Orientação: [Nome do orientador]
