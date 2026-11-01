# tech-challenge-fase3

FIAP AI Scientist — Fase 3: predição de alfabetização e inteligência analítica para educação no Brasil.

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![Fontes](https://img.shields.io/badge/fontes-INEP%20%7C%20IBGE-4285F4)
![License](https://img.shields.io/badge/license-MIT-green)

Modelo supervisionado para prever, no grão do estudante, se uma criança será
considerada alfabetizada ao final do 2º ano do Ensino Fundamental — e identificar os
fatores que mais influenciam esse resultado.

Consome a camada Gold construída no [Tech Challenge da Fase 2](https://github.com/luizafcunha/fiap-ai-scientist-fase-02)
e a enriquece com contexto socioeconômico municipal do Censo Demográfico 2022 e do PIB.

**Em uma leitura.** Os dados públicos de alfabetização são anonimizados: todo aluno de uma
cidade recebe as mesmas variáveis. Por isso o modelo estima a chance de alfabetização de uma
criança **naquele contexto**, não de uma criança específica — e nessa pergunta ele acerta. A
probabilidade prevista fica a cerca de um ponto da observada, separa contextos de 13% a 65% de
risco e aponta os municípios que ficarão abaixo da meta com 90% de precisão entre os dez
primeiros. O que nenhum modelo consegue com esses dados é distinguir duas crianças da mesma
cidade: 88% da variação está aí.

> **Legenda de status:** ✅ concluído · 🚧 em andamento · ⏳ planejado
> **Entrega:** 3 de novembro de 2026

---

## Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo analítico](#2-objetivo-analítico)
3. [Descrição da base utilizada](#3-descrição-da-base-utilizada)
4. [Arquitetura e fronteira com a Fase 2](#4-arquitetura-e-fronteira-com-a-fase-2)
5. [Etapas de modelagem](#5-etapas-de-modelagem)
6. [Escolha do algoritmo](#6-escolha-do-algoritmo)
7. [Métricas de avaliação](#7-métricas-de-avaliação)
8. [Interpretação dos resultados](#8-interpretação-dos-resultados)
9. [Insights encontrados](#9-insights-encontrados)
10. [Limitações do projeto](#10-limitações-do-projeto)
11. [Aplicação prática para políticas públicas](#11-aplicação-prática-para-políticas-públicas)
12. [Possíveis evoluções futuras](#12-possíveis-evoluções-futuras)
13. [Como executar](#13-como-executar)
14. [Estrutura do repositório](#14-estrutura-do-repositório)
15. [Fluxo de trabalho Git](#15-fluxo-de-trabalho-git)
16. [Evidências de execução](#16-evidências-de-execução)
17. [Roadmap e status](#17-roadmap-e-status)
18. [Equipe](#18-equipe)
19. [Licença](#19-licença)

---

## 1. Contexto do problema

O Brasil assumiu o compromisso de alfabetizar **todas as crianças até o final do 2º ano
do Ensino Fundamental até 2030**. A medição é feita por avaliação padronizada do INEP, e
considera-se alfabetizado o estudante que atinge **743 pontos na escala Saeb** de Língua
Portuguesa.

A Fase 2 construiu a pipeline que integra resultados, metas e contexto escolar, e
respondeu **o que aconteceu**: 53,3% dos municípios atingiram a meta de 2024 na rede
municipal, e 1.917 retrocederam entre 2023 e 2024.

Descrever o passado, porém, não muda o futuro. Um gestor que recebe o resultado da
avaliação já perdeu o ano letivo — a criança foi avaliada, o diagnóstico chegou depois.
**A lacuna que esta fase ataca é a antecipação:** identificar risco antes da avaliação,
quando ainda há tempo de intervir.

---

## 2. Objetivo analítico

Classificação binária no grão do aluno, prevendo `alfabetizado` (1 se `proficiencia >=
743` na escala Saeb, 0 caso contrário). O alvo está relativamente balanceado — 59,4% de
positivos —, então acurácia continua informativa, mas acompanhada de precisão e recall.

O modelo responde **quem está em risco de não alfabetizar**, a partir de contexto
disponível antes da avaliação (município, infraestrutura escolar, meta pactuada,
histórico do ano anterior). Ele **não** responde por que um aluno específico não
alfabetizou, nem substitui a avaliação em si — é sinal de risco, não diagnóstico.

Classificação, e não regressão sobre a proficiência, porque a decisão de política pública
é binária (aluno entra ou não em programa de apoio) e o corte de 743 pontos já é o
critério oficial de alfabetização — regredir a proficiência adicionaria complexidade sem
mudar a ação tomada a partir do resultado.

**O aluno provável.** O enunciado pede prever a alfabetização "utilizando variáveis
educacionais, territoriais e socioeconômicas" — e todas elas descrevem o município, não a
criança. Alunos da mesma cidade recebem o mesmo vetor de features (seção 7). Por isso o
modelo estima a **probabilidade de alfabetização para um aluno inserido em determinado
contexto**: a resposta para "um aluno deste território, desta condição socioeconômica e
desta condição educacional se alfabetiza?". Para essa pergunta, a métrica central é a
calibração — quando o modelo diz 40% de risco, cerca de 40% dos alunos daquele contexto de
fato não se alfabetizam —, reportada ao lado das métricas de classificação.

---

## 3. Descrição da base utilizada

### Fontes

| Nível | Origem | Tabela | Grão | Papel |
|---|---|---|---|---|
| Aluno | Fase 2 · Silver | `fato_aluno` | aluno × ano | Alvo e atributos individuais |
| Município | Fase 2 · Silver | `dim_territorio` | município | UF e região |
| Município | Fase 2 · Gold | `features_municipio` | município | Contexto educacional |
| Município | Fase 2 · Gold | `trajetoria_meta_2030` | município | Meta pactuada |
| Município | IBGE · Censo 2022 | `municipio` | município | População, domicílios, área, alfabetização adulta |
| Município | IBGE · PIB | `municipio` | município × ano | Capacidade econômica |

**O alvo vem da Silver; a Gold da Fase 3 é construída sobre ele.** O enunciado pede dados
da camada Gold, e a da Fase 2 é agregada por município — insuficiente para predição
individual. Em vez de contornar isso num script de pré-processamento, a Fase 3 declara
sua **própria camada Gold**, `fase3_gold.features_aluno`, no grão do estudante. A camada
que entrega dado pronto para consumo passa a entregar o que este consumo exige.

### `features_aluno` ✅

Uma linha por estudante avaliado em 2024, com 35 colunas. O significado, a faixa de valores e
o tratamento de cada coluna estão no [dicionário de dados](DICIONARIO.md).

| Métrica | Valor |
|---|---:|
| Linhas | 1.610.754 |
| Municípios | 5.450 |
| Colunas | 35 |
| Alvo positivo (`alfabetizado`) | 59,4% |
| Sem contexto municipal | 0 |

**Balanceamento saudável.** Com 59,4% de positivos, acurácia continua informativa — o que
não seria verdade numa base 90/10.

**Nulos parciais em 8 colunas.** Entre 0,001% e 2,6% (piores: `mun_meta_ano_alvo` e
`mun_distancia_meta_anterior`), abaixo do limiar que bloqueia a Gold — tratados por
imputação no pipeline de modelagem, não pela validação do Job.

| Grupo | Colunas |
|---|---|
| Chaves e agrupamento | `ano`, `id_aluno`, `id_escola`, `id_municipio`, `sigla_uf`, `regiao` |
| Alvo | `alfabetizado`, `peso_aluno` |
| Marcações | `uf_anomala` |
| Aluno | `rede_codigo`, `caderno` |
| Contexto educacional do município | `mun_taxa_ano_anterior`, `mun_total_escolas`, `mun_alunos_por_docente`, `mun_alunos_por_turma`, `mun_pct_integral`, `mun_indice_infraestrutura`, `mun_pct_rural`, `mun_pct_transporte` |
| Meta pactuada | `mun_meta_ano_alvo`, `mun_distancia_meta_anterior`, `mun_elegivel_meta` |
| Contexto socioeconômico | `mun_populacao`, `mun_densidade`, `mun_moradores_por_domicilio`, `mun_taxa_alfabetizacao_adulta`, `mun_idade_mediana`, `mun_indice_envelhecimento`, `mun_pct_indigena`, `mun_pct_quilombola`, `mun_pib_per_capita`, `mun_pct_va_agropecuaria`, `mun_pct_va_industria`, `mun_pct_va_servicos`, `mun_pct_va_administracao_publica` |

**Das treze colunas de `trajetoria_meta_2030`, apenas três entraram.** As demais —
`distancia_meta_2030`, `ritmo_necessario`, `indice_trajetoria`, `classificacao_trajetoria`
e `atingiu_meta_2024` — derivam de `taxa_2024`, que é o resultado a prever.

O que sobrevive ao corte é informação disponível antes da avaliação: a meta pactuada, a
elegibilidade do município, e `taxa_2023 − meta_2024`, que mede quanto faltava no fim de
2023 para alcançar a meta do ano seguinte. Essa última capta pressão institucional sobre
a rede — município longe da meta tende a mobilizar esforço.

**A taxa de alfabetização adulta é a da população de 15 anos ou mais**, como o Censo 2022
publica por município. Ela descreve a escolaridade do entorno da criança, não a dela: como a
faixa começa em 15 anos, **nenhuma criança avaliada aparece na tabela** — não há vazamento.

**Nenhuma contagem absoluta entra no modelo.** População indígena de 200 num município de
5 mil habitantes e de 200 em um de 2 milhões descrevem realidades opostas. Tudo vira
proporção, taxa ou densidade.

### Tratamento de vazamento de dados ✅

Requisito explícito do enunciado, e o item que mais distingue a maturidade do projeto.
Três tipos foram identificados e tratados.

**Vazamento direto — colunas que são o alvo disfarçado.** Na Silver da Fase 2,
`alfabetizado` é definido como `proficiencia >= 743`, verificado em 3.354.661 registros
com zero divergências.

| Coluna | Por quê |
|---|---|
| `proficiencia` | Define o alvo por construção |
| `distancia_corte` | `proficiencia − 743` |
| `faixa_proximidade` | Derivada de `distancia_corte` |

**As três não existem na saída.** São excluídas na construção da Gold, não no consumo —
o que impede o uso acidental. Uma regra bloqueante do Job derruba a execução se alguma
delas aparecer no contrato.

**Vazamento temporal.** `features_municipio` traz `alvo_taxa_2024`, que é a média do alvo
dos alunos daquele município. Usá-la para prever um aluno de 2024 vaza o resultado da
própria turma.

A solução foi restringir o dataset a **alunos de 2024, com a taxa de alfabetização municipal
de 2023**. Corta o volume à metade e, em troca, o modelo fica genuinamente preditivo: prevê o
ano corrente com o resultado do ano anterior, que é o uso que um gestor faria. As variáveis de
estrutura escolar vêm do Censo Escolar de 2024, que descreve escolas, turmas e docentes, e não
o resultado da avaliação.

**Vazamento de contexto — o mais sutil.** As features municipais se repetem em todos os
alunos do município. Medido em `sql/gold/pseudo_replicacao.sql`:

| Métrica | Valor |
|---|---:|
| Linhas | 1.610.754 |
| Municípios | 5.450 |
| Alunos por município (média) | 295,6 |
| Tamanho do município do aluno típico (mediana) | 944 |
| Máximo | 43.340 |

O tamanho efetivo da amostra para variáveis municipais é **5.450**, não 1,6 milhão. Um
`train_test_split` aleatório colocaria centenas de alunos do mesmo município em treino e
teste, e o modelo memorizaria o contexto sem generalizar — vazamento sem nenhuma coluna
proibida envolvida.

A correção é `GroupShuffleSplit` e `GroupKFold` agrupando por `id_municipio`, e ela muda
a pergunta que a métrica responde: de *"acerta sobre alunos de municípios que já
conhece"* para *"acerta sobre municípios que nunca viu"* — que é a pergunta de negócio
real, porque o gestor quer antecipar risco onde ainda não mediu.

---

## 4. Arquitetura e fronteira com a Fase 2

A pipeline da Fase 2 permanece em produção como fonte dos dados tratados. A Fase 3 lê e
não modifica: atua apenas sob o prefixo `fase3/` do mesmo bucket, e os crawlers da fase
anterior têm include paths explícitos que não o alcançam.

| Onde | O que acontece | Repositório |
|---|---|---|
| AWS — Glue, Catalog, S3 | Bronze → Silver → Gold da Fase 2 | Fase 2 |
| Local — Python | Ingestão socioeconômica, 54 MB | Fase 3 |
| AWS — Glue Job | Construção de `features_aluno`, 1,6 milhão de linhas | Fase 3 |
| Local — scikit-learn | Modelagem, avaliação, interpretabilidade | Fase 3 |

**Por que não reconstruir a base da Fase 2.** Refazer a tradução de rede, o unpivot das
metas e a agregação ponderada do Censo Escolar não acrescentaria nada e criaria risco de
divergência: dois números para a mesma coisa, sem saber qual está certo.

**Por que ingestão local e transformação em Glue.** A assimetria é deliberada.
Transformar 54 MB em Spark custaria mais em provisionamento do que a transformação
inteira, e o processamento local itera mais rápido. Já a construção de `features_aluno`
são 3,87 milhões de linhas e quatro joins — volume em que o Spark paga o provisionamento.

### Recursos declarados em Terraform ✅

Um `terraform apply` cria **9 recursos**:

| Recurso | Qtde | Papel |
|---|---:|---|
| `aws_glue_catalog_database` | 2 | `fase3_bronze` e `fase3_gold` |
| `aws_glue_crawler` | 1 | Cataloga as fontes socioeconômicas |
| `aws_glue_job` | 1 | Constrói `features_aluno` — Glue 4.0, 2× G.1X |
| `aws_s3_object` | 1 | Upload do script PySpark |
| `aws_glue_catalog_table` | 1 | `features_aluno`, com schema declarado |
| `aws_glue_workflow` | 1 | Orquestração |
| `aws_glue_trigger` | 2 | Encadeamento condicional |

**Crawler na Bronze, schema declarado na Gold.** Na Bronze o schema vem de quem produziu
o dado, e descobri-lo automaticamente é apropriado. Na Gold é decisão: as colunas de
vazamento simplesmente não existem no contrato.

### Histórico de ingestão preservado ✅

| Mecanismo | Efeito |
|---|---|
| Partição `dt_ingestao=AAAA-MM-DD` | Padrão Hive; cargas coexistem em vez de sobrescrever |
| Coluna `_ingerido_em` | Cada registro sabe quando foi lido |
| Manifesto em `reports/ingestao/manifesto.jsonl` | Volume, custo, caminho e a consulta exata de cada execução |

O manifesto é versionado e sobrevive ao laboratório, ao contrário do S3.

**A pasta da tabela vem antes da partição**, no padrão `<fonte>_<tabela>/dt_ingestao=…`.
O Crawler batiza a tabela pela última pasta antes da partição, então `censo_2022/municipio`
e `pib_municipal/municipio` colidiriam — as duas fontes têm uma tabela chamada
`municipio`. A verificação em `infra/executar_crawler.sh` confere que `dt_ingestao` foi
reconhecida como chave e falha se não for.

### Custo da ingestão ✅

Medido por *dry run* antes de cada execução, sem gerar custo:

| Fonte | Tabela | Linhas | Varredura |
|---|---|---:|---:|
| Censo 2022 | `municipio` | 5.570 | 0,41 MB |
| Censo 2022 | `alfabetizacao_grupo_idade_sexo_raca` | 779.800 | 47,42 MB |
| PIB | `municipio` (2021–2023) | 16.710 | 6,32 MB |
| **Total** | | **802.080** | **54,15 MB** |

O BigQuery cobra por coluna varrida: a seleção explícita de colunas é o que reduz custo, e
`LIMIT` corta o retorno, não a varredura.

### Orquestração ✅

O fluxo se encadeia dentro da AWS por **Glue Workflow**:

```
trigger ON_DEMAND
  └─ crawler da Bronze
      └─ (SUCCEEDED) job da Gold
```

A Gold só é gerada se a catalogação tiver sucesso. Sem o encadeamento, o Job leria um
Catalog desatualizado e produziria dado silenciosamente errado: a partição nova não
estaria registrada, e a leitura traria a carga anterior sem qualquer aviso.

**A ingestão fica fora do fluxo, e isso é limitação declarada.** O Glue Workflow encadeia
apenas crawlers e jobs; os dois scripts de ingestão são Python local. Convertê-los em
Glue Jobs Python Shell traria o ciclo inteiro para dentro do Workflow — foi avaliado, e o
custo seria baixo, 0,0625 DPU. Mas as ingestões somam 54 MB e rodam em segundos
localmente; empacotá-las em Glue para orquestrar duas etapas que executam uma vez por ano
não se justifica.

**Por que `ON_DEMAND` e não agendado.** O Censo Demográfico é decenal, o PIB é anual e a
avaliação do INEP é anual. Agendamento diário dispararia execuções reprocessando o mesmo
dado.

### Validação da Gold ✅

O Job valida a saída **antes de gravar**. Quatro regras bloqueantes derrubam a execução:

| Regra | Motivo |
|---|---|
| Chave `(ano, id_aluno)` duplicada | Grão comprometido |
| `id_municipio` fora de 7 dígitos | Join territorial falharia sem erro |
| Alvo nulo | Linha sem o que prever |
| Coluna de vazamento presente | Redundante com o schema, e de propósito |
| Coluna 100% nula | Feature sem valor desaparece do modelo sem erro |

A última virou bloqueante depois de um episódio concreto: as quatro proporções setoriais
do PIB ficaram 100% nulas em duas execuções, porque o IBGE publica o PIB total antes da
desagregação por atividade. O Job passou a escolher o ano mais recente em que cada bloco
de dados existe — PIB total de 2023, composição setorial de 2021.

---

## 5. Etapas de modelagem

> ✅ *Pré-processamento, baseline e comparação de algoritmos concluídos.*

**Seleção de features** — [`notebooks/02_selecao_features.ipynb`](notebooks/02_selecao_features.ipynb).
Das 35 colunas de `features_aluno`, **24 entram no modelo** e 11 ficam de fora, cada uma
com motivo e evidência no notebook:

| Grupo | Colunas | Motivo |
|---|---|---|
| Alvo | `alfabetizado` | Vira o `y` |
| Identificadores | `id_aluno`, `id_escola`, `id_municipio` | Rótulos sem significado; `id_municipio` define os grupos da divisão |
| Constantes | `ano`, `rede_codigo` | Um único valor em toda a base |
| Peso amostral | `peso_aluno` | Descreve o desenho da avaliação, não o aluno |
| Descartadas | `caderno`, `uf_anomala` | `caderno` só existe no dia da prova; `uf_anomala` é idêntica a `sigla_uf == "RS"` |
| Redundantes | `mun_meta_ano_alvo`, `mun_idade_mediana` | Correlação acima de 0,9 com outras features (a meta é calculável a partir da taxa e da distância) |

As decisões ficam em [`src/config/features_modelo.json`](src/config/features_modelo.json),
lido pelo notebook de modelagem: a lista de features existe em um único lugar.

**Alvo invertido.** `1 = não alfabetizado`. Precisão, recall e F1 passam a medir a classe
de risco, que é a que a política pública precisa encontrar.

**Divisão por município** — [`notebooks/03_modelagem_supervisionada.ipynb`](notebooks/03_modelagem_supervisionada.ipynb).
`GroupShuffleSplit` com 20% dos municípios no teste: treino com 4.360 municípios
(1.320.423 alunos) e teste com 1.090 (290.331), sem município em comum. Taxa de risco de
40,9% e 39,3%. O teste fica reservado para a avaliação final; a comparação de modelos usa
`GroupKFold` de 5 rodadas dentro do treino.

**Pipeline.** Pré-processamento e modelo num único `Pipeline` do scikit-learn, aprendendo
só com o treino de cada rodada:

| Grupo | Imputação | Transformação |
|---|---|---|
| 6 numéricas muito assimétricas (população, densidade, PIB per capita…) | Mediana | `log1p` + padronização |
| 11 numéricas | Mediana | Padronização |
| 4 razões do Censo Escolar com valores impossíveis | Mediana | Teto no percentil 99 do treino + padronização |
| `regiao`, `sigla_uf` | Mais frequente | One-hot |
| `mun_elegivel_meta` | Constante 0 | — |

**Estratégia de amostragem.** A base completa é usada, sem reamostragem: o alvo é
relativamente balanceado (59,4% × 40,6%). **Nenhum modelo usa `class_weight`.** O recall-alvo
da classe de risco é obtido pela escolha do limiar, por validação cruzada — não pelo peso das
classes. A seção 7 mostra o experimento que levou a essa decisão.

A busca de hiperparâmetros e a escolha do limiar usam uma amostra de até 30 alunos por
município. As features são municipais, e 1,3 milhão de alunos do treino equivalem a cerca de
4.400 pontos independentes: a amostra preserva a informação e torna a busca viável.

**Teto nas razões do Censo Escolar.** `mun_alunos_por_turma`, `mun_alunos_por_docente`,
`mun_pct_rural` e `mun_pct_transporte` têm numerador e denominador de populações diferentes e
assumem valores impossíveis (seção 10). O teto é aprendido no `fit`, portanto só com o
treino, e configurado em `limitar_extremos` no JSON de decisões. O efeito nas métricas foi
nulo — ROC-AUC de 0,6709 com e sem teto na regressão logística, o modelo mais sensível a
extremos por ser linear. A correção é de consistência, para que o modelo não receba valores
impossíveis, não de desempenho.

## 6. Escolha do algoritmo

Quatro algoritmos, avaliados no mesmo conjunto de teste — 1.090 municípios nunca vistos no
treino —, cada um com o limiar escolhido na validação cruzada para o mesmo recall-alvo de 70%:

| Algoritmo | ROC-AUC | PR-AUC | Recall | Precisão | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,674 | 0,551 | 0,753 | 0,491 | 0,594 | 0,2165 |
| Random Forest | 0,673 | 0,548 | 0,773 | 0,482 | 0,594 | 0,2170 |
| Regressão logística | 0,673 | 0,550 | 0,771 | 0,484 | 0,595 | 0,2166 |
| Árvore de decisão rasa | 0,664 | 0,528 | 0,801 | 0,474 | 0,596 | 0,2181 |

Métricas por aluno, no teste completo; o Brier é medido sem o RS. O recall supera o alvo de
70% porque o limiar é escolhido com cada município pesando igual e o teste é contado por aluno
— explicado na seção 7.

**Os quatro empatam.** A diferença de ROC-AUC entre o melhor e o pior é de 0,009, e entre os
três primeiros, de 0,001 — dentro do que uma nova divisão treino/teste poderia inverter. O F1
também empata, entre 0,594 e 0,596. Os modelos de árvores não extraem mais sinal que a
regressão logística, e a seção 7 mostra por quê: o limite está nos dados, não no algoritmo.

**Modelo escolhido: HistGradientBoosting.** Com desempenho equivalente, dois critérios
competiam. A interpretabilidade favoreceria a regressão logística, cujos coeficientes se leem
diretamente. Os números absolutos favorecem o HistGradientBoosting, com o melhor ROC-AUC e o
melhor Brier dos quatro. O grupo escolheu o segundo e trata a interpretabilidade com SHAP, na
seção 8.

---

## 7. Métricas de avaliação

**Métrica prioritária: recall da classe de risco.** O falso negativo, deixar de sinalizar
uma criança em risco, é o erro mais grave. O recall é acompanhado de precisão e F1, porque
um modelo que apontasse risco para todos teria recall de 100% e nenhuma utilidade. A
acurácia não serve sozinha: o modelo que responde sempre "alfabetizado" acerta 59%.

**Baseline** (validação cruzada por município, 5 rodadas, treino):

| Modelo | ROC-AUC | Recall | Precisão | F1 | Acurácia |
|---|---|---|---|---|---|
| Ingênuo (classe mais comum) | 0,500 | 0,000 | 0,000 | 0,000 | 0,591 |
| Regressão logística | **0,661** | 0,393 | 0,580 | 0,462 | 0,634 |

No limiar padrão de 0,5, a regressão logística encontra 214.273 das 540.192 crianças em risco
do treino — 39,7%. Dos alertas que emite, 57,7% estão corretos, contra 41% se fossem dados ao
acaso. O recall é baixo porque, sem peso nas classes, o modelo só sinaliza quem tem mais de 50%
de chance de risco; os modelos da seção 6 escolhem o limiar por validação cruzada e levam o
recall ao alvo. O ROC-AUC, que não depende de limiar, já mostra o poder de separação. Tabela
completa em [`reports/modelagem/`](reports/modelagem/) e matriz de confusão em
[`images/matriz_confusao_baseline.png`](images/matriz_confusao_baseline.png).

### O limite está nos dados

Das 29 features de `features_aluno`, **apenas uma varia entre alunos do mesmo município** — o
`caderno`, descartado por só existir no dia da prova. Todo aluno de uma cidade recebe o mesmo
vetor de features, e nenhum modelo pode distinguir duas crianças da mesma cidade.

No teste — 280.021 alunos em 1.009 municípios, sem o RS —, a variação do alvo se divide assim:

| Onde está a variação | Fração |
|---|---:|
| Entre municípios — alcançável pelas features | 11,6% |
| Dentro de cada município — inalcançável | 88,4% |

Para medir o que esse limite permite, imaginamos o melhor previsor possível com informação
municipal: um que soubesse a taxa real de 2024 de cada município. Ele não existe, mas define
o teto de qualquer modelo construído com estes dados.

### Duas medidas, dois tetos

O modelo responde a duas perguntas diferentes, e cada uma tem sua métrica.

| O que se mede | Métrica | Modelo | Teto | Alcançado |
|---|---|---:|---:|---:|
| Ordenar alunos por risco | ROC-AUC | 0,674 | 0,699 | 87% |
| Acertar a probabilidade | Brier skill | 8,9% | 11,6% | 76% |

"Alcançado" é a fração do ganho possível que o modelo obtém: sobre o acaso, no ROC-AUC; sobre
prever sempre a média, no Brier.

**Na ordenação, o modelo está perto do teto.** Uma única feature, a taxa do ano anterior, dá
0,663; o modelo completo, 0,674; o melhor possível, 0,699. Outro algoritmo ou mais busca de
hiperparâmetros disputam no máximo 0,025 — e a seção 6 mostra que os quatro modelos empatam.

O teto do Brier, 11,6%, coincide com a variação entre municípios, 11,6%. Não por acaso: o
melhor previsor municipal elimina exatamente a parte da variação que está entre os municípios,
e nada da que está dentro deles. Duas métricas independentes chegam ao mesmo limite.

### Calibração

Verificada em [`src/evaluation/calibracao.py`](src/evaluation/calibracao.py), sobre as
probabilidades gravadas pelos notebooks. **As probabilidades saem calibradas de origem**, sem
correção posterior:

| Modelo | Erro de calibração | Brier | Risco médio previsto | Risco observado |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0,87 | 0,2165 | 38,4% | 38,8% |
| Regressão logística | 0,70 | 0,2166 | 38,4% | 38,8% |
| Random Forest | 1,40 | 0,2170 | 38,9% | 38,8% |
| Árvore de decisão rasa | 0,85 | 0,2181 | 38,9% | 38,8% |

Erro de calibração em pontos percentuais: a distância média entre o risco previsto e o
observado, ponderada pelos alunos de cada faixa. Teste sem RS, 280.021 alunos.

Por decil de risco previsto, no HistGradientBoosting:

| Decil de risco previsto | Previsto | Observado |
|---|---:|---:|
| 1 — contextos mais favoráveis | 15,3% | 13,1% |
| 5 | 36,8% | 37,3% |
| 10 — contextos mais adversos | 62,2% | 64,8% |

**86,9% das crianças se alfabetizam nos contextos mais favoráveis; 35,2%, nos mais adversos.**
Diagrama de confiabilidade em [`images/calibracao_teste.png`](images/calibracao_teste.png).

### Por que os modelos não usam `class_weight`

Os quatro modelos foram construídos inicialmente com `class_weight="balanced"`, para priorizar
o recall da classe de risco. O peso faz o modelo aprender como se metade dos alunos estivesse
em risco, quando a taxa real é de cerca de 39%. O experimento em
[`src/evaluation/experimento_class_weight.py`](src/evaluation/experimento_class_weight.py)
treina cada modelo com e sem o peso, na mesma amostra, e compara no mesmo teste:

| Modelo | ROC-AUC com peso | sem peso | Risco médio previsto com peso | sem peso |
|---|---:|---:|---:|---:|
| Regressão logística | 0,671 | 0,671 | 50,2% | 38,5% |
| Árvore de decisão rasa | 0,657 | 0,658 | 49,2% | 38,1% |
| Random Forest | 0,672 | 0,673 | 50,5% | 39,2% |
| HistGradientBoosting | 0,673 | 0,672 | 50,2% | 38,7% |

Risco observado no teste: 38,8%.

**O peso não melhora a ordenação** — o ROC-AUC fica igual nos quatro —, e o recall-alvo pode
ser obtido só pela escolha do limiar. **E distorce as probabilidades**: com ele, todo modelo
prevê risco médio de cerca de 50%, o mundo balanceado que aprendeu; o erro de calibração passa
de 10 pontos.

Uma primeira versão do projeto mantinha o peso e corrigia as probabilidades depois do treino,
com uma correção analítica de prior. Funcionava, mas consertava um problema que o desenho mais
simples não cria. O peso foi retirado, e o histórico do repositório registra as duas versões.
Gráfico em [`images/experimento_class_weight.png`](images/experimento_class_weight.png).

### Validação e teste

A busca de hiperparâmetros e o limiar são escolhidos na amostra de até 30 alunos por município,
em que cada cidade pesa quase igual. O teste da seção 6 é contado por aluno, e os municípios
grandes pesam mais. Medir o teste dos dois jeitos, em
[`src/evaluation/otimizacao.py`](src/evaluation/otimizacao.py), separa o que é perda de
generalização do que é forma de contar:

| Medida | Contagem | ROC-AUC | Recall no limiar | Risco |
|---|---|---:|---:|---:|
| Validação cruzada da busca | Até 30 por município | 0,686 | — | 37,3% |
| Teste | Até 30 por município | 0,681 | 70,5% | 37,0% |
| Teste | Todos os alunos | 0,674 | 75,3% | 39,3% |

**Contado do mesmo jeito, o modelo generaliza.** A perda entre validação e teste é de 0,005 —
o otimismo normal de escolher a melhor de doze configurações —, e o limiar acerta o alvo de 70%
de recall.

**Contado por aluno, os números mudam por outro motivo.** Os municípios grandes têm mais risco,
39,3% contra 37,0%, e passam a pesar mais. O ROC-AUC cai, porque o modelo é mais fraco nas
cidades grandes (seções 8 e 9), e o recall sobe para 75%: o limiar sinaliza 60% dos alunos do
teste.

**A configuração escolhida não está em overfitting.** A busca foi plana — doze configurações
entre 0,684 e 0,686 — e escolheu árvores de profundidade 3, taxa de aprendizado de 0,03 e 300
iterações. Na curva dessa configuração, o pico da validação ocorre em 400 árvores, e as 300
escolhidas ficam a 0,0003 dele, com distância de 0,008 entre treino e validação. O notebook 05
mostra o overfitting com uma configuração agressiva, de propósito; a curva da configuração
escolhida está em [`images/hgb_curva_iteracoes_final.png`](images/hgb_curva_iteracoes_final.png).

## 8. Interpretação dos resultados

Interpretação do HistGradientBoosting com SHAP, em
[`src/evaluation/interpretacao.py`](src/evaluation/interpretacao.py). Os valores estão em
log-odds da classe de risco: positivo aumenta a chance de a criança não se alfabetizar.

### Duas perguntas diferentes

O SHAP responde **o que o modelo usa para prever**. O enunciado pergunta outra coisa: **quais
fatores mais impactam a alfabetização**. As respostas não coincidem, e a diferença entre elas é
o achado principal desta seção.

### O que o modelo usa: o ano anterior

No modelo completo, `mun_taxa_ano_anterior` domina, seguida de `mun_distancia_meta_anterior` —
que é a própria taxa de 2023 menos a meta. O município que alfabetizou bem em 2023 tende a
alfabetizar bem em 2024.

É informação legítima, disponível antes da avaliação. E é inútil como resposta para um gestor:
persistência não é alavanca de política.

### O que o contexto explica: retirando o passado

O mesmo modelo foi treinado sem as duas variáveis de histórico:

| | ROC-AUC | Ganho sobre o acaso |
|---|---:|---:|
| Com histórico | 0,672 | 0,172 |
| Sem histórico | 0,649 | 0,149 — 86% do anterior |

**Retirar o passado custa apenas 0,023 de ROC-AUC.** O contexto sozinho recupera 86% do poder
preditivo, porque o resultado do ano anterior não traz informação nova: é um resumo compacto do
próprio contexto. É por isso que as variáveis socioeconômicas não apareciam no topo do SHAP —
não por não importarem, mas por já estarem contidas na taxa de 2023.

A conclusão é robusta ao modelo: a análise "sem inércia" do notebook 04, com a Random Forest,
chega ao mesmo resultado — de 0,672 para 0,647. Os dois modelos desta seção são treinados como
o de produção, com até 30 alunos por município, e interpretados sobre alunos de fora do RS,
pelo motivo explicado abaixo.

### Os fatores de contexto, em ordem

| Variável | Peso | Relação com o risco |
|---|---:|---|
| UF | 0,292 | Varia por estado — ver abaixo |
| Índice de envelhecimento | 0,075 | Mais envelhecido, menos risco |
| Moradores por domicílio | 0,064 | Mais adensado, mais risco |
| Região | 0,044 | — |
| Total de escolas | 0,044 | Maior, mais risco |
| População | 0,034 | Maior, mais risco |
| Peso da administração pública no PIB | 0,031 | Maior, mais risco |
| Alfabetização adulta | 0,028 | Maior, menos risco |
| Tempo integral | 0,024 | Maior, menos risco |

Peso é o |SHAP| médio do modelo sem histórico. As colunas de UF e de região são somadas por
aluno antes do valor absoluto — sem isso, a UF apareceria fragmentada em 27 pesos pequenos e
pareceria irrelevante. Lista completa, com o modelo completo ao lado, em
[`reports/modelagem/interpretacao_ranking.csv`](reports/modelagem/interpretacao_ranking.csv);
gráfico em [`images/interpretacao_comparacao.png`](images/interpretacao_comparacao.png).

Três leituras:

- **A condição das famílias pesa.** Domicílios mais adensados, menor alfabetização adulta e
  economia dependente da administração pública — sinais de vulnerabilidade — aumentam o risco.
  O índice de envelhecimento provavelmente indica o estágio da transição demográfica, mais
  avançada nas regiões mais desenvolvidas, e não um efeito da idade em si.
- **A escala pesa contra.** Municípios maiores têm mais risco — o mesmo padrão encontrado de
  forma independente nos grupos da seção 9, onde o urbano mais rico alfabetiza menos que o
  interior estruturado.
- **Da escola, só o tempo integral.** É a única variável escolar com peso relevante, associada
  a menos risco. A infraestrutura, frequentemente apontada como prioridade, pesa pouco quando o
  contexto do município já é conhecido: 0,007.

### A UF: o que o contexto não explica

Sem o histórico, a UF é o fator mais forte — e seu efeito aparece **depois** de o modelo já
conhecer o contexto socioeconômico de cada município:

| UF | Efeito | Leitura |
|---|---:|---|
| Ceará | −1,31 | Muito menos risco do que o contexto indicaria |
| Goiás | −0,42 | Menos risco |
| Espírito Santo | −0,40 | Menos risco |
| Minas Gerais | −0,23 | Menos risco |
| Rio Grande do Norte | +0,67 | Mais risco |
| Sergipe | +0,71 | Mais risco |
| Bahia | +0,88 | Muito mais risco |

SHAP médio da coluna de cada UF entre os seus próprios alunos, em log-odds de risco. Tabela
completa em [`reports/modelagem/interpretacao_uf.csv`](reports/modelagem/interpretacao_uf.csv).

**O maior contraste está dentro de uma mesma região.** Ceará e Bahia são do Nordeste, com
contextos socioeconômicos parecidos, e ficam nos extremos opostos: o efeito do Ceará multiplica
a chance de risco por cerca de 0,27; o da Bahia, por cerca de 2,4.

Duas explicações são possíveis, e os dados não separam uma da outra:

- **Política estadual.** O Ceará mantém desde 2007 um programa de alfabetização em colaboração
  com os municípios, o PAIC. O efeito é coerente com ele.
- **Medição.** O indicador vem de avaliações aplicadas pelos sistemas estaduais e alinhadas
  nacionalmente. Diferenças entre esses sistemas também apareceriam como efeito de UF.

### O Rio Grande do Sul

O modelo de produção é treinado com o RS, e `sigla_uf_RS` aparece como a terceira variável mais
importante do SHAP do notebook 05, aumentando o risco. A queda do estado em 2024 tem duas
explicações compatíveis com os dados: uma alteração na origem da medição, apontada na Fase 2, ou
as enchentes que fecharam escolas por semanas, levantadas no notebook 02. Nas duas, é um choque
de 2024, não uma característica educacional estável do estado.

O grupo decidiu manter o RS na base — excluí-lo removeria justamente os alunos de maior risco,
se o efeito for real —, e a análise de sensibilidade do notebook 04 sustenta a decisão: treinar
com ou sem o estado dá o mesmo ROC-AUC nos demais. A interpretação desta seção é calculada
sobre alunos de fora do RS, para que o choque não se misture aos fatores de contexto.

### Leitura causal, não

O SHAP mostra o que o modelo usa, não o que causa a alfabetização. Tempo integral associado a
menos risco não prova que ampliá-lo alfabetiza: municípios que o oferecem podem diferir em
outras coisas que o modelo não vê. O valor desta seção é apontar onde olhar, não prescrever o
que fazer.

---

## 9. Insights encontrados

A EDA (`notebooks/01_eda.ipynb`, seção 14) levantou hipóteses que a modelagem depois confirmou:

- **O limite está nos dados.** Na base completa, só 10,6% da variação do alvo está entre
  municípios; 89,4% está
  entre crianças da mesma cidade — no conjunto de teste, sem o RS, a proporção é de 11,6% e
  88,4% (seção 7). Isso antecipa o teto de ROC-AUC e torna obrigatória
  a validação agrupada por município.
- **O histórico domina.** A taxa de 2023 tem correlação de 0,70 com a de 2024 (0,74 sem o RS),
  por isso o modelo também foi analisado sem o histórico (seção 8).
- **A UF tem efeito próprio.** O Ceará alfabetiza 85,4% e a Bahia 36,0%, com contextos
  parecidos — o efeito de UF que o SHAP isola na seção 8.
- **A escala pesa contra.** Municípios até 10 mil habitantes alfabetizam 66,4%; de 200 mil a
  1 milhão, 55,9%. O mesmo padrão aparece nos grupos abaixo.
- **Infraestrutura é associação, não causa.** A taxa sobe de 57,7% para 67,1% entre o primeiro
  e o quarto quartil do índice (6,9 pontos contando por criança), mas o gradiente some ou se
  inverte dentro de cada região — e no SHAP a infraestrutura pesa 0,007.
- **Tempo integral depende da região.** No país a correlação é −0,06; no Nordeste, onde ele
  existe de fato, 0,15. No SHAP, é a única variável escolar com peso relevante.
- **O RS sofreu um choque em 2024.** Na base da Fase 3, queda mediana de 18,9 pontos e 93,4%
  dos municípios em queda — a Fase 2, num recorte diferente de municípios, mediu 89,6% e 19,8
  pontos (seção 10). Os números variam com o recorte; a conclusão, não. O estado foi mantido no treino, com análise de sensibilidade (notebook 04).

### Regiões com padrões semelhantes

Resposta à pergunta de negócio do enunciado, em
[`src/modeling/clusterizacao.py`](src/modeling/clusterizacao.py). K-Means sobre o **contexto**
dos municípios — socioeconômico, educacional e territorial —, com a taxa de alfabetização
usada **só depois**, para descrever cada grupo. Agrupar pelo resultado tornaria tautológica
a conclusão de que contextos parecidos têm resultados parecidos.

O número de grupos foi escolhido pela silhueta, máxima em k = 4 — mas baixa, 0,188, e a
curva de inércia não tem cotovelo. Os contextos municipais formam um contínuo, não grupos
nítidos: os quatro grupos servem como tipologia, não como separação natural.

| Grupo | Municípios | Alunos | Alfabetização | Perfil |
|---|---:|---:|---:|---|
| 1 | 335 | 96.769 | 49,2% | Rural isolado, menor infraestrutura e densidade |
| 2 | 1.783 | 269.329 | 59,2% | Menor PIB e menor alfabetização adulta; mais tempo integral |
| 3 | 1.252 | 1.091.654 | 59,5% | Urbano, maior PIB, alfabetização adulta e infraestrutura |
| 4 | 2.080 | 153.002 | 71,9% | Interior estruturado, turmas menores |

Taxa ponderada pelos alunos, sem o RS.

**O grupo urbano mais rico alfabetiza menos que o interior estruturado.** O grupo 3 tem o
melhor contexto em quase tudo — PIB per capita mediano de R$ 48 mil, 95,4% de alfabetização
adulta — e chega a 59,5%. O grupo 4, com contexto um pouco inferior, alcança 71,9%. São doze
pontos contra o contexto, no grupo que concentra 1,09 milhão dos 1,6 milhão de alunos. A
capital paulista, maior município do grupo 3, alfabetiza 48,5%.

**O grupo do Nordeste empata com as grandes cidades com o pior contexto.** O grupo 2 tem o
menor PIB per capita mediano, R$ 14,9 mil, e a menor alfabetização adulta, 79,4% — e chega
aos mesmos 59,2% do grupo urbano. É também o grupo com mais tempo integral, 20,6% das
matrículas.

| Região | Grupo 1 | Grupo 2 | Grupo 3 | Grupo 4 |
|---|---:|---:|---:|---:|
| Centro-Oeste | 11% | 4% | 22% | 63% |
| Nordeste | 4% | 84% | 9% | 2% |
| Norte | 43% | 25% | 9% | 23% |
| Sudeste | 0% | 9% | 39% | 51% |
| Sul | 1% | 0% | 28% | 71% |

Sul, Sudeste e Centro-Oeste se distribuem pelos mesmos grupos e têm padrões semelhantes. O
Nordeste é o mais homogêneo, com 84% dos municípios num grupo só. O Norte é o mais
heterogêneo, espalhado por três grupos.

**Leitura causal, não.** O grupo 4 tem as menores turmas — 18,2 alunos, contra 22,2 no grupo
3 —, e é tentador atribuir a diferença a isso. O agrupamento mostra associação; a explicação
exigiria outro desenho de estudo. Projeção em
[`images/clusters_pca.png`](images/clusters_pca.png) e escolha de k em
[`images/clusters_silhueta.png`](images/clusters_silhueta.png).

---

## 10. Limitações do projeto

**Sem features no grão da escola.** O `id_escola` de `fato_aluno` não é o código INEP: os
42.497 valores começam todos com `60`, prefixo que não corresponde a nenhuma UF. É
identificador sintético, atribuído pela avaliação para impedir reidentificação de
crianças em escolas pequenas — o join com `fato_escola` retorna **zero linhas**,
verificado.

A limitação é real e reduz o poder do modelo: variação entre escolas do mesmo município
costuma explicar bastante em educação. Mas a causa é legítima, e o projeto respeitou a
proteção de privacidade em vez de contorná-la. O contexto escolar entra agregado por
município, em `mun_indice_infraestrutura` e `mun_alunos_por_docente`.

**Fontes alternativas foram buscadas e descartadas.** O Indicador de Nível Socioeconômico
(INSE) do INEP, construído a partir do questionário do Saeb, seria a candidata ideal, mas
está no grão da escola e não pode ser ligado aos alunos avaliados. Também foi testado se o
código sintético da escola se mantinha entre 2023 e 2024, o que permitiria usar o resultado
da escola no ano anterior: das 36.462 escolas com o mesmo código nos dois anos, **35.597
estavam em municípios diferentes** — o código é regenerado a cada ano, e a coincidência é
de numeração, não de escola. A anonimização decorre da política do INEP de proteção de
dados, em observância à LGPD.

**Cobertura temporal de dois anos.** A avaliação cobre 2023 e 2024. Dois pontos sustentam
comparação, não tendência.

**Defasagem das fontes econômicas.** O PIB total é de 2023 e a composição setorial de
2021, contra a avaliação de 2024. Riqueza municipal muda devagar, mas a defasagem existe.

**Duas UFs ausentes.** A base cobre 25 UFs. O Distrito Federal não tem rede municipal de
ensino, e a Gold só traz a rede municipal; Roraima não tem nenhum aluno que passe pelos
filtros da Gold em 2024, e a causa na fonte não foi investigada.

**Queda do Rio Grande do Sul em 2024.** Documentada na Fase 2: 89,6% dos municípios gaúchos
com queda, mediana de −19,8 pontos, deslocamento uniforme. Duas explicações são compatíveis com
os dados — alteração na origem da medição ou o efeito das enchentes de 2024, que suspenderam
aulas por semanas —, e os dados disponíveis não separam uma da outra. As 61.053 linhas do estado
ficam no treino, por decisão do grupo, e fora das avaliações e da interpretação (seções 7 e 8).

**Pseudo-replicação das variáveis municipais.** Tamanho efetivo de amostra de 5.450 para
features municipais, contra 1,6 milhão de linhas.

**Um município com composição setorial anulada.** Faina, em Goiás, com valor adicionado
industrial negativo de −477 milhões em 2021 — provável ajuste contábil do IBGE. As
proporções ficam nulas em vez de inverter de sinal.

**Features quase todas municipais.** Das 24 features, nenhuma descreve o aluno: crianças
do mesmo município recebem a mesma previsão. Como 88% da variação do alvo está dentro dos
municípios, o ROC-AUC tem teto de 0,699 com qualquer conjunto de features municipais
(seção 7). O modelo responde bem à probabilidade para o aluno de um contexto; não distingue
crianças da mesma cidade.

**Razões do Censo Escolar com valores impossíveis.** Quatro variáveis herdadas da Fase 2
têm numerador e denominador de populações diferentes:

| Variável | Causa | Extensão |
|---|---|---|
| `mun_pct_rural`, `mun_pct_transporte` | Numerador conta matrículas de todas as etapas; denominador, só anos iniciais | Acima de 100% em 1.279 e 1.105 municípios; máximos de 552,6% e 467,0% |
| `mun_alunos_por_turma`, `mun_alunos_por_docente` | Alunos de anos iniciais em escolas que não registram nenhuma turma de anos iniciais — provavelmente turmas multisseriadas, comuns em áreas indígenas e remotas | Jordão (AC), com 44,7% de população indígena: 109,9 alunos por turma, contra mediana de 20,3. Em 123 das 140 linhas de escola do município, há alunos de anos iniciais e nenhuma turma de anos iniciais |

As variáveis continuam crescendo com o fenômeno que medem e funcionam como índice, não como
percentual. O pipeline aplica um teto no percentil 99 do treino (seção 5), e o efeito nas
métricas foi nulo — a amostragem de até 30 alunos por município já limitava o peso de casos
como Jordão. A correção na origem fica pendente.

---

## 11. Aplicação prática para políticas públicas

### Municípios que podem não atingir a meta

Resposta à pergunta de negócio do enunciado, em
[`src/evaluation/metas_municipios.py`](src/evaluation/metas_municipios.py). Como as features
são do município, a média das probabilidades dos alunos de uma cidade estima a taxa de
alfabetização daquela cidade — e, comparada com a meta pactuada, indica quem tende a ficar
abaixo. A estimativa exige probabilidades certas em valor, não só em ordem, e é por isso que a
calibração da seção 7 importa aqui.

**Backtest.** O modelo usa o resultado de 2023 para prever 2024, então a previsão pode ser
conferida contra o que de fato aconteceu. Em 889 municípios do teste — nunca vistos no
treino, sem RS, com meta publicada e ao menos 30 alunos avaliados:

| Medida | Valor |
|---|---:|
| Correlação entre taxa prevista e observada | 0,812 |
| Erro médio da taxa | 5,8 pontos |
| Municípios de fato abaixo da meta | 373 (42%) |
| Municípios previstos abaixo da meta | 336 |
| Precisão nos 10 de maior risco previsto | 90% |
| Precisão nos 50 de maior risco | 80% |
| Precisão nos 100 de maior risco | 80% |

A acurácia geral, de 68,6%, subestima o que importa. Perto da meta, poucos pontos invertem
a classificação, e é ali que se concentram os erros. Um gestor não usa a lista inteira:
prioriza o topo, onde o acerto dobra a taxa-base de 42%. O número de municípios previstos
abaixo da meta, 336, fica perto dos 373 reais — consequência direta de as probabilidades
estarem calibradas. Gráfico em
[`images/metas_backtest_municipios.png`](images/metas_backtest_municipios.png).

**Para metas futuras**, o mesmo procedimento se aplica com o contexto de 2024 para projetar
2025. O backtest é o que dá confiança para usá-lo adiante.

### Uso responsável

O modelo estima a probabilidade para o aluno de um contexto, não o destino de uma criança.
Usá-lo para rotular alunos seria um erro de interpretação, além de um risco de
estigmatização. O uso legítimo é **alocar apoio a territórios e redes**, onde a estimativa é
calibrada e o ranking de municípios tem precisão demonstrada.

---

## 12. Possíveis evoluções futuras

Candidatos já identificados:

- Séries temporais, quando houver mais edições da avaliação
- **MUNIC — Pesquisa de Informações Básicas Municipais** (IBGE): capacidade institucional
  da prefeitura. Avaliada e adiada por custo de exploração — 306 colunas e edições de
  anos distintos
- **INSE e indicadores no grão da escola**, se a anonimização permitir ligá-los aos alunos
  avaliados — é o único caminho identificado para romper o teto de 0,699 de ROC-AUC
- **Correção na origem das razões do Censo Escolar**, com numerador e denominador da mesma
  etapa
- **Limiar escolhido por aluno**, ponderando cada município pelo número de alunos na escolha
  do limiar, para que o recall-alvo valha na mesma unidade em que o teste é reportado
- Reaprendizado periódico e publicação do modelo como serviço

---

## 13. Como executar

### Pré-requisitos

- Python 3.11
- **Terraform ≥ 1.5** — a infraestrutura AWS é declarada em `infra/terraform/`
- Projeto no GCP com a API do BigQuery habilitada
- Conta AWS com acesso ao bucket do data lake da Fase 2

### Instalação

```bash
git clone https://github.com/amandacleite/tech-challenge-fase3.git
cd tech-challenge-fase3

python -m venv .venv
source .venv/Scripts/activate    # Linux e macOS: source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env             # preencher com os valores do seu ambiente
```

### Variáveis de ambiente

| Variável | Descrição |
|---|---|
| `GCP_PROJECT_ID` | Projeto GCP que fatura as consultas ao BigQuery |
| `AWS_REGION` | Região do bucket — `us-east-1` no AWS Academy |
| `AWS_BUCKET` | Bucket do data lake, compartilhado com a Fase 2 |
| `PIPELINE_ENV` | `dev` ou `prod` |

> ⚠️ **Nenhuma credencial vai para o Git.** O `.env` está no `.gitignore`; o
> `.env.example` traz apenas os nomes das variáveis.

### Configuração da AWS

No AWS Academy, as credenciais expiram a cada sessão do laboratório. Copie o bloco de
**AWS Details → AWS CLI → Show**:

```bash
aws configure
```

Em ambientes AWS Academy configure também:

```bash
aws configure set aws_session_token <TOKEN>
```

Validação:

```bash
aws sts get-caller-identity
aws s3 ls s3://<bucket>/
```

O segundo comando precisa listar `bronze/`, `silver/` e `gold/` — as camadas da Fase 2,
que esta fase consome. Se o bucket não existir, o laboratório foi reiniciado e a Fase 2
precisa ser reconstruída antes.

### Configuração do Google Cloud

Application Default Credentials, sem arquivo de chave permanente:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project SEU_PROJETO
```

Validação:

```bash
python src/ingestion/buscar_dataset.py censo
```

Deve listar os datasets do IBGE e do INEP disponíveis na Base dos Dados.

### Terraform

Instale o binário, caso não tenha:

```bash
winget install HashiCorp.Terraform   # Windows
brew install terraform               # macOS
```

Feche e reabra o terminal depois de instalar — o PATH só é lido na inicialização. Confirme
com `terraform version`.

**Não é necessário `terraform.tfvars`.** O ARN da role é montado em tempo de execução a
partir de `data.aws_caller_identity`, e os demais valores têm default em `variables.tf`.
Para usar outro bucket ou prefixo, passe `-var` na linha de comando — **o default é o
bucket do autor original, não o seu**; use sempre `-var="bucket=<seu-bucket>"` (e a mesma
variável `BUCKET=<seu-bucket>` ao rodar os scripts de `infra/` e `scripts/consultar.sh`).

Se o bucket for trocado **depois** que o crawler já existir, o Terraform não consegue
atualizar o `s3_target` in-place (o recrawl `CRAWL_NEW_FOLDERS_ONLY` o torna imutável) —
use `terraform apply -replace="aws_glue_crawler.bronze"` nesse caso.

> ⚠️ O `terraform.tfstate` e o `.terraform/` estão no `.gitignore` de
> `infra/terraform/`. O *state* descreve a infraestrutura e o ID da conta.

No AWS Academy não é possível criar roles IAM — a `LabRole` já vem provisionada.

### Execução

**1. Sincronizar as camadas da Fase 2** — baixa as tabelas que a Gold desta fase consome:

```bash
python src/ingestion/sincronizar_fase2.py --listar    # confere o que existe
python src/ingestion/sincronizar_fase2.py
```

Validação:

```bash
ls data/raw/
cat reports/ingestao/manifesto.jsonl
```

**2. Ingerir as fontes socioeconômicas** — dry run antes, sem custo:

```bash
python src/ingestion/ingestao_censo_2022.py --medir
python src/ingestion/ingestao_censo_2022.py

python src/ingestion/ingestao_pib.py --medir
python src/ingestion/ingestao_pib.py
```

Validação:

```bash
aws s3 ls s3://<bucket>/fase3/bronze/ --recursive --region us-east-1
```

**3. Criar a infraestrutura** — 9 recursos:

```bash
cd infra/terraform
terraform init
terraform apply
cd ../..
```

**4. Executar o fluxo** — crawler e Gold encadeados:

```bash
bash infra/executar_workflow.sh
```

O script dispara o Workflow, acompanha as duas etapas e mostra a saída no S3 ao final.

Para depurar uma etapa isolada, sem rodar o fluxo:

```bash
bash infra/executar_crawler.sh
aws glue start-job-run --job-name fase3_job_gold --region us-east-1
```

Validação:

```bash
aws s3 ls s3://<bucket>/fase3/gold/features_aluno/ --recursive --region us-east-1
bash scripts/consultar.sh verificacao_features_aluno
```

**5. Baixar a Gold para uso local nos notebooks** — evita depender de sessão AWS/Athena a
cada execução do EDA:

```bash
aws s3 cp s3://<bucket>/fase3/gold/features_aluno/ data/sample/features_aluno/ --recursive
```

**6. Treinar e avaliar os modelos** — notebooks `03` a `06`, que gravam métricas e
probabilidades de teste em `reports/modelagem/`.

**7. Avaliação, interpretação e agrupamentos** — sobre as probabilidades gravadas pelos
notebooks; os módulos de experimento, otimização e interpretação treinam suas próprias versões
do modelo, na amostra de até 30 alunos por município:

```bash
python -m src.evaluation.experimento_class_weight  # por que os modelos não usam class_weight
python -m src.evaluation.calibracao                # calibração e teto dos dados
python -m src.evaluation.otimizacao                # curva da configuração final e validação x teste
python -m src.evaluation.metas_municipios          # backtest das metas por município
python -m src.evaluation.interpretacao             # SHAP com e sem histórico
python -m src.modeling.clusterizacao               # grupos de municípios por contexto
```

### Testes

Cinco testes conferem as decisões que sustentam a validação do modelo: nenhuma feature com
vazamento, split sem município repetido entre treino e teste, teto do pré-processamento
aprendido só no treino, limiar que atinge o recall-alvo e reprodutibilidade com a mesma
semente. Rodam sobre `data/sample/`, sem acesso à AWS, a partir da raiz do projeto:

```bash
pytest tests/ -v
```

### Consultas analíticas

As consultas ficam versionadas em `sql/<camada>/`, com comentários explicando as
restrições que cada camada impõe. Quem clonar o repositório e tiver acesso ao Catalog
reproduz os mesmos números.

```bash
bash scripts/consultar.sh                             # lista as da Gold
bash scripts/consultar.sh verificacao_features_aluno  # executa uma
bash scripts/consultar.sh --todas                     # executa todas

CAMADA=bronze bash scripts/consultar.sh particoes_ingestao
```

### Explorar outras fontes

```bash
python src/ingestion/buscar_dataset.py <termo>             # localiza o dataset
python src/ingestion/explorar_fonte.py <dataset>           # lista tabelas e volumes
python src/ingestion/explorar_fonte.py <dataset> <tabela>  # detalha as colunas
```

### Reprocessar uma carga passada

O histórico é preservado por partição:

```bash
ls data/raw/censo_2022_municipio/          # lista as particoes disponiveis
cat reports/ingestao/manifesto.jsonl       # consulta, volume e data de cada execucao
```

A função `writer.ultima_particao(fonte, tabela)` resolve a carga mais recente. Para outra
data, aponte diretamente para a partição desejada.

### Notebooks

Rodar na ordem: cada notebook usa decisões do anterior.

| Notebook | O que faz |
|---|---|
| [`01_eda`](notebooks/01_eda.ipynb) | O que os dados mostram sobre a alfabetização, e as hipóteses que isso gera para o modelo |
| [`02_selecao_features`](notebooks/02_selecao_features.ipynb) | Como cada coluna é tratada e quais entram no modelo; grava as decisões em `src/config/features_modelo.json` |
| [`03_modelagem_supervisionada`](notebooks/03_modelagem_supervisionada.ipynb) | Pipeline de pré-processamento e modelos de referência (baseline) |
| [`04_random_forest`](notebooks/04_random_forest.ipynb) | Random Forest: otimização, avaliação e interpretação |
| [`05_hist_gradient_boosting`](notebooks/05_hist_gradient_boosting.ipynb) | HistGradientBoosting, o modelo escolhido |
| [`06_arvore_decisao`](notebooks/06_arvore_decisao.ipynb) | Árvore de decisão rasa: regras que um gestor consegue ler |

As dependências de notebook (`jupyterlab`, `ipykernel`) já estão no `requirements.txt`
principal — não é necessário instalar nada à parte.

---

## 14. Estrutura do repositório

```
.
├── data/
│   ├── raw/          # Bronze — <fonte>_<tabela>/dt_ingestao=... (NAO versionada)
│   ├── analytics/    # dataset de modelagem (NAO versionado)
│   └── sample/       # amostra estratificada (VERSIONADA)
├── images/           # figuras geradas, para o README
├── infra/
│   ├── terraform/    # databases, crawler, job, tabela e workflow
│   ├── executar_workflow.sh
│   └── executar_crawler.sh
├── notebooks/        # EDA, modelagem e interpretabilidade
├── reports/
│   ├── ingestao/     # manifesto de execucoes (VERSIONADO)
│   └── modelagem/    # metricas dos modelos (VERSIONADO)
├── scripts/
│   └── consultar.sh  # executor de consultas no Athena
├── sql/
│   ├── bronze/       # verificacao do historico de ingestao
│   └── gold/         # verificacao e exploracao do dataset
├── src/
│   ├── config/       # settings e features_modelo.json
│   ├── ingestion/    # busca, exploracao, extracao e sincronizacao
│   ├── preprocessing/# pipeline de transformacao para ML
│   ├── transformation/ # Glue Job da Gold
│   ├── modeling/     # treino, otimizacao, selecao e clusterizacao
│   ├── evaluation/   # metricas, SHAP, calibracao, metas, interpretacao e otimizacao
│   └── visualization/# graficos reutilizaveis
├── tests/            # testes das decisoes de modelagem
├── requirements.txt
├── DICIONARIO.md     # dicionario de dados
└── README.md
```

**`data/sample/` é a exceção versionada**, de propósito: torna os notebooks executáveis
sem depender do S3, que expira junto com a sessão do laboratório.

---

## 15. Fluxo de trabalho Git

Nada é commitado direto na `main`. Toda branch entra por Pull Request.

### Branches

| Prefixo | Uso |
|---|---|
| `feature/` | Funcionalidade nova |
| `fix/` | Correção |
| `docs/` | Documentação |
| `chore/` | Configuração e manutenção |

### Padrão de commits

[Conventional Commits](https://www.conventionalcommits.org/pt-br/), no imperativo e em
minúsculas:

```
feat: ingere censo 2022 e pib municipal com particao por data
fix: separa ano do pib total do ano da composicao setorial
docs: documenta tratamento de vazamento de dados
```

### Pull Requests

A descrição explica **o que** foi feito, **por que** dessa forma, **como validar** e **o
que ficou de fora**. O PR é o registro da evolução do projeto — descrição vaga hoje é
contexto perdido amanhã.

---

## 16. Evidências de execução

| Evidência | Local | Status |
|---|---|---|
| Manifesto de ingestão | [`reports/ingestao/manifesto.jsonl`](reports/ingestao/manifesto.jsonl) | ✅ |
| Histórico de ingestão consultável por data | [`images/ConsultaAthenaCenso_2022_municipio.png`](images/ConsultaAthenaCenso_2022_municipio.png) | ✅ |
| Dados da Bronze no Athena | [`images/ConsultaAthenaCenso_2022_municipio_dados.png`](images/ConsultaAthenaCenso_2022_municipio_dados.png) | ✅ |
| Verificação de `features_aluno` no Athena | [`images/ConsultaAthenaFase3_gold.png`](images/ConsultaAthenaFase3_gold.png) | ✅ |
| Print — `terraform apply` | [`images/TerraformApply.png`](images/TerraformApply.png) | ✅ |
| Print — crawler com partição reconhecida | [`images/crawler.png`](images/crawler.png) | ✅ |
| Print — Glue Job da Gold concluído | [`images/GlueWorkflowConcluido.png`](images/GlueWorkflowConcluido.png) | ✅ |
| Notebook de EDA com saídas | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | ✅ |
| Notebooks de modelagem | [`notebooks/03`](notebooks/03_modelagem_supervisionada.ipynb) a [`06`](notebooks/06_arvore_decisao.ipynb) | ✅ |
| Matriz de confusão e curva ROC | [`images/hgb_avaliacao_teste.png`](images/hgb_avaliacao_teste.png) | ✅ |
| Gráficos SHAP | [`images/hgb_shap_summary.png`](images/hgb_shap_summary.png) | ✅ |
| Experimento com e sem `class_weight` | [`images/experimento_class_weight.png`](images/experimento_class_weight.png) | ✅ |
| Calibração das probabilidades | [`images/calibracao_teste.png`](images/calibracao_teste.png) | ✅ |
| Curva da configuração escolhida | [`images/hgb_curva_iteracoes_final.png`](images/hgb_curva_iteracoes_final.png) | ✅ |
| Backtest das metas por município | [`images/metas_backtest_municipios.png`](images/metas_backtest_municipios.png) | ✅ |
| Agrupamento de municípios | [`images/clusters_pca.png`](images/clusters_pca.png) | ✅ |
| Fatores com e sem histórico | [`images/interpretacao_comparacao.png`](images/interpretacao_comparacao.png) | ✅ |
| Dicionário de dados | [`DICIONARIO.md`](DICIONARIO.md) | ✅ |
| Vídeo executivo (até 5 min) | — | ⏳ |

A primeira delas é a que comprova a correção apontada na avaliação da Fase 2: a partição
por data de ingestão é reconhecida pelo Glue e consultável no Athena, então cargas
anteriores coexistem em vez de serem sobrescritas.

---

## 17. Roadmap e status

| Semana | Período | Entregável | Status |
|---|---|---|---|
| S0 | 11–13/set | Repositório, estrutura, dependências | ✅ |
| S1 | 14–20/set | Ingestão socioeconômica e catalogação | ✅ |
| S1 | 14–20/set | Gold no grão do aluno, com vazamento tratado | ✅ |
| S2 | 21–27/set | EDA | ✅ |
| S3 | 28/set–04/out | Pré-processamento e baseline | ✅ |
| S4 | 05–11/out | Modelagem — algoritmos comparados | ✅ |
| S5 | 12–18/out | Otimização de hiperparâmetros | ✅ |
| S6 | 19–25/out | Avaliação, calibração e interpretabilidade | ✅ |
| S7 | 26–30/out | Clusterização e perguntas de negócio | ✅ |
| S7 | 26–30/out | Documentação e vídeo executivo | ✅ |
| — | 31/out–03/nov | Revisão final e entrega | ✅ |

---

## 18. Equipe

| Nome | GitHub |
|---|---|
| Amanda Leite | [@amandacleite](https://github.com/amandacleite) |
| Antonio Lima | [@AntoniLima](https://github.com/AntoniLima) |
| Joviniano Lira | [@LiraJoviniano](https://github.com/LiraJoviniano) |
| Luiza Cunha | [@luizafcunha](https://github.com/luizafcunha) |
| Vinicius Moutinho | [@ViniciusMoutinhoDev](https://github.com/ViniciusMoutinhoDev) |

---

## 19. Licença

Distribuído sob a licença MIT.

Os dados utilizados são públicos, produzidos pelo INEP e pelo IBGE e disponibilizados
pela plataforma Base dos Dados, sujeitos aos termos de uso originais de cada fonte.
