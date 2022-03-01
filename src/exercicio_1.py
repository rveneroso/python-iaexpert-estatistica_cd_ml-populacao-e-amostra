import pandas as pd
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Função desenvolvida pelo professor.
def amostragem_sistematica(dataset, amostras):
  # Define o intervalo que deve existir entre uma amostra selecionada e a próxima.
  intervalo = len(dataset) // amostras
  # Define o valor inicial de seed() de forma que os resultados aleatórios sejam sempre os mesmos.
  random.seed(1)
  # Escolhe a primeira amostra.
  inicio = random.randint(0, intervalo)
  # Escolhe as demais amostras tendo como base a primeira amostra e o intervalo entre as amostras.
  indices = np.arange(inicio, len(dataset), step = intervalo)
  # Gera um novo DataFrame a partir do DataFrame original informando quais são os índices dos elementos que devem ser incluídos no novo
  # DataFrame.
  amostra_sistematica = dataset.iloc[indices]
  return amostra_sistematica

# Função desenvolvida pelo professor.
def amostragem_agrupamento(dataset, numero_grupos):
  intervalo = len(dataset) / numero_grupos

  grupos = []
  id_grupo = 0
  contagem = 0
  for _ in dataset.iterrows():
    grupos.append(id_grupo)
    contagem += 1
    if contagem > intervalo:
      contagem = 0
      id_grupo += 1

  dataset['grupo'] = grupos
  random.seed(1)
  grupo_selecionado = random.randint(0, numero_grupos)
  return dataset[dataset['grupo'] == grupo_selecionado]

# Função desenvolvida pelo professor.
def amostragem_estratificada(dataset, percentual, campo):
  split = StratifiedShuffleSplit(test_size=percentual, random_state=1)
  for _, y in split.split(dataset, dataset[campo]):
    df_y = dataset.iloc[y]
  return df_y

# DataFrame composto de 2000 linhas com 5 colunas cada uma.
dataset = pd.read_csv('../data/credit_data.csv')

# Dividimos o tamanho do DataFrame pelo número de amostras. Resultado: 2
# print(len(dataset) / 1000)

# Amostragem sistemática. Queremos uma amostra de 100 registros o que corresponde a 50% do total dos dados.
# Obtendo o primeiro elemento da amostra.
random.seed(1)
# O primeiro elemento selecionado deve estar entre 0 e 2.
indice_inicial = random.randint(0, 2)
# Agora precisamos obter os demais elementos usando um step = 2.
indices_amostras = np.arange(indice_inicial, len(dataset), step = 2)
# Gera o novo DataFrame a partir dos índices obtidos acima
dataset_amostras = dataset.iloc[indices_amostras]
#print(dataset_amostras)
# Comparando minha implementação com a função desenvolvida pelo professor
dataset_amostras = amostragem_sistematica(dataset,1000)
#print(dataset_amostras)
# Ok. Resultados iguais.

# Amostragem por grupos.
# Definindo a quantidade de elementos em cada grupo sabendo que trabalharemos com 2 grupos. A escolha do número
# de grupos foi baseada no fato de que queremos 1000 registros na amostra e o total de registros do DataFrame
# é 2000. Portanto, 2000 / 1000 = 2
numero_elementos_por_grupo = len(dataset) // 2 # Teremos 1000 elementos em cada grupo

grupos = []
grupo = 0
contador = 0
# Percorre cada uma das linhas do DataFrame.
for i in dataset.iterrows():
    grupos.append(grupo)
    contador += 1
    if(contador>numero_elementos_por_grupo):
        contador = 0
        grupo += 1
# Confere a distribuição de elementos dentro da lista grupos
#print(np.unique(grupos, return_counts=True))
# Cria um novo atributo no DataFrame recebendo os valores presentes na lista grupos
dataset['grupo'] = grupos
# Escolhe um número aleatório entre os números de grupos criados anteriormente
random.seed(1)
grupo_selecionado = random.randint(0, max(grupos))
# Obtém a lista de elementos que pertencem ao grupo selecionado
dataset_amostras = dataset[dataset['grupo'] == grupo_selecionado]
print(dataset_amostras.shape)
# Comparando minha implementação com a função desenvolvida pelo professor.
dataset_amostras = amostragem_agrupamento(dataset, 2)
print(dataset_amostras.shape)
# Ok. Resultados iguais

# Amostragem estratificada.
print(np.unique(dataset['c#default'], return_counts=True))
# Queremos uma amostra de 1000 o que corresponde a 50% de toda a base presente no DataFrame.
split = StratifiedShuffleSplit(test_size=0.5)
# A função split() coloca 50% dos registros na variável x (treinamento) e 50% na variável y (testes). A divisão
# será feita com base nos valores presentes na coluna 'c#default''.
for x, y in split.split(dataset, dataset['c#default']):
  df_x = dataset.iloc[x]
  df_y = dataset.iloc[y]
# O DataFrame que nos interessa é df_y que contém os registros utilizados na amostragem
print(df_x.shape)
print(df_y.shape)
# A solução do exercício apresentada pelo professor termina aqui. Não há um segundo processo de estratificação
# como foi feito na explicação desse tipo de coleta de amostras.
# Então, verificando a distribuição do conjunto de testes com base no atributo 'c#default':
print(np.unique(df_y['c#default'], return_counts=True))
# Temos:
# 858 registros de pessoas que pagaram o empréstimo: c#default=0
# 142 registros de pessoas que não pagaram o empréstimo: c#default=1
# Comparando minha implementação com a função desenvolvida pelo professo
df_y = amostragem_estratificada(dataset,0.5,'c#default')
print(np.unique(df_y['c#default'], return_counts=True))
# Ok. Resultados iguais

# Amostragem de reservatório. Não refiz essa implementação pois na aula de implementação desse método
# não foi feito nada além de criar a função amostragem_reservatorio
