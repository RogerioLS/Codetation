import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import sklearn.metrics as metrics

df_train = pd.read_csv('train.csv', sep="," , encoding="UTF8" )
df_test2 = pd.read_csv('test.csv', sep="," , encoding="UTF8" )

df_train1 = pd.read_csv('train.csv', sep="," , encoding="UTF8", usecols = [
    'NU_NOTA_MT',   
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_REDACAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP3',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_IDADE',
    'TP_COR_RACA',
    'TP_ESCOLA'] )
df_test1 = pd.read_csv('test.csv', sep="," , encoding="UTF8" , usecols = [
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_REDACAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP3',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_IDADE',
    'TP_COR_RACA',
    'TP_ESCOLA'])

#verifica o que foi importado
df_train1.head()
df_test1.head()

#vendo max, min, mean e etc!
calculo = df_train1.describe()

#Irei filtrar features de correlações para ver se o modelo vai ficar legal
melhores = df_train1.corr()
features = melhores[(melhores['NU_NOTA_MT'] <= 0.1) | (melhores['NU_NOTA_MT'] >= 0.1)
        & (melhores['NU_NOTA_MT'] < 1.0)]['NU_NOTA_MT']
features

# grafico que vai desmonstrar a relacao das colunas
features_list = features.index.to_list()
corr = df_train1[features_list].corr()
ax = plt.subplots(figsize=(10, 11))
sns.heatmap(corr,  annot=True, annot_kws={"size": 10},cmap="YlGnBu")

# contagem dos nao nulos
df_train1.count()

features_list.append('NU_NOTA_MT')

#excluindo valores null's, mas nesse caso vou utilizar as medias e fazer a atribuicao
#nas linhas que estão com NAN
df_train1 = df_train1.loc[(df_train1['NU_NOTA_CN'].notnull())]

# contagem dos nao nulos
df_test1.count()

# calculando a media fazendo de dois jeitos 
# 1 
df_train1['NU_NOTA_CN'].fillna(df_train1['NU_NOTA_CN'].mean(), inplace=True)
df_test1['NU_NOTA_CN'].fillna(df_test1['NU_NOTA_CN'].mean(), inplace=True)

#2 bom sem duvidas esse é a melhor opcao até mesmo porque se focemos 
# aderir a primeira iriamos fazer uma a uma e isso levaria tempo 
df1_train = df_train1.copy()
df1_test = df_test1.copy()

def zerandoColunas(data):
    for item in features_list:
        data[item] = data[item].fillna(0)
    return data
df1_train = zerandoColunas(df1_train)
df1_test = df1_test.fillna(0)

# Agora vamos atribuir as medias das colunas
df2_train = df_train1.copy()
df2_test = df_test1.copy()

def mediaColunas(data):
    for item in features_list:
        data[item] = data[item].fillna(data[item].mean())
    return data

df2_train = mediaColunas(df2_train)
df2_test = df1_test.fillna(df2_test.mean())

# Criando dataset de respostas
df_result = pd.DataFrame()

# Adicionando número de inscrição a ser salvo no arquivo de resposta final
df_result['NU_INSCRICAO'] = df_test2['NU_INSCRICAO']

# Removendo a variável target
features_list.remove('NU_NOTA_MT')


x_train = df1_train[features_list]
y_train = df1_train['NU_NOTA_MT']
x_test = df1_test[features_list]

pipe_RFR = Pipeline([('scaler',  StandardScaler()),
            ('RandomForestRegressor', RandomForestRegressor())])

CV_pipe_RFR = RandomizedSearchCV(estimator = pipe_RFR, param_distributions = {},
                                 cv = 5,return_train_score=True, verbose=0)

CV_pipe_RFR.fit(x_train, y_train)
ypred2 = CV_pipe_RFR.predict(x_test)

df_result['NU_NOTA_MT'] = np.around(ypred2,2)

df_result.to_csv('answer2.csv', index=False, header=True)













