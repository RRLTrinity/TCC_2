import pf_lib as tl
import numpy as np
import time
import concurrent.futures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import os
import joblib

def error_calc(Medida,real):
    return(np.abs(Medida/real-1)*100)

def select_SL(file_name):

    #Carregando A1
    
    df_resultados_existente = pd.read_csv(file_name)
    #print(df_sat_resultados_existente.to_string(index=False))
    col = ['M1BPM','M1SO1','M1SO2','M2BPM','M2SO1','M2SO2','M3BPM','M3SO1','M3SO2','M4BPM','M4SO1','M4SO2']
    Min_error_bpm = 4
    Min_error_sat = 0.3
    dp = []
    med = []
    for i in range(0,4):
        results = [df_resultados_existente[col[3*i]],df_resultados_existente[col[1+3*i]],df_resultados_existente[col[2+3*i]]]
        for j in range(0,3):
            med.append(np.mean(results[j]))
            dp.append(np.std(results[j]))

    par_list = {'BPMM':[],"SATM":[],'BPMDP':[],"SATDP":[],'BPMMT':[],"SATMT":[]}
    sat_i = 0
    for par in ['BPM',"SAT",'SAT']:
        
        
        if par =="BPM":
            for i in range(4):
                
                if med[3*i] <= Min_error_bpm:
                    

                    par_list["BPMDP"].append(dp[3*i])
                    par_list["BPMMT"].append(col[3*i])
                    par_list["BPMM"].append(med[3*i])
            print("BPM\nMédia: ", par_list['BPMM'][par_list['BPMDP'].index(min(par_list['BPMDP']))],"\nDP: ",min(par_list['BPMDP']),"\n")
        else:
            sat_i +=1
            for i in range(4):
                
                if med[3*i+sat_i] <= Min_error_sat:
                    

                    par_list["SATDP"].append(dp[3*i+sat_i])
                    par_list["SATMT"].append(col[3*i+sat_i])
                    par_list["SATM"].append(med[3*i+sat_i])
             
    
    
    print("SAT\nMédia: ", par_list['SATM'][par_list['SATDP'].index(min(par_list['SATDP']))],"\nDP: ",min(par_list['SATDP']),"\n")

    return[par_list['BPMMT'][par_list['BPMDP'].index(min(par_list['BPMDP']))],par_list['SATMT'][par_list['SATDP'].index(min(par_list['SATDP']))]]

def select_met(result,test):

    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/Treino3.csv"
    df_resultados_existente = pd.read_csv(file_name)

    match test:

        case 'L1':
            Treino = "Treino_luz1.csv"
            X_bpm = df_resultados_existente[['BPM']].iloc[0:10]
            y_bpm = df_resultados_existente['BPMReal'].iloc[0:10]

            X_sat1 = df_resultados_existente[['SO1']].iloc[0:10]
            X_sat2 = df_resultados_existente[['SO2']].iloc[0:10]
            y_sat = df_resultados_existente['SOreal'].iloc[0:10]

        case 'L2':
            Treino = "Treino_luz2.csv"
            X_bpm = df_resultados_existente[['BPM']].iloc[10:20]
            y_bpm = df_resultados_existente['BPMReal'].iloc[10:20]

            X_sat1 = df_resultados_existente[['SO1']].iloc[10:20]
            X_sat2 = df_resultados_existente[['SO2']].iloc[10:20]
            y_sat = df_resultados_existente['SOreal'].iloc[10:20]
        case 'L3':
            Treino = "Treino_luz3.csv"
            X_bpm = df_resultados_existente[['BPM']].iloc[20:30]
            y_bpm = df_resultados_existente['BPMReal'].iloc[20:30]

            X_sat1 = df_resultados_existente[['SO1']].iloc[20:30]
            X_sat2 = df_resultados_existente[['SO2']].iloc[20:30]
            y_sat = df_resultados_existente['SOreal'].iloc[20:30]
        case 'G':
            Treino = "Treino_geral.csv"
            X_bpm = df_resultados_existente[['BPM']].iloc[:]
            y_bpm = df_resultados_existente['BPMReal'].iloc[:]

            X_sat1 = df_resultados_existente[['SO1']].iloc[:]
            X_sat2 = df_resultados_existente[['SO2']].iloc[:]
            y_sat = df_resultados_existente['SOreal'].iloc[:]

    for resultado in result:
        print(resultado[0:2])
        match resultado[0:2]:

            case 'M1':
                modelo = LinearRegression()

            case 'M2':
                modelo = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            
            case 'M3':
                modelo = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

            case'M4':
                modelo = DecisionTreeRegressor()
        print(resultado[2:4])
        match resultado[2:4]:

            case 'BP':

                modelo.fit(X_bpm,y_bpm)
                joblib.dump(modelo, 'Resultados/Treino_final/modelo2_bpm_'+test)
                
            case 'SO':

                if resultado[4]=='1':
                    modelo.fit(X_sat1,y_sat)
                else:
                    modelo.fit(X_sat2,y_sat)              
                
                joblib.dump(modelo, 'Resultados/Treino_final/modelo2_sat_'+test)







def treino():      
    #nomedo arq
    name = ["Treino_luz1","Treino_geral"]

    #arquivo de resultados alimentação
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/Treino3.csv"
    file_name_teste = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/Teste_2.csv"
    file_name_result = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/Resultadof_geral2.csv"
    #file_name_resultg = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/Resultado_geral.csv"
    #Resultados Teste

    #Treino
    df_resultados_existente = pd.read_csv(file_name)

    X_bpm = df_resultados_existente[['BPM']].iloc[0:30]
    y_bpm = df_resultados_existente['BPMReal'].iloc[0:30]

    X_sat1 = df_resultados_existente[['SO1']].iloc[0:30]
    X_sat2 = df_resultados_existente[['SO2']].iloc[0:30]
    y_sat = df_resultados_existente['SOreal'].iloc[0:30]

    #Teste
    df_resultados_existente_teste = pd.read_csv(file_name_teste)

    bpm_test = df_resultados_existente_teste[['BPM']].iloc[0:12]
    Realidade_bpm = df_resultados_existente_teste[['BPMReal']].iloc[0:12]
    sat1_test = df_resultados_existente_teste[['SO1']].iloc[0:12]
    sat2_test = df_resultados_existente_teste[['SO2']].iloc[0:12]
    Realidade_sat = df_resultados_existente_teste[['SOreal']].iloc[0:12]

    #print(bpm_test['BPM'][0])
    #Regressão Linear

    modelo_1_bpm = LinearRegression()
    modelo_1_sat1 = LinearRegression()
    modelo_1_sat2 = LinearRegression()

    modelo_1_bpm.fit(X_bpm, y_bpm)
    modelo_1_sat1.fit(X_sat1, y_sat)
    modelo_1_sat2.fit(X_sat2, y_sat)

    #Regressão polinomial grau 2

    grau_polinomio = 2
    modelo_2_bpm = make_pipeline(PolynomialFeatures(degree=grau_polinomio), LinearRegression())
    modelo_2_sat1 = make_pipeline(PolynomialFeatures(degree=grau_polinomio), LinearRegression())
    modelo_2_sat2 = make_pipeline(PolynomialFeatures(degree=grau_polinomio), LinearRegression())

    modelo_2_bpm.fit(X_bpm, y_bpm)
    modelo_2_sat1.fit(X_sat1, y_sat)
    modelo_2_sat2.fit(X_sat2, y_sat)

    #Regressão polinomial Grau 3

    grau_polinomio = 3
    modelo_3_bpm = make_pipeline(PolynomialFeatures(degree=grau_polinomio), LinearRegression())
    modelo_3_sat1 = make_pipeline(PolynomialFeatures(degree=grau_polinomio), LinearRegression())
    modelo_3_sat2 = make_pipeline(PolynomialFeatures(degree=grau_polinomio), LinearRegression())

    modelo_3_bpm.fit(X_bpm, y_bpm)
    modelo_3_sat1.fit(X_sat1, y_sat)
    modelo_3_sat2.fit(X_sat2, y_sat)

    #Arvore de Decisões
    modelo_4_bpm = DecisionTreeRegressor()
    modelo_4_sat1 = DecisionTreeRegressor()
    modelo_4_sat2 = DecisionTreeRegressor()

    modelo_4_bpm.fit(X_bpm, y_bpm)
    modelo_4_sat1.fit(X_sat1, y_sat)
    modelo_4_sat2.fit(X_sat2, y_sat)


    for i in range(0,12):
        
        error_list = []

        novos_dados = pd.DataFrame({'BPM': [bpm_test['BPM'][i]]})
        previsao = modelo_1_bpm.predict(novos_dados)
        
        error_list.append(error_calc(previsao[0],Realidade_bpm['BPMReal'][i]))
        
        novos_dados = pd.DataFrame({'SO1': [sat1_test['SO1'][i]]})
        previsao = modelo_1_sat1.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))
        

        novos_dados = pd.DataFrame({'SO2': [sat2_test['SO2'][i]]})
        previsao = modelo_1_sat2.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))
        print("sat-",sat2_test['SO2'][i],previsao[0],Realidade_sat['SOreal'][i],np.abs(previsao[0]-Realidade_sat['SOreal'][i])/Realidade_sat['SOreal'][i]*100)

        novos_dados = pd.DataFrame({'BPM': [bpm_test['BPM'][i]]})
        previsao = modelo_2_bpm.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_bpm['BPMReal'][i]))

        novos_dados = pd.DataFrame({'SO1': [sat1_test['SO1'][i]]})
        previsao = modelo_2_sat1.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))

        novos_dados = pd.DataFrame({'SO2': [sat2_test['SO2'][i]]})
        previsao = modelo_2_sat2.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))

        novos_dados = pd.DataFrame({'BPM': [bpm_test['BPM'][i]]})
        previsao = modelo_3_bpm.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_bpm['BPMReal'][i]))

        novos_dados = pd.DataFrame({'SO1': [sat1_test['SO1'][i]]})
        previsao = modelo_3_sat1.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))

        novos_dados = pd.DataFrame({'SO2': [sat2_test['SO2'][i]]})
        previsao = modelo_3_sat2.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))

        novos_dados = pd.DataFrame({'BPM': [bpm_test['BPM'][i]]})
        previsao = modelo_4_bpm.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_bpm['BPMReal'][i]))
        print("bpm-",bpm_test['BPM'][i],previsao[0],Realidade_bpm['BPMReal'][i],np.abs(previsao[0]-Realidade_bpm['BPMReal'][i])/Realidade_bpm['BPMReal'][i]*100)

        novos_dados = pd.DataFrame({'SO1': [sat1_test['SO1'][i]]})
        previsao = modelo_4_sat1.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))

        novos_dados = pd.DataFrame({'SO2': [sat2_test['SO2'][i]]})
        previsao = modelo_4_sat2.predict(novos_dados)
        error_list.append(error_calc(previsao[0],Realidade_sat['SOreal'][i]))



        if os.path.exists(file_name_result):
                    
                df_resultados_existente = pd.read_csv(file_name_result)

                df_novos_resultados = [error_list[0],error_list[1],error_list[2],error_list[3],error_list[4],error_list[5],error_list[6],error_list[7],error_list[8],error_list[9],error_list[10],error_list[11]]
                df_resultados_existente.loc[df_resultados_existente.index[-1] + 1] = df_novos_resultados

                

                df_resultados_existente.to_csv(file_name_result, index=False)
                
        else:
                df_primeiros_resultados = pd.DataFrame({'M1BPM': [error_list[0]],
                                                        'M1SO1': [error_list[1]],
                                                        'M1SO2': [error_list[2]],
                                                        'M2BPM': [error_list[3]],
                                                        'M2SO1': [error_list[4]],
                                                        'M2SO2': [error_list[5]],
                                                        'M3BPM': [error_list[6]],
                                                        'M3SO1': [error_list[7]],
                                                        'M3SO2': [error_list[8]],
                                                        'M4BPM': [error_list[9]],
                                                        'M4SO1': [error_list[10]],
                                                        'M4SO2': [error_list[11]]})
                
                print(df_primeiros_resultados.to_string(index=False))
                df_primeiros_resultados.to_csv(file_name_result, index=False)


#treino()
result=select_SL("C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/Resultadof_geral2.csv")
print(result)
select_met(result,'G')

#bpm,sat = tl.valor_corrigido(110,97,'G')
#print(bpm,sat)