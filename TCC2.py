import numpy as np
import os 
from scipy import signal
import skimage
from skimage import data,io
import cv2 
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,filtfilt,sosfilt
import mediapipe as mp
from scipy import fft
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont, ExifTags
from sklearn.linear_model import LinearRegression
import pywt
from math import floor
import pandas as pd
from numba import njit
import concurrent.futures
import joblib
import concurrent.futures
import pf_lib as tl
import pywt
import time





if __name__ == '__main__':

        eC = 'lab'
        roi = 'G'
        filt = 'DA'

        #configurações
        begin = time.perf_counter()
        level = 5
        fb = 0.833333
        #fb = 0.4
        fh = 3
        #fh = 2.1666667
        #fh = 1.666667

        selector =7

        '''
        test_list = [0,4,2,3,4,5,6,7,8,9,
                     0,1,2,3,4,5,6,7,8,9,
                     0,15,2,3,4,5,6,7,8,9]
        #'''

        test_list = [13,14,12,0,3,15,16,17,3,4,1,7,16,17,18]
        #test_list = [0,9,1,5,3,8]
        #test_list = [0,7,1,5,3,8]
        #test_list = [0,17,1,5,3,10]
        #test_list = [0,3,7,17,1,4,8,5,3,6,11,10]
        #test_list = [0,0,1,2,3,0,0,1,2,3]
        #name_arq = ["5","10","15","20","25","face","10","15","20","25"]
        exp = test_list[selector]
        name_arq = str(exp+1)

        
        """
        if selector <1:       
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Natural/Lento/"+name_arq[selector]+".mp4";real_data = [[75,97]];alpha_test = 5
                #print(file_name)
        elif selector <5:       
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/FPS/Video_5_"+name_arq[selector]+"_FPS.mp4";real_data = [[75,97],[75,97],[75,97],[75,97]];alpha_test = 5
        elif selector <6:       
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/"+name_arq[selector]+".mp4";real_data = [[54,96]];alpha_test = 5
        elif selector <10:       
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/FPS/Video_face_"+name_arq[selector]+"_FPS.mp4";real_data = [[54,96],[54,96],[54,96],[54,96]];alpha_test = 5
        
        #file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/treino/Luz1/4.mp4"
        #file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/treino/Luz1/Acelerado/4.mp4" 
        #file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz3/Lento/7.mp4"
        #file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/face.mp4"
        #"""
        if selector <3:       
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz1/Lento/"+name_arq+".mp4";real_data = [[75,96],[77,97],[77,96],[83,97],[73,96],[103,96],[82,97],[91,97],[89,97],[93,96],[92,96],[88,97],[76,97],[71,96],[79,96]];alpha_test = 0.25 
                
        elif selector <5:
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz1/Acelerado/"+name_arq+".mp4";real_data = [[111,96],[96,96],[107,96],[150,96]];alpha_test = 100
                
        elif selector <8:
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz2/Lento/"+name_arq+".mp4";real_data = [[77,96],[78,97],[79,96],[79,96],[75,97],[76,97],[77,96],[77,97],[83,97],[80,96],[88,97],[78,96],[77,96],[83,97],[80,96],[80,97],[77,96],[69,96]];alpha_test = 5
                
        elif selector <10:
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz2/Acelerado/"+name_arq+".mp4";real_data = [[152,96],[118,97],[155,96],[107,96],[98,96],[110,97]];alpha_test = 5

        elif selector <12:
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz3/Acelerado/"+name_arq+".mp4";real_data = [[152,97],[118,97],[111,97],[106,96],[98,96],[119,97],[110,97],[150,96]];alpha_test = 3
                
        elif selector <15:
                file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Luz3/Lento/"+name_arq+".mp4";real_data = [[81,96],[88,96],[81,96],[85,96],[81,97],[78,96],[81,96],[78,96],[78,97],[77,96],[75,96],[76,96],[79,97],[77,97],[77,97],[78,96],[76,96],[78,97],[75,96],[69,96]];alpha_test = 3
        
        #"""
        #alpha_test =3
        #file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/tcc2.mp4"

        #Carregando video
        with concurrent.futures.ThreadPoolExecutor() as executor:
                funcao1 = executor.submit(tl.face_detection, file_name,1)
                funcao2 = executor.submit(tl.extract_video_info, file_name)



        indx = funcao1.result()
        rgb_vid, fps = funcao2.result()

        quarto = int(np.ceil(rgb_vid.shape[0]/4))-1

        
        with concurrent.futures.ThreadPoolExecutor() as executor:
                funcao1 = executor.submit(tl.wv_video_cc, rgb_vid[0:quarto],level)
                funcao2 = executor.submit(tl.wv_video_cc, rgb_vid[quarto:2*quarto+1],level)
                funcao3 = executor.submit(tl.wv_video_cc, rgb_vid[2*quarto+1:3*quarto+1],level)
                funcao4 = executor.submit(tl.wv_video_cc, rgb_vid[3*quarto+1:],level)

        wv_video = funcao1.result() + funcao2.result() + funcao3.result() + funcao4.result()


        #Filtragem

        ft_video = tl.fbp_wv2(wv_video,fb,fh,fps)

        if(len(ft_video)!= rgb_vid.shape[0]):
                ft_video = ft_video[0:-1]



        amp_video = tl.colorMagnification_wv(ft_video,alpha_test,1)


        #Reconstrução

        test = (len(ft_video),len(ft_video[0][0][0])*2**level,len(ft_video[0][0][0][0])*2**level,len(ft_video[0]))

        if test == rgb_vid.shape:


                final_video = tl.final_video_wv_cc(rgb_vid,wv_video,level) #Se desejar apenas processar o dado
                
        else:


                final_video = tl.final_video_wv_cc_nr(rgb_vid,amp_video,level) #Se desejar apenas processar o dado
#Calculo
        buffer1 = tl.extract_points(indx,final_video)


        signal1 = tl.avg_signal(buffer1)



        with concurrent.futures.ThreadPoolExecutor() as executor:
                func1 = executor.submit(tl.filtro_kerman, signal1[0])
                func2 = executor.submit(tl.filtro_kerman, signal1[1])
                func3 = executor.submit(tl.filtro_kerman, signal1[2])

        signalr1 = func1.result()
        signalg1 = func2.result()
        signalb1 = func3.result()
        signalf1 = [signalr1,signalg1,signalb1]

        Gsig = tl.hamming(signal1)

        with concurrent.futures.ThreadPoolExecutor() as executor:

                func5 = executor.submit(tl.bpm_1, Gsig,fps,fb,fh)
                func8 = executor.submit(tl.sat_oxi,signalf1)
                func7 = executor.submit(tl.sat_oxi2, signalf1)




        freq, fft_res,batimentos_t5= func5.result()
        
        sat_t2 = func7.result()
        sat_t3= func8.result()







        
        real_bpm = real_data[exp][0]
        real_sat = real_data[exp][1]

        #real_bpm =75
        #real_sat =97

        bpm,sat = tl.valor_corrigido(batimentos_t5,sat_t3,'G')
        print(bpm,sat)  

        print("BPM Obtido:",batimentos_t5,"\nBPM Real:",real_bpm,"\nSaturação Obtida:",sat_t2,"Saturação Real:",real_sat)
        final = time.perf_counter()
        tempo = final-begin
        print(f'Demorou:{tempo}')

        #tl.criar_imagem_com_texto(rgb_vid[-1],round(sat),bpm,"TCC2_img1")


        
        file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/FPS/valid.csv"

        if os.path.exists(file_name):
        
                df_resultados_existente = pd.read_csv(file_name)
                df_novos_resultados = [bpm,real_bpm,sat,real_sat]
                df_resultados_existente.loc[df_resultados_existente.index[-1] + 1] = df_novos_resultados

                print(df_resultados_existente.to_string(index=False))

                df_resultados_existente.to_csv(file_name, index=False)
                
        else:
                df_primeiros_resultados = pd.DataFrame({'BPM': [bpm],
                                                        'BPMReal': [real_bpm],
                                                        'SO2': [sat],
                                                        'SOreal': [real_sat]})
                
                print(df_primeiros_resultados.to_string(index=False))
                df_primeiros_resultados.to_csv(file_name, index=False)
        print(fps)

        

