import cv2
import os 
import time
from moviepy.editor import *

if __name__ == '__main__':

#configurações
    
    #"""
    
    

    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/face.mp4"

    for nova_taxa in [25,20,15,10]:

        new_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/FPS/Video_face_"+str(nova_taxa)+"_FPS"
        
        #Método 3
        start = time.perf_counter()
        # Carregue o vídeo
        clip = VideoFileClip(file_name)

        # Escreva o novo vídeo com a taxa de fps desejada (por exemplo, 30 fps)
        clip.write_videofile(new_name+".mp4", fps=nova_taxa)
        novo_video = cv2.VideoCapture(new_name+".mp4")
        fps = int(novo_video.get(cv2.CAP_PROP_FPS))
        end = time.perf_counter()
        tempo = end-start
        print(f'Tempo gasto Metodo 2: {tempo}')
        print(f'FPS estimado: {fps}')
            #"""
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Final_cut/Natural/Lento/5.mp4"
    vid = cv2.VideoCapture(file_name)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(fps)