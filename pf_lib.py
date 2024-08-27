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
import copy

landmark_list = [103,67,109,10,338,297,332,104,69,108,151,337,299,333,280,425,411,427,50,205,207,187]
forehead_list = [103,67,109,10,338,297,332,104,69,108,151,337,299,333]
cheek_right_list = [280,425,411,427]
cheek_left_list = [50,205,207,187]
cheek_list = cheek_left_list+ cheek_right_list


def face_detection(file_name,n=1):
    vid = cv2.VideoCapture(file_name)
    indexes = []
    t = 0
    
    mp_face_mesh = mp.solutions.face_mesh
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
   
    drawing_specs = drawing.DrawingSpec(color = (100, 255, 0), thickness = 1)

    with mp_face_mesh.FaceMesh(
            max_num_faces = n,
            refine_landmarks = True,
            min_detection_confidence = 0.6,
            min_tracking_confidence = 0.6,
     
        ) as face_mesh:


        while(vid.isOpened()):
            read, image = vid.read()
            indexes.append([])
            

            

            #Verificando a leitura do arquivo
            if (not read):
                break
            M,N,_ = image.shape
            #face detection LandMarks
            results = face_mesh.process(image)
            face = mp_face_mesh.FACEMESH_FACE_OVAL
            
            for face_landmarks in results.multi_face_landmarks:
                





                

                drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = drawing_specs

                )
                for roi_landmark in range(0,len(landmark_list)):
                    indexx = face_landmarks.landmark[landmark_list[roi_landmark]].x*N
                    indexy = face_landmarks.landmark[landmark_list[roi_landmark]].y*M
                    image = np.array(image)
                    
                    

                    for i in range(int(indexx-9),int(indexx+10)):
                        for j in range(int(indexy-9),int(indexy+10)):
                            #image[j][i] = image[j][i]*0
                            #buffer[t].append(image[j][i])
                            if (j,i) not in indexes:
                                indexes[t].append((j,i))


            #cv2.imshow('image_window_name',image)
            t += 1
            #identificando interrupção
            if cv2.waitKey(10) & 0xFF == ord("0"):
                break

            
        
    vid.release()
    cv2.destroyAllWindows()
    
    return indexes



def extract_points(indexes, vid):
    #
    buffer = np.zeros((vid.shape[0],len(indexes[0]),3),dtype='float')
    
    for i in range(0,vid.shape[0]):
        frame = vid[i]
        for j in range(0,len(indexes[1])):
  
            buffer[i,j] = frame[indexes[i][j]]
            
            
    
    return buffer

def extract_points_isolated(indexes, vid):

    forehead = len(forehead_list)
    cheek_r = len(cheek_right_list)+forehead
    cheek_l = len(cheek_left_list)+cheek_r

    forehead_limit = forehead*19*19
    
    cheek_llimit = cheek_l*19*19
    
    cheek_rlimit = cheek_r*19*19
    
    
    buffer_fh = np.zeros((vid.shape[0],len(forehead_list)*19*19,3),dtype='float')
    np.shape(buffer_fh)
    buffer_chl = np.zeros((vid.shape[0],len(cheek_left_list)*19*19,3),dtype='float')
    buffer_chr = np.zeros((vid.shape[0],len(cheek_right_list)*19*19,3),dtype='float')
    for i in range(0,vid.shape[0]):
        frame = vid[i]
        for j in range(0,forehead_limit-1):
            
            buffer_fh[i,j] = frame[indexes[i][j]]
    for i in range(0,vid.shape[0]):
        frame = vid[i]
        for j in range(0,len(cheek_right_list)-1):
            
            buffer_chr[i,j] = frame[indexes[i][forehead_limit+j]]
    for i in range(0,vid.shape[0]):
        frame = vid[i]
        for j in range(0,len(cheek_left_list)-1):
            
            buffer_chl[i,j] = frame[indexes[i][j+cheek_rlimit]]           
            
    
    return buffer_fh,buffer_chl,buffer_chr


def extract_video_info(file_name):
    vid = cv2.VideoCapture(file_name)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    H, W = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frames_video = np.zeros((frame_count,H,W,3),dtype='float')
    n = 0

    while(vid.isOpened()):
        read, image = vid.read()
        
        if (not read):
            break
        else:
            frames_video[n] = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            n += 1

    return frames_video, fps
        


def gaussianPyrDown (image,levels):
    cI = image.copy()
    gp = [image]

    for i in range(0,levels):
        cI = cv2.pyrDown(cI)     
        gp.append(cI)
    return gp


def gp_video(frames_video,levels):
    
    for i in range(0,frames_video.shape[0]):
        frame = frames_video[i]
        pyr = gaussianPyrDown(frame,levels)
        if i == 0:
            
            gp_vid = np.zeros((frames_video.shape[0],pyr[-1].shape[0], pyr[-1].shape[1],3))
            
        
        gp_vid[i] = pyr[-1]
    return gp_vid


def gp_video_CC(frames_video,levels):
    
    for i in range(0,frames_video.shape[0]):
        frame = frames_video[i]
        
        #frame = frames_video[i]/255
        
        #yiq = skimage.color.rgb2yiq(frame)
        ycc = skimage.color.rgb2ycbcr(frame)
        #lab = skimage.color.rgb2lab(frame)
        #pyr = gaussianPyrDown(yiq,levels)
        pyr = gaussianPyrDown(ycc,levels)
        #pyr = gaussianPyrDown(lab,levels)
        if i == 0:
            
            gp_vid = np.zeros((frames_video.shape[0],pyr[-1].shape[0], pyr[-1].shape[1],3))
            
        
        gp_vid[i] = pyr[-1]
        
    return gp_vid

def wv_video_cc(video,level):
    wv_vid_coeffs = []
    for i in range(0,video.shape[0]):
        #frame = video[i]
        frame = video[i]/255
        #framecc = skimage.color.rgb2yiq(frame)
        #framecc = skimage.color.rgb2ycbcr(frame)
        framecc = skimage.color.rgb2lab(frame)
        #wv = pywt.dwt2(yiq,'db1')
        #wv = pywt.dwt2(ycc,'db1')
        #coeffs = [pywt.wavedec2(lab[ :, :, j], 'db1',level=level) for j in range(0,3)]
        coeffs = [[pywt.wavedec2(framecc[ :, :, j], 'db4',level=level)[0] for j in range(0,3)]]+[[[pywt.wavedec2(framecc[ :, :, j], 'db4',level=level)[l][k] for j in range(0,3)]for k in range(3)]for l in range(1,level+1)]
        
            
        wv_vid_coeffs.append(coeffs)
        
    return wv_vid_coeffs
 
def wv_video_cc2(video,level):
    wv_vid_coeffs = []

    for i in range(0,video.shape[0]):
        
        #frame = frames_video[i]
        frame = video[i]/255
        #ncm = skimage.color.rgb2yiq(frame)
        #ncm = skimage.color.rgb2ycbcr(frame)
        ncm = skimage.color.rgb2lab(frame)
        #wv = pywt.dwt2(yiq,'db1')
        #wv = pywt.dwt2(ycc,'db1')
        coeffsA = np.array([pywt.wavedec2(ncm[ :, :, j], 'db1',level=level)[0] for j in range(0,3)])
        coeffsD = [[np.moveaxis(np.array([pywt.wavedec2(ncm[ :, :, j], 'db1',level=level)[l][k] for j in range(0,3)]),0,-1) for k in range(3)] for l in range(1,level+1)]
        acoeffsA = np.moveaxis(coeffsA, (0), (-1))

        #acoeffsD = np.moveaxis(coeffsD, (0), (-1))    
        wv_vid_coeffs.append([acoeffsA,coeffsD])

        
    return np.array(wv_vid_coeffs)

def wv_video_cc3(video,level):
    wv_vid_coeffs = []

    for i in range(0,video.shape[0]):
        
        #frame = frames_video[i]
        frame = video[i]/255
        #ncm = skimage.color.rgb2yiq(frame)
        #ncm = skimage.color.rgb2ycbcr(frame)
        ncm = skimage.color.rgb2lab(frame)
        #wv = pywt.dwt2(yiq,'db1')
        #wv = pywt.dwt2(ycc,'db1')
        coeffsA = np.array([pywt.wavedec2(ncm[ :, :, j], 'db1',level=level)[0] for j in range(0,3)])
        coeffsD = [[np.moveaxis(np.array([pywt.wavedec2(ncm[ :, :, j], 'db1',level=level)[l][k] for j in range(0,3)]),0,-1) for k in range(3)] for l in range(1,level+1)]
        acoeffsA = np.moveaxis(coeffsA, (0), (-1))

        #acoeffsD = np.moveaxis(coeffsD, (0), (-1))    
        wv_vid_coeffs.append([acoeffsA,coeffsD])

        
    return wv_vid_coeffs

def res_img_gauss(image,levels):
    cI = image.copy()

    for i in range(0,levels):
        cI = cv2.pyrUp(cI)     
    return cI
def colorvideoMagnification(video,amp,atn):
    
    video[:,:,:,0] *= amp
    video[:,:,:,1] *= atn
    video[:,:,:,2] *= atn

    return video


def colorvideoMagnificationmult(video,amp,atn,sel):
    match sel:
        case 0:
            video[:,:,:,0] *= amp
            video[:,:,:,1] *= atn
            video[:,:,:,2] *= atn
            
        case 1:
            video[:,:,:,0] *= atn
            video[:,:,:,1] *= amp
            video[:,:,:,2] *= amp
            
        case 2:
            video[:,:,:,0] *= amp
            video[:,:,:,1] *= amp
            video[:,:,:,2] *= amp
            

    return video

def colorMagnification(image,amp,atn):
    image = image*amp
    image[:,:,1] *= atn
    image[:,:,2] *= atn
    
    return image


def colorMagnification_wv(video,amp,atn):
    
    for f in range(len(video)):
        for cf in range(len(video[f])):
            
            if cf == 0:
              
            #"""
               for c in range(len(video[f][cf])):
                    if c ==1 or c==2:
                        video[f][cf][c]=video[f][cf][c]*amp*atn
                   
                    
                    else:
                        
                        video[f][cf][c]=video[f][cf][c]*amp

            """
            #else:
            if cf>0:
                for d in range(len(video[f][cf])):
                        for c in range(len(video[f][cf][d])):
                            if c==1 or c==2:
                                video[f][cf][d][c]=video[f][cf][d][c]*amp*atn
                            else:
                                video[f][cf][d][c]*=amp 
            
                
            
                #"""
   

    return video

def filt_temp_id(data,low,high, fps):
    fft_vid = fft.fft(data,axis = 0)
    freqs = fft.fftfreq(data.shape[0],d = 1/(fps))
    fb = (np.abs(freqs-low)).argmin()
    fa = (np.abs(freqs-high)).argmin()
    fft_vid[:fb] = 0
    fft_vid[fa:-fa] = 0
    fft_vid[-fb:] = 0
    return np.abs(fft.ifft(fft_vid,axis=0)), fft_vid,freqs,low,high

def filt_temp_id_2(data,low,high, fps):
    filt_result = np.zeros_like(data)
    for i in range(3):
        fft_vid = fft.fft(data[:,:,:,i],axis = 0)
        freqs = fft.fftfreq(data.shape[0],d = 1/(fps))
        indices_frequencias_desejadas = np.where((freqs >= low) & (freqs <= high) )[0]
        fft_result_filtrado = np.zeros_like(fft_vid)
        fft_result_filtrado[indices_frequencias_desejadas] = fft_vid[indices_frequencias_desejadas]

        filt_result[:,:,:,i] = np.abs(fft.ifft(fft_result_filtrado,axis=0))
        

        #np.abs(fft.ifft(fft_vid,axis=0))
    return filt_result, fft_result_filtrado,freqs,low,high

def filt_temp_id2_wv(data,low,high, fps):
    filt_result = np.zeros_like(data)
    fft_vid = fft.fft(data,axis = 0)
    freqs = fft.fftfreq(data.shape[0],d = 1/(fps))
    indices_frequencias_desejadas = np.where((freqs >= low) & (freqs <= high))[0]
    fft_result_filtrado = np.zeros_like(fft_vid)
    fft_result_filtrado[indices_frequencias_desejadas] = fft_vid[indices_frequencias_desejadas]
    print(freqs[indices_frequencias_desejadas])
    filt_result = np.abs(fft.ifft(fft_result_filtrado,axis=0))

        #np.abs(fft.ifft(fft_vid,axis=0))
    return filt_result, fft_result_filtrado,freqs,low,high

def filt_temp_id_wv(data,low,high, fps):
    
    fft_vid = fft.fft(data,axis = 0)
    freqs = fft.fftfreq(len(data),d = 1/(fps))
    fb = (np.abs(freqs-low)).argmin()
    fa = (np.abs(freqs-high)).argmin()
    fft_vid[:fb] = 0
    fft_vid[fa:-fa] = 0
    fft_vid[-fb:] = 0
    return np.abs(fft.ifft(fft_vid,axis=0))

def filt_temp_id_wv2(data,low,high, fps):
    
    filt_result = np.zeros_like(data)
    
    fft_vid = fft.fft(data,axis = 0)
    freqs = fft.fftfreq(len(data),d = 1/(fps))
    indices_frequencias_desejadas = np.where((freqs >= low) & (freqs <= high) )[0]
    fft_result_filtrado = np.zeros_like(fft_vid)
    fft_result_filtrado[indices_frequencias_desejadas] = fft_vid[indices_frequencias_desejadas]

    filt_result = np.abs(fft.ifft(fft_result_filtrado,axis=0))
        

        #np.abs(fft.ifft(fft_vid,axis=0))
    return filt_result

def filt_temp_id_wv_D(data,d,low,high, fps):
    
    
    array_data = np.moveaxis(data.tolist(), (0), (1))
    
    fft_vid = fft.fft(array_data[d].tolist(),axis = 0)
    
    freqs = fft.fftfreq(len(array_data[d].tolist()),d = 1/(fps))
    
    fb = (np.abs(freqs-low)).argmin()
    
    fa = (np.abs(freqs-high)).argmin()
    
    fft_vid[:fb] = 0
    
    fft_vid[fa:-fa] = 0
    
    fft_vid[-fb:] = 0
    
    result = np.abs(fft.ifft(fft_vid,axis=0))
    
    #print(result.shape)
    return np.array(result)


def fbp_wv(data,low,high, fps):
    base=[]
    
    for cf in range(len(data[0])):
        base.append([])
        if cf ==0:
            filtro = [data[i][cf] for i in range(len(data))]
            filtro = np.moveaxis(filtro,(1),(0))
            for c in range(len(data[0][cf])):
                
                base[cf].append(filt_temp_id_wv(filtro[c],low,high,fps))
        else:
            for n in range(len(data[0][cf])):
                base[cf].append([])
                filtro = [data[i][cf][n] for i in range(len(data))]
                filtro = np.moveaxis(filtro,(1),(0)) 
                for c in range(len(data[0][cf][n])):
                    base[cf][n].append(filt_temp_id_wv(filtro[c],low,high,fps))
    
    filtrado = []
    for t in range(len(base[0][0])):
        filtrado.append([])
        for cf in range(0,len(base)):
            filtrado[t].append([])
            if cf == 0:
                for c in range(3):
                    filtrado[t][cf].append(base[cf][c][t])
            else:    
                for n in range(len(base[1])):
                    filtrado[t][cf].append([])
                    for c in range(3):
                        filtrado[t][cf][n].append(base[cf][n][c][t])

        
    return filtrado

def fbp_wv2(data,low,high,fps):

    coefs=[]
    
    for cf in range(len(data[0])):
        
        """
        if cf ==0:
            filtro = [data[i][cf] for i in range(len(data))]
            filtro = np.moveaxis(filtro,(1),(0))
            filtrado0 = filt_temp_id_wv(filtro[0],low,high,fps)
            filtrado1 = filt_temp_id_wv(filtro[1],low,high,fps)
            filtrado2 = filt_temp_id_wv(filtro[2],low,high,fps)    
            coefs.append([filtrado0,filtrado1,filtrado2])
        
        #""" 
        #else:
        if cf>0:
            coefs.append([])
            for n in range(len(data[0][cf])):
                
                filtro = [data[i][cf][n] for i in range(len(data))]
                filtro = np.moveaxis(filtro,(1),(0)) 
                filtrado0 = filt_temp_id_wv2(filtro[0],low,high,fps)
                filtrado1 = filt_temp_id_wv2(filtro[1],low,high,fps)
                filtrado2 = filt_temp_id_wv2(filtro[2],low,high,fps)    
                #coefs[cf].append([filtrado0,filtrado1,filtrado2])
                coefs[cf-1].append([filtrado0,filtrado1,filtrado2])
        #"""
                
    #print(len(coefs),len(coefs[0]),len(coefs[0][0]),len(coefs[0][0][0]),len(data[0]),len(data[0][1]))
    result = []
    for t in range(len(data)):
        result.append([])
        for cf in range(len(data[0])):


            #"""
            if cf==0:
                result[t].append([data[t][cf][0],data[t][cf][1],data[t][cf][2]])

            else :
                result[t].append([])
                for n in range(len(data[t][cf])):
                    result[t][cf].append([coefs[cf-1][n][0][t],coefs[cf-1][n][1][t],coefs[cf-1][n][2][t]])
            #"""
            """
            if cf==0:
                result[t].append([coefs[cf][0][t],coefs[cf][1][t],coefs[cf][2][t]])
            
            else:
                result[t].append([])
                for n in range(len(data[t][cf])):
                    result[t][cf].append([data[t][cf][n][0],data[t][cf][n][1],data[t][cf][n][2]])
            
            #"""
            """
            if cf==0:
                result[t].append([coefs[cf][0][t],coefs[cf][1][t],coefs[cf][2][t]])


            else:
                result[t].append([])
                for n in range(len(data[t][cf])):
                    result[t][cf].append([coefs[cf][n][0][t],coefs[cf][n][1][t],coefs[cf][n][2][t]])


            #""" 
            
            
            

        
    return result

def filt_temp_id_wv20(data,low,high, fps):
    ext_sig = []
    k=-1
    for i in range(len(data[0][0])):
        if i ==0:
            ext_sig.append([])
            for j in range(len(data)):

                ext_sig[i].append([])

                for m in range(len(data[0][0][i])):
                    ext_sig[i][j].append([])
                    for n in range(len(data[0][0][i][m])):
                        ext_sig[i][j][m].append([data[j][0][i][m][n],data[j][1][i][m][n],data[j][2][i][m][n]])
        if i != 0:
            ext_sig.append([])
            for k in range(len(data[0][0][i])):
                ext_sig[i].append([])
                for j in range(len(data)):

                                ext_sig[i][k].append([])

                                for m in range(len(data[0][0][i][k])):
                                    ext_sig[i][k][j].append([])
                                    for n in range(len(data[0][0][i][k][m])):
                                        ext_sig[i][k][j][m].append([data[j][0][i][k][m][n],data[j][1][i][k][m][n],data[j][2][i][k][m][n]])


    
    filtrado = ext_sig
    freqs = fft.fftfreq(len(data),d=1/(fps))
    fb = (np.abs(freqs-low)).argmin()
    fa = (np.abs(freqs-high)).argmin()

    for i in range(len(ext_sig)):
        if i == 0:
            fft_vid = fft.fft(ext_sig[i],axis = 0)
            #freqs = fft.fftfreq(len(ext_sig[0]),d = 1/(fps))
            #fb = (np.abs(freqs-low)).argmin()
            #fa = (np.abs(freqs-high)).argmin()
            fft_vid[:fb] = 0
            fft_vid[fa:-fa] = 0
            fft_vid[-fb:] = 0
            filtrado[i] = np.abs(fft.ifft(fft_vid,axis=0))
        else :
            for k in range(len(data[0][0][i])):
                fft_vid = fft.fft(ext_sig[i][k],axis = 0)
                #freqs = fft.fftfreq(len(ext_sig[i][k]),d = 1/(fps))
                #fb = (np.abs(freqs-low)).argmin()
                #fa = (np.abs(freqs-high)).argmin()
                fft_vid[:fb] = 0
                fft_vid[fa:-fa] = 0
                fft_vid[-fb:] = 0
                filtrado[i][k] = np.abs(fft.ifft(fft_vid,axis=0))
    
    dat_filt = data
    
    for i in range(len(data)):

        for j in range(len(data[0][0])):
            if j == 0:
                for m in range(len(data[0][0][j])):
                    for n in range(len(data[0][0][j][m])):
                        
                        dat_filt[i][0][j][m][n] = filtrado[j][i][m][n][0]
                        dat_filt[i][1][j][m][n] = filtrado[j][i][m][n][1]
                        dat_filt[i][2][j][m][n] = filtrado[j][i][m][n][2]

            else: 
                for k in range(len(data[0][0][i])):

                    for m in range(len(data[0][0][k][j])):
                        for n in range(len(data[0][0][k][j][m])):
                            dat_filt[i][0][j][k][m][n] = filtrado[j][k][i][m][n][0]
                            dat_filt[i][1][j][k][m][n] = filtrado[j][k][i][m][n][1]
                            dat_filt[i][2][j][k][m][n] = filtrado[j][k][i][m][n][2]

    return dat_filt, fft_vid,freqs,low,high

def new_video(vid_ori, vid_amp, levels,name,fps):
    result = np.zeros(vid_ori.shape)

    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        
        frame = res_img_gauss(frame,levels=levels)

        rgb = skimage.color.yiq2rgb(frame)
        #rgb = skimage.color.ycbcr2rgb(frame)#YCC
        #rgb = skimage.color.lab2rgb(frame)#lab
        #rgb = skimage.color.lab2rgb(frame)*255#lab255
        
        rgb = cv2.resize(rgb, dsize=(vid_ori[i].shape[1],vid_ori[i].shape[0]), interpolation=cv2.INTER_CUBIC)
        
        frame = rgb + vid_ori[i]
        
        result[i] = cv2.cvtColor(frame.astype(np.uint8),cv2.COLOR_RGB2BGR)
    h,w = result[0].shape[0:2]
    writer = cv2.VideoWriter("resultado"+name+".avi",cv2.VideoWriter_fourcc('M','J','P','G'),27,(w,h),1)
    for i in range(0,result.shape[0]):
        writer.write(cv2.convertScaleAbs(result[i]))
    writer.release()
    return result

def final_video(vid_ori, vid_amp, levels):
    result = np.zeros(vid_ori.shape)
    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        frame = res_img_gauss(frame,levels=levels)
        frame = frame + vid_ori[i]
        result[i] = frame

    return result

def final_video_cc(vid_ori, vid_amp, levels):
    result = np.zeros(vid_ori.shape)
    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        
        frame = res_img_gauss(frame,levels=levels)
        #print(frame.shape)
        #rgb = skimage.color.yiq2rgb(frame)
        ycc = skimage.color.ycbcr2rgb(frame)
        #lab = skimage.color.lab2rgb(frame)
        #lab = skimage.color.lab2rgb(frame)*255
        #frame = rgb + vid_ori[i]
        frame = ycc + vid_ori[i]
        #frame = lab + vid_ori[i]
        result[i] = frame
    
    return result

#soma com resize
def final_video_cc_nr(vid_ori, vid_amp, levels):
    result = np.zeros(vid_ori.shape)

    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        
        frame = res_img_gauss(frame,levels=levels)

        #rgb = skimage.color.yiq2rgb(frame)
        rgb = skimage.color.ycbcr2rgb(frame)#YCC
        #rgb = skimage.color.lab2rgb(frame)#lab
        #rgb = skimage.color.lab2rgb(frame)*255#lab255
        
        rgb = cv2.resize(rgb, dsize=(vid_ori[i].shape[1],vid_ori[i].shape[0]), interpolation=cv2.INTER_CUBIC)
        
        frame = rgb + vid_ori[i]
        result[i] = frame
    
    return result

def final_video_wv_cc(vid_ori,wv_video,level):
    result = np.zeros(vid_ori.shape)

    for i in range(0,len(wv_video)):
        
        wave = [[wv_video[i][0][j]]+[tuple(wv_video[i][l][:][j])for l in range(1,level+1)]for j in range(3)]
        canais = [pywt.waverec2(wave[j], 'db4') for j in range(3)]
        #canais = [pywt.waverec2(wv_video[i][j], 'db1') for j in range(3)]
        frame = np.stack(canais, axis=-1)
        #print(len(canais),frame.shape,canais[0][50][50],frame[50,50,0] )

        
        #rgb = skimage.color.yiq2rgb(frame)
        rgb = skimage.color.ycbcr2rgb(frame)#YCC
        #rgb = skimage.color.lab2rgb(frame)#lab
        #rgb = skimage.color.lab2rgb(frame)*255#lab255
        frame = rgb + vid_ori[i]
        result[i] = frame        

    return result

def final_video_wv_cc_nr(vid_ori,wv_video,level):
    result = np.zeros(vid_ori.shape)

    for i in range(0,len(wv_video)):
        
        wave = [[wv_video[i][0][j]]+[tuple(wv_video[i][l][:][j])for l in range(1,level+1)]for j in range(3)]
        canais = [pywt.waverec2(wave[j], 'db4') for j in range(3)]
        
        frame = np.stack(canais, axis=-1)
        
        #rgb = skimage.color.yiq2rgb(frame)
        rgb = skimage.color.ycbcr2rgb(frame)#YCC
        #rgb = skimage.color.lab2rgb(frame)#lab
        #rgb = skimage.color.lab2rgb(frame)*255#lab255
        rgb = cv2.resize(rgb, dsize=(vid_ori[i].shape[1],vid_ori[i].shape[0]), interpolation=cv2.INTER_CUBIC)
        

            
        frame = rgb + vid_ori[i]
        result[i] = frame        

    return result

@njit
def soma_video(frame,vid):
    return frame + vid

def final_video3(vid_ori, vid_amp, levels):
    result = np.zeros(vid_ori.shape)
    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        frame = res_img_gauss(frame,levels=levels)
        frame = soma_video(frame,vid_ori[i])
        result[i] = frame

    return result

def final_video2(vid_ori, vid_amp, levels):
    result = np.zeros(vid_ori.shape)
    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        frame = res_img_gauss(frame,levels=levels)
        
        result[i] = frame

    return result

def final_video_yiq(vid_ori, vid_amp, levels):
    result = np.zeros(vid_ori.shape)
    for i in range(0,vid_amp.shape[0]):
        frame = vid_amp[i]
        frame = res_img_gauss(frame,levels=levels)
        frame = yiq_rgb(frame)
        frame = frame + vid_ori[i]
        result[i] = frame

    return result

def carregar_vid(file_name):
    vid = cv2.VideoCaptrue(file_name)
    return vid

def rgb_yiq(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        yiq = skimage.color.rgb2yiq(frame)
        if i == 0:
            
            yiq_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        
        yiq_vid[i] = yiq
    return yiq_vid  

def yiq_rgb(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        rgb = skimage.color.yiq2rgb(frame)
        if i == 0:
            
            rgb_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        rgb_vid[i] = rgb
    return rgb_vid 

def rgb_hsv(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        yiq = skimage.color.rgb2hsv(frame)
        if i == 0:
            
            yiq_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        
        yiq_vid[i] = yiq
    return yiq_vid  

def hsv_rgb(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        rgb = skimage.color.hsv2rgb(frame)
        if i == 0:
            
            rgb_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        rgb_vid[i] = rgb
    return rgb_vid 

def rgb_yuv(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        yiq = skimage.color.rgb2yuv(frame)
        if i == 0:
            
            yiq_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        
        yiq_vid[i] = yiq
    return yiq_vid  

def yuv_rgb(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        rgb = skimage.color.yuv2rgb(frame)
        if i == 0:
            
            rgb_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        rgb_vid[i] = rgb
    return rgb_vid 

def rgb_ycbcr(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        ycc = skimage.color.rgb2ycbcr(frame)
        if i == 0:
            
            ycc_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        
        ycc_vid[i] = ycc
    return ycc_vid  

def ycbcr_rgb(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        rgb = skimage.color.ycbcr2rgb(frame)
        if i == 0:
            
            rgb_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        rgb_vid[i] = rgb
    return rgb_vid 

def rgb_lab(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        lab = skimage.color.rgb2lab(frame)
        if i == 0:
            
            lab_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        
        lab_vid[i] = lab
    return lab_vid 

def lab_rgb(video):
    for i in range(0,video.shape[0]):
        frame = video[i]
        rgb = skimage.color.lab2rgb(frame)
        if i == 0:
            
            rgb_vid = np.zeros((video.shape[0],video.shape[1], video.shape[2],3))
            
        rgb_vid[i] = rgb
    return rgb_vid 


def avg_signal(vid):

    avg = np.zeros((3,vid.shape[0]))

    for i in range(0,vid.shape[0]):
        
        n = vid[i].shape[0]
        sum = [0,0,0]

        for j in range(0,n):

            sum[0]+=vid[i,j,0]
            sum[1]+=vid[i,j,1]
            sum[2]+=vid[i,j,2]   

        avg[0,i] = sum [0]/(n) 
        avg[1,i] = sum [1]/(n) 
        avg[2,i] = sum [2]/(n) 
    return avg

def avg_signal1(vid):

    avg = np.zeros((3,vid.shape[0]))

    for i in range(0,vid.shape[0]):
        
        n = vid[i].shape[0]
        print(vid.shape)
        print(vid[i,0].shape)
        Média = [np.mean(canal, axis= 0) for canal in vid[i] ]
        print(len(Média))

        #avg[0,i] = sum [0]/(n) 
        #avg[1,i] = sum [1]/(n) 
        #avg[2,i] = sum [2]/(n) 
    return avg

def avg_signal2(vid):

    if vid.shape[0]%4 !=0:
        N = int(np.ceil(vid.shape[0]/4)+1)
    
    else:
        N = int(vid.shape[0]/4)

    avg = np.zeros((3,vid.shape[0]))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        funcao1 = executor.submit(avg_loop,0,N,vid,N)
        funcao2 = executor.submit(avg_loop,N,N*2,vid,N)
        funcao3 = executor.submit(avg_loop,N*2,N*3,vid,N)
        funcao4 = executor.submit(avg_loop,N*3,min(N*4,vid.shape[0]),vid,N)
    mold1 = funcao1.result() 
    mold2 = funcao2.result() 
    mold3 = funcao3.result()
    mold4 = funcao4.result()

    for i in range(0,vid.shape[0]):
        if i<N:
            avg[0,i] = mold1[0,i]
            avg[1,i] = mold1[1,i]
            avg[2,i] = mold1[2,i] 
        elif i<2*N:
            avg[0,i] = mold2[0,i-N]
            avg[1,i] = mold2[1,i-N]
            avg[2,i] = mold2[2,i-N] 
        elif i<3*N:
            avg[0,i] = mold3[0,i-2*N]
            avg[1,i] = mold3[1,i-2*N]
            avg[2,i] = mold3[2,i-2*N] 
        elif i<4*N:
            avg[0,i] = mold4[0,i-3*N]
            avg[1,i] = mold4[1,i-3*N]
            avg[2,i] = mold4[2,i-3*N] 


    return avg

def avg_loop(Ni,Nf,vid,k):


    avg = np.zeros((3,k))


    for i in range(0,k):
        
        if i + Ni>= Nf:
            break
        n = vid[i+Ni].shape[0]
        sum = [0,0,0]

        for j in range(0,n):

            sum[0]+=vid[i+Ni,j,0]
            sum[1]+=vid[i+Ni,j,1]
            sum[2]+=vid[i+Ni,j,2]   

        avg[0,i] = sum [0]/(n) 
        avg[1,i] = sum [1]/(n) 
        avg[2,i] = sum [2]/(n) 
    return avg



def freq_card(fft,freqs,f_high,f_low):
    fft_max = []

    for n in range(fft.shape[0]):
        if f_low <= freqs[n] <= f_high:
            fft_max.append(np.abs(fft[n]).max())

        else:
            fft_max.append(0)

    
    picos,prop = signal.find_peaks(fft_max)
    pico_max = -1
    freq_max = 0

    for pico in picos:
        if fft_max[pico] > freq_max:
            pico_max = pico
            freq_max = fft_max[pico]
    
    return floor(freqs[pico_max]*60)


def freq_card2(canal,fps):

    picos,_ = signal.find_peaks(canal[1])
    peak_mean = 0
    for pico in picos:
        peak_mean += canal[1][pico]
    peak_mean /= len(picos)

    batidas = []
    for pico in picos:
        if canal[1][pico]>= peak_mean:
            batidas.append(canal[1][pico]) 
    
    return floor(len(batidas)/(len(canal[1])/fps)*60)

def freq_card2_1(canal,fps):

    picos,_ = signal.find_peaks(canal[1])

    
    return floor(len(picos)/(len(canal[1])/fps)*60)

def freq_card3(canal,fps):

    picos,_ = signal.find_peaks(canal[1])
    peak_interval_mean = 0
    i = 0
    n = 0
    for pico in picos:
        if i == 2:
            peak_interval_mean += (pico-last_peak)/fps
            i=0
            n += 1
        if i == 0:
            last_peak = pico
        i += 1

    peak_interval_mean /= n
 
    
    return floor(1/(peak_interval_mean/60))

def freq_card4(canal,fps):

    picos,_ = signal.find_peaks(canal[1])
    peak_interval_mean = 0
    i = 0
    start = True
    n = 0
    for pico in picos:

        if start:
            last_peak = pico
            i += 1
            start = False
        else:
            if i % 2 == 0:
                peak_interval_mean += (pico-last_peak)/fps
                last_peak = pico
                
                n += 1
            i+=1

    peak_interval_mean /= n
 
    
    return floor(1/(peak_interval_mean/60))

def hamming(gsig):
    for i in range(3):
        if i ==1:
            janela = signal.hamming(len(gsig[1]))
            gsig[1]=gsig[1]*janela
    return gsig


def bpm_1(canal,fps,fb,fh):
    fft_result = np.fft.fft(canal[1])
    freqs = np.fft.fftfreq(len(canal[1]), 1/fps)  
    
    indices_frequencias_desejadas = np.where((freqs >= fb) & (freqs <= fh))[0]
    fft_result_filtrado = np.zeros_like(fft_result)
    fft_result_filtrado[indices_frequencias_desejadas] = fft_result[indices_frequencias_desejadas]
    
    peak_idx = np.argmax(np.abs(fft_result_filtrado))
    


    peak_freq = freqs[peak_idx]
    



    bpm = round(peak_freq * 60)

    
    peaks, _ = signal.find_peaks(np.abs(fft_result_filtrado))  

    
    peak_freqs = freqs[peaks]
    """
    # Plotando o espectro de frequência filtrado
    plt.figure(figsize=(10, 6))
    plt.stem(freqs, np.abs(fft_result_filtrado)**2)
    plt.plot(peak_freqs, np.abs(fft_result_filtrado[peaks])**2, "x")
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Espectro de Frequência Filtrado')

    # Marcando o pico máximo
    plt.plot(peak_freq, np.abs(fft_result_filtrado[peak_idx]**2), 'ro', label='Pico Máximo')

    # Configurando os limites do eixo x para a faixa de frequência de interesse
    plt.xlim(fb, fh)

    # Adicionando legenda
    plt.legend()

    plt.show()
    #"""
    peak_indices = np.argsort(np.abs(fft_result_filtrado))[-10:]


    peak_intensities = np.abs(fft_result_filtrado[peaks])[0:7]
    
    peak_freqs = freqs[peaks][0:7]

    vales, _ = signal.find_peaks(-np.abs(fft_result_filtrado))

    vales_freqs = freqs[vales][0:7]
    vales_inten = np.abs(fft_result_filtrado[vales])[0:7]


    data = [peak_freqs,peak_intensities,vales_freqs,vales_inten]

    return freqs,fft_result_filtrado,bpm

def bpm_2(canal,fps,fb,fa):
    
    wavelet_name = 'db4'  
    level = 5  

    
    coeffs = []
    for i in range(3):
        cA, cD = pywt.dwt(canal[i], wavelet_name)
        coeffs.append((cA, cD))
    def butter_bandpass_filter(data, cutoff_low, cutoff_high, sampling_rate, order=4):
        nyquist = 0.5 * sampling_rate
        low = cutoff_low / nyquist
        high = cutoff_high / nyquist
        b, a = butter(order, [low, high], btype='band', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    sampling_rate = fps  
    total_samples = len(canal[0])
    frequencies = np.fft.fftfreq(total_samples, d=1/sampling_rate)

    
    frequencies_hertz = frequencies * sampling_rate
    indices_frequencias_desejadas = np.where((frequencies_hertz >= 0) & (frequencies_hertz <= 50))[0]
    print(frequencies_hertz)
    fft_result_filtrado_cA = np.zeros_like(np.fft.fft(coeffs[i][0]))
    fft_result_filtrado_cA[indices_frequencias_desejadas] =np.fft.fft(coeffs[i][0])[indices_frequencias_desejadas]
    fft_result_filtrado_cD = np.zeros_like(np.fft.fft(coeffs[i][1]))
    fft_result_filtrado_cD[indices_frequencias_desejadas] =np.fft.fft(coeffs[i][1])[indices_frequencias_desejadas]

    """
    # Plotando os coeficientes em função da frequência
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.plot(frequencies_hertz[:total_samples//2], np.abs(fft_result_filtrado_cA)[:total_samples//2])
        plt.title(f'Detalhes (cA) - Canal {i+1}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 2, 2*i+2)
        plt.plot(frequencies_hertz[:total_samples//2], np.abs(fft_result_filtrado_cD)[:total_samples//2])
        plt.title(f'Aproximação (cD) - Canal {i+1}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    """
    # Aplicando o filtro passa-faixa
    filtered_cA = butter_bandpass_filter(coeffs[1][0], fb, fa, fps)
    filtered_cD = butter_bandpass_filter(coeffs[1][1], fb, fa, fps)

    peak_idx = np.argmax(np.abs(coeffs[1][1]))
    peak_freq = frequencies_hertz[peak_idx]
    print(peak_freq)

# Plotando os coeficientes em função da frequência
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.plot(frequencies_hertz[:total_samples//2], np.abs(np.fft.fft(filtered_cD))[:total_samples//2])
        plt.title(f'Detalhes (cA) - Canal {i+1}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 2, 2*i+2)
        plt.plot(frequencies_hertz[:total_samples//2], np.abs(np.fft.fft(filtered_cA))[:total_samples//2])
        plt.title(f'Aproximação (cD) - Canal {i+1}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def sat_oxi(canais):
    vermelho = canais[0]
    DC_vermelho = np.mean(vermelho)
    AC_vermelho = np.std(vermelho)
    Infra_Vermelho = canais[2]
    DC_infra_vermelho = np.mean(Infra_Vermelho)
    AC_infra_vermelho = np.std(Infra_Vermelho)

    sat_oxi = (AC_vermelho*DC_infra_vermelho)/(AC_infra_vermelho*DC_vermelho)
    
    return round(sat_oxi*100)


def sat_oxi2(canais):
    vermelho = canais[0]
    Med_vermelho = np.mean(vermelho)

    Infra_Vermelho = canais[2]
    Med_infra_vermelho = np.mean(Infra_Vermelho)

    num = np.sum((vermelho-Med_vermelho)*(Infra_Vermelho-Med_infra_vermelho))
    den =  np.sqrt(np.sum((vermelho-Med_vermelho)**2)*np.sum((Infra_Vermelho-Med_infra_vermelho)**2))



    sat_oxi = abs(num/den)
    
    return floor(sat_oxi*100)


def filtro_kerman(sinais):

    
    dt = 1  
    A = np.array([[1, dt], [0, 1]])  
    H = np.array([[1, 0]])  

    
    Q = np.array([[0.1, 0], [0, 0.1]]) 
    R = np.array([[1]])  

    
    x = np.array([[0], [0]])  
    P = np.array([[1, 0], [0, 1]])  

    
    
    filtered_signals = []
    
    
    for z in sinais:
        

        
        x_pred = np.dot(A, x)
        P_pred = np.dot(np.dot(A, P), A.T) + Q
        
        
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(np.dot(np.dot(H, P_pred), H.T) + R))
        x = x_pred + np.dot(K, z - np.dot(H, x_pred))
        P = np.dot((np.identity(2) - np.dot(K, H)), P_pred)

        filtered_signals.append(x[0, 0])
    

    
    return filtered_signals

def criar_imagem_com_texto(imagem,saturacao,bpm,filename):
    
    save_image(imagem,filename)
    imagem = Image.open("C:/Users/Rodrigo/Documents/rocketseat/TCC/imagem_sem_texto_"+ filename+".png")
    desenho = ImageDraw.Draw(imagem)

    
    fonte = ImageFont.load_default()  
    texto = f"Frequência Cardíaca: {bpm} bpm\nSaturação de oxigênio: {saturacao}%"
    largura_texto, altura_texto = desenho.textsize(texto, font=fonte)
    posicao = (imagem.width - largura_texto - 10, 10)
    
    coordenadas_retangulo = [posicao[0] - 5, posicao[1] - 5, imagem.width, posicao[1] + altura_texto + 5]

    
    desenho.rectangle(coordenadas_retangulo, fill="black")
    desenho.text(posicao, texto, fill="white", font=fonte)
    
    imagem.save("imagem_com_texto_"+filename+".png") 

def save_image(imagem,filename):
    image = cv2.cvtColor(imagem.astype('float32'),cv2.COLOR_RGB2BGR)
    cv2.imwrite("imagem_sem_texto_"+ filename+".png", image)

def corrigir_orientacao(imagem):
    try:
        for tag, valor in imagem._getexif().items():
            if ExifTags.TAGS.get(tag) == 'Orientation':
                if valor == 3:
                    imagem = imagem.rotate(180, expand=True)
                elif valor == 6:
                    imagem = imagem.rotate(270, expand=True)
                elif valor == 8:
                    imagem = imagem.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        
        pass
    
    return imagem

def redimensionar_para_mesma_altura(imagem1, imagem2):
    
    img1 = Image.open(imagem1)
    img2 = Image.open(imagem2)

    
    img1 = corrigir_orientacao(img1)
    img2 = corrigir_orientacao(img2)

    
    altura_img1 = img1.height
    altura_img2 = img2.height

    
    altura_minima = min(altura_img1, altura_img2)

    
    img1 = img1.resize((int((altura_minima / altura_img1) * img1.width), altura_minima))
    img2 = img2.resize((int((altura_minima / altura_img2) * img2.width), altura_minima))

    return img1, img2


def paste_image(file1,file2):
    
    img1, img2 = redimensionar_para_mesma_altura(file1, file2)

    
    largura_img1, largura_img2 = img1.width, img2.width

    
    largura_total = largura_img1 + largura_img2
    img_final = Image.new('RGB', (largura_total, img1.height))
    
    
    img_final.paste(img1, (0, 0))
    img_final.paste(img2, (largura_img1, 0))

    
    img_final.save("imagem_resultante.png")

def salvar_dados_bpm(bpm1,bpm2,bpm3,bpm4,bpm5,real,name):
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/resultados_encontrados_bpm"+name+".csv"
    
    if os.path.exists(file_name):
    
        df_resultados_existente = pd.read_csv(file_name)
        df_novos_resultados = [bpm1,bpm2,bpm3,bpm4,bpm5,real]
        df_resultados_existente.loc[df_resultados_existente.index[-1] + 1] = df_novos_resultados
        
        print(df_resultados_existente.to_string(index=False))

        df_resultados_existente.to_csv(file_name, index=False)
                
    else:
        df_primeiros_resultados = pd.DataFrame({'Técnica 1': [bpm1],
                                                'Técnica 2': [bpm2],
                                                'Técnica 3': [bpm3],
                                                'Técnica 4': [bpm4],
                                                'Técnica 5': [bpm5],
                                                'Real': [real]})
                
        print(df_primeiros_resultados.to_string(index=False))

        df_primeiros_resultados.to_csv(file_name, index=False)

def limpar_dados_bpm(name):
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/resultados_encontrados_bpm"+name+".csv"
    os.remove(file_name)

def limpar_dados_sat(name):
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/resultados_encontrados_sat"+name+".csv"
    os.remove(file_name)

def salvar_dados_sat(sat1,sat2,real,name):
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/resultados_encontrados_sat"+name+".csv"
    
    if os.path.exists(file_name):
    
        df_resultados_existente = pd.read_csv(file_name)
        df_novos_resultados = [sat1,sat2,real]
        df_resultados_existente.loc[df_resultados_existente.index[-1] + 1] = df_novos_resultados

        print(df_resultados_existente.to_string(index=False))

        df_resultados_existente.to_csv(file_name, index=False)
                
    else:
        df_primeiros_resultados = pd.DataFrame({'Técnica 1': [sat1],
                                                'Técnica 2': [sat2],
                                                'Real': [real]})
        
        print(df_primeiros_resultados.to_string(index=False))

        df_primeiros_resultados.to_csv(file_name, index=False)

def freq_detection(name_arq):
	file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/"+name_arq+".mp4"
	rgb_vid, fps = extract_video_info(file_name)
	AGEs = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	age_limit = [2,6,12,20,32,43,53,100]

	genders = ['Masculino', 'Feminino']



	prototxtPath = os.path.sep.join(["age_detector", "deploy_age.prototxt"])
	weightsPath = os.path.sep.join(["age_detector", "age_net.caffemodel"])
	ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)


	prototxtPath = os.path.sep.join(["gender_detector", "deploy_gender.prototxt"])
	weightsPath = os.path.sep.join(["gender_detector", "gender_net.caffemodel"])
	genNet = cv2.dnn.readNet(prototxtPath, weightsPath)


	prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt.txt"])
	weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


	
	image = rgb_vid[-1]
	image = image.astype(np.float32)

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()



	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:

		
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			if (startX <image.shape[0]) and (startY <image.shape[1]) :
				face = image[startY:endY, startX:endX]

				faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
					(78.4263377603, 87.7689143744, 114.895847746),
					swapRB=False)
						
				# Detecção da idade
				ageNet.setInput(faceBlob)
				preds_age = ageNet.forward()
				
				i = preds_age[0].argmax()
				age = AGEs[i]
				age_max = age_limit[i]
				ageConfidence = preds_age[0][i]
				#Detecção do genero
				genNet.setInput(faceBlob)
				preds_gen = genNet.forward()
				
				i = preds_gen[0].argmax()
				gen = genders[i]
				genConfidence = preds_gen[0][i]
				

			
			


	if gen == "Masculino":

		if age_max <=2:
			freq_b = 120/60
			freq_a = 140/60
		
		elif age_max <= 20:
			freq_b = 80/60
			freq_a = 100/60

		elif age_max <= 60:
			freq_b = 70/60
			freq_a = 90/60

		else:
			freq_b = 50/60
			freq_a = 60/60	

	if gen == "Feminino":

		if age_max <=2:
			freq_b = 120/60
			freq_a = 140/60
		
		elif age_max <= 20:
			freq_b = 80/60
			freq_a = 100/60

		elif age_max <= 60:
			freq_b = 73/60
			freq_a = 80/60

		else:
			freq_b = 50/60
			freq_a = 60/60		
	return(freq_b,freq_a)

def correacao_ml(bpm,sat):

    name = "Rods1"
    
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/resultados_encontrados_sat"+name+".csv"
    df_sat_resultados_existente = pd.read_csv(file_name)

    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/resultados_encontrados_bpm"+name+".csv"
    df_bpm_resultados_existente = pd.read_csv(file_name)


    X_bpm_1 = df_bpm_resultados_existente[['Técnica 3']]
    y_bpm = df_bpm_resultados_existente['Real']
    X_sat_1 = df_sat_resultados_existente[['Técnica 1']]
    y_sat = df_sat_resultados_existente['Real']

    modelo_bpm = LinearRegression()
    modelo_sat = LinearRegression()
    modelo_bpm.fit(X_bpm_1, y_bpm)
    modelo_sat.fit(X_sat_1, y_sat)

    novos_dados = pd.DataFrame({'Técnica 3': [bpm]})
    previsao = modelo_bpm.predict(novos_dados)
    new_bpm = previsao[0]

    novos_dados = pd.DataFrame({'Técnica 1': [sat]})
    previsao = modelo_sat.predict(novos_dados)
    new_sat = previsao[0]

    return new_bpm,new_sat

def valor_corrigido(bpm,sat,cond):
    
    #Carregamento do Modelo
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/modelo2_bpm_"+cond
    modelo_bpm = joblib.load(file_name)
    file_name = "C:/Users/Rodrigo/Documents/rocketseat/TCC/Resultados/Treino_final/modelo2_sat_"+cond
    modelo_sat = joblib.load(file_name)


    dado_bpm = pd.DataFrame({'BPM':[bpm]})
    #Correção
    match cond:
        case 'L1':
            dado_sat = pd.DataFrame({'SO1':[sat]})
        case 'L2':
            dado_sat = pd.DataFrame({'SO2':[sat]})
        case 'L3':
            dado_sat = pd.DataFrame({'SO1':[sat]})
        case 'G':
            dado_sat = pd.DataFrame({'SO2':[sat]})


    
        

    
    novo_bpm = modelo_bpm.predict(dado_bpm)[0]
    
    novo_sat = modelo_sat.predict(dado_sat)[0]

    return novo_bpm,novo_sat

