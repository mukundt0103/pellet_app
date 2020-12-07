import streamlit as st
import numpy as np
import math
#import matplotlib
# matplotlib.use('Agg')
import cv2
# import tensorflow as tf
import time
import os
from PIL import Image
# import click
from collections import Counter
import matplotlib.pyplot as plt
# import altair as alt
import pandas as pd

from model_tiny import *
from deblur_unet import *

from skimage.morphology import extrema
from skimage.morphology import watershed as skwater

# from deblurgan.model import generator_model
# from deblurgan.utils import load_image, deprocess_image, preprocess_image

def watershed(img,mask,pdict,sh_ar,cr_ar):
    hsv = mask
    Z = hsv.reshape((-1))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 4
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    label = label.reshape(img.shape[0:2])

    cell_colour    = np.array([0, 0, 0])
    cell_label    = (np.inf,-1)
    for l,c in enumerate(center):
        dist_cell = np.sum(np.square(c-cell_colour)) #Euclidean distance between colours
        if dist_cell<cell_label[0]:
            cell_label=(dist_cell,l)


    cell_label   = cell_label[1]

    thresh = 1*(label==cell_label)
    thresh = np.uint8(thresh)

    # kernel  = np.ones((3,3),np.uint8)


    h_fraction = 0.3
    dist     = cv2.distanceTransform(thresh,cv2.DIST_L2,3)
    maxima   = extrema.h_maxima(dist, h_fraction*dist.max())


    unknown = cv2.subtract(thresh,maxima)


    ret, markers = cv2.connectedComponents(maxima)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    markers[unknown==np.max(unknown)] = 0
    markers = skwater(-dist,markers,watershed_line=True)

    imgout = img.copy()
    imgout[markers == 0] = [0,0,255] #Label the watershed_line

    fl=0
    total_pellets = 0
    pellet_area = [0] * 5
    for l in np.unique(markers):
        if l==0:      #Watershed line
            continue
        if l==1:      #Background
            continue
        temp = label.copy()
        temp[markers!=l]=-1
        area    = np.sum(temp==cell_label)
        if area > 0:
            total_pellets += 1
            sh_ar.append(area)
            if area < 200 and area >= 100:
                pellet_area[0] += 1
            if area < 300 and area >= 200:
                pellet_area[1] += 1
            if area < 400 and area >= 300:
                pellet_area[2] += 1
            if area < 500 and area >= 400:
                pellet_area[3] += 1
            if area <= 1000 and area >= 500:
                fl=1
                pellet_area[4] += 1
    if fl==1:
        print("Output image",imgout.shape)
        #print(imgout)
        cr_ar.append(imgout)
               
        # print("pixels for pellet {0} is {1}".format(l,area))

    # print('Pellets detected: ', total_pellets)
    # print('Pellet area 100-200', pellet_area[0])
    # print('Pellet area 200-300', pellet_area[1])
    # print('Pellet area 300-400', pellet_area[2])
    # print('Pellet area 400-500', pellet_area[3])
    # print('Pellet area 500-1000', pellet_area[4])
    pdict['s1']+=pellet_area[0]
    pdict['s2']+=pellet_area[1]
    pdict['s3']+=pellet_area[2]
    pdict['s4']+=pellet_area[3]
    pdict['s5']+=pellet_area[4]
    pdict['s6']+=total_pellets

    return imgout,pdict,sh_ar,cr_ar
st.beta_set_page_config(page_title="Pellet Detection",page_icon=open('logo-small.png', 'rb').read())
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
weight_path = 'generator.h5'
input_dir = 'static/crp/'
output_dir = 'static/deblur/'

reg = [[350, 606, 606, 862], [350, 606, 862, 1118], [606, 862, 606, 862], [
    606, 862, 862, 1118], [350, 606, 1118, 1374], [606, 862, 1118, 1374]]

# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
st.markdown("<h1 style='text-align: center;'>Pellet Detection</h1>", unsafe_allow_html=True)
# sub_but=st.sidebar.button("Detect on existing video!")
if st.sidebar.button("Clear Cache"):
    st.caching.clear_cache()
st.sidebar.title("Settings")
fr= st.sidebar.slider("Output video duration in seconds", 1, 20, 2, 1)
fps= st.sidebar.slider("Output video FPS", 1,30,1,1)



# Non-interactive elements return a placeholder to their location
# in the app. Here we're storing progress_bar to update it later.
progress_bar = st.sidebar.progress(0)

# These two elements will be filled in later, so we create a placeholder
# for them using st.empty()
frame_text = st.sidebar.empty()

fr_selector=st.sidebar.empty()
fr_sel=fr_selector.slider("Frame selector", 1,fps*fr,1,1)
det=st.sidebar.empty()

files=st.file_uploader("Upload video:")
show_file = st.empty()
# if sub_but==False:
if True:
    while not files:
        show_file.info("Please upload a video file!")
    content = files.getvalue()
    vid_sub=open("static/subvideo.mp4","wb")
    vid_sub.write(content)
    show_file.empty()


text1=st.empty()
image1 = st.empty()
text2=st.empty()
image2 = st.empty()
text3=st.empty()
image3 = st.empty()
crop_text=st.empty()
crops=st.empty()
crop1_text=st.empty()
crops1=st.empty()

cap=None
# if sub_but==False:
#     cap = cv2.VideoCapture("static/subvideo.mp4")
# else:
#     cap = cv2.VideoCapture("subvideo.mp4")
cap = cv2.VideoCapture("static/subvideo.mp4")
ret, frame = cap.read()
video_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
Width = frame.shape[1]
Height = frame.shape[0]


vid_info="<h2>Input video details<h2><p>FPS: "+str(math.ceil(video_fps))+"</p>"\
    "<p>Video resolution: "+str(Height)+"x"+str(Width)+"</p>"
        
show_file.markdown(vid_info,unsafe_allow_html=True)

@st.cache
def run_app():

    crop_text.text("")
    crops.text("")

    idx = 1
    cap=None
    # if sub_but==False:
    #     cap = cv2.VideoCapture("static/subvideo.mp4")
    # else:
    #     cap = cv2.VideoCapture("subvideo.mp4")
    cap = cv2.VideoCapture("static/subvideo.mp4")
    inp = cv2.VideoWriter("static/orgvid.mp4",
                            cv2.VideoWriter_fourcc(*'avc1'), fps, (Width, Height))
    outp = cv2.VideoWriter("static/predvid.mp4",
                            cv2.VideoWriter_fourcc(*'avc1'), fps, (Width, Height))
    graph = cv2.VideoWriter("static/graph.mp4",
                            cv2.VideoWriter_fourcc(*'avc1'), fps, (640,480))

    g = deblur_unet(pretrained_weights='deblur_unet.h5')
    model = unet(pretrained_weights='ep057.h5')
    sh_ar = []
    crops_full=[]
    adict={'s1':0,'s2':0,'s3':0,'s4':0,'s5':0,'s6':0,'t1':0,'t2':0}

    while(cap.isOpened()):

        cr_ar = []
        ret, frame = cap.read()
        if ret is not True or idx == (fr*fps+1):
            cap.release()
            outp.release()
            graph.release()
            inp.release()
            print("Exiting program!")
            break

        orgfr=frame.copy()

        progress_bar.progress(int(((idx-1)/(fr*fps))*100))
        frame_text.text("Frame "+str(idx)+"/"+str(fr*fps))
        # fr_sel.slider("Frame selector", 0,idx,1,1)

        crp=np.zeros((6,256,256,3))
        for i in range(6):
            crp[i,:,:,:]=frame[reg[i][0]:reg[i][1],reg[i][2]:reg[i][3],:]
        

        db_pre=crp/255
        start = time.time()
        deblur=g.predict(db_pre)
        temp=time.time()-start
        print(" Deblur Prediction speed:",temp)
        db_dep=deblur * 255
        db_dep=db_dep.astype('uint8')
        

        db_dep=db_dep / 255.
        start = time.time()
        seg=model.predict(db_dep)
        temp1=time.time()-start
        seg=seg*255
        seg=seg.astype('uint8')
        db_dep=deblur * 255
        db_dep=db_dep.astype('uint8')
        print("UNet Prediction speed:",temp1)
        #print("Images",seg[0,:,:,0],db_dep[i-1,:,:,:])

        crp1=db_dep.copy()

        num_pel=0
        pdict={'s1':0,'s2':0,'s3':0,'s4':0,'s5':0,'s6':0}

        for i in range(1,7):
            crp1[i-1,:,:,:],pdict,sh_ar,cr_ar=watershed(db_dep[i-1,:,:,:],seg[i-1,:,:,0],pdict,sh_ar,cr_ar)
            frame[reg[i-1][0]:reg[i-1][1], reg[i-1][2]:reg[i-1][3], :] = crp1[i-1,:,:,:]
            
        for i in range(1,7):
            adict['s'+str(i)]+=pdict['s'+str(i)]
        adict['t1']+=temp
        adict['t2']+=temp1

        det_text="<h2>Inference times<h2><p>Deblur inference speed: <span class='highlight blue bold'>"+str(float("{:.2f}".format(temp)))+"s</span></p>"\
            "<p>UNet inference speed: <span class='highlight blue bold'>"+str(float("{:.2f}".format(temp1)))+"s</span></p>"\
        "<h2>Detections<h2><p>Pellets detected: <span class='highlight red bold'>"+str(pdict['s6'])+"</span></p>"\
            "<p>Pellet area 100-200: <span class='highlight red bold'>"+str(pdict['s1'])+"</span></p>"\
                "<p>Pellet area 200-300: <span class='highlight red bold'>"+str(pdict['s2'])+"</span></p>"\
                    "<p>Pellet area 300-400: <span class='highlight red bold'>"+str(pdict['s3'])+"</span></p>"\
                        "<p>Pellet area 400-500: <span class='highlight red bold'>"+str(pdict['s4'])+"</span></p>"\
                            "<p>Pellet area 500-1000: <span class='highlight red bold'>"+str(pdict['s5'])+"</span></p>"

        det.markdown(det_text,unsafe_allow_html=True)

        print("Number of pellets:",num_pel)
        cv2.imwrite("static/fr"+str(idx)+".jpg",frame)
        outp.write(frame)
        inp.write(orgfr)

        counter = Counter(sh_ar)
        xrang=list(counter.keys())
        yrang=[]
        crang=list(np.ones(len(xrang)))
        for z in xrang:
            yrang.append(counter.get(z,0))

        plt.figure(num=None,figsize=(8, 6), dpi=80)
        plt.scatter(xrang,yrang)
        plt.ylabel('No. of pellets')
        plt.xlabel('Area')
        plt.savefig('foo.jpg')
        plt.close()
        graph.write(cv2.imread('foo.jpg'))


        idx += 1

        text1.subheader("Original Image")
        text2.subheader("Predicted Image")
        text3.subheader("Scatter Plot")
        image1.image(orgfr, use_column_width=True,channels='BGR')
        image2.image(frame, use_column_width=True,channels='BGR')
        image3.image(cv2.imread('foo.jpg'), use_column_width=True,channels='BGR')
        crop1_text.subheader("Crops with pellet area >500")
        #print(cr_ar)
        crops1.image(cr_ar)
        crops_full.append(cr_ar)
    return crops_full,adict 

crops_full,adict=run_app()

def disp():

    det_text="<h2>Average inference times<h2><p>Deblur inference speed: <span class='highlight blue bold'>"+str(float("{:.2f}".format(adict['t1']/(fr*fps))))+"s</span></p>"\
        "<p>UNet inference speed: <span class='highlight blue bold'>"+str(float("{:.2f}".format(adict['t2']/(fr*fps))))+"s</span></p>"\
    "<h2>Total detections<h2><p>Pellets detected: <span class='highlight red bold'>"+str(adict['s6'])+"</span></p>"\
        "<p>Pellet area 100-200: <span class='highlight red bold'>"+str(adict['s1'])+"</span></p>"\
            "<p>Pellet area 200-300: <span class='highlight red bold'>"+str(adict['s2'])+"</span></p>"\
                "<p>Pellet area 300-400: <span class='highlight red bold'>"+str(adict['s3'])+"</span></p>"\
                    "<p>Pellet area 400-500: <span class='highlight red bold'>"+str(adict['s4'])+"</span></p>"\
                        "<p>Pellet area 500-1000: <span class='highlight red bold'>"+str(adict['s5'])+"</span></p>"

    
    text1.subheader("Original Video")
    text2.subheader("Predicted Video")
    text3.subheader("Dynamic Scatter Plot")
    image1.video(open('static/orgvid.mp4', 'rb').read())
    image2.video(open('static/predvid.mp4', 'rb').read())
    image3.video(open('static/graph.mp4', 'rb').read())
    
    det.markdown(det_text,unsafe_allow_html=True)
    crop_text.subheader("Frame #"+str(fr_sel))
    crops.image(cv2.imread("static/fr"+str(fr_sel)+".jpg"),use_column_width=True,channels='BGR')
    crop1_text.subheader("Pellets with area>500")
    crops1.image(crops_full[fr_sel-1])

disp()
# We clear elements by calling empty on them.
progress_bar.empty()
frame_text.empty()

