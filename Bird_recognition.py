import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

#Functions
N_FFT = 1024
HOP_SIZE = 1024  
N_MELS = 128             
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel' 
FMIN = 1200

#MFCC
def get_mfcc(mp3_file_path):
  y, sr = librosa.load(mp3_file_path, offset=0, duration=30)
  mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE, 
                                        n_mels=N_MELS, 
                                        htk=True, 
                                        fmin=FMIN,
                                        fmax=sr/2))
  return mfcc
#Mel spectogram
def get_melspectrogram(mp3_file_path):
  y, sr = librosa.load(mp3_file_path, offset=0, duration=30)
  melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE, 
                                        n_mels=N_MELS, 
                                        htk=True, 
                                        fmin=FMIN,
                                        fmax=sr/2))
  return melspectrogram
#Chroma vector
def get_chroma_vector(mp3_file_path):
  y, sr = librosa.load(mp3_file_path)
  chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE 
                                        ))
  return chroma
#Tonal Centroid Features
def get_tonnetz(mp3_file_path):
  y, sr = librosa.load(mp3_file_path)
  tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=sr,
                                        hop_length=HOP_SIZE, 
                                       ))
  return tonnetz

#Feature
def get_feature(file_path):
  # Extracting MFCC feature
  mfcc = get_mfcc(file_path)
  mfcc_mean = mfcc.mean(axis=1)
  mfcc_min = mfcc.min(axis=1)
  mfcc_max = mfcc.max(axis=1)
  mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

  # Extracting Mel Spectrogram feature
  melspectrogram = get_melspectrogram(file_path)
  melspectrogram_mean = melspectrogram.mean(axis=1)
  melspectrogram_min = melspectrogram.min(axis=1)
  melspectrogram_max = melspectrogram.max(axis=1)
  melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

  # Extracting chroma vector feature
  chroma = get_chroma_vector(file_path)
  chroma_mean = chroma.mean(axis=1)
  chroma_min = chroma.min(axis=1)
  chroma_max = chroma.max(axis=1)
  chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

  # Extracting tonnetz feature
  tntz = get_tonnetz(file_path)
  tntz_mean = tntz.mean(axis=1)
  tntz_min = tntz.min(axis=1)
  tntz_max = tntz.max(axis=1)
  tntz_feature = np.concatenate( (tntz_mean, tntz_min, tntz_max) ) 
  
  feature = np.concatenate( (chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
  return feature



#Tensorflow Model Prediction
def model_prediction(test_mp3):
    a = get_feature(test_mp3)
    b = a.reshape(1,len(a))
    # load, no need to initialize the loaded_rf
    rf = joblib.load("./random_forest.joblib")
    predictions = rf.predict(b)
    return predictions[0] #Return index of max element


#@st.cache(suppress_st_warning=True)

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

## Main Page
if(app_mode=="Home"):
    st.header("BIRD SPECIES RECOGNITION BASED ON THERE SOUND PATTERNS")



#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains files of the following bird species:")
    st.text("Motacillaalba, Phoenicurusochruros, Cardueliscarduelis, Turduspilaris, Sturnusvulgaris, Fringillacoelebs")
    st.text("Sittaeuropaea, Streptopeliadecaocto, Luscinialuscinia, Troglodytestroglodytes, Columbapalumbus, Alaudaarvensis, Parusmajor")
    st.text("Turdusmerula, Passermontanus, Phylloscopustrochilus, Turdusphilomelos, Passerdomesticus, Erithacusrubecula")


#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
    if test_mp3 is not None:
            path_in = 'test/'+test_mp3.name
            

    #Show Button
    if(st.button("Play Audio")):
        st.audio(test_mp3)
    
    #Predict Button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(path_in)
        label =['Motacillaalba', 'Phoenicurusochruros', 'Cardueliscarduelis', 'Turduspilaris', 'Sturnusvulgaris', 'Fringillacoelebs', 'Sittaeuropaea', 'Streptopeliadecaocto', 'Luscinialuscinia', 'Troglodytestroglodytes', 'Columbapalumbus', 'Alaudaarvensis', 'Parusmajor',
                 'Turdusmerula', 'Passermontanus', 'Phylloscopustrochilus', 'Turdusphilomelos', 'Passerdomesticus', 'Erithacusrubecula']
        st.success("Model is predicting it's a {} ".format(label[result_index]))

       
