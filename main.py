import streamlit as st
import pandas as pd
import os
from typing import List
from pathlib import Path
import pickle
class MultipleInputs:
    Pace: List[float]
    Cadence: List[float]
   


def load_regression_model():
    import_dir = Path("models/reg_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model


def load_classifier_model():
    import_dir = Path("models/class_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model


def multi_pred(item: MultipleInputs):
    # reshape inputs
    model_input = []
    for d, s in zip(item.Pace, item.Candence,):
        model_input.append([d, s])
    reg_model = load_regression_model()
    #class_model = load_classifier_model()

    reg_pred = reg_model.predict(model_input)
    #class_pred = class_model.predict(model_input)

    return {
        "regression_predictions": [float(i) for i in list(reg_pred)],
       # "class_predictions": [float(i) for i in list(class_pred)],
    }
def file_selector(folder_path='./', type="Health"):
        folder_path = folder_path + type
        filenames = os.listdir(folder_path)
        csvFiles = []
        for file in filenames:
            if "csv" in file:
                csvFiles.append(file)
        selected_filename = st.selectbox('Select ' + type, csvFiles)
        return os.path.join(folder_path, selected_filename)
fileName = file_selector(type="Data")
if fileName:
    df = pd.read_csv(fileName)
    df = df.drop('Steps', 1)
    df = df.drop('Distance', 1)

    
    audioOnDf = df[df["AudioOn"] == True ]
    audioOffDf = df[df["AudioOn"] == False]
    avgAudioOnCadence = audioOnDf["Candence"].sum() / len(audioOnDf["Candence"])
    avgAudioOffCadence = audioOffDf["Candence"].sum() / len(audioOffDf["Candence"])
    st.header("Audio On Candence: " + str(avgAudioOnCadence))
    st.header("Audio Off Candence: " + str(avgAudioOffCadence))
    st.text("The lower the better")

    avgAudioOnPace = audioOnDf["Pace"].sum() / len(audioOnDf["Pace"])
    avgAudioOffPace = audioOffDf["Pace"].sum() / len(audioOffDf["Pace"])

    st.header("Audio On Pace: " + str(avgAudioOnPace))
    st.header("Audio Off Pace: " + str(avgAudioOffPace))
    st.text("The higher the better")
    load_regression_model()
    predictDf = multi_pred(df)
    st.line_chart(predictDf)
    st.header("Audio Off")
    st.line_chart(audioOffDf)
    

