"""
author: Quang Vo
e-mail: quang.vo@sinch.com
last modified: Mar 01, 2022
"""
from utils import SimilarityFunctions
from pathlib import Path
from data_reader import DataReader
from unidecode import unidecode
from preprocessor import Preprocessor
import helper_function
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import easyocr
from paddleocr import PaddleOCR
import pytesseract
from pytesseract import Output
from model.donut_ocr.donut_ocr import CustomDonutOCR

# Config
DATA_PATH = 'data'
IMG_OUT_DIR = 'outputs'
CLASS = ['CNH_Frente', 'CPF_Frente', 'RG_Frente']
DATA_TYPE = ['gt_ocr', 'gt_segmentation', 'in']
preprocess_params = dict(
    gray_scale=True,
    gaussian_blurring=True,
    ksize=(3, 3),
    equalization=True,
    clip_limit=2.0,
    tile_grid_size=(12, 12),
    thresholding=True,
    thresh=165,
    maxval=255,
    type_=cv2.THRESH_TRUNC + cv2.THRESH_OTSU
)

def run_ocr_reader(reader = 'easyOCR', gpu=False):
    # Load data
    dataReader = DataReader(CLASS, DATA_TYPE)
    dataReader.to_table(DATA_PATH)
    dataReader.csv_to_tuple('gt_ocr')
    df = dataReader.get_data()

    # Preprocess
    prProcessor = Preprocessor(preprocess_params)
    allImgArrOri = [cv2.imread(imgPath) for imgPath in df['in'].tolist()]
    allImgArr = [prProcessor.transform(imgArr) for imgArr in allImgArrOri]

    # Run OCR engines
    if reader == 'easyOCR':
        easyOCRReader = easyocr.Reader(['pt'], gpu=gpu)
        output = [easyOCRReader.readtext(imgArr, rotation_info=[90, 180, 270]) for imgArr in allImgArr]

        # Extract text
        outputText = []
        for example in output:
            exampleText = []
            for preds in example:
                exampleText.append(unidecode(preds[1].strip().lower()))
            outputText.append(exampleText)
        outputText = [' '.join(s) for s in outputText]

    if reader == 'paddleOCR':
        paddleOCRReader = PaddleOCR(
            lang = 'en',
            det_model_dir = 'model/paddle_ocr/en_PP-OCRv3_det_infer',
            rec_model_dir = 'model/paddle_ocr/en_PP-OCRv3_rec_infer',
            cls_model_dir = 'model/paddle_ocr/ch_ppocr_mobile_v2.0_cls_infer',
            use_angle_cls=True
        )
        output = [paddleOCRReader.ocr(imgArr, cls=True) for imgArr in allImgArr]

        # Extract text
        outputText = []
        for example in output:
            exampleText = []
            for preds in example[0]:
                exampleText.append(unidecode(preds[1][0].strip().lower()))
            outputText.append(exampleText)
        outputText = [' '.join(s) for s in outputText]

    if reader == 'Tesseract':
        output = [pytesseract.image_to_data(imgArr, output_type=Output.DICT) for imgArr in allImgArr]
        outputText = []
        for example in output:
            exampleText = [unidecode(s.strip().lower()) for s in example['text'] if s != '']
            outputText.append(exampleText)
        outputText = [' '.join(s) for s in outputText]

    if reader == 'Donut':
        preTrainedProcessor = "naver-clova-ix/donut-base-finetuned-cord-v2"
        preTrainedModel = "naver-clova-ix/donut-base-finetuned-cord-v2"
        donutReader = CustomDonutOCR(preTrainedProcessor, preTrainedModel)
        output = [donutReader.read_text(imgArr) for imgArr in allImgArrOri]
        # Extract text
        outputText = []
        for example in output:
            exampleText = []
            for preds in example:
                exampleText.append(unidecode(preds.strip().lower()))
            outputText.append(exampleText)
        outputText = [' '.join(s) for s in outputText]


    df[reader] = outputText
    df.to_csv(f'output/{reader}_output.csv', sep=',', index=False)
    return df


def plot_boxplot(df, reader='easyOCR'):
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    df.sort_values(by='value', inplace=True, ascending=False)
    plot = sns.boxplot(ax=ax, x="class", y="value", hue="metric", data=df, order=CLASS)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('%')
    plt.xlabel('')
    plot.xaxis.set_ticklabels(plot.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'output/{reader}_similarity.png')
    return

def compute_similarity(df, reader = 'easyOCR'):
    resultDf = pd.DataFrame(columns=['metric', 'value', 'class'])
    for cl in df['class'].unique():
        subDf = df[df['class'] == cl]
        groundTruth = subDf['gt_ocr_text'].tolist()
        func = SimilarityFunctions(groundTruth)

        for idx, row in subDf.iterrows():
            iou, tfidfSim = func.similarity_word_level(row['gt_ocr_text'], row[reader])
            resultDf = pd.concat([resultDf,
                                  pd.DataFrame([{'metric':'IOU', 'value':iou, 'class':cl}]),
                                  pd.DataFrame([{'metric':'TFIDF Sim', 'value':tfidfSim, 'class':cl}]),
                                  ], ignore_index=True)

    return pd.DataFrame(resultDf)

def main():
    reader = 'Tesseract'
    path = f'output/{reader}_output.csv'
    file = Path(path)
    if not file.exists():
        df = run_ocr_reader(reader=reader, gpu=False)
    else:
        df = pd.read_csv(path, index_col=False)
    df = compute_similarity(df, reader=reader)
    print(tabulate(df.head(), headers='keys'))
    plot_boxplot(df, reader=reader)
    return

if __name__ == '__main__':
    main()