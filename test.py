from data_reader import DataReader
from preprocessor import Preprocessor
import helper_function
from tabulate import tabulate
import numpy as np
import torch
import cv2
from unidecode import unidecode
import easyocr
from paddleocr import PaddleOCR,draw_ocr
import pytesseract
from pytesseract import Output
from model.donut_ocr.donut_ocr import CustomDonutOCR


# Config
PRINT_DATA = True
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

# Load data
reader = DataReader(CLASS, DATA_TYPE)
reader.to_table(DATA_PATH)
reader.csv_to_tuple('gt_ocr')
data = reader.get_data()

if PRINT_DATA:
    print(tabulate(data.head(), headers='keys'))

# Sample for testing
IDXS = []
for c in CLASS:
    IDXS.extend(list(data[data['class'] == c].sample(n=2).index.values))
samples = data.iloc[IDXS]

# Preprocess
prProcessor = Preprocessor(preprocess_params)
sampleImgArrOri = [cv2.imread(imgPath) for imgPath in samples['in'].tolist()]
sampleImgArr = [prProcessor.transform(imgArr) for imgArr in sampleImgArrOri]

# # Donut
# preTrainedProcessor = "naver-clova-ix/donut-base-finetuned-cord-v2"
# preTrainedModel = "naver-clova-ix/donut-base-finetuned-cord-v2"
# donutReader = CustomDonutOCR(preTrainedProcessor, preTrainedModel)
#
# image = sampleImgArrOri[0]
# output = donutReader.read_text(image)
# print(output)


# Pyteseract
output = [pytesseract.image_to_data(imgArr, output_type=Output.DICT) for imgArr in sampleImgArrOri]
outputText = []
for example in output:
    exampleText = [unidecode(s.strip().lower()) for s in example['text'] if s != '']
    outputText.append(exampleText)
outputText = [' '.join(s) for s in outputText]
print(outputText)

# for i in range(len(output)):
#     d = output[i]
#     n_boxes = len(d['text'])
#     for j in range(n_boxes):
#         if int(d['conf'][j]) > 60:
#             (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
#             img = cv2.rectangle(sampleImgArrOri[i], (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imwrite(f'output/result_{i+1}.jpg', img)

# # Paddle OCR
# paddleOCRReader = PaddleOCR(
#     lang = 'en',
#     det_model_dir = 'model/paddle_ocr/en_PP-OCRv3_det_infer',
#     rec_model_dir = 'model/paddle_ocr/en_PP-OCRv3_rec_infer',
#     cls_model_dir = 'model/paddle_ocr/ch_ppocr_mobile_v2.0_cls_infer',
#     use_angle_cls=True
# )
# output = [paddleOCRReader.ocr(imgArr, cls=True) for imgArr in sampleImgArr]
#
# from PIL import Image
# from matplotlib import cm
#
# for i in range(len(output)):
#     result = output[i][0]
#     image = Image.fromarray(np.uint8(sampleImgArrOri[i]))
#     boxes = [line[0] for line in result]
#     txts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     im_show = draw_ocr(image, boxes, txts, scores, font_path='model/paddle_ocr/fonts/simfang.ttf')
#     im_show = Image.fromarray(im_show)
#     im_show.save(f'output/result_{i+1}.jpg')


# for i in range(len(output)):
#     print(np.mean([t[-1] for t in output[i]]))

# Visuallieze
# i = 0
# for i in range(len(output)):
#     # draw rectangle on easyocr results
#     image = sampleImgArrOri[i]
#     results = output[i]
#     for res in results:
#         top_left = (int(res[0][0][0]), int(res[0][0][1])) # convert float to int
#         bottom_right = (int(res[0][2][0]), int(res[0][2][1])) # convert float to int
#         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
#         cv2.putText(image, res[1], (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
#
#     # write image
#     cv2.imwrite(f'output/{i}.jpg', image)

# Preprocess visualize
# for i in range(len(sampleImgArr)):
#     cv2.imshow('image', sampleImgArrOri[i])
#     cv2.waitKey(0)
#     cv2.imshow('image', sampleImgArr[i])
#     cv2.waitKey(0)


# easyOCR
# easyOCRReader = easyocr.Reader(['pt'], gpu=False)
# output = [easyOCRReader.readtext(imgArr, rotation_info=[90,180,270]) for imgArr in sampleImgArr]
# outputText = []
# for example in output:
#     exampleText = []
#     for preds in example:
#         exampleText.append(preds[1])
#     outputText.append(exampleText)
#
# outputText = [' '.join(s) for s in outputText]