import os
import json
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from shapely.geometry import Polygon
import glob
from PIL import Image
from pytesseract import pytesseract
from lxml import etree
import ast

from sklearn.model_selection import train_test_split
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Getting Label Studio output from json file
f = open('project-2-at-2024-07-11-20-12-5822c964.json')
label_studio_data = json.load(f)

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area,poly_2.area)
    return iou/min_area
    
    
def hocr_to_dataframe(fp):
    doc = etree.parse(fp)
    words = []
    wordConf = []
    coords_list = []
    for path in doc.xpath('//*'):
        if 'ocrx_word' in path.values():
            coord_text = path.values()[2].split(';')[0].split(' ')[1:] 
            word_coord = list(map(int, coord_text)) #x1, y1, x2, y2
            conf = [x for x in path.values() if 'x_wconf' in x][0]
            wordConf.append(int(conf.split('x_wconf ')[1]))
            words.append(path.text)
            coords_list.append(word_coord)

    dfReturn = pd.DataFrame({'word' : words,
                             'coords': coords_list,
                             'confidence' : wordConf})

    return(dfReturn)

# Creating dataFrame
document_data = dict()
document_data['file_name'] = []
document_data['labelled_bbox']= []

for i in range(len(label_studio_data)):
    row = label_studio_data[i]
    file_name = os.path.basename(row['data']['image'])
    label_list, labels, bboxes = [], [], []

    for label_ in row['annotations'][0]['result']:
        label_value = label_['value']
        x, y, w, h = label_value['x'], label_value['y'], label_value['width'], label_value['height']
        original_w , original_h = label_['original_width'], label_['original_height']

        x1 = int((x * original_w) / 100)
        y1 = int((y * original_h) / 100)
        x2 = x1 + int(original_w*w / 100)
        y2 = y1 + int(original_h*h / 100)
        
        label = label_value['rectanglelabels']
        label_list.append((label, (x1,y1,x2,y2), original_h, original_w))
        
    document_data['file_name'].append(file_name)    
    document_data['labelled_bbox'].append(label_list)        

custom_dataset = pd.DataFrame(document_data)
custom_dataset.to_csv("Data.csv",index=True)

# Defining labeltoid and idtolabel
label2id = {"Title": 1, "Heading": 1, "Subheading": 2,"Image":4,"NonText_footer":3,"NonText_Header":5,"Text":6,"Page_Heading":7,"Page_Footer":8}
id2label = {v:k for k, v in label2id.items()}

# Extract text from images using Tesseract HOCR
final_list = []
    
for i in tqdm(custom_dataset.iterrows(), total=custom_dataset.shape[0]):
    custom_label_text = {}
    word_list = []
    ner_tags_list  = []
    bboxes_list = []
    
    file_name = i[1]['file_name']
    #  Iterates over all images in the specified directory.
    for image in glob.glob('./data/input_images/*.jpg'): #Make sure you add your extension or change it based on your needs 
        frame_file_name = os.path.basename(image)
        if frame_file_name == file_name:
            custom_label_text['id'] = i[0]
            image_basename = os.path.basename(image)
            custom_label_text['file_name'] = image_basename
            annotations = []
            label_coord_list = i[1]['labelled_bbox']
            for label_coord in label_coord_list:
                (x1,y1,x2,y2) = label_coord[1]
                box1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 
                label = label_coord[0][0]
                base_name = os.path.join('./data', 'layoutlmv3_hocr_output',os.path.basename(image).split('.')[0])
                pytesseract.run_tesseract(image, base_name, extension='box', lang=None, config="hocr")
                hocr_file = os.path.join(base_name+'.hocr')
                hocr_df = hocr_to_dataframe(hocr_file)
                for word in hocr_df.iterrows():
                    coords = word[1]['coords']
                    (x1df,y1df,x2df,y2df) = coords
                    box2 = [[x1df, y1df], [x2df, y1df], [x2df, y2df], [x1df, y2df]]
                    words = word[1]['word']
                    overlap_perc = calculate_iou(box1,box2)
                    temp_dic = {}
                    if overlap_perc > 0.80:
                        if words != '-':
                            word_list.append(words)
                            bboxes_list.append(coords)
                            label_id = label2id[label]                              
                            ner_tags_list.append(label_id)
                        
                        custom_label_text['tokens'] = word_list
                        custom_label_text['bboxes'] = bboxes_list
                        custom_label_text['ner_tags'] = ner_tags_list

    final_list.append(custom_label_text)

# Splitting data and writing final data to train and test files
train, test = train_test_split(final_list, random_state=21, test_size=0.3)

for detail  in final_list:
    with open('final_list_text.txt', 'a') as f:
        f.write(str(detail))
        f.write('\n')
        
for detail  in train:
    with open('train.txt', 'a') as f:
        f.write(str(detail))
        f.write('\n')
        
for detail  in test:
    with open('test.txt', 'a') as f:
        f.write(str(detail))
        f.write('\n')