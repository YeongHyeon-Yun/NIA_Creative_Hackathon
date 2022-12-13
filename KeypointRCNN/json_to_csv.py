import json
import csv
import os
from tqdm import tqdm


dir_path = '/workspace/seohyeong/ASAP/pigs/pig_Jason train/'
dir_list = os.listdir('/workspace/seohyeong/ASAP/pigs/pig_Jason train')

ana_txt_save_path = '/workspace/seohyeong/ASAP/testfolder/pigs'

with open('pigs_keypoints2.csv', 'w', newline = '') as output_file:
    f = csv.writer(output_file)
    f.writerow(['image', 'x_0', 'y_0', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4',
                    'x_5', 'y_5', 'x_6', 'y_6', 'x_7', 'y_7'])

for i in range(len(dir_list)):
    data = json.load(open(f'{dir_path}{dir_list[i]}', 'r'))

    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    new_file = dir_list[i].split('.')
    # music.json 파일을 읽어서 melon.csv 파일에 저장
    with open(f'{dir_path}{dir_list[i]}', 'r', encoding = 'utf-8') as input_file, open('pigs_keypoints2.csv', 'a', newline = '') as output_file:
        data = json.load(input_file)
        
        '''
        data[0] 은 json 파일의 한 줄을 보관 {"title:"Super Duper", "songId": ...}
        data[0]['컬럼명'] 은 첫 번째 줄의 해당 컬럼 element 보관
        '''
        
        f = csv.writer(output_file)
        for annotations in tqdm(data['ANNOTATION_INFO']):
            image = data['IMAGE']['IMAGE_FILE_NAME']
            image_id = annotations["ID"]
            keypoints = annotations["KEYPOINTS"]
        
            while True:
                if 1 not in keypoints and 2 not in keypoints:
                    break
                elif 1 in keypoints:
                    keypoints.remove(1)
                else:
                    keypoints.remove(2)
            f.writerow([image, *keypoints])

            # print(image, keypoints)
        # write each row of a json file
        # for keypoint in keypoints:
            
                
            # f.writerow([datum["title"], datum["songId"], datum["artist"], datum["img"]])
