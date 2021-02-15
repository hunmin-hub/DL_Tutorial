import os
from PIL import Image
from tqdm import tqdm
folder_path = './datasets'
extensions = []
# 손상된 파일 전처리
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    count=0
    for filee in tqdm(os.listdir(sub_folder_path)):
        file_path = os.path.join(sub_folder_path, filee)
        #print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        try :
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
            if filee.split('.')[1] not in extensions:
                extensions.append(filee.split('.')[1])
        except :
            print(f'Target : {file_path}\n')
            os.remove(file_path)
            print('Remove.. Done\n')