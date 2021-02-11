import os
import urllib.request
import tarfile
import shutil
root_path='./'
url='http://yaroslavvb.com/upload/notMNIST/'
train_filename='notMNIST_large.tar.gz'
test_filename='notMNIST_small.tar.gz'

def data_download(filename,url,root_path):
    download_url=url+filename
    download_path=root_path+filename
    if os.path.exists(download_path) :
        print(f'Already download : {filename}')
    else :
        print(f'Start Download : Filename:{filename}')
        urllib.request.urlretrieve(download_url,download_path)
        print("Download Complete")
    return download_path

def data_extract(filename):
    # 압축 해제
    install_path=os.path.splitext(os.path.splitext(filename)[0])[0] # remove tar.gz
    if os.path.isdir(install_path) :
        print(f'Already Extract and install Data Set :{filename}')
    else :
        tar=tarfile.open(filename)
        print(f'Start Exract on : {filename}')
        tar.extractall(root_path)
        print(f'Done.')
    return

def make_data_folder(filename,case_data):
    ## 설치한 이미지 파일들을 dataset folder에 case에 맞게 폴더생성 후 복사
    ## case_date -> str
    file_path=os.path.splitext(os.path.splitext(filename)[0])[0]
    target_path=root_path+"dataset/"+case_data
    if os.path.exists(target_path) :
        print(f'Already exists {case_data} folder.')
    else :
        print(f'Copy to {case_data} folder')
        shutil.copytree(file_path,target_path)
        print('Done.')
    return

def main(train_filename,test_filename,url,root_path) :
    train_filename=data_download(train_filename,url,root_path)
    test_filename=data_download(test_filename,url,root_path)

    data_extract(train_filename)
    data_extract(test_filename)

    make_data_folder(train_filename,"train")
    make_data_folder(test_filename,"test")

if __name__ == "__main__" :
    main(train_filename,test_filename,url,root_path)