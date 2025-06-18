import requests

url1 = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
url2 = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'

def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

download_file(url1, 'deploy.prototxt.txt')
download_file(url2, 'res10_300x300_ssd_iter_140000.caffemodel')
