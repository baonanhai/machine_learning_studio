import os
import re
import time

import requests

session_request = requests.session()
headers = {'authority': 'www.v2ex.com',
           'method': 'GET',
           'scheme': 'https',
           'accept-encoding': 'gzip, deflate, br',
           'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
           'referer': 'https://www.v2ex.com/signin',
           'user-agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36'}


def get_train_img(once):
    url = 'https://www.v2ex.com/_captcha?once=' + str(once)
    session_request.headers['path'] = '/_captcha?once=' + str(once)
    session_request.headers['accept'] = 'image/webp,image/apng,image/*,*/*;q=0.8'
    r = session_request.get(url, headers=session_request.headers)
    if r.status_code == 200:
        file_name = int(time.time())
        file_path = os.path.join('train_img', str(file_name) + '.png')
        with open(file_path, 'wb') as img:
            img.write(r.content)
            print('下载图片:', file_path)


def get_once():
    r = session_request.get('https://www.v2ex.com/signin', headers=headers)
    once_info = re.search(r'once=(\d+)', r.text)
    if once_info:
        return once_info[1]
    else:
        return -1


if __name__ == '__main__':
    for i in range(1000):
        print('开始获取第', i, '个数据')
        once = get_once()
        get_train_img(once)
