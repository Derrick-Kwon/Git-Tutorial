#Day_15_02_Flask #인터페이스인듯 하다.
#https://tutorial.djangogirls.org/ko/ #튜토리얼로 장고걸스 여기서 인터페이스를 만들어보자

# static, templates 두개의 폴더를 만든다(폴더이름 틀리면 ㄴㄴ)
# static은 정적: 바뀌지 않는데이터를 여기 넣는다.
# templates: html5 파일을 여기 넣는다.
# templates -> html 파일 만들어봐라!

#파이썬 애니웨어


import random
import requests
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
@app.route('/') #이 / 는 루트라고 한다 ㅇ우리 서버가 가질 수 있는 가장 짧은 주소다.
def index(): #함수를 매핑시키는 것이다 #api 어플리케이션 프로그래밍 인디케이션(??)
    return 'hello, flask!!!'

@app.route('/randoms')
def show_randoms():
    a = make_randoms()
    return str(a)

def make_randoms():
    return [random.randrange(100) for _ in range(10)]



@app.route('/html')
def show_html():
    a = [random.randrange(100) for _ in range(10)]
    return render_template('sample_01.html', numbers=a) #render_template은 html이 바뀌는것을 계속 업데이트 해준다!
    #numbers는 그냥 매개변수 이름이다! 우리가 함수 안에서 사용할

    #파이썬 플라스크 이미지 삽입하기.. 이런식으로 검색 ㄱㄱ

@app.route('/copy')
def show_html_02():
    return render_template('sample_02.html')

@app.route('/upload')
def upload():
    return render_template('sample_03.html')

#퀴즈 브라우저로부터 이미지를 수신해서 서버에 저장하세요.
@app.route('/save', methods=['POST']) #파일 주고받는거는 get방식과 post 방식이 있다
def save_image():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join('static', filename))
        # 저장할 경로 + 파일명
        f.save(os.path.join('static', secure_filename((f.filename))))
        return 'uploads 디렉토리 -> 파일 업로드 성공!'
    pass


if __name__ == '__main__':
    app.run(debug=True)  # debug = True를 주면 개발자 서버가 되고 자동으로 수정업데이트가 된다.

#파이썬 애니웨어에서는 it __name__ 이거는 치지 마라. 거기 안에 알아서 돌려주는 툴이 있다.
gudwls5863
kwon6821!@


#DDNS 서비스:  사설 ip 공개하게 해줌