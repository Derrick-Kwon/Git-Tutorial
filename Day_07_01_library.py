# Day_07_01_library.py
# 웹 크롤링 = 웹상의 정보를 가져오는 것!
#html 등은 위 아래로 양쪽에 같은 코드가 나온다! <body> 태그 같은!

import requests
import re

url = 'http://211.251.214.176:8800/index.php?room_no=2'  #url의 원래 주소!
received = requests.get(url)
print(received)  #<Response [200]> #이 뜻은 ok 라는 뜻이다!  #http 상태코드
print(received.text) #문자열로 나오는데 여기서 정보를 가져올 수 있다!

results = re.findall(r'<td align="center" style="border: 1px #000000 solid;">([0-9]+) 석</td>', received.text) #정규표현식= 거의다 검색or 치환이다. 이중 findall 은 검색! 치환은 거의 쓰이지 않는다
#둥근 괄호를 쓰면 괄호안의 원하는 것만 가져올 수 있다!.
#괄호 안에있는것이 유니크하다는 것이 보장되어야 한다!
print(results)
print('빈 좌석 :', results[-2])


#퀴즈7-01 흥덕 도서관의 노트북 열람실의 빈 좌석 번호를 알려주세요!

# print(empty_data)

empty = re.findall(r'color:green;font-size:13pt;font-family:Arial"><b>([0-9]+)</b></font>', received.text)
empty = [int(n) for n in empty]
print(empty)

#강사님 코드 이렇게도 표현가능하다
empty = re.findall(r'.+:green.+"><b>([0-9]+)</b></font>', received.text) #. = 문자 + = 추가로 뒤에쭉 온다는 뜻
empty = [int(n) for n in empty]
print(empty)


results_3 = re.findall(r'<body  style="background-repeat: no-repeat;">(.+)</body>', received.text) #이 코드는 서로 다른줄들이 많아서 안된다. 이런 경우엔 전체를 다 한줄로 만들어줘야한다
#하나로 만들어 주기 위해 DOTALL 기능을 쓴다!

body = re.findall(r'<body  style="background-repeat: no-repeat;">(.+)</body>', received.text, re.DOTALL)

# print(results_3_multi_line)

#퀴즈 7-02 바디테그 안쪽에 있는 table 태그를 찾으세요

result_3_table_tag = re.findall(r'<table .+?>(.+?)</table>', body[0], re.DOTALL) #body는 리스트 형태로 findall 에서 나온다 이 body[0] 을 찾고자 하는거 뒤에 넣으면, 아까 찾았던 그 파일이아니라, 이 사이에서 찾게 된다!!!
print(result_3_table_tag)

#***중요!
# .+ : 탐욕적(greedy)
# .+?: 비탐욕적(non-greedy) ?를 쓰지 않으면 맨 마지막 > 를 찾는다!  ?를 쓰면 처음 >가 나오면 멈춘다!
# 위의 두개를 구분할 수 있으면 왠만한 크롤링은 가능!