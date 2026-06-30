---
title: streamlit
category: general
tags:
  - dev
url: https://velog.io/@kupulau/streamlit
created_at: 2024-09-12
related_notes:
---

데이터 사이언티스트가 다른 조직(개발 부서 등)의 도움 없이 빠르게 웹 서비스를 만들기 위해 등장한 것이 streamlit이다. (→ 파이썬으로 프론트 구현하기)
프로토타입을 배포하기에는 좋지만, 정식 서비스를 출시할 때는 당연히 프론트엔드/백엔드 전문가와 협업하는 과정이 필요할 것이다. 

### 설치

`pip3 install streamlit==1.36.0`

<br>

### import 
```python
import streamlit as st
```

<br>

### 기본 구현 (UI)


- 제목
`st.title("Title")`


- header
`st.header("header")`

- subheader
`st.subheader("subheader")`


- 텍스트
`st.write("Hello, world!")`



- 사용자에게 텍스트를 입력받기
`text1 = st.text_input("텍스트를 입력하세요.")`
`type="password"` 옵션을 넣을 수도 있다.
`password = st.text_input("비밀번호를 입력하세요.", type="password")`



- 사용자에게 숫자를 입력받기
`num1 = st.number_input("숫자를 입력하세요.")`



- 사용자에게 날짜/시간 입력받기
`date = st.date_input("날짜를 입력하세요.")`
`time = st.time_input("시간을 입력하세요.")`


- 사용자에게 파일 업로드받기
`file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])`


- Radio Button : 여러 버튼 중 하나를 선택하게 한다.
```python
color = st.radio("Choose a color", ("Red", "Green", "Blue"))

if color == "Red":
	st.write("You chose Red")
elif color == "Green":
	st.write("You chose Green")
elif color == "Blue":
	st.write("You chose Blue")
```

- Select Box : 여러 선택지 중 하나를 선택하게 한다.
```python
color = st.selectbox("Choose a color", ("Red", "Green", "Blue"))
```

- multi select box : 선택지들 중에서 복수 선택을 할 수 있게 함
```python
multi = st.multiselect("Choose", ["Red", "Green", "Blue"])
```

- 체크박스 : True/False를 입력받을 때
`check = st.checkbox("완료 시 체크")`
`value` 옵션을 추가해서 default 값을 설정할 수 있다. (ex) `value = True`)


- button : 버튼을 눌러 특정 동작을 실행하고 싶을 경우
```python
if st.button("Press the button to check the result"):
	st.write("You win!")
```

- form : 설문처럼 여러 입력을 한꺼번에 받고 싶을 경우
```python
with st.form():
	id = st.text_input("Input id")
    password = st.text_input("Input password", type="password")
    st.form_submit_button("Submit")
```

- table 표기
`st.dataframe(df)`
`st.table(df)`
df라는 pandas DataFrame이 있을 때 `.dataframe()`과 `.table()`로 표를 보여줄 수 있는데 `.dataframe()`은 표에 interactive하게 접근(클릭, 정렬 등)이 가능하다. 


- line chart 그리기
`st.line_chart(df)`
그려진 결과물은 pvg, png 등으로 저장도 할 수 있다!


- map 그리기
`st.map(df)`
df에 x좌표와 y좌표가 있으면 산점도처럼 지도가 그려진다.


- code
`st.code(print("Hello, world!"))`
```python
print("Hello, world!")
```

- latex 표기
`st.latex("\int a x^2 \,dx")`
$\int ax^2 dx$


- spinner : 진행 중일 때 메시지를 표시
```python
import time

with st.spinner("Loading..."):
	time.sleep(10)    # 10s 동안 진행
```


- status 표시
`st.success("Success!")`
`st.info("Information")`
`st.warning("Warning!")`
`st.error("Error!")`


- expander : 클릭해서 확장하는 부분이 필요할 때
```python
with st.expander("click to check more information"):
	st.write("contents")
```

<br>

### session state

streamlit은 화면에서 어떤 업데이트가 생기면 전체 streamlit 코드를 다시 실행한다. 즉, 이전에 했던 액션이 저장되지 않는 것이다. 예를 들어 사용자가 버튼을 눌러 클릭 수를 센다고 했을 때, 클릭을 하는 순간 클릭 수가 0으로 초기화되었다가 클릭을 인식하여 클릭 수가 1이 된다. 그리고 다음 클릭에서 다시 클릭 수가 0으로 초기화되고 카운트 되기 때문에 클릭을 몇 번을 하던 클릭 수가 1로 카운트 되어 버리는 것이다....
따라서 이전 상태를 저장해 상태를 유지하고 싶을 때 session state 코드를 이용해야 한다.

```python
# default value init
if 'count' not in st.session_state:
	st.session_state.count = 0

# click이 눌리면 count 수를 늘리고, 이를 계속 유지함
click = st.button('click')
if click:
	st.session_state.count += 1 
```

<br>

### caching

앞서 session state에서 설명한 것과 같이, streamlit은 업데이트가 생기면 코드를 다시 실행하기 때문에 데이터를 읽어오는 작업이 포함되어 있다면 데이터를 매번 다시 읽어오게 되어 비효율적이고 시간이 오래 걸리게 된다. 이 때 캐싱을 사용해서 함수 호출 결과를 local에 저장하면 더 빠르고 효율적이게 실행될 수 있다.

`@st.cache_data`, `@st.cache_resource`를 이용해서 캐싱을 할 수 있는데, 전자는 데이터, API request의 response에 사용되고 후자는 DB 연결, ML model을 load할 때 사용한다. 아래와 같이 데코레이터 함수처럼 사용한다.

```python
@st.cache_data
def load_large_data():
	df = pd.read_csv('large_data.csv')
    return df
```

<br>

### 실행

위의 코드들을 이용해 만든 result.py를 터미널에서 `streamlit run result.py`를 하면 local URL과 network URL을 반환한다. 
local URL은 현재 개발 중인 컴퓨터에서만 접속 가능한 주소이고, network URL은 동일한 네트워크에 연결된 다른 기기에서 접속 가능한 주소이다.
이 URL을 브라우저에 복사 붙여넣기 하면 웹페이지가 동작하는 것을 볼 수 있다. 


<br>

### References

https://streamlit.io
https://docs.streamlit.io
https://github.com/whitphx/streamlit-webrtc
https://cheat-sheet.streamlit.app