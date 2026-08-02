---
title: Apache Airflow
category: general
tags:
  - dev
  - python
url: https://velog.io/@kupulau/Apache-Airflow
created_at: 2024-12-15
related_notes:
---

Apache Airflow는 workflow 관리 및 스케줄링 도구이다. 
airflow를 사용하면 학습을 1주일에 1번, 예측을 10분에 1번씩과 같이 주기적으로 실행되는 [[2024-12-14-Product serving#Batch serving|batch serving]]을 구현해 볼 수 있다. 

### 기본 개념

#### DAG (Directed Acylic Graph)
- airflow에서 작업을 정의하는 방법
- 작업의 흐름과 순서 정의

#### Operator
- airflow의 작업 유형을 나타내는 클래스
- ex) BashOperator, PythonOperator, SQLOperator 등

#### Scheduler
- DAG를 보며 현재 실행해야 하는 스케줄을 확인

#### Executor
- 작업이 실행되는 환경
- ex) LocalExecutor, CeleryExecutor 등

<br>

### 설치
`$ pip3 install apache-airflow==2.6.3`

⚠️ Apache Airflow는 최신 버전에서 다른 라이브러리와의 충돌 등 버그가 많다. 반드시 자신의 파이썬 환경과 호환되면서도 오래된 (안정적인) 버전으로 설치하도록 하자. Python 3.11.7 기준 Airflow 2.6.3 문제없이 호환됨을 확인.

<br>

### 환경 변수 설정

Airflow의 환경 변수를 설정해주자. bashrc에 등록해서 영구적으로 경로를 참조하도록 할지는 사용자의 선택이지만 매번 실행할 때마다 export 하기 귀찮기도 하고 만약 export하는 것을 잊기라도 하면 엉뚱한 곳에 파일이 작성되는 경우도 있으니 bashrc에 등록하는 것을 기준으로 작성하겠다.

`$ vi ~/.bashrc`

위 명령어를 사용해서 bashrc 파일을 열고 맨 아래에 원하는 디렉토리를 적어서 airflow home directory를 설정해준다.

`export AIRFLOW_HOME=your_directory`

터미널을 껐다 켜서 새로 시작하거나 `$ source ~/.bashrc`로 변경사항 저장하면 끝. 

<br>

### 데이터 베이스 초기화

`$ airflow db init`

위의 명령어로 데이터 베이스를 초기화하면 airflow 설정 파일인 `airflow.cfg`와 데이터 베이스 (default : SQLite) 파일인 `airflow.db`가 생성된다. 

<br>

### Admin 계정 생성
`$ airflow users create --username admin --password your_password --firstname gildong --lastname hong --role Admin --email id@gmail.com`

위 명령어로 admin 계정을 생성한다. 위의 계정 정보는 airflow webserver에 로그인 할 때 사용된다.

<br>

### Airflow webserver 실행
`$ airflow webserver --port 8080`

위의 명령어를 이용하면 airflow webserver가 실행되는데, 웹 브라우저를 켜서 주소창에 `http://localhost:8080`을 입력하면 airflow webserver를 띄울 수 있다. 
브라우저에서 webserver를 띄우면 가장 먼저 로그인 페이지가 뜨는데 위에서 설정한 계정 정보로 로그인을 하면 된다.

<br>

### Airflow scheduler 실행
별도의 터미널 창에 `$ airflow scheduler`를 입력하여 airflow scheduler를 실행한다. 

<br>

### DAG 작성
DAG는 airflow에서 작업의 흐름과 순서를 정의하는 구조라고 했다. DAG를 정의할 파일을 저장할 디렉토리를 먼저 생성하자. 이름은 `dags`여야한다.
이제 `dags`에서 DAG를 정의하는 파일을 작성해보자. 이 파일은 크게 DAG 정의, Task 정의, Task 순서 정의의 3가지 파트로 나뉜다. 
아래는 날짜를 출력하고 "Hello world!"를 출력하는 작업을 하는 간단한 DAG 파일의 예시이다. 이 DAG는 2024년 1월 1일부터 2024년 1월 8일까지 매일 UTC 00:30 AM에 bash 명령어인 date를 실행해 날짜를 출력한 후 python 함수 print_hello를 실행하도록 스케줄링한 것이다.  

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    "owner": "gildong",
    "depends_on_past": False,    # 이전 DAG의 task 성공 여부에 따라서 현재 task를 실행할지 결정. False는 과거 task의 성공 여부와 상관 없이 실행
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 8)
}

def print_hello():
	print("Hello world!")

#####################################################################
# Part 1. DAG 정의

with DAG(
    dag_id = "basic_dag",
    default_args=default_args,
    schedule_interval="30 0 * * *",     # 매일 UTC 00:30 AM에 실행 / 한 번만 실행하고 싶다면 "@once"
    tags=["my_dags"]
) as dag:

#####################################################################
# Part 2. task 정의
    task1 = BashOperator(
        task_id="print_date",   
        bash_command="date"   # 실행할 bash command
    )
    
    task2 = PythonOperator(
        task_id="print_hello",
        python_callable=print_hello
    )
        
#####################################################################
# Part 3. task 순서 정의    
    task1 >> task2
```

위에서 schedule_interval = "30 0 * * * "으로 표기된 것은 Cron 표현식이다.
Cron 표현식은 5자리로 구성된 batch proessing 스케줄링을 정의한 표현식이다. 
첫째 자리 : minute (0-59)
둘째 자리 : hour (0-23)
셋째 자리 : day of the month (1-31)
넷째 자리 : month (1-12)
다섯째 자리 : day of the week (0-6) (일요일 - 토요일)

Cron 표현식 예시

|Cron expression|Schedule|
|:--:|:--:|
|\*****|every minute|
|0****|every hour|
|00***|every day at 12:00 AM|
|00**FRI|at 12:00 AM, only on Friday|
|001**|at 12:00 AM, on day 1 of the month|

Cron 표현식을 어떻게 작성해야할지 헷갈린다면, [CronMaker](http://www.cronmaker.com) 사이트를 이용하자.

위의 DAG 파일을 `basic.py`로 저장하고 airflow webserver를 확인해보면 DAGs 탭에 `basic_dag`라는 이름으로 DAG이 생성된 것을 확인할 수 있다. DAG 이름 좌측의 토글을 클릭하면 DAG이 실행되며, 클릭해보면 실행한 결과, 로그 등을 확인할 수 있다. 

이외에도 DAG를 더 잘 작성하기 위해서 [DummyOperator](https://airflow.apache.org/docs/apache-airflow/2.2.4/_api/airflow/operators/dummy/index.html), [SimpleHttpOperator](https://airflow.apache.org/docs/apache-airflow-providers-http/stable/operators.html), [BranchPythonOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.BranchPythonOperator), variable, connection, hook, sensor, XComs, jinja template,  [documentation](https://airflow.apache.org/docs/#providers-packages-docs-apache-airflow-providers-index-html) 등을 참고하자. 

<br>

### Slack과의 연동
DAG이 실행되다가 어떤 이슈가 생겼을 때 알람이 오면 바로 확인할 수 있어 더 편리할 것이다. DAG가 실행되다가 task가 실패했을 때 slack으로 알림이 오도록 연동시켜 보자. 

#### Airflow slack provider 설치
이 역시 버전 문제에 민감하므로, 여기에서는 python==3.11.7, airflow==2.6.3과 문제없이 호환되는 8.6.0 버전을 설치하겠다. 
`$ pip3 install 'apache-airflow-providers-slack[http]'==8.6.0`

#### Slack API key 발급
[Slack API Apps](https://api.slack.com/apps) 페이지에서 API 키를 발급 받자. 우측 상단의 Create New App > From scratch > App Name 설정 > workspace 지정하면 slack 메시지가 도착할 workspace를 지정할 수 있다. 그 후 Basic information 탭에서 Incoming Webhooks를 클릭하고 Activate incoming webhooks의 토글을 클릭해 webhook을 활성화한다. 그러면 하단에 Add New Webhook to Workspace 버튼이 보이는데 여기서 아까 등록한 workspace의 채널을 하나 지정하여 등록하면 Webhook URL이 생기게 된다. URL은 대략 `https://hooks.slack.com/services/ABCDEFG/1234567` 이런 형태인데 `/ABCDEFG/1234567` 부분은 password이므로 노출되지 않도록 조심하자.

#### webhook 등록
이제 airflow webserver의 Admin > Connections 탭에 들어가서 좌측 상단의 + 버튼(Add a new record)을 눌러 webhook connection을 추가하자. 아래 정보를 입력한 후 Save를 누르면 webhook이 등록된다. 이제 DAG 실행 중 문제가 생기면 slack으로 알림을 받을 준비가 되었다. 

>Connection id : 본인이 식별 가능한 이름 (ex) slack_webhook 등)
Connection type : HTTP
Host : https://hooks.slack.com/services
Password : /ABCDEFG/1234567

#### slack alert 코드 작성
이제 webhook을 이용해 slack에 알림을 보내는 코드를 작성해야 한다.

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

SLACK_DAG_CONN_ID = "slack_webhook"    # Connection id에 입력한 본인이 식별 가능한 이름

def send_message(slack_msg):
    return SlackWebhookOperator(
        task_id="slack_webhook",
        slack_webhook_conn_id=SLACK_DAG_CONN_ID,
        message=slack_msg,
        username="Airflow-alert"
    )
    
def fail_alert(context):
    slack_msg = """
            Task Failed!
            Task: {task}
            Dag: `{dag}`
            Execution Time: {exec_date}
            """.format(
                task=context.get("task_instance").task_id, 
                dag=context.get("task_instance").dag_id,
                exec_date=context.get("execution_date")
            )
            
    alert = send_message(slack_msg)
    
    return alert.execute(context=context)
```

위와 같은 파일을 `utils/slack_alert.py`로 저장하자. 그리고 실제로 작업하는 DAG 파일에서는 `from utils.slack_alert import fail_alert`와 같이 import하고 DAG 코드에 `on_failure_callback` 인자에 `fail_alert` 함수를 등록하면 된다.

```python
with DAG(
    dag_id = "basic_dag",
    default_args=default_args,
    schedule_interval="30 0 * * *",     
    tags=["my_dags"],
    on_failure_callback=fail_alert
) as dag:
```

만약 성공했을 때 알람을 보내고 싶다면 메시지 내용을 성공으로 변경하고 `on_failure_callback` 대신 `on_success_callback` 인자를 이용하면 된다.

<br>


### References
https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/tree/main/01-batch-serving(airflow)
https://airflow.apache.org/docs/#providers-packages-docs-apache-airflow-providers-index-html
https://velog.io/@insutance/Airflow-Airflow-간단하게-설치하기#5-webserver-띄우기
https://blog.somideolaoye.com/building-pipeline-for-data-harvesting-with-apache-airflow

