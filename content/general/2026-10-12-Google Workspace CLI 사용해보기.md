---
title: Google Workspace CLI 사용해보기
category: general
tags:
url:
created_at: 2026-10-12
related_notes:
---

[Google Workspace CLI](https://github.com/googleworkspace/cli)를 사용해보자. Google Workspace CLI는 Google Drive, Gmail, Calendar, Sheets 등 Google Workspace의 모든 REST API를 터미널에서 제어할 수 있는 CLI 도구이다. 

## 설치

- npm 사용
```bash
npm install -g @googleworkspace/cli
```

- homebrew 사용 (macOS / Linux)
```bash
brew install googleworkspace-cli
```

- [github releases 페이지](https://github.com/googleworkspace/cli/releases)에서 OS에 맞는 바이너리를 직접 다운로드해서 `PATH`에 등록하여 사용할 수도 있다. 


## 인증 설정

Google API 접근 권한이 필요하므로 최초 1회 인증 설정이 필요하다.

1. [Google Cloud Console](https://console.cloud.google.com/) 접속 후 새 프로젝트 생성
2. API 및 서비스 > 라이브러리에서 `Google Drive API` 사용 설정
3. OAuth 동의 화면 구성 (앱 유형: 외부, 테스트 사용자에 본인 구글 계정 이메일 등록)
4. 사용자 인증 정보 > 사용자 인증 정보 만들기 > OAuth 클라이언트 ID
	- 애플리케이션 유형: **데스크톱 앱(Desktop app)**
5. 발급된 클라이언트 JSON을 다운로드하여 아래 경로에 이름을 바꿔 저장:
    - Mac/Linux: `~/.config/gws/client_secret.json`
    - Windows: `%USERPROFILE%\.config\gws\client_secret.json`
6. 로그인 실행 : `gws auth login -s drive`



## 활용 예시


### Google Drive

- 최근 10개 파일 조회 (json 출력)
```bash
gws drive files list --params '{"pageSize": 10}'
```

- 파일 이름만 추출하여 출력
```bash
gws drive files list --params '{"pageSize": 20}' | jq -r '.files[].name'
```

- 전체 파일 페이지네이션 조회
```bash
gws drive files ilst --params '{"pageSize": 100}' --page-all | jq -r '.files[].name'
```

- 특정 폴더나 이름 조건 검색
```bash
# 파일명에 '보고서'가 포함된 파일 검색
gws drive files list --params '{"q": "name contains '\''보고서'\'' and trashed = false"}'
```

- 파일 업로드
```bash
gws drive +upload ./document.pdf --name "업로드문서.pdf"
```

- 폴더 ID 알아내기 → 폴더 내 파일 개수, 용량 계산에 필요
```bash
gws drive files list --params '{
  "q": "mimeType = '\''application/vnd.google-apps.folder'\'' and name = '\''내폴더이름'\'' and trashed = false",
  "fields": "files(id, name)"
}'
```

 - 폴더 내 파일 개수 세기
```bash
gws drive files list \
  --params '{"q": "'\''<FOLDER_ID>'\'' in parents and trashed = false"}' \
  --page-all | jq -s 'map(.files // []) | flatten | length'
```

- 폴더 내 파일들의 용량 계산
```bash  
# MB 단위로 환산 출력
gws drive files list \
  --params '{"q": "'\''<FOLDER_ID>'\'' in parents and trashed = false", "fields": "nextPageToken, files(id, name, size)"}' \
  --page-all | jq -s '[map(.files // []) | flatten[] | (.size // 0 | tonumber)] | add / (1024 * 1024) | tostring + " MB"'  
```
⚠️ Google Workspace 전용 문서(Docs, Sheets 등)는 원본 드라이브 파일 용량이 `null`로 반환되어 합산에서 0으로 계산된다. 
