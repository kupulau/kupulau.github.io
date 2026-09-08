---
title: VS code로 LaTex 편집하기
category: general
tags:
  - latex
url: https://velog.io/@kupulau/VS-code로-LaTex-편집하기
created_at: 2025-09-02
related_notes:
---


논문이나 보고서를 작성할 때 LaTex을 많이 사용한다. overleaf 등의 다양한 LaTex editor가 있지만, 분량이 많아지게 되면 무료 버전으로는 감당하기 어렵게 된다. 이 때 Visual Studio Code의 LaTex를 다루는 extension을 사용하면 VS code를 LaTex editor로 사용할 수 있게 된다.

<br>

### LaTex Workshop extension
Latex Workshop은 VS code에서 tex 문서를 편집하고 빌드할 수 있는 편집기 역할을 하는 확장 프로그램이다. extension 탭에서 설치할 수 있다.

<br>

### TeX Live
그러나 위의 extension은 tex 파일을 pdf로 컴파일하지는 못하는데, 이를 위해서 Tex Live(Tex 배포판)를 따로 설치해야 한다. LaTex Workshop extension이 이 컴파일러를 불러서 tex를 pdf로 컴파일한다. 
macOS에서 사용하는 Tex 배포판은 MacTeX와 BasicTeX가 있는데, 전자는 컴파일에 필요한 모든 패키지를 포함하고 있지만 용량이 3-4GB로 무겁고, 후자는 꼭 필요한 최소의 컴파일러만 들어있는 경량 버전이다.
MacTeX는 공식 사이트에서 pkg 파일을 받아 설치하고, BasicTex는 brew로 설치 가능하다. 
여기서는 가벼운 BasicTex를 기준으로 설치해보자.

<br>

### BasicTeX 설치
터미널에서 아래 명령어를 이용해 BasicTeX를 설치한다.

`$ brew install --cask basictex`

설치 후 PATH를 설정해줘야 한다.

`$ echo 'export PATH="/usr/local/texlive/2025basic/bin/universal-darwin:$PATH"' >> ~/.zshrc`
`$ source ~/.zshrc`

터미널에 `which pdflatex`, `pdflatex --version`를 입력했을 때 설치된 위치와 버전이 출력되면 잘 설치된 것이다.

<br>

### 패키지 설치
BasicTeX는 최소한의 컴파일러만 들어있기 때문에 컴파일하다 보면 패키지 없음 에러가 자주 발생한다. 그럴 때는 필요한 패키지를 그 때 그 때 설치하면 된다.
`$ sudo tlmgr install <필요한 패키지>`




### References
https://success-now.tistory.com/17
