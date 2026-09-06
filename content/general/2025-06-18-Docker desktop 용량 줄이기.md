---
title: Docker desktop 용량 줄이기
category: general
tags:
  - docker
  - DevOps
url: https://velog.io/@kupulau/Docker-desktop-용량-줄이기
created_at: 2025-06-18
related_notes:
---

Docker는 사용하면서 이미지, 컨테이너, 볼륨, 네트워크 등을 계속 저장하므로, 주기적으로 정리하지 않으면 디스크 용량을 크게 차지하게 된다. docker 때문에 꽉 찬 디스크 용량을 줄이는 방법을 알아보자.

### `docker system prune`
사용하지 않는 모든 이미지, 중지된 컨테이너, 안 쓰는 네트워크를 삭제한다.
뒤에 `-a --volumes` 옵션을 붙이면 볼륨(로컬 PC에 저장된 파일들)까지 삭제된다.
꽤 위험한 명령이므로 반드시 백업 후 실행하자.

<br>

### `docker system df`
`docker system df`를 이용하면 아래와 같이 docker에서 어떤 부분이 얼마 만큼의 용량을 차지하고 있는지 확인할 수 있다. 
```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          0         0         0B        0B
Containers      0         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B
```

#### docker 이미지 삭제
- `docker image prune` : 사용하지 않는 이미지 삭제
- `docker rmi <image_id>` : 특정 이미지 삭제

#### docker container 삭제
- `docker container prune` : 중지된 컨테이너 삭제

#### docker 볼륨 삭제
- `docker volume ls`
이 명령어를 이용하면 아래와 같이 볼륨 목록이 나온다. 
```
DRIVER    VOLUME NAME
local     pgdata
local     myapp_data
local     1f29e22bc8f...
```
그러나 무작위로 이름이 붙여져 있거나 하면 어떤 볼륨인지 확인하기가 어렵다. 이럴 때는 `docker volume inspect <volume_name>`으로 상세 정보를 확인할 수 있다. 
- `docker volume rm <volume_name>` : 특정 볼륨 삭제
- `docker volume prune` : 사용하지 않는 볼륨 삭제

#### docker 캐시 삭제
Build cache는 이미지 빌드 과정에서 생성된 중간 레이어, 캐시, 임시 파일들이다. 빌드 속도를 빠르게 할 수 있지만, 디스크 공간이 부족할 때는 삭제하는 것도 방법이다.
`docker builder prune`하여 삭제하면 되고, 
```
WARNING! This will remove all dangling build cache.
Are you sure you want to continue? [y/N]
```
이 경고문이 떴을 때 `y`를 입력해서 캐시를 삭제하면 된다.

<br>


### References
https://new-pow.tistory.com/79
https://manchann.tistory.com/35