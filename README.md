# AI - 10
- main.py는 절대 수정 X
- 기타 파일이 있을 경우 같이 제출
- 선공: 첫 번째로 보드에 piece을 place하는 플레이어(P1)
- 후공: 두 번째로 보드에 piece을 place하는 플레이어(P2), 즉 시작하자마자 piece을 select하는 플레이어

### Issue
생각나는 이슈가 있다면 적어주시고, 카톡으로 남겨주시면 의견 나눠봅시다.

- 현재 보드에 놓인 말의 개수에 따라 MCTS에서 Minimax로 알고리즘 변경
- MCTS 중 Simulation 설정(현재는 완전 랜덤)
- (최적화: 보드 대칭 및 회전)
- (유전 알고리즘 적용 및 결과 분석): -ing

### Closed Issue
- ~~MCTS 구현~~
- ~~MCTS에서 당장 이길 수 있다면, 그것을 선택~~
- ~~MCTS 중 Select가 이상함~~
- ~~Simulation Main 만들어서 승률 분석~~
- ~~pieces 분류~~
- ~~P2인 경우 첫 시작은 MCTS을 하지 않고 랜덤하게 선택~~

# Github 협업 가이드
## 1. Issue 생성
1. Github 페이지에서 Issues에 들어간다.
2. New issue 클릭 후 제목을 다음의 Prefix중 하나를 사용하여 정한다. Prefix는 필수이며 뒤의 내용은 생략가능하다.
예시)
`[Fix] MCTS 오류 패치`
```
[Feat]: 새로운 기능 개발
[Design]: 프론트엔드 디자인 구현 및 수정
[Fix]: 버그, 오류 해결, 코드 수정
[Refactor]: 코드 리팩토링 (기능에 큰 변화는 없으나 알고리즘, 구조를 전면적 수정)
[Chore]: 간단한 수정 및 기타
[Setting]: 환경 및 프로젝트 세팅
[Add]: 파일 추가
[Del]: 파일 제거
[Docs]: README.md, 주석 등 문서 작성
```


3. 내용으로는 다음을 기본 틀로 적는다. 제목에 표현하였다면 내용은 생략가능하다.
```
## 🛠 Issue
<!-- 이슈에 대해 간략하게 설명해주세요 -->

## 📝 To-do
<!-- 진행할 작업에 대해 적어주세요 -->
- [ ] todo!
```
4. Submit new issue을 통해 issue을 생성한다.
5. 해당 페이지에서 제목 옆에 있는 #5와 같은 숫자를 기억한 후 Edit을 눌러 제목을 수정한다.
6. 수정할 제목은 Prefix 다음에 번호를 붙여준다.
예시)
`[Fix] #5 - MCTS 오류 패치`

## 2. Branch 생성
1. Branch에 들어가 New branch를 누른다.
2. Branch는 방금 만든 issue을 따라간다. 형식은 Prefix/Issue번호 이다.
예시)
`Fix/#5`

## 3. Commit
Commit에 대해서는 다른 블로그를 참고바란다. \
Commit이 끝나면 전에 만든 Branch로 Pull한다. \
형식은 [Prefix] #Issue번호 - 한 일 이다.

예시)
`[Fix] #5 - MCTS 오류 패치`

Prefix로는 다음이 있다.
```
- [Design]: 프론트엔드 디자인 구현 및 수정
- [Feat]: 새로운 기능 구현
- [Fix]: 버그, 오류 해결, 코드 수정
- [Add]: Feat 이외의 부수적인 코드 추가, 라이브러리 추가, 파일 추가
- [Del]: 쓸모없는 코드, 주석 삭제
- [Refactor]: 전면 수정이 있을 때 사용
- [Remove]: 파일 삭제 (코드 몇 줄이 아닌 파일 하나를 지울 때)
- [Chore]: 그 이외의 잡일/ 버전 코드 수정, 패키지 구조 변경, 파일 이동, 파일이름 변경
- [Docs]: README나 WIKI 등의 문서 개정
- [Setting]: 세팅
- [Test]: 테스트 코드
```

## 4. Pull Request
Branch에 Pull을 하였다면 Pull Request에 자동으로 뜬다. \
뜨지 않을 경우 Pull Request을 클릭하여 해당하는 Branch를 클릭한다. \
제목은 Branch와 동일하게 한다. \
본문 내용은 다음과 같다. 필요없는 부분은 생략가능하다. \
Issue번호에는 생성한 Issue의 번호를 기입한다.
```
### Issue
closed #Issue번호
<br/>

### Motivation

<br/>

### Key Changes

<br/>
```
예시)
```
### Issue
closed #5
<br/>
```

다 하였다면 카톡으로 알린 후 이상이 없다면 merge한다.
Conflict가 뜬다면 카톡으로 알린다.
