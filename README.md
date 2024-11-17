# AI - 10
- main.py는 절대 수정 X
- 기타 파일이 있을 경우 같이 제출
- 선공: 첫 번째로 보드에 piece을 place하는 플레이어(P1)
- 후공: 두 번째로 보드에 piece을 place하는 플레이어(P2), 즉 시작하자마자 piece을 select하는 플레이어

# Issue
생각나는 이슈가 있다면 적어주시고, 카톡으로 남겨주시면 의견 나눠봅시다.

- 현재 보드에 놓인 말의 개수에 따라 MCTS에서 Minimax로 알고리즘 변경
- P2인 경우 MCTS을 하지 않고 랜덤하게 선택
- MCTS에서 당장 이길 수 있다면, 그것을 선택
- (최적화: 보드 대칭 및 회전, pieces 분류)

# Closed Issue
- ~~MCTS 구현~~

# Github 협업 가이드
1. Github 페이지에서 Issues에 들어간다.
2. 
