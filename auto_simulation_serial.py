import itertools
import subprocess

# 각 변수의 범위 설정
P1_MCTS_ITERATIONS_RANGE = range(500, 601, 100)
P1_SWITCH_POINT_RANGE = range(8, 10)
P2_MCTS_ITERATIONS_RANGE = range(600, 601)
P2_SWITCH_POINT_RANGE = range(9, 10)
ITERATION_RANGE = range(2, 3)

# 모든 조합 생성
combinations = itertools.product(
    P1_MCTS_ITERATIONS_RANGE,
    P1_SWITCH_POINT_RANGE,
    P2_MCTS_ITERATIONS_RANGE,
    P2_SWITCH_POINT_RANGE,
    ITERATION_RANGE,
)

for i, (p1_mcts, p1_swtich, p2_mcts, p2_switch, iteration) in enumerate(combinations):
    print(f"[실행 시작] Params: {p1_mcts}, {p1_swtich}, {
          p2_mcts}, {p2_switch}, {iteration}")

    process = subprocess.Popen(
        ["python", "simulation.py", str(p1_mcts), str(
            p1_swtich), str(p2_mcts), str(p2_switch), str(iteration)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 실시간 출력 처리
    for line in process.stdout:
        print(line, end="")

    # 에러 메시지 출력
    process.wait()  # 프로세스 종료를 대기
    if process.returncode != 0:
        print(f"오류 발생: {process.stderr.read()}")
        break
