import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor

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


def run_simulation(params):
    # simulation.py를 실행하고 실시간 로그를 출력
    p1_mcts, p1_switch, p2_mcts, p2_switch, iteration = params
    print(f"[실행 시작] Params: {params}")

    try:
        process = subprocess.Popen(
            ["python", "simulation.py", str(p1_mcts), str(
                p1_switch), str(p2_mcts), str(p2_switch), str(iteration)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 실시간 출력 처리
        for line in process.stdout:
            print(line, end="")

        process.wait()  # 프로세스 종료 대기
        if process.returncode != 0:
            print(f"[오류] Params: {params}, 에러 메시지: {process.stderr.read()}")

    except Exception as e:
        print(f"[예외 발생] Params: {params}, Exception: {e}")


# 병렬 실행
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_simulation, comb)                   : comb for comb in combinations}
        for future in futures:
            params = futures[future]
            try:
                future.result()  # 결과 가져오기
            except Exception as exc:
                print(f"[에러] Params: {params}, Exception: {exc}")
