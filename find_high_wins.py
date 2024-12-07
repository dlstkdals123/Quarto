import os
import shutil

def extract_high_wins(log_folder, target_folder, win_count):
    # log 폴더와 결과 저장 폴더 설정
    if not os.path.exists(log_folder):
        print(f"폴더 '{log_folder}'가 존재하지 않습니다.")
        return

    target_folder = f"log_more_than_{win_count}wins"  # 복사할 폴더명 (win_count 반영)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)  # 결과 저장 폴더 생성

    high_wins_files = []  # Wins가 win_count 이상인 파일 저장용 리스트

    for file_name in os.listdir(log_folder):
        file_path = os.path.join(log_folder, file_name)
        if not os.path.isfile(file_path):
            continue  # 파일이 아닌 경우 건너뜀

        try:
            with open(file_path, 'r') as file:
                content = file.read()

                # P1과 P2의 Wins 값 찾기
                for player in ["P1", "P2"]:
                    win_line = f"Player: {player}\n    Wins:"
                    start_index = content.find(win_line)
                    if start_index != -1:
                        start_index += len(win_line)
                        end_index = content.find('\n', start_index)
                        wins_value = int(content[start_index:end_index].strip())

                        if wins_value >= win_count:
                            high_wins_files.append(file_name)
                            shutil.copy(file_path, os.path.join(target_folder, file_name))
                            break  # P1의 전적이 조건을 만족하는 경우, P2는 확인할 필요 없음
        except Exception as e:
            print(f"파일 '{file_name}'을 처리하는 중 오류 발생: {e}")

    # 결과 출력
    if high_wins_files:
        print(f"Wins가 {win_count} 이상인 파일이 다음 폴더로 복사되었습니다:")
        print(target_folder)
        for file in high_wins_files:
            print(file)
    else:
        print(f"Wins가 {win_count} 이상인 파일이 없습니다.")

# 프로그램 실행
log_folder = "log"  # 로그 파일들이 저장된 폴더명
win_count = 9  # 최소 승리 횟수 설정
extract_high_wins(log_folder, "", win_count)
