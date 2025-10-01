import math

# 외부에서 받아오는 real distance도 받아와야 함....

IV_HEIGHT = 173
P_HEIGHT = None
GAP = 47.5
REAL_DISTANCE = 0.0


def calculate_range(id: str, Depth: float):
    global REAL_DISTANCE
    # 근데 뎁스 어떤 주기로 받아와야 할지 생각 못함

    if id is None:
        return 0
    else:
        P_HEIGHT = patients[id]["height"]
        pow_range = pow(Depth,2)-pow(IV_HEIGHT-P_HEIGHT+GAP, 2)
        REAL_DISTANCE = math.sqrt(pow_range)
        return REAL_DISTANCE

        
def add_patient():
    pid = input("환자 ID: ").strip()
    if pid in patients:
        print(f"ID {pid} 는 이미 존재합니다.\n")
        return

    first_name = input("이름(First name): ").strip()
    last_name = input("성(Last name): ").strip()
    sex = input("성별(M/F): ").strip().upper() == "M"
    blood = input("혈액형: ").strip()
    height = float(input("키(cm): ").strip())
    weight = float(input("체중(kg): ").strip())
    is_warning = input("경고 환자입니까? (Y/N): ").strip().upper() == "Y"

    patients[pid] = {
        "marker_id": pid,
        "first_name": first_name,
        "last_name": last_name,
        "sex": sex,
        "blood": blood,
        "height": height,
        "weight": weight,
        "is_warning_patient": is_warning
    }
    print(f"환자 {pid} 저장 완료\n")




def list_patients():
    # 전역 변수 읽기(표시용)
    try:
        curr_id = current_patient_id
    except NameError:
        curr_id = None

    if not patients:
        print("등록된 환자가 없습니다.\n")
        return

    # 헤더
    header = f"{'ID':<6}{'Name':<22}{'Sex':<6}{'Blood':<7}{'Ht(cm)':>8}{'Wt(kg)':>9}{'Warn':>7}"
    print(header)
    print("-" * len(header))

    # 숫자 ID 우선 정렬(문자 ID도 안전하게 처리)
    def _sort_key(k):
        return (0, int(k)) if k.isdigit() else (1, k)

    for pid in sorted(patients.keys(), key=_sort_key):
        p = patients[pid]
        name = f"{p.get('last_name','') } {p.get('first_name','')}".strip()
        sex = "M" if p.get("sex", False) else "F"
        blood = p.get("blood", "-")
        height = p.get("height", 0)
        weight = p.get("weight", 0.0)
        warn = "Y" if p.get("is_warning_patient", False) else "N"
        mark = "*" if (curr_id is not None and str(pid) == str(curr_id)) else " "

        print(f"{pid:<5}{mark} {name:<20}{sex:<6}{blood:<7}{height:>8.0f}{weight:>9.1f}{warn:>7}")
    print()



patients = {
    "1" : {
        "marker_id" : "1",
        "first_name" : "jieun",
        "last_name" : "kim",
        "sex" : False,
        "blood": "A+",     
        "height" : 152,
        "weight" : 123.123,
        "is_warning_patient" : False
    }
}

current_patient_id = None

# ----- CLI 진입점 (임포트 시 실행되면 안 됨!) -----
def run_cli():
    while True:
        print("=== 환자 관리 프로그램 ===")
        print("1. 환자 추가")
        print("2. 환자 목록 보기")
        print("3. 종료")
        choice = input("선택: ").strip()

        if choice == "1":
            add_patient()
        elif choice == "2":
            list_patients()
        elif choice == "3":
            print("프로그램 종료")
            break
        else:
            print("잘못된 입력입니다.\n")

# 임포트 시에는 실행 안 되고 파일 직접 실행할 때만 메뉴가 뜨도록
if __name__ == "__main__":
    run_cli()