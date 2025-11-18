# patient_info.py

import math
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool


"""
# 추후 에러 예외사항 검증용으로 추가할지도
import mysql.connector
from mysql.connector import errorcode

try:
  cnx = mysql.connector.connect(user='scott',
                                database='employ')
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cnx.close()

"""
"""
import logging
로깅도 추가사항이므로 추후 구현 시 덧붙일 것

"""


"""
mysql.connector.connect(
    user='jieun', password='1234', database='hospital',
    unix_socket='/var/run/mysqld/mysqld.sock', connection_timeout=5
)

"""



#=====================================================================
# pool connection
#=====================================================================
pool = MySQLConnectionPool(
                                  pool_name="main", pool_size=5,
                                  user='jieun', password='1234',
                                  host='127.0.0.1',
                                  database='hospital',
                                  connection_timeout=5)


#=====================================================================
# 변수선언 및 query 구현
#=====================================================================
IV_HEIGHT = 173
GAP = 47.5
_last_distance: float | None = None

query = ("SELECT marker_id, first_name, last_name, sex, blood, height, weight, is_warning_patient FROM patient WHERE marker_id = %s")
# marker_id, first_name, last_name, sex, blood, height, weight, is_warning_patient

def get_pool_connection():
    
    cnx = pool.get_connection()
    cursor = cnx.cursor(dictionary=True)
    return cursor, cnx




def calculate_range(marker_id: str, Depth: float)-> float | None:
    global _last_distance
    id = marker_id.strip()

    if not id:
        return 0

    cnx = None
    cursor = None

    if id:
        try:
            cursor, cnx = get_pool_connection()
            cursor.execute(query, (id,))
            row = cursor.fetchone()

            if not row:
                    print(f"[MISS] No patient row for marker_id={id!r}")
                    return _last_distance
            if row["height"] is None:
                    print(f"[MISS] height is NULL for marker_id={id!r}")
                    return _last_distance

            # 계산
            P_HEIGHT = float(row["height"])
            pow_range = pow(Depth,2)-pow(IV_HEIGHT-P_HEIGHT+GAP, 2)
            if pow_range < 0:
                return _last_distance
            REAL_DISTANCE = math.sqrt(pow_range)
            _last_distance = REAL_DISTANCE

        
        finally:
            try:
                if cursor:
                    cursor.close()
            finally:
                if cnx:
                    cnx.close()

    else:
        return _last_distance
    
    return REAL_DISTANCE

def get_patient_info(marker_id: str)-> str | None:
    id = marker_id.strip()
    cnx = pool.get_connection()
    cursor = cnx.cursor(dictionary=True)
    cursor.execute(query, (id,))
    
    row = cursor.fetchone()

    if not row:
        print(f"[MISS] No patient row for marker_id={id!r}")
        return NULL

    try:
        if cursor:
            cursor.close()
    finally:
        if cnx:
            cnx.close()

    final_name = f"{row['first_name'].strip()} {row['last_name'].strip()}"
      
    row.pop('first_name', None)
    row.pop('last_name', None)

    row = {'final_name': final_name, **row}


    return row

def add_patient(marker_id_add_patient: str):      # 환자 추가 메서드
    

    p_info, row = get_patient_info(marker_id_add_patient)
    return p_info
 
def load_patient_list():
    """모든 환자의 ID와 이름만 불러오기"""
    cnx = pool.get_connection()
    cursor = cnx.cursor(dictionary=True)
    cursor.execute("SELECT marker_id, first_name, last_name FROM patient ORDER BY marker_id ASC")
    rows = cursor.fetchall()

    cursor.close()
    cnx.close()

    result = []
    for r in rows:
        full_name = f"{r['first_name'].strip()} {r['last_name'].strip()}"
        result.append({
            "marker_id": r["marker_id"],
            "final_name": full_name
        })
    return result

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