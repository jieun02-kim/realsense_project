
# patient_ui.py
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget
)
from PyQt6.QtGui import QShortcut, QKeySequence, QPixmap, QFont
from PyQt6.QtCore import Qt
import sys, signal
from PyQt6.QtCore import QTimer

import patient_info as info
#import gown_marker_pipeline as pipeline





class BasePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #F4F6F9;")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 20, 30, 20)
        self.layout.setSpacing(20)
        self.main_layout = self.layout

        # 헤더 (Back + Search)
        self.header_layout = QHBoxLayout()

        self.back_button = QPushButton("← Back to Dashboard")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #037091;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search Patient...")
        self.search_bar.setFixedWidth(400)
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 8px;
                padding: 5px 10px;
            }
        """)

        self.header_layout.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.header_layout.setSpacing(50)
        self.header_layout.addWidget(self.search_bar, alignment=Qt.AlignmentFlag.AlignLeft)
        self.layout.addLayout(self.header_layout)

    def set_main_window(self, main_window):
        self.main_window = main_window
        self.back_button.clicked.connect(self.go_to_dashboard)

    def go_to_dashboard(self):
        if hasattr(self, "main_window"):
            self.main_window.stack.setCurrentIndex(0)

# ====================================================================
# DASHBOARD PAGE
# ====================================================================
class DashboardPage(BasePage):
    def __init__(self):
        super().__init__()
        self.cache_mid: str = None
        self.mid: str = "1"

        title = QLabel("Patient Report")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333333;")
        self.main_layout.addWidget(title)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)

        # 왼쪽 박스
        self.left_box = QFrame()
        self.left_box.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 0px;
            }
        """)
        self.left_box_layout = QVBoxLayout(self.left_box)
        self.left_box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_name = QLabel("Name: ")
        self.lbl_name.setFont(QFont("Arial", 25, QFont.Weight.Bold))
        self.lbl_name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_name.setStyleSheet("color: #222222;")

        self.lbl_id = QLabel(f"Patient ID: ")
        self.lbl_id.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_id.setStyleSheet("color: #000000; font-size: 17px;")

        self.lbl_status = QLabel(f"Status: ")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_status.setStyleSheet("color: #037091; font-weight: bold; font-size: 17px;")

        view_btn = QPushButton("View Profile")
        edit_btn = QPushButton("Edit Profile")
        for btn in [view_btn, edit_btn]:
            btn.setFixedHeight(30)
            btn.setFixedWidth(220)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #037091;
                    color: white;
                    border-radius: 6px;
                    font-size: 20px;
                }
                QPushButton:hover {
                    background-color: #0499C1;
                }
            """)

        self.left_box_layout.addSpacing(50)
        self.left_box_layout.addWidget(self.lbl_name)
        self.left_box_layout.addSpacing(4)
        self.left_box_layout.addWidget(self.lbl_id)
        self.left_box_layout.addSpacing(30)
        self.left_box_layout.addWidget(self.lbl_status)
        self.left_box_layout.addSpacing(27)
        self.left_box_layout.addWidget(view_btn)
        self.left_box_layout.addSpacing(5)
        self.left_box_layout.addWidget(edit_btn)



        # 오른쪽 박스 (처음엔 비어 있음)
        self.info_box = QFrame()
        self.info_box.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 0px;
            }
        """)
        self.info_box.setContentsMargins(30, 20, 30, 20)


        self.info_layout = QVBoxLayout(self.info_box)
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    
        self.info_layout.setSpacing(14)

        # 버튼 동작 연결
        view_btn.clicked.connect(lambda: self.load_patient_info())
        edit_btn.clicked.connect(lambda: print("Edit profile clicked! (추후 구현 예정)"))

        # 전체 배치
        bottom_layout.addWidget(self.left_box, 2)
        bottom_layout.addWidget(self.info_box, 3)
        self.main_layout.addLayout(bottom_layout)
        self.main_layout.addStretch(0)

        # ======================================================
        # 하단: 전체 환자 리스트 테이블
        # ======================================================
        self.table_frame = QFrame()
        self.table_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
            }
        """)
        self.table_layout = QVBoxLayout(self.table_frame)
        self.table_layout.setContentsMargins(20, 20, 20, 20)
        self.table_layout.setSpacing(10)

        table_title = QLabel("Registered Patients")
        table_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        table_title.setStyleSheet("color: #222222;")
        self.table_layout.addWidget(table_title)

        self.patient_table = QTableWidget()
        self.patient_table.setColumnCount(2)
        self.patient_table.setHorizontalHeaderLabels(["Patient ID", "Name"])
        self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.patient_table.verticalHeader().setVisible(False)
        self.patient_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.patient_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.patient_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: #FAFAFA;
                font-size: 14px;
            }
        """)
        self.table_layout.addWidget(self.patient_table)

        # 전체 환자 정보 불러오기
        self.load_patient_list()

        # 테이블 클릭 시 해당 ID로 정보 로드
        self.patient_table.cellClicked.connect(self.on_patient_selected)

        # 메인 레이아웃에 추가
        self.main_layout.addWidget(self.table_frame)



        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.load_patient_info_clock())
        self.timer.start(300)  # 0.3초마다 mid 상태 확인


    def load_patient_list(self):
        """DB에서 환자 목록을 불러와 UI 테이블에 표시"""
        patients = info.load_patient_list()  # patient_info.py 의 함수 호출
        self.patient_table.setRowCount(len(patients))
        
        for i, p in enumerate(patients):
            self.patient_table.setItem(i, 0, QTableWidgetItem(str(p["marker_id"])))
            self.patient_table.setItem(i, 1, QTableWidgetItem(p["final_name"]))



    def on_patient_selected(self, row, column):
        """리스트에서 클릭된 환자의 mid로 갱신 + 왼쪽 정보 즉시 반영"""
        mid_item = self.patient_table.item(row, 0)
        if not mid_item:
            return

        self.mid = mid_item.text().strip()
        data = info.get_patient_info(self.mid)

        # 🔹 왼쪽 라벨 즉시 갱신
        self.lbl_name.setText(f"{data.get('final_name', 'Unknown')}")
        self.lbl_id.setText(f"Patient ID: {data.get('marker_id', 'N/A')}")
        self.lbl_status.setText(
            f"Status: {'Warning' if data.get('is_warning_patient') else 'Normal'}"
        )

        # 오른쪽은 그대로 (View Profile 눌러야 표시)
        self.clear_widget()
        self.show_no_patient()
        self.cache_mid = None

        self.left_box.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                padding: 0px;
            }
        """)


   



    # ======================================================
    #  right 정보 박스 업데이트 함수
    # ======================================================
    def load_patient_info_clock(self):
        if self.mid is None or self.mid != self.cache_mid:
            # 기존 내용 삭제
            self.clear_widget()
            self.show_no_patient()
            self.cache_mid = None
            return
        
        if self.mid == self.cache_mid:
            return

    def load_patient_info(self):
        from patient_info import get_patient_info
        
        if self.mid is None:
            # 기존 내용 삭제
            for i in reversed(range(self.info_layout.count())):
                widget = self.info_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            self.show_no_patient()
            self.cache_mid = None
            return
        
        if self.mid == self.cache_mid:
            return
        
        # 기존 내용 삭제
        self.clear_widget()


        # ================================
        # 오른쪽 Info 테이블 표시
        # ================================

        info_title = QLabel(f"Patient Information (ID: {self.mid})")
        info_title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        info_title.setStyleSheet("color: #222222;")
        self.info_layout.addWidget(info_title)
        self.info_layout.setSpacing(14)
        

        # QTableWidget 생성
        grid = QTableWidget()
        grid.setColumnCount(2)
        grid.setRowCount(6)
        grid.horizontalHeader().setVisible(False)
        grid.verticalHeader().setVisible(False)
        grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        grid.setShowGrid(False)
        grid.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                font-size: 14px;
            }
        """)
    
        for i, (key, value) in enumerate(get_patient_info(self.mid).items()):
                grid.setItem(i, 0, QTableWidgetItem(str(key)))
                grid.setItem(i, 1, QTableWidgetItem(str(value)))

        self.info_layout.addWidget(grid)

        self.left_box.setStyleSheet("""
            QFrame {
                background-color: #E9ECEF;
                border-radius: 12px;
                padding: 0px;
            }
        """)




        self.cache_mid = self.mid
    

    
    def clear_widget(self):
        for i in reversed(range(self.info_layout.count())):
                widget = self.info_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

                   
    
    def show_no_patient(self):
        label = QLabel("No patient selected")
        label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        label.setStyleSheet("color: #999999;")
        self.info_layout.addWidget(label)

        
# 나머지 페이지들 동일
class PatientInfoPage(BasePage):
    def __init__(self):
        super().__init__()
        title = QLabel("Patient Info Page")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333333;")
        self.layout.addWidget(title)
        self.layout.addStretch()

class ReportsPage(BasePage):
    def __init__(self):
        super().__init__()
        title = QLabel("Reports Page")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333333;")
        self.layout.addWidget(title)
        self.layout.addStretch()

class PrescriptionsPage(BasePage):
    def __init__(self):
        super().__init__()
        title = QLabel("Prescriptions Page")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333333;")
        self.layout.addWidget(title)
        self.layout.addStretch()

# 메인 윈도우
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Management Dashboard")
        self.resize(1200, 720)
        self.setStyleSheet("background-color: #F4F6F9;")

        quit_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        quit_shortcut.activated.connect(QApplication.quit)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 사이드바
        sidebar = QFrame()
        sidebar.setFixedWidth(170)
        sidebar.setStyleSheet("""
            QFrame { background-color: #037091; color: white; }
            QPushButton {
                background-color: transparent; color: white; border: none;
                font-size: 15px; text-align: left; padding: 10px 20px;
            }
            QPushButton:hover { background-color: #0096A6; }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        logo_label = QLabel()
        pixmap = QPixmap('marker-hospital.png')
        scaled_pixmap = pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(logo_label)

        title_label = QLabel("electronics hospital")
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sidebar_layout.addWidget(title_label)
        sidebar_layout.addSpacing(30)

        btn_dashboard = QPushButton("Dashboard")
        btn_patient = QPushButton("Patient Info")
        btn_reports = QPushButton("Reports")
        btn_prescriptions = QPushButton("Prescriptions")
        btn_settings = QPushButton("Settings")

        for b in [btn_dashboard, btn_patient, btn_reports, btn_prescriptions, btn_settings]:
            sidebar_layout.addWidget(b)
        sidebar_layout.addStretch()

        # 스택 구조
        self.stack = QStackedWidget()
        self.dashboard_page = DashboardPage()
        self.patient_page = PatientInfoPage()
        self.reports_page = ReportsPage()
        self.prescriptions_page = PrescriptionsPage()

        for page in [self.dashboard_page, self.patient_page, self.reports_page, self.prescriptions_page]:
            page.set_main_window(self)

        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.patient_page)
        self.stack.addWidget(self.reports_page)
        self.stack.addWidget(self.prescriptions_page)

        btn_dashboard.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_patient.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_reports.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        btn_prescriptions.clicked.connect(lambda: self.stack.setCurrentIndex(3))

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack)

# 실행부
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

