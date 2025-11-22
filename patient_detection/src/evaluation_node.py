import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import csv
import time
import math

class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')

        # ============================
        # 데이터 로그
        # ============================
        self.cmd_log = []
        self.odom_log = []
        self.need_dist_log = []

        # 이전 상태 (delta 계산용)
        self.prev_time_cmd = None
        self.prev_x = None
        self.prev_y = None

        # 시작 시각 (30초 타이머)
        self.start_time = time.time()

        # ============================
        # ROS 구독 설정
        # ============================
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/evaluation/target_distance', self.need_dist_callback, 10)

        # 타이머 (0.1초마다 체크)
        self.create_timer(0.1, self.check_timeout)

        self.get_logger().info("EvaluationNode started. Waiting for data...")

    # ============================================================
    # 1) cmd_vel → Δdistance_cmd
    # ============================================================
    def cmd_callback(self, msg):
        now = time.time()

        if self.prev_time_cmd is None:
            self.prev_time_cmd = now
            return

        dt = now - self.prev_time_cmd
        d_cmd = msg.linear.x * dt

        self.cmd_log.append({
            "time": now,
            "distance_cmd": d_cmd
        })

        self.prev_time_cmd = now

    # ============================================================
    # 2) odom → Δdistance_odom
    # ============================================================
    def odom_callback(self, msg):
        now = time.time()

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is None:
            self.prev_x = x
            self.prev_y = y
            self.get_logger().info("Odom first received.")
            return

        dx = x - self.prev_x
        dy = y - self.prev_y
        d_odom = math.sqrt(dx * dx + dy * dy)

        self.odom_log.append({
            "time": now,
            "distance_odom": d_odom
        })

        self.prev_x = x
        self.prev_y = y

    # ============================================================
    # 3) need_dist (실제 pipeline에서 보내는 목표 이동거리)
    # ============================================================
    def need_dist_callback(self, msg):
        now = time.time()
        self.need_dist_log.append({
            "time": now,
            "need_dist": msg.data
        })

    # ============================================================
    # 4) 30초 후 자동 종료 및 CSV 저장
    # ============================================================
    def check_timeout(self):
        if time.time() - self.start_time >= 30.0:
            self.get_logger().info("30s done. Saving CSV...")
            self.save_csv()
            self.get_logger().info("Shutting down evaluation_node.")
            rclpy.shutdown()

    # ============================================================
    # 5) CSV 저장
    # ============================================================
    def save_csv(self):
        filename = "evaluation_results.csv"

        n = min(len(self.cmd_log), len(self.odom_log), len(self.need_dist_log))

        if n == 0:
            self.get_logger().warn("No data collected. CSV will not be saved.")
            return

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "distance_cmd",
                "distance_odom",
                "need_dist",
                "error_cmd_vs_odom",
                "error_odom_vs_need"
            ])

            for i in range(n):
                t = self.cmd_log[i]["time"]
                d_cmd = self.cmd_log[i]["distance_cmd"]
                d_odom = self.odom_log[i]["distance_odom"]
                d_need = self.need_dist_log[i]["need_dist"]

                err1 = d_cmd - d_odom
                err2 = d_odom - d_need

                writer.writerow([t, d_cmd, d_odom, d_need, err1, err2])

        self.get_logger().info(f"CSV saved: {filename}")


# ============================================================
# Main
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
