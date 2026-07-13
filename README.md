# neupan_uav

`neupan_uav` 是 NeuPAN UAV 版本的 C++ runtime 工作空间源码根。它把 UAV 规划核心
从原 `NeuPAN` Python 中抽出，形成一个可独立构建、测试和部署的 C++
规划器 core，并提供 ROS 2 / PX4 封装。

当前仓库包含两个 package：

- `libneupan_uav`：ROS-independent C++ UAV planner core。
- `neupan_ros`：ROS 2 Humble / MAVROS / PX4 wrapper。


## 快速开始

依赖：

- ROS 2 Humble
- Eigen3
- yaml-cpp
- [OSQP](https://github.com/osqp/osqp) C API
- Rockchip RKNN runtime（启用真实 DUNE RKNN 推理时需要）
- MAVROS message package（`mavros_msgs`）

OSQP 推荐安装到系统路径：

```bash
cd ~
git clone https://github.com/osqp/osqp
cd osqp
mkdir -p build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo cmake --build . --target install
```

构建本工作空间：

```bash
cd /home/orangepi/neupan_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 测试

```bash
cd /home/orangepi/neupan_ws
source /opt/ros/humble/setup.bash
colcon test --packages-select libneupan_uav neupan_ros --event-handlers console_direct+
```

## 使用

默认启动 C++ UAV planner node 和 PX4 control owner：

```bash
cd /home/orangepi/neupan_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch neupan_ros bringup_cpp.launch.py
```

默认 launch 使用 `neupan_ros` 安装目录下的资源：

- `share/neupan_ros/config/planner.yaml`
- `share/neupan_ros/config/robot.yaml`
- `share/neupan_ros/models/uav_robot_default/obs_point_net_T25_N160_fp.rknn.metadata.json`
- `share/neupan_ros/models/uav_robot_default/obs_point_net_T25_N160_fp.rknn`

主要话题：

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 输入 | `/Odometry` | `nav_msgs/Odometry` | UAV pose / attitude |
| 输入 | `/cloud_registered_body` | `sensor_msgs/PointCloud2` | body frame 障碍点云 |
| 输入 | `/neupan/control/applied_cmd_vel` | `geometry_msgs/TwistStamped` | 实际发布控制量反馈 |
| 输出 | `/neupan/planner/cmd_vel` | `geometry_msgs/TwistStamped` | planner setpoint |
| 输出 | `/neupan/planner/arrived` | `std_msgs/Bool` | 到达标志 |
| PX4 | `/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | MAVROS velocity setpoint |

`prev_u` / warm-start 语义：planner 使用上一周期**实际发布给 PX4**的控制量作为下一帧
seed，而不是直接使用 planner 自己算出的第一列控制量。

## 架构

```text
neupan_uav/
├── README.md
├── libneupan_uav/              # ROS-independent C++ planner core
│   ├── include/neupan_uav/
│   ├── src/
│   ├── tests/
│   └── tests/data/
└── neupan_ros/                 # ROS 2 / PX4 wrapper
    ├── include/neupan_ros/
    ├── src/
    ├── config/
    ├── launch/
    ├── models/uav_robot_default/
    └── test/
```

`libneupan_uav` 是 plain CMake package；`neupan_ros` 是 `ament_cmake` package。
两者可以在同一个 colcon workspace 中构建，`neupan_ros` 通过
`find_package(libneupan_uav REQUIRED)` 依赖 core。

## 项目状态与路线图

**已实现：**

- C++ UAV planner core
- C++ ROS 2 / PX4 wrapper
- OSQP C API NRMP backend
- C++ RKNN runtime runner
- C++ farfield guide
- 当前实机 `planner.yaml` / `robot.yaml` 的 C++ runtime 配置加载
- `/home/orangepi/neupan_ws` 独立构建和测试

## 致谢与许可

本项目是 **[NeuPAN](https://github.com/hanruihua/NeuPAN)** 的 UAV C++ runtime
迁移产物。如果你使用本工作，请引用 NeuPAN 原论文和相关项目。
