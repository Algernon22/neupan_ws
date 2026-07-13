# NeuPAN UAV C++ Runtime

This source root contains the C++ UAV runtime extracted from the migration
workspaces:

- `libneupan_uav`: ROS-independent UAV planner core.
- `neupan_ros`: ROS2/PX4 wrapper around `libneupan_uav`.

The original `NeuPAN` and `neupan_uav_ros2_ws` trees remain useful for Python
legacy/debug entry points, training/export tools, replay comparison, and
migration history. They are not required by this C++ runtime source layout.

## Build

```bash
cd /home/orangepi/neupan_ws
source /opt/ros/humble/setup.bash
colcon build
```

## Test

```bash
cd /home/orangepi/neupan_ws
source /opt/ros/humble/setup.bash
colcon test --packages-select libneupan_uav neupan_ros --event-handlers console_direct+
```

`libneupan_uav` is a plain CMake package. `neupan_ros` is an ament package and
uses `find_package(libneupan_uav REQUIRED)` after the workspace is sourced or
when both packages are built together by colcon.

## Launch

```bash
cd /home/orangepi/neupan_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch neupan_ros bringup_cpp.launch.py
```

The launch file uses the installed `neupan_ros/config/planner.yaml`,
`neupan_ros/config/robot.yaml`, and the installed RKNN metadata/model under
`neupan_ros/models/uav_robot_default` by default. For no-RKNN smoke tests:

```bash
ros2 launch neupan_ros bringup_cpp.launch.py dune_rknn_metadata_file:=__disabled__
```
