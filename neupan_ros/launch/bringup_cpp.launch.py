"""Bring up the C++ NeuPAN planner wrapper and PX4 control owner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_USE_ROBOT_YAML = "__from_robot_yaml__"
_DISABLE_RKNN = "__disabled__"


def _launch_setup(context, *args, **kwargs):
    log_level = LaunchConfiguration("log_level")
    planner_config_file = LaunchConfiguration("planner_config_file")
    dune_rknn_metadata_file = LaunchConfiguration("dune_rknn_metadata_file")
    dune_rknn_core_mask = LaunchConfiguration("dune_rknn_core_mask")
    dune_rknn_require_device = LaunchConfiguration("dune_rknn_require_device")

    pkg_share = FindPackageShare("neupan_ros")
    config_dir = PathJoinSubstitution([pkg_share, "config"])
    robot_yaml = PathJoinSubstitution([config_dir, "robot.yaml"])

    common_arguments = ["--ros-args", "--log-level", log_level]
    planner_overrides = {
        "robot_config_dir": config_dir,
    }
    planner_file_value = planner_config_file.perform(context)
    if planner_file_value != _USE_ROBOT_YAML:
        planner_overrides["planner_config_file"] = planner_config_file

    metadata_value = dune_rknn_metadata_file.perform(context)
    if metadata_value == _DISABLE_RKNN:
        planner_overrides["dune_rknn_metadata_file"] = ""
    elif metadata_value != _USE_ROBOT_YAML:
        planner_overrides["dune_rknn_metadata_file"] = dune_rknn_metadata_file

    core_mask_value = dune_rknn_core_mask.perform(context)
    if core_mask_value != _USE_ROBOT_YAML:
        planner_overrides["dune_rknn_core_mask"] = dune_rknn_core_mask

    require_device_value = dune_rknn_require_device.perform(context)
    if require_device_value != _USE_ROBOT_YAML:
        planner_overrides["dune_rknn_require_device"] = dune_rknn_require_device

    return [
        Node(
            package="neupan_ros",
            executable="neupan_uav_node",
            name="neupan_uav_node",
            output="screen",
            emulate_tty=True,
            parameters=[robot_yaml, planner_overrides],
            arguments=common_arguments,
        ),
        Node(
            package="neupan_ros",
            executable="px4_control",
            name="px4_control",
            output="screen",
            emulate_tty=True,
            parameters=[robot_yaml],
            arguments=common_arguments,
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument(
                "planner_config_file", default_value=_USE_ROBOT_YAML
            ),
            DeclareLaunchArgument(
                "dune_rknn_metadata_file", default_value=_USE_ROBOT_YAML
            ),
            DeclareLaunchArgument(
                "dune_rknn_core_mask", default_value=_USE_ROBOT_YAML
            ),
            DeclareLaunchArgument(
                "dune_rknn_require_device", default_value=_USE_ROBOT_YAML
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
