import importlib.util
import unittest
from pathlib import Path

from launch import LaunchContext


LAUNCH_PATH = (
    Path(__file__).resolve().parents[1] / "launch" / "bringup_cpp.launch.py"
)


def load_launch_module():
    spec = importlib.util.spec_from_file_location("bringup_cpp_launch", LAUNCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured_context(**values):
    context = LaunchContext()
    defaults = {
        "log_level": "info",
        "command_frame": "camera_init",
        "planner_config_file": "__from_robot_yaml__",
        "dune_rknn_metadata_file": "__from_robot_yaml__",
        "dune_rknn_core_mask": "__from_robot_yaml__",
        "dune_rknn_require_device": "__from_robot_yaml__",
    }
    defaults.update(values)
    context.launch_configurations.update(defaults)
    return context


def parameter_override_keys(node, context):
    keys = set()
    for params in node._Node__parameters:
        if not isinstance(params, dict):
            continue
        for key in params:
            keys.add("".join(part.perform(context) for part in key))
    return keys


class BringupLaunchTest(unittest.TestCase):
    def test_defaults_do_not_override_robot_yaml_planner_settings(self):
        module = load_launch_module()
        context = configured_context()

        planner_node, _ = module._launch_setup(context)

        self.assertEqual(parameter_override_keys(planner_node, context),
                         {"robot_config_dir", "command_frame"})

    def test_explicit_arguments_override_robot_yaml(self):
        module = load_launch_module()
        context = configured_context(
            planner_config_file="custom_planner.yaml",
            dune_rknn_metadata_file="custom.metadata.json",
            dune_rknn_core_mask="CORE_2",
            dune_rknn_require_device="false",
        )

        planner_node, _ = module._launch_setup(context)

        self.assertEqual(
            parameter_override_keys(planner_node, context),
            {
                "robot_config_dir",
                "command_frame",
                "planner_config_file",
                "dune_rknn_metadata_file",
                "dune_rknn_core_mask",
                "dune_rknn_require_device",
            },
        )

    def test_disable_rknn_overrides_metadata_with_empty_string_only(self):
        module = load_launch_module()
        context = configured_context(dune_rknn_metadata_file="__disabled__")

        planner_node, _ = module._launch_setup(context)

        self.assertEqual(
            parameter_override_keys(planner_node, context),
            {"robot_config_dir", "command_frame", "dune_rknn_metadata_file"},
        )


if __name__ == "__main__":
    unittest.main()
