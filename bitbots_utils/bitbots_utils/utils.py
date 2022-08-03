from typing import Any, Dict, List

import os

import yaml
import rclpy
from rclpy.node import Node
from ament_index_python import get_package_share_directory
from ros2param.api import parse_parameter_dict
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterValue as ParameterValueMsg
from rcl_interfaces.msg import ParameterType as ParameterTypeMsg
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.parameter import parameter_value_to_python, Parameter


def read_urdf(robot_name):
    urdf = os.popen(f"xacro {get_package_share_directory(f'{robot_name}_description')}"
                    f"/urdf/robot.urdf use_fake_walk:=false sim_ns:=false").read()
    return urdf


def load_moveit_parameter(robot_name):
    moveit_parameters = get_parameters_from_plain_yaml(
        f"{get_package_share_directory(f'{robot_name}_moveit_config')}/config/kinematics.yaml",
        "robot_description_kinematics.")
    robot_description = ParameterMsg()
    robot_description.name = "robot_description"
    robot_description.value = ParameterValueMsg(string_value=read_urdf(robot_name),
                                                type=ParameterTypeMsg.PARAMETER_STRING)
    moveit_parameters.append(robot_description)
    robot_description_semantic = ParameterMsg()
    robot_description_semantic.name = "robot_description_semantic"
    with open(f"{get_package_share_directory(f'{robot_name}_moveit_config')}/config/{robot_name}.srdf", "r") as file:
        value = file.read()
        robot_description_semantic.value = ParameterValueMsg(string_value=value, type=ParameterTypeMsg.PARAMETER_STRING)
    moveit_parameters.append(robot_description_semantic)
    return moveit_parameters


def get_parameters_from_ros_yaml(node_name, parameter_file, use_wildcard):
    # Remove leading slash and namespaces
    with open(parameter_file, 'r') as f:
        param_file = yaml.safe_load(f)
        param_keys = []
        if use_wildcard and '/**' in param_file:
            param_keys.append('/**')
        if node_name in param_file:
            param_keys.append(node_name)

        if param_keys == []:
            raise RuntimeError('Param file does not contain parameters for {}, '
                               ' only for nodes: {}'.format(node_name, param_file.keys()))
        param_dict = {}
        for k in param_keys:
            value = param_file[k]
            if type(value) != dict or 'ros__parameters' not in value:
                raise RuntimeError('Invalid structure of parameter file for node {}'
                                   'expected same format as provided by ros2 param dump'.format(k))
            param_dict.update(value['ros__parameters'])
        return parse_parameter_dict(namespace='', parameter_dict=param_dict)


def get_parameters_from_plain_yaml(parameter_file, namespace=''):
    with open(parameter_file, 'r') as f:
        param_dict = yaml.safe_load(f)
        return parse_parameter_dict(namespace=namespace, parameter_dict=param_dict)


def get_parameter_dict(node, prefix):
    parameter_config = node.get_parameters_by_prefix(prefix)
    config = {}
    for param in parameter_config.values():
        if "." in param.name[len(prefix) + 1:]:
            # this is a nested dict with sub values
            subparameter_name = param.name.split(".")[1]
            config[subparameter_name] = get_parameter_dict(node, prefix + "." + subparameter_name)
        else:
            # just a normal single parameter
            config[param.name[len(prefix) + 1:]] = param.value
    return config


def get_parameters_from_other_node(own_node: Node,
                                   other_node_name: str,
                                   parameter_names: List[str],
                                   service_timeout_sec: float = 20.0) -> Dict:
    """
    Used to receive parameters from other running nodes.
    Returns a dict with requested parameter name as dict key and parameter value as dict value.
    """
    client = own_node.create_client(GetParameters, f'{other_node_name}/get_parameters')
    ready = client.wait_for_service(timeout_sec=service_timeout_sec)
    if not ready:
        raise RuntimeError(f'Wait for {other_node_name} parameter service timed out')
    request = GetParameters.Request()
    request.names = parameter_names
    future = client.call_async(request)
    rclpy.spin_until_future_complete(own_node, future)
    response = future.result()

    results = {}  # Received parameter
    for i, param in enumerate(parameter_names):
        results[param] = parameter_value_to_python(response.values[i])
    return results


def set_parameters_of_other_node(own_node: Node,
                                 other_node_name: str,
                                 parameter_names: List[str],
                                 parameter_values: List[Any],
                                 service_timeout_sec: float = 20.0) -> List[bool]:
    """
    Used to set parameters of another running node.
    Returns a list of booleans indicating success or failure.
    """
    client = own_node.create_client(SetParameters, f'{other_node_name}/set_parameters')
    ready = client.wait_for_service(timeout_sec=service_timeout_sec)
    if not ready:
        raise RuntimeError(f'Wait for {other_node_name} parameter service timed out')
    request = SetParameters.Request()

    for name, value in zip(parameter_names, parameter_values):
        param = Parameter(name=name, value=value)
        parameter_value_msg = param.to_parameter_msg()
        request.parameters.append(parameter_value_msg)

    future = client.call_async(request)
    rclpy.spin_until_future_complete(own_node, future)
    response = future.result()
    return [res.success for res in response.results]
