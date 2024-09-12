"""Utilities for loading and parsing JSON files with

Currently it only supports "pi" and basic arithmetic operations.
"""

import sys
import os
import numpy as np
import json


def safe_eval(expr):
    """Safely evaluate an expression with certain names and operations"""
    allowed_names = {"pi": np.pi}
    # Only allow certain names to be used in the expression
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of name {name} is not allowed")
    return eval(expr, {"__builtins__": {}}, allowed_names)


def load_json(robot, json_file):
    """Load json file"""
    # Load json file
    pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        pardir + "/problems/" + robot + "/" + json_file + ".json", "rb"
    ) as f:
        data = json.load(f)

    # Replace the placeholders "pi", "+", "-", etc., with actual calculations
    for i, config in enumerate(data["init_configs"]):
        data["init_configs"][i] = [
            (
                safe_eval(str(val))
                if isinstance(val, str)
                and any(op in val for op in ["pi", "+", "-", "*", "/"])
                else val
            )
            for val in config
        ]
    # Add fixed rotation if not exists
    if "fixed_rotation" not in data:
        data["fixed_rotation"] = None

    # Add robot name and json file name to the data
    data["robot_name"] = robot
    data["problem_type"] = json_file
    # Remove unnecessary comments
    del data["_comments"]

    return data
