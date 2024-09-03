import ctypes
import os


class Opt(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("length", ctypes.c_int),
    ]


class RelaxedIKS(ctypes.Structure):
    pass


dir_path = os.path.dirname(os.path.realpath(__file__))
# If this is a windows system, load the dll file
if os.name == "nt":
    lib = ctypes.cdll.LoadLibrary(
        dir_path + "/target/debug/relaxed_ik_lib.dll"
    )
# If this is a linux system, load the so file
elif os.name == "posix":
    lib = ctypes.cdll.LoadLibrary(
        dir_path + "/target/debug/librelaxed_ik_lib.so"
    )
else:
    raise Exception("Unsupported OS")

lib.relaxed_ik_new.restype = ctypes.POINTER(RelaxedIKS)
lib.relaxed_ik_free.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.solve.argtypes = [
    ctypes.POINTER(RelaxedIKS),  # relaxed iK pointer
    ctypes.POINTER(ctypes.c_double),  # target position
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # target rotation
    ctypes.c_int,
]
lib.solve.restype = Opt
lib.solve_precise.argtypes = [
    ctypes.POINTER(RelaxedIKS),  # relaxed iK pointer
    ctypes.POINTER(ctypes.c_double),  # target position
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # target rotation
    ctypes.c_int,
    ctypes.c_int,
]
lib.solve_precise.restype = Opt
lib.reset.argtypes = [
    ctypes.POINTER(RelaxedIKS),  # relaxed iK pointer
    ctypes.POINTER(ctypes.c_double),  # new home joint state
    ctypes.c_int,
]
lib.get_current_poses.restype = Opt
lib.get_goal_poses.restype = Opt
lib.get_init_poses.restype = Opt


class RelaxedIKRust:
    def __init__(self, setting_file_path=None):
        """
        setting_file_path (string): path to the setting file
            if no path is given, the default setting file will be used
            /configs/loaded_robot
        """
        if setting_file_path is None:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p())
        else:
            self.obj = lib.relaxed_ik_new(
                ctypes.c_char_p(setting_file_path.encode("utf-8"))
            )

    def __exit__(self, exc_type, exc_value, traceback):
        lib.relaxed_ik_free(self.obj)

    def solve(self, positions, orientations):
        """
        Assuming the robot has N end-effectors1
        positions (1D array with length as 3*N): list of end-effector positions
        orientations (1D array with length as 4*N): list of end-effector
            orientations (in quaternion xyzw format)
        """
        pos_arr = (ctypes.c_double * len(positions))()
        quat_arr = (ctypes.c_double * len(orientations))()
        for i in range(len(positions)):
            pos_arr[i] = positions[i]
        for i in range(len(orientations)):
            quat_arr[i] = orientations[i]

        xopt = lib.solve(
            self.obj, pos_arr, len(pos_arr), quat_arr, len(quat_arr)
        )
        return xopt.data[: xopt.length]

    def solve_precise(self, positions, orientations, max_iter):
        """
        Assuming the robot has N end-effectors1
        positions (1D array with length as 3*N): list of end-effector positions
        orientations (1D array with length as 4*N): list of end-effector
            orientations (in quaternion wzyz format)
        max_iter (int): maximum number of iterations
        """
        # Convert orientation from xyzw to wxyz
        pos_arr = (ctypes.c_double * len(positions))()
        quat_arr = (ctypes.c_double * len(orientations))()
        for i in range(len(positions)):
            pos_arr[i] = positions[i]
        for i in range(len(orientations)):
            quat_arr[i] = orientations[i]
        xopt = lib.solve_precise(
            self.obj, pos_arr, len(pos_arr), quat_arr, len(quat_arr), max_iter
        )
        return xopt.data[: xopt.length]

    def get_current_poses(self):
        xopt = lib.get_current_poses(self.obj)
        poses = xopt.data[: xopt.length]
        return poses

    def get_goal_poses(self):
        xopt = lib.get_goal_poses(self.obj)
        poses = xopt.data[: xopt.length]
        return poses

    def get_init_poses(self):
        xopt = lib.get_init_poses(self.obj)
        poses = xopt.data[: xopt.length]
        return poses

    def reset(self, joint_state):
        js_arr = (ctypes.c_double * len(joint_state))()
        for i in range(len(joint_state)):
            js_arr[i] = joint_state[i]
        lib.reset(self.obj, js_arr, len(js_arr))


if __name__ == "__main__":
    relaxed_ik = RelaxedIKRust()

    pos = [0.5, 0, 0.5]
    quat = [0.5, 0.5, 0.5, 0.5]

    ori_q = [0, 0.8, 0, 0.8, 0, 0.8, 0]
    # relaxed_ik.reset(ori_q)
    print(relaxed_ik.get_init_poses())

    # for _ in range(100):
    #     joint_angles = relaxed_ik.solve(pos, quat)
    joint_angles = relaxed_ik.solve_precise(pos, quat, 1000)

    print()
    print(relaxed_ik.get_init_poses())
    print(relaxed_ik.get_goal_poses())
    print(relaxed_ik.get_current_poses())
