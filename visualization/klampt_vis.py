"""Visualization for redundancy resolution using Klampt visualization"""

import time
import numpy as np

from OpenGL.GL import glEnable, glDisable, glColor3f, GL_LIGHTING
from OpenGL.GL import glBegin, glEnd
from OpenGL.GL import glLineWidth, GL_LINE_STRIP, GL_LINES
from OpenGL.GL import glPointSize, GL_POINTS
from OpenGL.GL import glVertex3f, glVertex3fv

from klampt import vis
from klampt import TransformPoser
from klampt.vis.glcommon import GLWidgetPlugin
from klampt.vis import gldraw

from grr.utils import matrix_to_quat

RED = [0.6350, 0.0780, 0.1840]
GREEN = [0, 0.6078, 0.3333]
BLUE = [0, 0.4470, 0.7410]
YELLOW = [0.9290, 0.6940, 0.1250]


class GLRedundancyProgram(GLWidgetPlugin):
    """A program for visualizing redundancy resolution"""

    def __init__(self, resolution, width=1280, height=720):
        GLWidgetPlugin.__init__(self)
        self.resolution = resolution
        self.robot = resolution.robot
        self.klampt_robot = self.robot.robot
        self.config = self.klampt_robot.getConfig()

        # Set up the viewport
        self.width = width
        self.height = height
        self.clipping_planes = (0.1, 50)

        # Teleoperation end-effector widget
        self.xform_widget = TransformPoser()
        self.mode = "teleoperation"  # "teleoperation" or "inspection"

        # Visualization
        # roadmap
        self.domain = np.array(self.robot.domain)
        self.visualize_workspace = False
        self.roadmap_lists = self.load_roadmap_vertices_and_edges()
        self.disconnection_lists = self.load_disconnected_vertices_and_edges()
        # trajectory
        self.walk_path = []
        self.walk_workspace_path = []
        self.walk_progress = 0
        self.walk_workspace_progress = 0
        # time step
        self.display_timestep = 0.1

    def load_roadmap_vertices_and_edges(self):
        """Load the roadmap vertices and edges from the resolution graph"""
        graph = self.resolution.graph

        points = set()
        edges = set()
        for _, d in graph.nodes(data=True):
            points.add(tuple(d["point"][:3]))
        for i, j, _ in graph.edges(data=True):
            edges.add(
                (
                    tuple(graph.nodes[i]["point"][:3]),
                    tuple(graph.nodes[j]["point"][:3]),
                )
            )
        return list(points), list(edges)

    def load_disconnected_vertices_and_edges(self):
        """Load the disconnected vertices and edges from the resolution graph"""
        graph = self.resolution.solver.graph
        if graph is None:
            return [], []

        points = set()
        edges = set()
        for i, j, _ in graph.edges(data=True):
            # skip connected edges
            if graph.edges[i, j]["connected"]:
                continue
            # skip unfesiable nodes
            if (
                graph.nodes[i]["config"] is None
                or graph.nodes[j]["config"] is None
            ):
                continue

            points.add(tuple(graph.nodes[i]["point"][:3]))
            points.add(tuple(graph.nodes[j]["point"][:3]))
            edges.add(
                (
                    tuple(graph.nodes[i]["point"][:3]),
                    tuple(graph.nodes[j]["point"][:3]),
                )
            )

        return list(points), list(edges)

    def initialize(self):
        """Initialize the program"""
        super().initialize()
        # white background
        self.window.program.clearColor = [1, 1, 1, 1]

        # End-effector poser
        self.xform_widget.enableTranslation(True)
        self.xform_widget.enableRotation(True)
        self.xform_widget.set(*self.resolution.robot.robot_ee.getTransform())
        self.addWidget(self.xform_widget)

        return True

    def run(self):
        """Run the program"""
        vis.setPlugin(self)

        # Set up starting viewport
        vp = vis.getViewport()
        vp.w, vp.h = self.width, self.height
        vp.clippingplanes = self.clipping_planes
        # plane - top-down view
        vp.set_transform(((1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 5)), "openGL")
        # 3d - side view
        # matrix = euler_to_matrix((-90, 0, 0), degrees=True).flatten().tolist()
        # vp.set_transform((matrix, (0, -3, 0)), "openGL")
        # matrix = euler_to_matrix((-90, 90, 0), degrees=True).flatten().tolist()
        # vp.set_transform((matrix, (-3, 0.2, 0)), "openGL")

        vis.setViewport(vp)

        # Run the visualization
        vis.show()
        while vis.shown():
            time.sleep(self.display_timestep)

        # End the visualization
        vis.setPlugin(None)
        vis.kill()

    def display(self):
        """Handles the display event"""
        # Update the robot
        self.klampt_robot.setConfig(self.config)
        self.klampt_robot.drawGL()

        # Draw workspace graph
        if self.visualize_workspace:
            # Draw vertices
            glDisable(GL_LIGHTING)
            glColor3f(*YELLOW)
            glPointSize(3.0)
            glBegin(GL_POINTS)
            for point in self.roadmap_lists[0]:
                glVertex3fv(point)
            glEnd()

            # Draw edges
            glDisable(GL_LIGHTING)
            glColor3f(*GREEN)
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for point_pair in self.roadmap_lists[1]:
                glVertex3fv(point_pair[0])
                glVertex3fv(point_pair[1])
            glEnd()

        # Draw discontinuous boundaries
        else:
            # Draw vertices
            glDisable(GL_LIGHTING)
            glColor3f(*YELLOW)
            glPointSize(3.0)
            glBegin(GL_POINTS)
            for point in self.disconnection_lists[0]:
                glVertex3fv(point)
            glEnd()

            # Draw edges
            glDisable(GL_LIGHTING)
            glColor3f(*RED)
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for point_pair in self.disconnection_lists[1]:
                glVertex3fv(point_pair[0])
                glVertex3fv(point_pair[1])
            glEnd()

        # Draw workspace bound
        glDisable(GL_LIGHTING)
        glColor3f(1, 0.5, 0)
        gldraw.box(
            self.domain[:3, 0],
            self.domain[:3, 1],
            lighting=False,
            filled=False,
        )

        # Draw the workspace path if given
        if self.walk_workspace_path:
            # Draw the whole path
            glDisable(GL_LIGHTING)
            glColor3f(*RED)
            glLineWidth(10.0)
            glBegin(GL_LINE_STRIP)
            for w in self.walk_workspace_path:
                glVertex3f(w[0], w[1], w[2])
            glEnd()
            glLineWidth(1.0)

            # Draw the progress
            glDisable(GL_LIGHTING)
            glColor3f(*GREEN)
            glPointSize(20.0)
            glBegin(GL_POINTS)
            point = self.walk_workspace_path[self.walk_workspace_progress]
            glVertex3fv(point)
            glEnd()

        GLWidgetPlugin.display(self)

    # def motionfunc(self, x, y, dx, dy):
    #     """Handles mouse motion events."""
    #     super().motionfunc(x, y, dx, dy)

    def keyboardfunc(self, c, x, y):
        """Handles keyboard events"""
        if c == "h":
            print("Keyboard help:")
            print("- i: toggles between teleoperation and inspection mode")
            print("- g: toggles drawing the workspace graph")
            print("- w: performs the given trajectory")

        elif c == "i":
            if self.mode == "inspection":
                self.mode = "teleoperation"
            elif self.mode == "teleoperation":
                self.mode = "inspection"
            print("Toggled visualization mode to", self.mode)

        elif c == "g":
            self.visualize_workspace = not self.visualize_workspace

        elif c == "w":
            self.walk_workspace_progress = 0
            self.walk_progress = 0

    def set_workspace_path(self, path, interpolate=3):
        """Set the workspace path to follow"""
        num_div = interpolate
        self.walk_workspace_path = []
        # Interpolation
        for i in range(len(path) - 1):
            for _ in range(num_div):
                self.walk_workspace_path.append(path[i])
        self.walk_workspace_path.append(path[-1])
        # pause the path following
        self.walk_workspace_progress = len(self.walk_workspace_path) - 1

    def set_path(self, path, interpolate=3):
        """Set the path to follow"""
        num_div = interpolate
        self.walk_path = []
        # Interpolation
        for i in range(len(path) - 1):
            for j in range(num_div):
                u = float(j) / num_div
                self.walk_path.append(
                    self.robot.interpolate(path[i], path[i + 1], u)
                )
        self.walk_path.append(path[-1])
        # pause the path following
        self.walk_progress = len(self.walk_path) - 1

    def idle(self):
        """Called when the program is idle"""
        # If moving the widget, update the configuration
        if self.xform_widget.hasFocus():
            rotation_matrix, x = self.xform_widget.get()

            # get current point
            point = x
            if self.robot.rotation == "variable":
                rot = matrix_to_quat(rotation_matrix)
                point = np.concatenate((point, rot))

            # Teleoperation mode
            if self.mode == "teleoperation":
                config = self.resolution.teleop_solve(
                    point,
                    self.config_to_active_config(self.config),
                    max_change=0.03,
                )
            # Inspection mode
            else:
                config = self.resolution.solve(
                    point,
                    self.config_to_active_config(self.config),
                    nearest_node_only=True,
                )

            # Update the current configuration
            if config is not None:
                self.config = self.active_config_to_config(config)

            self.refresh()

        # If there is a given path, move the robot along the path
        if (self.walk_path is not None and len(self.walk_path) != 0) and (
            self.walk_progress < len(self.walk_path)
        ):
            # Update configuration path
            self.config = self.active_config_to_config(
                self.walk_path[self.walk_progress]
            )
            # Configuration path will be updated in the display() function

            self.walk_progress += 1
            self.refresh()

        # If there is a given workspace path, highlight the given target
        if (
            self.walk_workspace_path is not None
            and len(self.walk_workspace_path) != 0
        ) and (self.walk_workspace_progress < len(self.walk_workspace_path)):
            # Workspace path will be updated in the display() function

            self.walk_workspace_progress += 1
            self.walk_workspace_progress = min(
                self.walk_workspace_progress, len(self.walk_workspace_path) - 1
            )
            self.refresh()

    def config_to_active_config(self, config):
        """Convert full config to active config"""
        if config is None:
            return None

        config = np.array(config)
        return config[self.robot.active_joints]

    def active_config_to_config(self, active_config):
        """Convert active config to full config"""
        if active_config is None:
            return None

        config = np.array(self.config)
        config[self.robot.active_joints] = active_config
        return config
