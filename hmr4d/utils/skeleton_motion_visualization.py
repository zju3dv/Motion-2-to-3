import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hmr4d.utils.pylogger import Log
import logging

logger = logging.getLogger("matplotlib.animation")
logger.disabled = True

# TODO: make this file more elegant


class SkeletonAnimationGenerator:
    def __init__(self, x, fps, margin=0.5, obs_dis=0, dpi=120) -> None:
        # TODO: fit more kind of skeleton
        self.fps = fps
        self.dpi = dpi
        self.skeleton = x
        self.frame_number = x.shape[0]

        # 1. prepare meta data of motion
        self.x_min, self.x_max = self.skeleton[:, :, 0].min() - margin, self.skeleton[:, :, 0].max() + margin + obs_dis
        self.y_min, self.y_max = self.skeleton[:, :, 1].min(), self.skeleton[:, :, 1].max()
        self.z_min, self.z_max = self.skeleton[:, :, 2].min() - margin, self.skeleton[:, :, 2].max() + margin
        self.floor = self.y_min

        # 2. prepare the painting canvas
        self.fig = plt.figure(figsize=(5, 5))
        plt.tight_layout()
        ax = self.fig.add_subplot(111, projection="3d")
        ax.view_init(elev=0, azim=30, roll=90)

        # 3. style settings
        ax.set_axis_off()
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_aspect("equal")
        ax.grid(False)
        self.ax = ax

        self.update_fn = self.get_update_fn()
        self.ani = FuncAnimation(
            self.fig, self.update_fn, frames=self.frame_number, interval=1000 / self.fps, repeat=False
        )

    def get_xz_plane(self):
        # 2. generate the plane
        xz_plane = Poly3DCollection(
            # fmt:off
            [[[self.x_min, self.floor, self.z_min], 
              [self.x_max, self.floor, self.z_min], 
              [self.x_max, self.floor, self.z_max], 
              [self.x_min, self.floor, self.z_max]]]
            # fmt:on
        )
        xz_plane.set_facecolor([0.5, 0.5, 0.5, 0.5])
        return xz_plane

    def get_motions(self):
        """Get lines describing the motions."""
        # TODO: fix other format, now only Human3DML format.
        frame_num = self.skeleton.shape[0]
        motions = []
        kinematic_chain = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        ]
        # get chains: (frame_num, chain_num, 3, chain_len)
        for fid in range(frame_num):
            chains = []
            for chain in kinematic_chain:
                verts = self.skeleton[fid, chain]  # (chain_len, 3)
                verts = verts.transpose(1, 0)  # to (3, chain_len)
                chains.append(verts)
            motions.append(chains)
        return motions

    def get_trajs(self):
        """Get lines describing the trajectories."""
        trajs = self.skeleton[:, 0]
        trajs[:, 1] = self.floor
        return trajs

    def get_update_fn(self):
        """Get the function to generate things to be plotted in a frame."""
        # Prepare all the needed data here.
        scenes = [self.get_xz_plane()]
        motions = self.get_motions()
        trajs = self.get_trajs()

        def update_fn(fid):
            """Update the frame with frame id."""
            # 0. clear the canvas (dont't clear trajs)
            for line in self.ax.lines:
                line.remove()
            for collection in self.ax.collections:
                collection.remove()

            # 1. draw the scene
            for scene in scenes:
                self.ax.add_collection3d(scene)
            # 2. draw the skeleton
            chains = motions[fid]
            # 2.1 generate the colors of each chain
            colors = plt.cm.rainbow(np.linspace(0, 1, len(chains) + 1))
            colors_chain = colors[:-1]
            color_traj = colors[-1]
            # 2.2. draw the chains
            for chain, color in zip(chains, colors_chain):
                self.ax.plot(chain[0], chain[1], chain[2], color=color)
            # 3. draw the trajectory(a xz-point)
            self.ax.plot(trajs[: fid + 1, 0], trajs[: fid + 1, 1], trajs[: fid + 1, 2], color=color_traj)

        return update_fn

    def save(self, save_path):
        """Save the animation."""
        self.ani.resume()
        self.ani.save(save_path, fps=self.fps, dpi=self.dpi)

    def show(self):
        """Show the animation."""
        self.ani.resume()
        plt.show()


if __name__ == "__main__":
    """Sample of usage."""
    # prepare config
    # load data
    x0 = np.load("in.npy")
    x0 = x0.reshape(-1, 22, 3)  # the shape of x0 should be (frame, joints, xyz), i.e. (f, 22, 3)
    ag = SkeletonAnimationGenerator(x0, fps=30)
    ag.save("test.mp4")
    ag.show()
