import torch

import hmr4d.utils.matrix as matrix


class TUMTrajHelper:
    """
    Here we use this to transform the output of DPVO: https://github.com/princeton-vl/DPVO to
    transformation matrix. Thus we ignore the timestaps cues in the TUM format.

    Ref: [TUM format docs](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
    """

    def __init__(
        self,
        path: str,  # path to TUM.txt file
    ) -> None:
        self._data = self._read_data(path)
        self._frame_cnt = len(self._data["timestamp"])

        self._T_c2w = self._calculate_T_c2w(self._data)
        self._T_w2c = self._T_c2w.inverse()

    def _read_data(self, path: str):
        data = {
            "timestamp": [],
            "traj": [],
            "orientation_quat": [],
        }
        # read line by line
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # line: time x y z qx qy qz qw
                str_datas = line.split(" ")
                float_datas = [float(s) for s in str_datas]
                # extract data
                timestamp = float_datas[0]
                traj = float_datas[1:4]
                orientation = float_datas[4:8]
                # store data
                data["timestamp"].append(timestamp)
                data["traj"].append(traj)
                data["orientation_quat"].append(orientation)
        # to tensor
        for k, v in data.items():
            data[k] = torch.tensor(v)
        return data

    def _calculate_T_c2w(self, traj):
        xyz = traj["traj"]  # global position
        quat = traj["orientation_quat"]  # unit quaternion with respect to the world origin
        rot = matrix.rot_matrix_from_quaternion(quat)

        # compose transformation matrix
        """
        T = ( rot xyz )
            ( 000   1 )
        """
        T_c2w = torch.eye(4)[None].repeat(self._frame_cnt, 1, 1)
        T_c2w[:, :3, :3] = rot.inverse()
        T_c2w[:, :3, 3] = xyz
        return T_c2w

    @property
    def T_c2w(self):
        return self._T_c2w

    @property
    def T_w2c(self):
        return self._T_w2c

    @property
    def timestamps(self):
        return self._data["timestamp"]

    @property
    def traj(self):
        return self._data["traj"]
