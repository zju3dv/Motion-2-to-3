COLOR_NAMES = ["red", "green", "blue", "cyan", "magenta", "orange", "black"]


NBA_SKELETON = {
    "joints": [
        "Hip",  # 0
        "RHip",  # 1
        "RKnee",  # 2
        "RAnkle",  # 3
        "LHip",  # 4
        "LKnee",  # 5
        "LAnkle",  # 6
        "Neck",  # 7
        "Nose",  # 8
        "Head",  # 9
        "LShoulder",  # 10
        "LElbow",  # 11
        "LWrist",  # 12
        "RShoulder",  # 13
        "RElbow",  # 14
        "RWrist",  # 15
        "next_root",  # 16
    ],
    "kinematic_chain": [
        ["RAnkle", "RKnee", "RHip", "Hip"],
        ["LAnkle", "LKnee", "LHip", "Hip"],
        ["Hip", "Neck", "Nose", "Head"],
        ["RWrist", "RElbow", "RShoulder", "Neck"],
        ["LWrist", "LElbow", "LShoulder", "Neck"],
    ],
    "joints_category": [
        ["RHip", "RKnee", "RAnkle"],
        ["LHip", "LKnee", "LAnkle"],
        ["Neck", "Nose", "Head"],
        ["RShoulder", "RElbow", "RWrist"],
        ["LShoulder", "LElbow", "LWrist"],
        ["next_root"],
        ["Hip"],
    ],
}


GYM_SKELETON = {
    "joints": [
        "Nose",  # 0
        "LeftEye",  # 1
        "RightEye",  # 2
        "LeftEar",  # 3
        "RightEar",  # 4
        "LeftShoulder",  # 5
        "RightShoulder",  # 6
        "LeftElbow",  # 7
        "RightElbow",  # 8
        "LeftWrist",  # 9
        "RightWrist",  # 10
        "LeftHip",  # 11
        "RightHip",  # 12
        "LeftKnee",  # 13
        "RightKnee",  # 14
        "LeftAnkle",  # 15
        "RightAnkle",  # 16
        "Ball",  # 17
    ],
    "kinematic_chain": [
        ["RightHip", "RightKnee", "RightAnkle"],
        ["LeftHip", "LeftKnee", "LeftAnkle"],
        ["LeftShoulder", "LeftEar", "LeftEye", "Nose", "RightEye", "RightEar", "RightShoulder"],
        ["RightShoulder", "RightElbow", "RightWrist"],
        ["LeftShoulder", "LeftElbow", "LeftWrist"],
        ["LeftShoulder", "RightShoulder", "RightHip", "LeftHip", "LeftShoulder"],
    ],
    "joints_category": [
        ["RightHip", "RightKnee", "RightAnkle"],
        ["LeftHip", "LeftKnee", "LeftAnkle"],
        ["LeftEar", "LeftEye", "Nose", "RightEye", "RightEar"],
        ["RightShoulder", "RightElbow", "RightWrist"],
        ["LeftShoulder", "LeftElbow", "LeftWrist"],
        ["Ball"],
    ],
}

H36M_SKELETON = {
    "joints": [
        "Hip",  # 0
        "RHip",  # 1
        "RKnee",  # 2
        "RAnkle",  # 3
        "LHip",  # 4
        "LKnee",  # 5
        "LAnkle",  # 6
        "Spine",  # 7
        "Neck",  # 8
        "Nose",  # 9
        "Head",  # 10
        "LShoulder",  # 11
        "LElbow",  # 12
        "LWrist",  # 13
        "RShoulder",  # 14
        "RElbow",  # 15
        "RWrist",  # 16
        "next_root",  # 17
    ],
    "kinematic_chain": [
        ["RAnkle", "RKnee", "RHip", "Hip"],
        ["LAnkle", "LKnee", "LHip", "Hip"],
        ["Hip", "Spine", "Neck", "Nose", "Head"],
        ["RWrist", "RElbow", "RShoulder", "Neck"],
        ["LWrist", "LElbow", "LShoulder", "Neck"],
    ],
    "joints_category": [
        ["RHip", "RKnee", "RAnkle"],
        ["LHip", "LKnee", "LAnkle"],
        ["Spine", "Neck", "Nose", "Head"],
        ["RShoulder", "RElbow", "RWrist"],
        ["LShoulder", "LElbow", "LWrist"],
        ["next_root"],
        ["Hip"],
    ],
}



SMPL_SKELETON = {
    "joints": [
        "pelvis",  # 0
        "left_hip",  # 1
        "right_hip",  # 2
        "spine1",  # 3
        "left_knee",  # 4
        "right_knee",  # 5
        "spine2",  # 6
        "left_ankle",  # 7
        "right_ankle",  # 8
        "spine3",  # 9
        "left_foot",  # 10
        "right_foot",  # 11
        "neck",  # 12
        "left_collar",  # 13
        "right_collar",  # 14
        "head",  # 15
        "left_shoulder",  # 16
        "right_shoulder",  # 17
        "left_elbow",  # 18
        "right_elbow",  # 19
        "left_wrist",  # 20
        "right_wrist",  # 21
        "next_root",  # 22 if exists
    ],
    "kinematic_chain": [
        ["pelvis", "right_hip", "right_knee", "right_ankle", "right_foot"],
        ["pelvis", "left_hip", "left_knee", "left_ankle", "left_foot"],
        ["pelvis", "spine1", "spine2", "spine3", "neck", "head"],
        ["spine3", "right_collar", "right_shoulder", "right_elbow", "right_wrist"],
        ["spine3", "left_collar", "left_shoulder", "left_elbow", "left_wrist"],
    ],
    "joints_category": [
        ["right_hip", "right_knee", "right_ankle", "right_foot"],
        ["left_hip", "left_knee", "left_ankle", "left_foot"],
        ["spine1", "spine2", "spine3", "neck", "head"],
        ["right_collar", "right_shoulder", "right_elbow", "right_wrist"],
        ["left_collar", "left_shoulder", "left_elbow", "left_wrist"],
        ["next_root"],
        ["pelvis"],
    ],
}
