import decord
import imageio
import numpy as np


def read_video_np(video_path, num_frames=-1, frame_id=None):
    """
    Args:
        video_path: str
        num_frames: int, number of frames to read, -1 means all frames
        frame_id: np.array, (N, ) int, the frame indices to read
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(video_path), decord.cpu(0))

    # get frame_id
    if frame_id is not None:  # use user specified frame_id
        assert num_frames == -1
    else:  # calculate frame_id
        if num_frames != -1:  # less than video length
            assert num_frames <= len(vr)
        else:  # all video
            num_frames = len(vr)
        frame_id = range(num_frames)

    # read
    frames = vr.get_batch(frame_id).asnumpy()
    return frames


def save_video(images, video_path, fps=30, quality=6):
    images = np.array(images)

    # remove right-bottom to make sure the size is multiple of 16
    H, W = images[0].shape[:2]
    H = H // 16 * 16
    W = W // 16 * 16
    images = images[:, :H, :W, :]  # Do not change left-upper corner

    # Save to target
    imageio.mimsave(video_path, images, fps=fps, quality=quality)
