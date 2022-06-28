import os
import shutil
import glob


def delete_video_files():
    """
    删除检测生成的历史数据
    :return:
    """
    work_path = os.getcwd()
    frame_split_path = os.path.join(work_path, 'frame_split/*')
    file = glob.glob(frame_split_path)
    for f in file:
        os.remove(f)

    file = glob.glob(os.path.join(work_path, 'runs/detect/*'))
    for f in file:
        os.remove(f)
