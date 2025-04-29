import os
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):

    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
            pts_filename=info['lidar_path'],
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsics = []
            lidar2cam_rts = []
            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar = np.eye(4)
                cam2lidar[:3, :3] = cam2lidar_r
                cam2lidar[:3, 3] = cam2lidar_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(np.linalg.inv(cam2lidar))

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                cam_intrinsics=cam_intrinsics,
                lidar2cam=np.stack(lidar2cam_rts),
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict


@DATASETS.register_module()
class CustomNuScenesDatasetVal(CustomNuScenesDataset):
    """Custom NuScenes dataset for validation visualization."""

    def __len__(self):
        return 4
    
    def __getitem__(self, idx):
        idx = 80
        return super().__getitem__(idx)
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        return {"bbox_mAP": 0.5, 
                "bbox_mAP_50": 0.5, 
                "bbox_mAP_75": 0.5}


@DATASETS.register_module()
class CustomNuScenesDatasetOverfit(CustomNuScenesDataset):
    """Custom NuScenes dataset for overfitting."""

    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        idx = 16934
        return super().__getitem__(idx)
    

@DATASETS.register_module()
class CustomNuScenesDatasetOverfitVal(CustomNuScenesDatasetOverfit):
    """Custom NuScenes dataset for overfitting for validation set just for 
    visualization.
    """

    def __len__(self):
        return 4
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        return {"bbox_mAP": 0.5, 
                "bbox_mAP_50": 0.5, 
                "bbox_mAP_75": 0.5}
    