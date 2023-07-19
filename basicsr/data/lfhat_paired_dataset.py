import cv2
import numpy as np
import os.path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torchvision.transforms import ToTensor

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import DATASET_REGISTRY
# from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
# from basicsr.utils.registry import DATASET_REGISTRY

from skimage import io


@DATASET_REGISTRY.register()
class LightFieldPairedDataset(data.Dataset):

    def __init__(self, opt):
        super(LightFieldPairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        lq_path = gt_path.replace(self.gt_folder, self.lq_folder)
        # img_gt_bytes = self.file_client.get(gt_path, 'gt')
        # img_gt = imfrombytes(img_gt_bytes, float32=True)
        # img_lq_bytes = self.file_client.get(lq_path, 'lq')
        # img_lq = imfrombytes(img_lq_bytes, float32=True)
        img_gt = io.imread(gt_path).astype(np.float32) / 255.0
        img_lq = io.imread(lq_path).astype(np.float32) / 255.0
        # img_lq = imfrombytes(img_lq_bytes, float32=True)
        # # modcrop
        # size_h, size_w, _ = img_gt.shape
        # size_h = size_h - size_h % scale
        # size_w = size_w - size_w % scale
        # img_gt = img_gt[0:size_h, 0:size_w, :]

        # # generate training pairs
        # size_h = max(size_h, self.opt['gt_size'])
        # size_w = max(size_w, self.opt['gt_size'])
        # img_gt = cv2.resize(img_gt, (size_w, size_h))
        # img_lq = imresize(img_gt, 1 / scale)

        # img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        # img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)

        # # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # # color space transform
        # if 'color' in self.opt and self.opt['color'] == 'y':
        #     img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
        #     img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = ToTensor()(img_gt)
        img_lq = ToTensor()(img_lq)
        # img_lq = img_lq.permute(1, 0, 2)
        img_lq = img_lq.permute(1, 2, 0)
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)

# @DATASET_REGISTRY.register()
# class LightFieldPairedDataset(data.Dataset):
#     """Paired image dataset for image restoration.

#     Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

#     There are three modes:

#     1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
#     2. **meta_info_file**: Use meta information file to generate paths. \
#         If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
#     3. **folder**: Scan folders to generate paths. The rest.

#     Args:
#         opt (dict): Config for train datasets. It contains the following keys:
#         dataroot_gt (str): Data root path for gt.
#         dataroot_lq (str): Data root path for lq.
#         meta_info_file (str): Path for meta information file.
#         io_backend (dict): IO backend type and other kwarg.
#         filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
#             Default: '{}'.
#         gt_size (int): Cropped patched size for gt patches.
#         use_hflip (bool): Use horizontal flips.
#         use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
#         scale (bool): Scale, which will be added automatically.
#         phase (str): 'train' or 'val'.
#     """

#     def __init__(self, opt):
#         super(LightFieldPairedDataset, self).__init__()
#         self.opt = opt
#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None

#         self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'

#         if self.io_backend_opt['type'] == 'lmdb':
#             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
#             self.io_backend_opt['client_keys'] = ['lq', 'gt']
#             self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
#         elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
#             self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                                                           self.opt['meta_info_file'], self.filename_tmpl)
#         else:
#             self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         scale = self.opt['scale']

#         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
#         # image range: [0, 1], float32.
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         img_gt = imfrombytes(img_bytes, float32=True)
#         lq_path = self.paths[index]['lq_path']
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         img_lq = imfrombytes(img_bytes, float32=True)

#         # augmentation for training
#         if self.opt['phase'] == 'train':
#             gt_size = self.opt['gt_size']
#             # random crop
#             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
#             # flip, rotation
#             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

#         # color space transform
#         if 'color' in self.opt and self.opt['color'] == 'y':
#             img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
#             img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

#         # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
#         # TODO: It is better to update the datasets, rather than force to crop
#         if self.opt['phase'] != 'train':
#             img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

#         # BGR to RGB, HWC to CHW, numpy to tensor
#         img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
#         # normalize
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_gt, self.mean, self.std, inplace=True)

#         return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

#     def __len__(self):
#         return len(self.paths)

