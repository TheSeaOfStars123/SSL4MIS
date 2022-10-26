'''
  @ Date: 2022/3/24 10:29
  @ Author: Zhao YaChen
'''
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.config import KeysCollection
from monai.data import Dataset, DataLoader
from scipy import ndimage
from torchvision.utils import make_grid

'''
将多模态的数据以子图的方式展示在一张图像上
'''
# def show_data_multimodal():
#     # 尝试读取文件
#     file_path = '/Users/zyc/Desktop/breast_dataset_patient/31084 SONG XIANG QING/ph1/1.2.840.113619.2.388.57473.14116753.13368.1538914746.885.dcm'
#     file_path = '/Volumes/Elements/breastMR_202003_sort/77336 SUN JIAN YING/Ph1_dyn Ax Vibrant 5p/1.2.840.113619.2.388.57473.14116753.12824.1561900597.956.dcm'
#     file_path = '/Users/zyc/Downloads/manifest-PyHQgfru6393647793776378748/ISPY1/ISPY1_1001/01-10-1985-690199-MR BREASTUNI UE-38479/42000.000000-PE Segmentation thresh70-88078/1-1.dcm'
#     ds = dicom.dcmread(file_path, force=True)
#     ds = dicom.read_file(file_path)
#     print(ds)
#
#     sample_filename = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph1.nii'
#     sample_filename_mask = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_seg.nii'  # 512,512,108
#     sample_img = nib.load(sample_filename)
#     sample_img = np.asanyarray(sample_img.dataobj)
#     sample_img = np.rot90(sample_img)
#     sample_mask = nib.load(sample_filename_mask)
#     sample_mask = np.asanyarray(sample_mask.dataobj)
#     sample_mask = np.rot90(sample_mask)
#     print("img shape ->", sample_img.shape)
#     print("mask shape ->", sample_mask.shape)
#
#     sample_filename2 = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph2.nii'
#     sample_img2 = nib.load(sample_filename2)
#     sample_img2 = np.asanyarray(sample_img2.dataobj)
#     sample_img2 = np.rot90(sample_img2)
#
#     sample_filename3 = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph3.nii'
#     sample_img3 = nib.load(sample_filename3)
#     sample_img3 = np.asanyarray(sample_img3.dataobj)
#     sample_img3 = np.rot90(sample_img3)
#
#     sample_filename4 = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph4.nii'
#     sample_img4 = nib.load(sample_filename4)
#     sample_img4 = np.asanyarray(sample_img4.dataobj)
#     sample_img4 = np.rot90(sample_img4)
#
#     mask_WT = sample_mask.copy()
#     show_data(sample_img, sample_img2, sample_img3, sample_img4, mask_WT, 78)
#
# def show_data(sample_img, sample_img2, sample_img3, sample_img4, mask_WT, index):
#     # https://matplotlib.org/3.3.2/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
#     # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
#     fig = plt.figure(figsize=(20, 10))
#     gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])
#
#     #  Varying density along a streamline
#     ax0 = fig.add_subplot(gs[0, 0])
#     flair = ax0.imshow(sample_img[:, :, index], cmap='bone')
#     ax0.set_title("FLAIR", fontsize=18, weight='bold', y=-0.2)
#     fig.colorbar(flair)
#
#     #  Varying density along a streamline
#     ax1 = fig.add_subplot(gs[0, 1])
#     t1 = ax1.imshow(sample_img2[:, :, index], cmap='bone')
#     ax1.set_title("T1", fontsize=18, weight='bold', y=-0.2)
#     fig.colorbar(t1)
#
#     #  Varying density along a streamline
#     ax2 = fig.add_subplot(gs[0, 2])
#     t2 = ax2.imshow(sample_img3[:, :, index], cmap='bone')
#     ax2.set_title("T2", fontsize=18, weight='bold', y=-0.2)
#     fig.colorbar(t2)
#
#     #  Varying density along a streamline
#     ax3 = fig.add_subplot(gs[0, 3])
#     t1ce = ax3.imshow(sample_img4[:, :, index], cmap='bone')
#     ax3.set_title("T1 contrast", fontsize=18, weight='bold', y=-0.2)
#     fig.colorbar(t1ce)
#
#     #  Varying density along a streamline
#     ax4 = fig.add_subplot(gs[1, 1:3])
#
#     # ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
#     l1 = ax4.imshow(mask_WT[:, :, index], cmap='summer', )
#
#     ax4.set_title("", fontsize=20, weight='bold', y=-0.1)
#
#     _ = [ax.set_axis_off() for ax in [ax0, ax1, ax2, ax3, ax4]]
#
#     colors = [im.cmap(im.norm(1)) for im in [l1]]
#     labels = ['Non-Enhancing tumor core']
#     patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
#     # put those patched as legend-handles into the legend
#     plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize='xx-large',
#                title='Mask Labels', title_fontsize=18, edgecolor="black", facecolor='#c5c6c7')
#
#     plt.suptitle("Multimodal Scans -  Data | Manually-segmented mask - Target", fontsize=20, weight='bold')
#     fig.savefig("data_sample.png", format="png", pad_inches=0.2, transparent=False, bbox_inches='tight')
#     fig.savefig("data_sample.svg", format="svg", pad_inches=0.2, transparent=False, bbox_inches='tight')


"""
使用monai-1.0.0运行 需要切换虚拟环境py37_torch运行
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from monai.transforms import LoadImaged, AddChanneld, SpatialCropD, Compose, ScaleIntensityd, EnsureTyped, \
    MapTransform, SpatialCrop
from skimage.util import montage

'''
对单个nii文件进行crop
'''
def monai_crop():
    file = {}
    file['image'] = 'D:\Desktop\BREAST\BREAST/breast-dataset-training-validation\Breast_TrainingData\Breast_Training_005\Breast_Training_005_ph3.nii'
    file['label'] = 'D:\Desktop\BREAST\BREAST/breast-dataset-training-validation\Breast_TrainingData\Breast_Training_005\Breast_Training_005_seg.nii'
    data_dicts = []
    data_dicts.append(file)
    # 加载一个图像
    loader = LoadImaged(keys=["image", "label"], dtype=np.float32)
    sample_data = loader(data_dicts[0])
    # 计算质心
    CM = ndimage.measurements.center_of_mass(sample_data['label'].squeeze(axis=0).numpy())
    add_channel = AddChanneld(keys=["image", "label"])
    sample_data = add_channel(sample_data)
    crop = SpatialCropD(keys=["image", "label"], roi_center=CM, roi_size=(128, 128, 48))
    data_crop = crop(sample_data)
    print(data_crop['image'].shape)
    # 保存
    nib.save(nib.Nifti1Image(data_crop['image'].squeeze().numpy(), np.eye(4)), 'D:\Desktop\BREAST\BREAST/breast-dataset-training-validation'
                                                   '\Breast_TrainingData\Breast_Training_005\Breast_Training_005_ph3_monai_crop.nii')# (1, 256, 256, 30)
    nib.save(nib.Nifti1Image(data_crop['label'].squeeze().numpy(), np.eye(4)), 'D:\Desktop\BREAST\BREAST/breast-dataset-training-validation'
                                                   '\Breast_TrainingData\Breast_Training_005\Breast_Training_005_seg_monai_crop.nii')# (1, 256, 256, 30)

'''
根据breast_name_mapping.csv文件中的数据，进行mass区域展示，保存为montage形式
'''
data_types = ['_ph1_voi.nii', '_ph3_voi_128x128x48.nii', '_ph5_voi_128x128x48.nii',
              '_t2_sitk_voi_128x128x48.nii', '_dwi_sitk_voi_128x128x48.nii', '_seg_voi_128x128x48.nii']
data_types_name = ['dceph1', 'dceph3', 'dceph5', 't2', 'dwi', 'label']
def get_file_list():
    default_prefix = 'D:/Desktop/BREAST/BREAST/'
    dce_train_data = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'
    name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_path = name_mapping_df['Breast_subject_ID'].tolist()

    val_files = []
    train_files = []
    for idx, id_ in enumerate(name_mapping_path):
        file = {}
        for data_type, data_type_name in zip(data_types, data_types_name):
            file[data_type_name] = os.path.join(dce_train_data, id_, id_ + data_type)

        train_files.append(file)
    return train_files, val_files


def save_as_montage():
    pin_memory = torch.cuda.is_available()
    train_files, val_files = get_file_list()
    train_transforms = Compose(
        [
            LoadImaged(keys=["dceph3", "t2", "dwi", "label"]),
            AddChanneld(keys=["dceph3", "t2", "dwi", "label"]),
            ScaleIntensityd(keys=["dceph3", "t2", "dwi", "label"]),
            EnsureTyped(keys=["dceph3", "t2", "dwi", "label"]),  # 确保输入数据为PyTorch Tensor或numpy数组
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=pin_memory)
    for i, data in enumerate(train_loader):
        #  torch.Size([1, 1, 128, 128, 48])
        #  torch.Size([1, 1, 128, 128, 48])
        print(data["dceph3"].shape, data["label"].shape)
        dce_tensor = data['dceph3'].squeeze().cpu().detach().numpy()
        t2_tensor = data['t2'].squeeze().cpu().detach().numpy()
        dwi_tensor = data['dwi'].squeeze().cpu().detach().numpy()
        mask_tensor = data['label'].squeeze().cpu().detach().numpy()

        mask = np.rot90(montage(mask_tensor))
        fig_name = data['dceph1'][0].split("\\")[1]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 30))
        ax1.imshow(np.rot90(montage(dce_tensor)), cmap='bone')
        ax1.imshow(np.ma.masked_where(mask==False, mask), cmap='cool', alpha=0.6)
        ax1.set_title(fig_name+'_dceph3', fontsize=10)
        ax2.imshow(np.rot90(montage(t2_tensor)), cmap='bone')
        ax2.imshow(np.ma.masked_where(mask == False, mask), cmap='cool', alpha=0.6)
        ax2.set_title(fig_name + '_t2', fontsize=10)
        ax3.imshow(np.rot90(montage(dwi_tensor)), cmap='bone')
        ax3.imshow(np.ma.masked_where(mask == False, mask), cmap='cool', alpha=0.6)
        ax3.set_title(fig_name + '_dwi', fontsize=10)
        plt.savefig(fig_name + '.png')
        plt.show()

'''
根据breast_name_mapping_test.csv文件中的数据，进行mass区域展示，保存为v2和v3
'''
data_types_all = ['_ph3.nii',
                  '_ph3_voi_128x128x48.nii',
                  '_t2_sitk.nii',
                  '_dwi_sitk.nii',
                  '_seg.nii',
                  '_seg_voi_128x128x48.nii',
                  '_ph3.nii',
                  '_seg.nii']
data_types_name_all = ['dceph3',
                       'dceph3_voi',
                       't2',
                       'dwi',
                       'label',
                       'label_voi',
                       'dceph3_ori',
                       'label_ori']
# data_types_all = ['_ph3.nii', '_seg.nii']
# data_types_name_all = ['dceph3', 'label']
def get_file_list_test():
    default_prefix = 'D:/Desktop/BREAST/BREAST/'
    dce_train_data = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'
    name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping_test.csv'
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_path = name_mapping_df['Breast_subject_ID'].tolist()

    val_files = []
    train_files = []
    for idx, id_ in enumerate(name_mapping_path):
        file = {}
        for data_type, data_type_name in zip(data_types_all, data_types_name_all):
            file[data_type_name] = os.path.join(dce_train_data, id_, id_ + data_type)

        train_files.append(file)
    return train_files, val_files

class SpatialCropByRoiD(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        CM = tuple(map(int, ndimage.measurements.center_of_mass(d['label'].squeeze().numpy())))
        # CM = (448, 235, 64)  # Breast_Traning_271
        for key in self.keys:
            img = d[key]
            crop = SpatialCrop(roi_center=CM, roi_size=(128, 128, 48))
            d[key] = crop(img)
        return d

def save_center_as_pic():
    pin_memory = torch.cuda.is_available()
    train_files, val_files = get_file_list_test()
    keys = ["dceph3", "label", "dceph3_voi", "label_voi", "dceph3_ori", "label_ori"]
    train_transforms = Compose(
        [
            LoadImaged(keys=keys, dtype=np.float32),
            AddChanneld(keys=keys),
            SpatialCropByRoiD(keys=["dceph3", "label"]),
            ScaleIntensityd(keys=keys),
            EnsureTyped(keys=keys),  # 确保输入数据为PyTorch Tensor或numpy数组
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=pin_memory)
    for i, data in enumerate(train_loader):
        #  torch.Size([1, 1, 128, 128, 48])
        #  torch.Size([1, 1, 128, 128, 48])
        print(data["dceph3"].shape, data["label"].shape)

        # show data dceph3_voi(1, 1, 128, 128, 48) + decph3(1, 1, 128, 128, 48)
        # (4堆叠而成, 128, 128, 5个slice) -> (4, 5, 128, 128) -> (20个图像, 128, 128) -> (20, 1, 128, 128)
        dce_image = torch.unsqueeze(torch.stack((data["dceph3_voi"][0, 0, :, :, 10:31:5],
                                                 data["label_voi"][0, 0, :, :, 10:31:5],
                                                 data["dceph3"][0, 0, :, :, 10:31:5],
                                                 data["label"][0, 0, :, :, 10:31:5]))
                                    .permute(0, 3, 1, 2).reshape(20, 128, 128), dim=1)
        # nrow一行放5个 -> (1通道, 128*4行, 128*5列) ->(1, 512, 640) -> 转置为(512, 640, 1)
        grid_dce_image = make_grid(dce_image, nrow=5, padding=0, normalize=False)
        plt.imshow(np.transpose(grid_dce_image.numpy(), (1, 2, 0)))
        fig_name = data['t2'][0].split("\\")[1]
        plt.title(fig_name)
        plt.savefig(fig_name + '_v2.png')
        # plt.show()

        # show data decph3_ori(1, 1, 512, 512, 112)
        # (2, 512, 512, 3) -> (2, 3, 512, 512) -> (6, 512, 512) -> (6, 1, 512, 512)
        CM = ndimage.measurements.center_of_mass(data['label_ori'].squeeze().numpy())
        z = int(CM[2])
        IMAGE_WIDTH_ORI = data['dceph3_ori'].shape[2]
        dce_image = torch.unsqueeze(torch.stack((data["dceph3_ori"][0, 0, :, :, z-1:z+2],
                                                 data["label_ori"][0, 0, :, :, z-1:z+2]))
                                    .permute(0, 3, 1, 2).reshape(6, IMAGE_WIDTH_ORI, IMAGE_WIDTH_ORI), dim=1)
        # nrow一行放3个 -> (1, 512*2行, 512*3列) -> (1, 1024, 1536) -> 转置为(1024, 1536, 1)
        grid_dce_image = make_grid(dce_image, nrow=3, padding=0, normalize=False)
        plt.imshow(np.transpose(grid_dce_image.numpy(), (1, 2, 0)))
        fig_name = data['t2'][0].split("\\")[1]
        plt.title(fig_name)
        plt.savefig(fig_name + '_v3.png')
        # plt.show()

if __name__ == '__main__':
    # show_data_multimodal
    # monai_crop()
    # save_as_montage()
    save_center_as_pic()