'''
  @ Date: 2022/3/23 19:05
  @ Author: Zhao YaChen
'''
import dicom2nifti
import pydicom as dicom
import nibabel as nib
import numpy as np
from io import BytesIO
from pydicom import read_file
import os
import re
import shutil
import pandas as pd
import SimpleITK as sitk
import h5py
import scipy.ndimage as ndimage

from nipype.interfaces.ants import N4BiasFieldCorrection

import radiomics.featureextractor as FEE
from radiomics.imageoperations import getWaveletImage, getLoGImage
from radiomics.firstorder import RadiomicsFirstOrder
from dicom2nifti.exceptions import ConversionError
'''
    文件路径定义
'''
default_prefix = 'D:/Desktop/BREAST/BREAST/'
name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
default_directory = 'F:/breastMR_202003_sort'
fune_directory = 'F:/breast_dataset_patient'
output_label_info = 'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/output_label_info.csv'

root_path = 'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_TrainingData'

'''
    得到文件夹内各个文件名称
'''
def get_file_list_by_time(file_path):
  dir_list = os.listdir(file_path)
  if not dir_list:
      return
  else:
      # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
      # os.path.getmtime() 函数是获取文件最后修改时间
      # os.path.getctime() 函数是获取文件最后创建时间
      dir_list = sorted(dir_list,key=lambda x: os.path.getctime(os.path.join(file_path, x)))
      print(dir_list)
      return dir_list

def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)

# 用于生成罗列数据集的每个病人的名称
def get_file_list_by_name(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # dir_list = sorted(os.listdir(file_path))  # 文件名按字母排序
        dir_list = os.listdir(file_path)
        print(dir_list)
        dir_list.sort(key=sort_key)
        print(dir_list)
        return dir_list


'''
    将.dcm格式序列图像转化为.nii格式
'''
def addFileMetaInfo(file_path, new_floder, new_name):
    # bytestream = b'\x02\x00\x02\x00\x55\x49\x16\x00\x31\x2e\x32\x2e\x38\x34\x30\x2e\x31' \
    #              b'\x30\x30\x30\x38\x2e\x35\x2e\x31\x2e\x31\x2e\x39\x00\x02\x00\x10\x00' \
    #              b'\x55\x49\x12\x00\x31\x2e\x32\x2e\x38\x34\x30\x2e\x31\x30\x30\x30\x38' \
    #              b'\x2e\x31\x2e\x32\x00\x20\x20\x10\x00\x02\x00\x00\x00\x01\x00\x20\x20' \
    #              b'\x20\x00\x06\x00\x00\x00\x4e\x4f\x52\x4d\x41\x4c'
    #
    # fp = BytesIO(bytestream)
    # ds = read_file(fp, force=True)
    # print(ds)

    # Manually add the preamble
    fp = BytesIO()
    fp.write(b'\x00' * 128)
    fp.write(b'DICM')

    # Add the contents of the file
    f = open(file_path, 'rb')
    fp.write(f.read())
    f.close()
    fp.seek(0)

    # Read the dataset
    ds = read_file(fp)
    # print(ds)
    if not os.path.exists(new_floder):
        os.makedirs(new_floder)
    ds.save_as(os.path.join(new_floder, new_name))

# 默认参数降低调用函数的难度
def addFloderMetaInfo(old_folder, new_folder, new_name=""):
    # 得到旧文件夹的每个文件
    # 按照字母排序得到指定文件夹下包含文件的名字的列表
    dcm_file_list = listDir(old_folder)
    print(dcm_file_list)
    print(dcm_file_list.__len__())
    # 遍历list
    for x in dcm_file_list:
        addFileMetaInfo(os.path.join(old_folder, x), new_folder, x)
    return dcm_file_list.__len__()


def listDir(folder_path):
    # 只保留以dicom结尾的文件
    # return os.listdir(folder_path)
    return list(filter(lambda f: str(f).endswith('dcm'), os.listdir(folder_path)))

#.dicom转换为.nii
def dicom2nii(original_dicom_directory: str, output_file: str):
    dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=False)

def dicom2nii2(original_dicom_directory: str, output_file: str):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(original_dicom_directory)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    image_array = sitk.GetArrayFromImage(image2) # z, y, x
    origin = image2.GetOrigin() # x, y, z
    spacing = image2.GetSpacing() # x, y, z
    # image3=sitk.GetImageFromArray(image3)##其他三维数据修改原本的数据，
    sitk.WriteImage(image2,output_file) #这里可以直接换成image2 这样就保存了原来的数据成了nii格式了。

def breast_dicom2nii():
    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(
        'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_update.csv')
    dir_list = pCR_info_df["影像号"].astype(str) + " " +pCR_info_df['病人姓名'].astype(str)
    mass_labe_name_list = pCR_info_df['label名称']
    slice = 0
    # isAddMetaInfo = False
    # isAddMetaInfo = True
    isCopyLabelFile = True

    for number in range(253, 308):
        patient_name = dir_list.get(number)
        mass_labe_name = mass_labe_name_list.get(number)
        number = number + 1
        nii_directory = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d' % number
        if not os.path.exists(nii_directory):
            os.makedirs(nii_directory)
        if isCopyLabelFile:
            label_nii_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii' % (
                number, number)
            # 首先将mass-label.nii复制到目标文件夹中
            shutil.copyfile(os.path.join(default_directory, patient_name, mass_labe_name + "-label.nii"), label_nii_path)
        for j in range(1, 6):
            basic_ph_name_list = [
                # 'Ph%d_Dyn Ax Vibrant 5p' % j,
                'Ph%d_dyn Ax Vibrant 5p' % j,
                'ax VIBRANT-FLEX Ph%d' % j,
                'Ph%d:Dyn Ax Vibrant 5p' % j,
                'Ax VIBRANT-Flex Ph%d' % j,
                'Dyn Ax Vibrant Ph%d' % j,
                '50%d-Ph%d/dyn Ax Vibrant 5p' % (j, j),
                '50%d-WATER_ Ph%d/Ax VIBRANT-Flex' % (j, j),
                '50%d-WATER_ Ph%d' % (j, j),
                '60%d-Ph%d/Sag Vibrant SinglePhase' % (j, j),
                '60%d-Ph%d/dyn Ax Vibrant 5p' % (j, j),
                '60%d-WATER_ Ph%d/Sag VIBRANT-Flex' % (j, j),
                '70%d-Ph%d/dyn Ax Vibrant 5p' % (j, j),
                '70%d-Ph%d/Sag Vibrant SinglePhase' % (j, j),
                'WATER_ Ph%d_Ax VIBRANT-Flex' % j,
            ]
            for basic_ph_name in basic_ph_name_list:
                original_dicom_directory = os.path.join(default_directory, patient_name, basic_ph_name)
                if os.path.exists(original_dicom_directory):
                    basic_ph_name1 = basic_ph_name
                    break

            fune_dicom_directory = os.path.join(fune_directory, patient_name, 'ph%d' % j)
            final_directory = os.path.join(nii_directory, 'Breast_Training_%03d_ph%d.nii' % (number, j))
            try:
                isAddMetaInfo = True
                # original_dicom_directory->fune_dicom_directory
                slice = addFloderMetaInfo(original_dicom_directory, new_folder=fune_dicom_directory)
                # fune_dicom_directory->final_directory
                dicom2nii(fune_dicom_directory, final_directory)
            except (ConversionError, FileNotFoundError, ValueError):
                print("Oops! FileNotFoundError: [WinError 3] 系统找不到指定的路径。")
                isAddMetaInfo = False
                slice = listDir(original_dicom_directory).__len__()
                dicom2nii(original_dicom_directory, final_directory)
            print(final_directory + ' Done!')
        # 每处理完一个病人的病例进行csv记录
        add_data = [{'Number': str(number),
                     'File_Path': basic_ph_name1,
                     'Patient_ID Patient_Name': patient_name,
                     'Breast_subject_ID': 'Breast_Training_%03d' % number,
                     'Slice': slice,
                     'remark': isAddMetaInfo}]
        df = pd.DataFrame(add_data)
        df.to_csv(name_mapping_path, index=None, mode='a', header=None)


def breast_dicom2nii_t2():
    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(
        'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_update.csv')
    dir_list = pCR_info_df["影像号"].astype(str) + " " +pCR_info_df['病人姓名'].astype(str)
    mass_labe_name_list = pCR_info_df['label名称']
    slice = 0
    isAddMetaInfo = False
    # isAddMetaInfo = True
    isCopyLabelFile = False

    for number in range(0, 308):
        patient_name = dir_list.get(number)
        mass_labe_name = mass_labe_name_list.get(number)
        number = number + 1
        nii_directory = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d' % number
        if not os.path.exists(nii_directory):
            os.makedirs(nii_directory)
        if isCopyLabelFile:
            label_nii_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii' % (
                number, number)
            # 首先将mass-label.nii复制到目标文件夹中
            shutil.copyfile(os.path.join(default_directory, patient_name, mass_labe_name + "-label.nii"), label_nii_path)
        basic_ph_name_list = [
            'IDEAL T2',
        ]
        for basic_ph_name in basic_ph_name_list:
            original_dicom_directory = os.path.join(default_directory, patient_name, basic_ph_name)
            if os.path.exists(original_dicom_directory):
                basic_ph_name1 = basic_ph_name
                break

        fune_dicom_directory = os.path.join(fune_directory, patient_name, 't2')
        final_directory = os.path.join(nii_directory, 'Breast_Training_%03d_t2.nii' % (number))
        if isAddMetaInfo:
            # original_dicom_directory->fune_dicom_directory
            slice = addFloderMetaInfo(original_dicom_directory, new_folder=fune_dicom_directory)
            # fune_dicom_directory->final_directory
            dicom2nii(fune_dicom_directory, final_directory)
        else:
            slice = listDir(original_dicom_directory).__len__()
            dicom2nii(original_dicom_directory, final_directory)
        print(final_directory + ' Done!')
        # 每处理完一个病人的病例进行csv记录
        add_data = [{'Number': str(number),
                     'File_Path': basic_ph_name1,
                     'Patient_ID Patient_Name': patient_name,
                     'Breast_subject_ID': 'Breast_Training_%03d' % number,
                     'Slice': slice,
                     'remark': isAddMetaInfo}]
        df = pd.DataFrame(add_data)
        df.to_csv(name_mapping_path, index=None, mode='a', header=None)

'''
    .h5转化为.nii文件
'''
def h5ToNII():
    dataset = h5py.File('D:\Desktop\SSL4MIS\data\BraTS2019\data\BraTS19_TCIA01_131_1.h5', 'r')  # 指定h5文件的路径
    savepath = "D:\Desktop\SSL4MIS\data\BraTS2019\data(nii)"  # 另存为nii文件的路径
    first_level_keys = [key for key in dataset.keys()]
    for first_level_key in first_level_keys:
        if not os.path.exists(os.path.join(savepath, first_level_key)):
            os.makedirs(os.path.join(savepath, first_level_key))

        image_arr = np.array(dataset[first_level_key])
        img = sitk.GetImageFromArray(image_arr)
        img.SetSpacing([1.0, 1.0, 1.0])  # 根据需求修改spacing
        sitk.WriteImage(img, os.path.join(savepath, first_level_key, "BraTS19_TCIA01_131_1.nii.gz"))
        print(first_level_key)

'''
    取出要预测的病变区域，向四个方向延申64像素，得到voi区域=128*128*64
'''
def crop_mass_area():

    # ERROR_LIST = [10, 59, 64, 67, 73, 76, 85, 162, 165, 175, 259]
    # for i in ERROR_LIST:
    for i in range(1, 309):
        folder_path = "D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d" % (i)
        if(os.path.exists(folder_path)):
            file_path = "D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii" % (i, i)
            mri_default_path = "D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d" % (i, i)
            # nilearn读取数据
            roi_img = nib.load(file_path)
            affine = roi_img
            roi_data = roi_img.get_data()
            # 得到有label标记的slice和总层数
            label_slice_list = []
            label_slice_sum = 0
            for k in range(roi_data.shape[2]):
                test_slice = roi_data[:, :, k]
                if test_slice.max() > 0:
                    label_slice_list.append(k)
                    label_slice_sum += 1
            print('roi data shape:', label_slice_sum)
            # ndimage获取数组的质心
            CM = ndimage.measurements.center_of_mass(roi_data)
            # 手动调用内置函数来强制类型转换
            x = int(CM[0])
            y = int(CM[1])
            z = int(CM[2])
            # 定义偏移量
            offsetX = 96
            offsetY = 96
            offsetZ = 24
            # 保存为nii文件
            # nib.Nifti1Image(voi_mri_data).to_filename('Breast_Training_002_ph2_voi.nii')
            # voi_mri_data.to_filename('Breast_Training_002_ph2_voi.nii')
            # nib.save(voi_mri_data, os.path.join(folder_path, 'Breast_Training_002_ph2_voi.nii'))
            # 对6个阶段
            for j in range(1, 6):
                mri_path = mri_default_path + "_ph" + str(j) + ".nii"
                mri_img = nib.load(mri_path)
                mri_data = mri_img.get_data()
                # 三维数组的切片 mri shape -> (512, 512, 92) voi mri shape -> (256, 256, 64)/(255, 255, 63)
                # 处理特殊情况
                # 当 mri shape -> (136, 256, 256)
                # 当 roi shape -> (256, 256, 136)
                if mri_data.shape != roi_data.shape:
                    if mri_data.shape[0] == roi_data.shape[2]:
                        # mri_data = np.moveaxis(mri_data, 0, -1)
                        mri_data = np.moveaxis(mri_data, (0, 1, 2), (2, 1, 0))

                xshape = mri_data.shape[0]
                # 因为图像和label在x方向上是反的mri_data = np.moveaxis(mri_data, (0, 1, 2), (2, 1, 0))
                # 需要将图像数据进行左右反转
                # 一个example:
                # 当x=165 y=272 z=66时 此时shape_x=512 shape_y=512 shape_z=104
                # mri_data取[(512-165)-64:(512-165)+64, 272-64:272+64, 66-4:66+4]
                # roi_data取[165-64:165+64, 272-64:272+64, 66-4:66+4]

                # 处理要截取的层数超出总层数范围的情况：
                # 如果z-offsetZ<0||z+offsetZ>shape_z
                # 首先得到delta=0-(z-offsetZ)||delta=(z+offsetZ)-shape_z
                # 重新得到质心z的层数：z+delta||z-delta
                if (xshape - x) - offsetX < 0:
                    delta = 0 - ((xshape - x) - offsetX)
                    x = x - delta
                elif (xshape - x) + offsetX > mri_data.shape[0]:
                    delta = (xshape - x) + offsetX - mri_data.shape[0]
                    x = x + delta
                if y - offsetY < 0:
                    delta = 0 - (y - offsetY)
                    y = y + delta
                elif y + offsetY > mri_data.shape[1]:
                    delta = y + offsetY - mri_data.shape[1]
                    y = y - delta
                if z - offsetZ < 0:
                    delta = 0 - (z - offsetZ)
                    z = z + delta
                elif z + offsetZ > mri_data.shape[2]:
                    delta = z + offsetZ - mri_data.shape[2]
                    z = z - delta
                voi_mri_data = mri_data[(xshape - x) - offsetX:(xshape - x) + offsetX,
                               y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
                voi_mri_data = np.flip(voi_mri_data, axis=0)
                # voi_mri_data = mri_data[x - offsetX:x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
                # voi_mri_data = mri_data[(-x) - offsetX:(-x) + offsetX, (-y) - offsetY:(-y) + offsetY, z - offsetZ:z + offsetZ]
                pair_img = nib.Nifti1Pair(voi_mri_data, np.eye(4))
                nib.save(pair_img, os.path.join(folder_path, 'Breast_Training_%03d_ph%d_voi_192x192x48.nii' % (i, j)))

            # 对label三维数组同样切片
            voi_data = roi_data[x - offsetX:x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
            pair_img = nib.Nifti1Pair(voi_data, np.eye(4))
            nib.save(pair_img, os.path.join(folder_path, 'Breast_Training_%03d_seg_voi_192x192x48.nii' % (i)))

            # 输出切片后大小
            print("mri shape ->", mri_data.shape)
            print("roi shape ->", roi_data.shape)
            print("voi mri shape ->", voi_mri_data.shape)

        def list_or_tuple_to_string(list1):
            return '('+' '.join(str(e) for e in list1)+')'

        # 每处理完一个病人的病例进行csv记录
        add_data = [{'Number': 'Breast_Training_%03d' % i, 'CM': list_or_tuple_to_string(CM), 'Shape:': list_or_tuple_to_string(mri_data.shape), 'labelSlice': list_or_tuple_to_string(label_slice_list), 'labelSlicesum': label_slice_sum, 'voiDataShape': voi_mri_data.shape}]
        df = pd.DataFrame(add_data)
        df.to_csv(output_label_info, index=None, mode='a', header=None)

'''
    for seg h5 dataset
    将.nii转化为.h5文件
    输入训练集的原图和label，输出h5文件
    先选取作为示例
    一个h5中，包含一个ph1/ph3/ph5阶段image和label
    "image":(3, 128, 128, 64)
    "label":(1, 128, 128, 64)
'''
data_types = ['_ph1_voi_debug.nii', '_ph3_voi_debug.nii', '_ph5_voi_debug.nii']
def nii2h5_seg():
    # 修改h5存放路径
    h5py_path = 'D:\Desktop\BREAST\BREAST\data'
    # h5py_path = 'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation-h5/Breast_TrainingData'
    id_num = 0
    for id_ in sorted(os.listdir(root_path)):
        # 载入所有image模态
        images = []
        for data_type in data_types:
            img_path = os.path.join(root_path, id_, id_ + data_type)
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # (64, 128, 128)
            # img = (img - img.min()) / (img.max() - img.min())
            images.append(img)
        img = np.stack(images)  # (3, 64, 128, 128)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))  # (3, 128, 128, 64)
        # 载入label模态
        mask_path = os.path.join(root_path, id_, id_ + '_seg_voi_debug.nii')
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))  # (64, 128, 128)
        # mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
        # mask = np.clip(mask, 0, 1)
        mask = mask[None, :, :, :]  # (1, 128, 128, 64)
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        print('img.shape[1:3]:', img.shape[1:4], mask.shape[1:4])
        if img.shape[1:4] != mask.shape[1:4]:
            print("Error")
        f = h5py.File(os.path.join(h5py_path, id_ + '_ph135_seg.h5'), 'w')
        f.create_dataset('image', data=img, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()
        id_num += 1
    print("Converted total {} niis to h5 files".format(id_num))

'''
    for classification h5 dataset
    将.nii转化为.h5文件
    输入训练集的原图和label，输出h5文件
    先选取作为示例
    一个h5中，包含一个ph1/ph3/ph5阶段image和label
    "image":(3, 128, 128, 64)
    "label":(1,) 0-pCR 1-non-pCR
'''
def nii2h5_classification():
    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv('D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_ori.csv')
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_df.rename({'Number': 'ID'}, axis=1, inplace=True)
    df = pCR_info_df.merge(name_mapping_df, on="ID", how="right")
    # 修改h5存放路径
    h5py_path = 'D:\Desktop\BREAST\BREAST\data'
    # h5py_path = 'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation-h5/Breast_TrainingData'
    id_num = 0
    for id_ in sorted(os.listdir(root_path)):
        # 载入所有image模态
        images = []
        for data_type in data_types:
            img_path = os.path.join(root_path, id_, id_ + data_type)
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # (64, 128, 128)
            # img = (img - img.min()) / (img.max() - img.min())
            images.append(img)
        img = np.stack(images)  # (3, 64, 128, 128)
        # img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))  # (3, 128, 128, 64)
        # 载入label模态
        # 根据Breast_subject_ID返回ID
        index = df['Breast_subject_ID'][df['Breast_subject_ID'].values == id_].index
        # 根据ID返回“病理完全缓解”
        pCR_label = (df['病理完全缓解'][index.values[0]], )   # 0/1
        f = h5py.File(os.path.join(h5py_path, id_ + '_ph135_cls.h5'), 'w')
        f.create_dataset('image', data=img, compression="gzip")
        f.create_dataset('label', data=pCR_label, compression="gzip")
        f.close()
        id_num += 1
    print("Converted total {} niis to h5 files".format(id_num))

# 使用N4BiasFieldCorrection可以校正偏差域
def correct_bias():
    for i in range(1, 309):
        folder_path = "D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d" % (i)
        if(os.path.exists(folder_path)):
            mri_default_path = "D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d" % (i, i)
            correct = N4BiasFieldCorrection()
            # 对6个阶段
            for j in range(1, 6):
                mri_path = mri_default_path + "_ph" + str(j) + ".nii"
                out_mri_path = mri_default_path + "_ph" + str(j) + "_N4Bias"+".nii"
                # 使用N4BiasFieldCorrection校正MRI图像的偏置场
                correct.inputs.input_image = mri_path
                correct.inputs.output_image = out_mri_path
                done = correct.run()
                # done.outputs.output_image
                # input_image = sitk.ReadImage(mri_path, sitk.sitkFloat64)
                # output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
                # sitk.WriteImage(output_image, out_mri_path)


# 进行特征匹配并将匹配的特征保存到csv文件中
# pyradiomics 使用示例
data_types = ['_ph1.nii', '_ph3.nii', '_ph5.nii', '_seg.nii']
data_types_name = ['dceph1', 'dceph3', 'dceph5', 'seg']
def feature_and_save_as_csv():
    default_prefix = 'D:/Desktop/BREAST/BREAST/'
    dce_train_data = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'
    name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
    para_path = 'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/Params.yml'

    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(
        'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_ori.csv')
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_df.rename({'Number': 'ID'}, axis=1, inplace=True)
    df = pCR_info_df.merge(name_mapping_df, on="ID", how="right")

    # 文件全部路径
    files = []
    for id_ in sorted(os.listdir(dce_train_data)):
        file = {}
        for data_type, data_type_name in zip(data_types, data_types_name):
            file[data_type_name] = os.path.join(dce_train_data, id_, id_ + data_type)
        index = df['Breast_subject_ID'][df['Breast_subject_ID'].values == id_].index
        pCR_label = df['病理完全缓解'][index.values[0]]  # 0/1
        file["id"] = id_
        file["label"] = pCR_label
        files.append(file)

    # 对于每个病例使用配置文件初始化特征抽取器
    result = {}
    all_dic = []
    extractor = FEE.RadiomicsFeatureExtractor(para_path)
    # print("Extraction parameters:\n\t", extractor.settings)
    # print("Enabled filters:\n\t", extractor.enabledImagetypes)
    # print("Enabled features:\n\t", extractor.enabledFeatures)
    for file in files:
        data_type_seg = data_types_name[-1]
        for data_type_except_seg in data_types_name[:-1]:
            # 运行
            result = extractor.execute(file[data_type_except_seg], file[data_type_seg])  # 抽取特征
            # print("Result type:", type(result))  # result is returned in a Python ordered dictionary
            print("Calculated features:", file["id"], data_type_except_seg)
            # for key, value in result.items():  # 输出特征
            #     print("\t", key, ":", value)
            #     dic[key] = value
            result['pid'] = file["id"]
            result['modal'] = data_type_seg
            # result['age'] =
            # result['sex'] =
            result['label'] = file["label"]
            all_dic.append(result)

    df = pd.DataFrame(all_dic)
    print(df)
    df.to_csv('D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/breast_input_ph135.csv')

if __name__ == '__main__':
    breast_dicom2nii()
    # breast_dicom2nii_t2()
    # correct_bias()
    # crop_mass_area()
    # nii2h5_seg()
    # nii2h5_classification()
    # feature_and_save_as_csv()

