'''
  @ Date: 2022/3/23 19:05
  @ Author: Zhao YaChen
'''
import dicom2nifti
import nibabel as nib
from io import BytesIO

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
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
from dicom2nifti.exceptions import ConversionError
from skimage import measure

'''
    文件路径定义
'''
default_prefix = 'D:/Desktop/BREAST/BREAST/'
default_prefix2 = 'D:/Desktop/'
# default_prefix = '/Users/zyc/Desktop/'

name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
name_mapping_path_t2 = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping_t2.csv'
name_mapping_path_dwi = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping_dwi.csv'
name_mapping_path_ph0 = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping_ph0.csv'
output_label_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/output_label_info.csv'
pCR_label_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_update.csv'

root_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'
h5py_path = default_prefix + 'breast-dataset-training-validation-h5/Breast_TrainingData'

default_directory = 'F:/breastMR_202003_sort'
fune_directory = 'F:/breast_dataset_patient'

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
        dir_list = sorted(dir_list, key=lambda x: os.path.getctime(os.path.join(file_path, x)))
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


'''
    .dicom转换为.nii
'''


def dicom2nii(original_dicom_directory: str, output_file: str):
    dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=False)


def breast_dicom2nii():
    # 使用df读取Breast_MR_list_update.csv文件
    pCR_info_df = pd.read_csv(pCR_label_path)
    dir_list = pCR_info_df["影像号"].astype(str) + " " + pCR_info_df['病人姓名'].astype(str)
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
            shutil.copyfile(os.path.join(default_directory, patient_name, mass_labe_name + "-label.nii"),
                            label_nii_path)
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
    pCR_info_df = pd.read_csv(pCR_label_path)
    dir_list = pCR_info_df["影像号"].astype(str) + " " + pCR_info_df['病人姓名'].astype(str)
    mass_labe_name_list = pCR_info_df['label名称']
    slice = 0
    # isAddMetaInfo = False
    isAddMetaInfo = True
    isCopyLabelFile = False

    for number in range(105, 106):
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
            shutil.copyfile(os.path.join(default_directory, patient_name, mass_labe_name + "-label.nii"),
                            label_nii_path)
        basic_ph_name_list = [
            'IDEAL T2',
            'my T2'
            'WATER_ Ax T2 FRFSE IDEAL',
            'Ax T2 STIR',
            'Ax IDEAL T2',
            '3-WATER_ IDEAL T2',
            'WATER_ IDEAL T2',
            '4-WATER_ Ax T2 FRFSE IDEAL',
            '3-WATER_ Ax T2 FRFSE IDEAL',
        ]
        for basic_ph_name in basic_ph_name_list:
            original_dicom_directory = os.path.join(default_directory, patient_name, basic_ph_name)
            if os.path.exists(original_dicom_directory):
                basic_ph_name1 = basic_ph_name
                break

        fune_dicom_directory = os.path.join(fune_directory, patient_name, 't2')
        final_directory = os.path.join(nii_directory, 'Breast_Training_%03d_t2.nii' % (number))
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
        # df.to_csv(name_mapping_path_t2, index=None, mode='a', header=None)


def breast_dicom2nii_dwi():
    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(pCR_label_path)
    dir_list = pCR_info_df["影像号"].astype(str) + " " + pCR_info_df['病人姓名'].astype(str)
    mass_labe_name_list = pCR_info_df['label名称']
    slice = 0
    # isAddMetaInfo = False
    isAddMetaInfo = True
    isCopyLabelFile = False

    for number in range(208, 209):
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
            shutil.copyfile(os.path.join(default_directory, patient_name, mass_labe_name + "-label.nii"),
                            label_nii_path)
        basic_ph_name_list = [
            'Ax DWI',
            'my DWI',
            'AX DWI B=800 224',
            'FOCUS DWI',
            '4-AX DWI B=800 224',
            'AX DWI TEST 01',
            'Ax DWI 800',
            'AX DWI TEST',
            '4-Ax STIR DWI-Dual',
            '5-AX DWI B=1000',
            '5-Ax STIR DWI-Dual',
            '5-AX DWI B=800 224',
            'AX FOCUS DWI',
            '3-AX DWI B=800 224',
            'Ax STIR DWI-Dual'
        ]
        for basic_ph_name in basic_ph_name_list:
            original_dicom_directory = os.path.join(default_directory, patient_name, basic_ph_name)
            if os.path.exists(original_dicom_directory):
                basic_ph_name1 = basic_ph_name
                break

        fune_dicom_directory = os.path.join(fune_directory, patient_name, 'dwi')
        final_directory = os.path.join(nii_directory, 'Breast_Training_%03d_dwi.nii' % (number))
        try:
            isAddMetaInfo = True
            # original_dicom_directory->fune_dicom_directory
            slice = addFloderMetaInfo(original_dicom_directory, new_folder=fune_dicom_directory)
            # fune_dicom_directory->final_directory
            dicom2nii(fune_dicom_directory, final_directory)
        except (ConversionError, FileNotFoundError, ValueError, IndexError):
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
        # df.to_csv(name_mapping_path_dwi, index=None, mode='a', header=None)


def breast_dicom2nii_ph0():
    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(pCR_label_path)
    dir_list = pCR_info_df["影像号"].astype(str) + " " + pCR_info_df['病人姓名'].astype(str)
    mass_labe_name_list = pCR_info_df['label名称']
    isCopyLabelFile = False

    for number in range(307, 309):
        patient_name = dir_list.get(number)
        mass_labe_name = mass_labe_name_list.get(number)
        if pCR_info_df['label是否排除'][number] != '是' and pCR_info_df['dce是否排除'][number] != '是' and pCR_info_df[
            't2是否排除'][number] != '是':
            number = number + 1
            nii_directory = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d' % number
            if not os.path.exists(nii_directory):
                os.makedirs(nii_directory)
            if isCopyLabelFile:
                label_nii_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii' % (
                    number, number)
                # 首先将mass-label.nii复制到目标文件夹中
                shutil.copyfile(os.path.join(default_directory, patient_name, mass_labe_name + "-label.nii"),
                                label_nii_path)
            basic_ph_name_list = [
                'my ph0',
                'ax VIBRANT-FLEX',
                'dyn Ax Vibrant 5p',
                '500-dyn Ax Vibrant 5p',
                'Dyn Ax Vibrant',
                'Dyn Ax Vibrant Ph0',
                '500-WATER_ Ax VIBRANT-Flex',
                '600-dyn Ax Vibrant 5p',
                '700-dyn Ax Vibrant 5p',
                'WATER_ Ax VIBRANT-Flex',
            ]
            for basic_ph_name in basic_ph_name_list:
                original_dicom_directory = os.path.join(default_directory, patient_name, basic_ph_name)
                if os.path.exists(original_dicom_directory):
                    basic_ph_name1 = basic_ph_name
                    break

            fune_dicom_directory = os.path.join(fune_directory, patient_name, 'ph0')
            final_directory = os.path.join(nii_directory, 'Breast_Training_%03d_ph0.nii' % (number))
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
            df.to_csv(name_mapping_path_ph0, index=None, mode='a', header=None)
        else:
            number = number + 1


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
        folder_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d" % (
            i)
        if os.path.exists(os.path.join(folder_path, 'Breast_Training_%03d_ph1_voi_128x128x48.nii' % i)):
            continue
        if os.path.exists(folder_path):
            file_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii" % (
            i, i)
            mri_default_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d" % (
            i, i)
            # nilearn读取数据
            roi_img = nib.load(file_path)
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
            offsetX = 64
            offsetY = 64
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
                nib.save(pair_img, os.path.join(folder_path, 'Breast_Training_%03d_ph%d_voi_128x128x48.nii' % (i, j)))

            # 对label三维数组同样切片
            voi_data = roi_data[x - offsetX:x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
            pair_img = nib.Nifti1Pair(voi_data, np.eye(4))
            nib.save(pair_img, os.path.join(folder_path, 'Breast_Training_%03d_seg_voi_128x128x48.nii' % (i)))

            # 输出切片后大小
            print("mri shape ->", mri_data.shape)
            print("roi shape ->", roi_data.shape)
            print("voi mri shape ->", voi_mri_data.shape)

        def list_or_tuple_to_string(list1):
            return '(' + ' '.join(str(e) for e in list1) + ')'

        # 每处理完一个病人的病例进行csv记录
        add_data = [{'Number': 'Breast_Training_%03d' % i, 'CM': list_or_tuple_to_string(CM),
                     'Shape:': list_or_tuple_to_string(mri_data.shape),
                     'labelSlice': list_or_tuple_to_string(label_slice_list), 'labelSlicesum': label_slice_sum,
                     'voiDataShape': voi_mri_data.shape}]
        df = pd.DataFrame(add_data)
        df.to_csv(output_label_path, index=None, mode='a', header=None)


'''
    取出要预测的病变区域，向四个方向延申64像素，得到voi区域=128*128*64
'''
def crop_mass_area_t2_or_dwi():
    # LABEL = '_t2_sitk'
    LABEL = '_dwi_sitk'

    # ERROR_LIST = [10, 59, 64, 67, 73, 76, 85, 162, 165, 175, 259]
    # for i in ERROR_LIST:
    for i in range(1, 309):
        folder_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d" % (
            i)
        # if os.path.exists(os.path.join(folder_path, ('Breast_Training_%03d' + LABEL + '_voi_128x128x48.nii') % i)):
        #     continue
        if os.path.exists(folder_path):
            file_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii" % (
            i, i)
            mri_default_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d" % (
            i, i)
            # nilearn读取数据
            roi_img = nib.load(file_path)
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
            offsetX = 64
            offsetY = 64
            offsetZ = 24

            mri_path = mri_default_path + LABEL + ".nii"
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
            if x - offsetX < 0:
                delta = 0 - (x - offsetX)
                x = x + delta
            elif x + offsetX > mri_data.shape[0]:
                delta = x + offsetX - mri_data.shape[0]
                x = x - delta
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
            voi_mri_data = mri_data[x - offsetX: x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
            # voi_mri_data = np.flip(voi_mri_data, axis=0)
            # voi_mri_data = mri_data[x - offsetX:x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
            # voi_mri_data = mri_data[(-x) - offsetX:(-x) + offsetX, (-y) - offsetY:(-y) + offsetY, z - offsetZ:z + offsetZ]
            pair_img = nib.Nifti1Pair(voi_mri_data, np.eye(4))
            nib.save(pair_img, os.path.join(folder_path, ('Breast_Training_%03d' + LABEL + '_voi_128x128x48.nii') % i))

            # 输出切片后大小
            print("mri shape ->", mri_data.shape)
            print("roi shape ->", roi_data.shape)
            print("voi mri shape ->", voi_mri_data.shape)

        def list_or_tuple_to_string(list1):
            return '(' + ' '.join(str(e) for e in list1) + ')'

        # 每处理完一个病人的病例进行csv记录
        add_data = [{'Number': 'Breast_Training_%03d' % i, 'CM': list_or_tuple_to_string(CM),
                     'Shape:': list_or_tuple_to_string(mri_data.shape),
                     'labelSlice': list_or_tuple_to_string(label_slice_list), 'labelSlicesum': label_slice_sum,
                     'voiDataShape': voi_mri_data.shape}]
        df = pd.DataFrame(add_data)
        # df.to_csv(output_label_path, index=None, mode='a', header=None)


'''
    for seg h5 dataset
    将.nii转化为.h5文件
    输入训练集的原图和label，输出h5文件
    先选取作为示例
    一个h5中，包含一个ph1/ph3/ph5阶段image和label
    "image":(3, 128, 128, 64)
    "label":(1, 128, 128, 64)
'''


def nii2h5_seg():
    data_types = ['_ph1_voi_debug.nii', '_ph3_voi_debug.nii', '_ph5_voi_debug.nii']
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
    data_types = ['_ph1_voi_debug.nii', '_ph3_voi_debug.nii', '_ph5_voi_debug.nii']
    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(pCR_label_path)
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_df.rename({'Number': 'ID'}, axis=1, inplace=True)
    df = pCR_info_df.merge(name_mapping_df, on="ID", how="right")
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
        pCR_label = (df['病理完全缓解'][index.values[0]],)  # 0/1
        f = h5py.File(os.path.join(h5py_path, id_ + '_ph135_cls.h5'), 'w')
        f.create_dataset('image', data=img, compression="gzip")
        f.create_dataset('label', data=pCR_label, compression="gzip")
        f.close()
        id_num += 1
    print("Converted total {} niis to h5 files".format(id_num))


'''
    使用N4BiasFieldCorrection可以校正偏差域
'''


def correct_bias():
    for i in range(1, 309):
        folder_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d" % (
            i)
        if (os.path.exists(folder_path)):
            mri_default_path = default_prefix + "breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d" % (
            i, i)
            correct = N4BiasFieldCorrection()
            # 对6个阶段
            for j in range(1, 6):
                mri_path = mri_default_path + "_ph" + str(j) + ".nii"
                out_mri_path = mri_default_path + "_ph" + str(j) + "_N4Bias" + ".nii"
                # 使用N4BiasFieldCorrection校正MRI图像的偏置场
                correct.inputs.input_image = mri_path
                correct.inputs.output_image = out_mri_path
                done = correct.run()
                # done.outputs.output_image
                # input_image = sitk.ReadImage(mri_path, sitk.sitkFloat64)
                # output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
                # sitk.WriteImage(output_image, out_mri_path)


def resample_sitk():
    resample_data_types = ['_t2.nii', '_dwi.nii']
    after_resample_data_types = ['_t2_sitk.nii', '_dwi_sitk.nii']
    file_list = sorted(os.listdir(root_path))
    for id_ in file_list[195:]:
        ph1_img = sitk.ReadImage(os.path.join(root_path, id_, id_ + '_ph1.nii'))
        for data_type, after_data_type in zip(resample_data_types, after_resample_data_types):
            resample_img = sitk.ReadImage(os.path.join(root_path, id_, id_ + data_type))
            if len(resample_img.GetSize()) == 4:
                resample_img = resample_img[:, :, :, 0]
                print(id_ + ' has 4 dimension.')
            resample_out_img = resample_image(resample_img, ph1_img.GetSize(), ph1_img.GetOrigin(),
                                              ph1_img.GetSpacing())
            sitk.WriteImage(resample_out_img, os.path.join(root_path, id_, id_ + after_data_type))
        print(id_ + ' finished!')


def resample_image(itk_image, out_size, out_origin, out_spacing=[1.0, 1.0, 2.0]):
    # 根据输出out_spacing设置新的size
    # out_size = [
    #     int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
    #     int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
    #     int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    # ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)


'''
将数据集拷贝到monailabel_dataset文件夹内
'''


def ready_for_monai_dataset():
    # 首先读取name_mapping.csv文件 排除非优质数据
    name_mapping_df = pd.read_csv(name_mapping_path, encoding='unicode_escape')
    # 遍历每行
    for idx, data in name_mapping_df.iterrows():
        if data['Exclude'] != 1.0:
            image_file_path = os.path.join(root_path, data['Breast_subject_ID'], data['Breast_subject_ID'] + '_ph3.nii')
            label_file_path = os.path.join(root_path, data['Breast_subject_ID'], data['Breast_subject_ID'] + '_seg.nii')
            destinate_image_path = default_prefix2 + 'MONAILabel_datasets/myTask01_Breast'
            destinate_label_path = default_prefix2 + 'MONAILabel_datasets/myTask01_Breast/labels/final'
            destinate_name = data['Breast_subject_ID'] + '_ph3.nii'
            shutil.copyfile(image_file_path, os.path.join(destinate_image_path, destinate_name))
            shutil.copyfile(label_file_path, os.path.join(destinate_label_path, destinate_name))
            print(destinate_name + 'Done!')


import cv2

"""
dirtype:proposed/deepedit 29 2 4 3 24
029:
    center_point = [128 - 66, 128 - 61, 23]
    fore_points = [[128 - 58, 128 - 48, 23]]
    back_points = [[128 - 49, 128 - 56, 23]]
"""
def getResultPicture(dir_type, i, label_number, fore_number, back_number, index, need_point, center_point, fore_points,
                     back_points):
    img_file = "D:\Desktop\MONAILabel_datasets1\myTask06_BreastCrop\Breast_Training_%03d_ph3.nii" % (i)
    gt_file = "D:\Desktop\MONAILabel_datasets1\myTask06_BreastCrop\labels/final/Breast_Training_%03d_ph3.nii" % (i)
    label_file = 'D:\Desktop\MONAILabel_datasets1\myTask06_BreastCrop\labels\%s_test_labels\Breast_Training_%03d_ph3.nii' % (
    dir_type, i)
    label_scibble_file = 'D:\Desktop\MONAILabel_datasets1\myTask06_BreastCrop\labels\%s_test_labels\Breast_Training_%03d_ph3_with_scribble.nii' % (
    dir_type, i)
    img = nib.load(img_file)
    img = np.asarray(img.dataobj)
    img = np.rot90(img)
    img = np.fliplr(img)
    gt = nib.load(gt_file)
    gt = np.asarray(gt.dataobj)
    gt = np.rot90(gt)
    gt = np.fliplr(gt)
    if os.path.exists(label_scibble_file):
        label = nib.load(label_scibble_file)
        label = np.asarray(label.dataobj)
        label = np.rot90(label)
        label = np.fliplr(label)
        label_mass = np.where((label == label_number) | (label == fore_number), label_number, 0)
        label_fore_scr = np.where(label == fore_number, label, 0)
        label_back_scr = np.where(label == back_number, label, 0)
    else:
        label = nib.load(label_file)
        label = np.asarray(label.dataobj)
        label = np.rot90(label)
        label_mass = np.fliplr(label)
    # ================================================================= slice_index =================================================================
    slice_index = index - 1
    gt_index = gt[:, :, slice_index, None].astype(np.uint8)
    label_mass = label_mass[:, :, slice_index, None].astype(np.uint8)
    if os.path.exists(label_scibble_file):
        label_fore_scr = label_fore_scr[:, :, slice_index, None].astype(np.uint8)
        label_back_scr = label_back_scr[:, :, slice_index, None].astype(np.uint8)
    img_index = img[:, :, slice_index, None]
    img_index *= 255
    img_index = img_index.astype(np.uint8)
    plt.imshow(img_index, cmap="gray")
    # ================================================================ cv2.findContours =================================
    # contours, _ = cv2.findContours(gt_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img_index, contours, -1, (189, 183, 107), 1)
    # plt.imshow(img_index, cmap="gray")
    # ================================================================= canny ================================================================
    edges = cv2.Canny(gt_index, 0, 2)
    plt.plot(np.nonzero(edges)[1], np.nonzero(edges)[0], 'gd', markersize=2)
    edges2 = cv2.Canny(label_mass, 0, 2)
    plt.plot(np.nonzero(edges2)[1], np.nonzero(edges2)[0], 'yd', markersize=2)
    # ================================================================= meature =================================================================
    # img_index = img_index.squeeze()
    # gt_index = gt_index.squeeze()
    # contours = measure.find_contours(gt_index, 0.5)
    # for c in contours:
    #     c = np.around(c).astype(np.int)
    #     img_index[c[:, 0], c[:, 1]] = np.array((255))
    # cv2.imwrite('test.png', img_index)
    # ================================================================= point =================================================================
    if need_point:
        plt.plot(center_point[0], center_point[1], 'o', markersize=5, color='cyan')
        for fore_point in fore_points:
            plt.plot(fore_point[0], fore_point[1], 'o', markersize=3, color='cyan')
        for back_point in back_points:
            plt.plot(back_point[0], back_point[1], 'o', markersize=4, color='orangered')
    # ================================================================= scribbles ================================================================
    if os.path.exists(label_scibble_file):
        plt.plot(np.nonzero(label_fore_scr)[1], np.nonzero(label_fore_scr)[0], 'o', markersize=2, alpha=0.5,
                 color='blue')
        plt.plot(np.nonzero(label_back_scr)[1], np.nonzero(label_back_scr)[0], 'o', markersize=2, alpha=0.5,
                 color='red')
    plt.axis('off')
    plt.savefig(
        'D:\Desktop\MONAILabel_datasets1\myTask06_BreastCrop\labels\%s_test_labels\Breast_Training_%03d_ph3.png' % (
        dir_type, i)
        , bbox_inches='tight', pad_inches=-0.1)
    plt.show()


def getResultPictureForMI(NAME, index, extreme_points, back_points, CM):
    img_path = "D:\Desktop\MONAILabel_datasets1\myTask06_Breast\MIDeepSeg_image\%s_slice%d.png" % (NAME, index)
    gt_file = "D:\Desktop\MONAILabel_datasets1\myTask06_Breast\labels/final\%s.nii" % (NAME)
    label_file = "D:\Desktop\MONAILabel_datasets1\myTask06_Breast\labels\MIDeepSeg_test_labels\%s_result.png" % (NAME)
    save_file = "D:\Desktop\MONAILabel_datasets1\myTask06_Breast\labels\MIDeepSeg_test_labels\%s.png" % (NAME)
    # 画img
    img = np.asarray(Image.open(img_path))
    img = img[CM[1] - 64:CM[1] + 64, CM[0] - 64:CM[0] + 64, 0]
    plt.imshow(img, cmap="gray")
    # 画gt轮廓
    gt = nib.load(gt_file)
    gt = np.asarray(gt.dataobj)
    gt = np.rot90(gt)
    gt = np.fliplr(gt)
    slice_index = index - 1
    gt_index = gt[:, :, slice_index].astype(np.uint8)
    # 画label轮廓
    label_mass = np.asarray(Image.open(label_file))
    # ================================================================= canny ================================================================
    edges = cv2.Canny(gt_index, 0, 2)
    edges = edges[CM[1] - 64:CM[1] + 64, CM[0] - 64:CM[0] + 64]
    plt.plot(np.nonzero(edges)[1], np.nonzero(edges)[0], 'gd', markersize=2)
    edges2 = cv2.Canny(label_mass, 0, 2)
    edges2 = edges2[CM[1] - 64:CM[1] + 64, CM[0] - 64:CM[0] + 64]
    plt.plot(np.nonzero(edges2)[1], np.nonzero(edges2)[0], 'yd', markersize=2)
    # 画极值点
    xs = CM[0] - 64
    ys = CM[1] - 64
    for point in extreme_points:
        plt.plot(point[0] - xs, point[1] - ys, 'mo', markersize=4)
    for back_point in back_points:
        if back_point[0] - xs > 0 and back_point[1] - ys > 0 and back_point[0] - xs < 128 and back_point[1] - ys < 128:
            plt.plot(back_point[0] - xs, back_point[1] - ys, 'o', markersize=4, color='orangered')
    plt.axis('off')
    plt.savefig(save_file, bbox_inches='tight', pad_inches=-0.1)
    plt.show()


def getScribbleData():
    # file_data = pd.read_csv('C:/Users/10099/Desktop/proposes_Dataset - 副本.csv')
    # file_data = pd.read_csv('C:/Users/10099/Desktop/econet_Dataset - 副本.csv')
    file_data = pd.read_csv('C:/Users/10099/Desktop/graphcuts_Dataset - 副本.csv')
    temp_y = []
    start_index = 400
    end_index = start_index + 100
    add_data = []
    save_path = 'C:/Users/10099/Desktop/testtest3.csv'
    for idx, data in file_data.iterrows():
        if data[0] >= start_index:
            if data[0] < end_index:
                temp_y.append(data[1])
            else:
                add_data.append({'index': (start_index + end_index) / 2, 'value': np.average(temp_y)})
                start_index = end_index
                end_index = start_index + 100
                print('new start:', start_index)
                temp_y = []
                temp_y.append(data[1])
    df = pd.DataFrame(add_data)
    df.to_csv(save_path, index=None, header=None)


"""
dirtype:proposed/deepedit 29 2 4 3 24
029:
    center_point = [128 - 66, 128 - 61, 23]
    fore_points = [[128 - 58, 128 - 48, 23]]
    back_points = [[128 - 49, 128 - 56, 23]]
"""


def getResultPictureForAbation(flag, direction, i, label_number, fore_number, back_number, index, paint_scribbles = 0):
    img_file = "D:\Desktop\MONAILabel_datasets1\myTask07_BreastCrop\Breast_Training_%03d_ph3_128x128x96.nii" % (i)
    gt_file = "D:\Desktop\MONAILabel_datasets1\myTask07_BreastCrop\labels/final/Breast_Training_%03d_ph3_128x128x96.nii" % (i)
    label_file = 'D:\Desktop\MONAILabel_datasets1\myTask07_BreastCrop\labels\logits\Breast_Training_%03d_ph3_128x128x96_final_label.nii' % (i)
    label_scibble_file = 'D:\Desktop\MONAILabel_datasets1\myTask07_BreastCrop\labels\logits\Breast_Training_%03d_ph3_128x128x96_with_scribble.nii' % (i)
    label_final = 'D:\Desktop\MONAILabel_datasets1\myTask07_BreastCrop\labels\logits\Breast_Training_%03d_ph3_128x128x96_final.nii' % (i)
    img = nib.load(img_file)
    img = np.asarray(img.dataobj)
    gt = nib.load(gt_file)
    gt = np.asarray(gt.dataobj)
    if flag == 0:
        label = nib.load(label_scibble_file)
        label = np.asarray(label.dataobj)
        label_mass = np.where((label == label_number)|(label == back_number), label_number, 0)
        label_fore_scr = np.where(label == fore_number, label, 0)
        label_back_scr = np.where(label == back_number, label, 0)
    else:
        label = nib.load(label_scibble_file)
        label = np.asarray(label.dataobj)
        label_final = nib.load(label_final)
        label_final = np.asarray(label_final.dataobj)
        label_mass = np.where(label_final == label_number, label_number, 0)
        # ================================================================= slice_index =================================================================
    if direction == 0:
        img = np.rot90(img)
        img = np.fliplr(img)
        gt = np.rot90(gt)
        gt = np.fliplr(gt)
        label_mass = np.rot90(label_mass)
        label_mass = np.fliplr(label_mass)
        slice_index = index - 1
        gt_index = gt[:, :, slice_index, None].astype(np.uint8)
        label_mass = label_mass[:, :, slice_index, None].astype(np.uint8)
        if flag == 0:
            label_back_scr = np.rot90(label_back_scr)
            label_back_scr = np.fliplr(label_back_scr)
            label_fore_scr = label_fore_scr[:, :, slice_index, None].astype(np.uint8)
            label_back_scr = label_back_scr[:, :, slice_index, None].astype(np.uint8)
        img_index = img[:, :, slice_index, None]
        plt.imshow(img_index, cmap="gray")
    elif direction == 1:
        gt = np.where((label == 2) | (label == label_number) | (label == fore_number), 2, 0)
        slice_index = index - 1
        gt_index = gt[:, slice_index, :, None].astype(np.uint8)
        label_mass = label_mass[:, slice_index, :, None].astype(np.uint8)
        if flag == 0:
            label_fore_scr = label_fore_scr[:, slice_index, :, None].astype(np.uint8)
            label_back_scr = label_back_scr[:, slice_index, :, None].astype(np.uint8)
        img_index = img[:, slice_index, :, None]
        img_index = np.rot90(img_index)
        img_index = np.fliplr(img_index)
        gt_index = np.rot90(gt_index)
        gt_index = np.fliplr(gt_index)
        label_mass = np.rot90(label_mass)
        label_mass = np.fliplr(label_mass)
        plt.imshow(img_index, cmap="gray")
    else:
        gt = np.where((label == 2) | (label == label_number) | (label == fore_number), 2, 0)
        slice_index = index - 1
        gt_index = gt[slice_index, :, :, None].astype(np.uint8)
        label_mass = label_mass[slice_index, :, :, None].astype(np.uint8)
        if flag == 0:
            label_fore_scr = label_fore_scr[slice_index, :, :, None].astype(np.uint8)
            label_back_scr = label_back_scr[slice_index, :, :, None].astype(np.uint8)
            label_fore_scr = np.rot90(label_fore_scr)
            label_fore_scr = np.fliplr(label_fore_scr)
        img_index = img[slice_index, :, :, None]
        img_index = np.rot90(img_index)
        img_index = np.fliplr(img_index)
        gt_index = np.rot90(gt_index)
        gt_index = np.fliplr(gt_index)
        label_mass = np.rot90(label_mass)
        label_mass = np.fliplr(label_mass)

        plt.imshow(img_index, cmap="gray")
    # ================================================================= canny ================================================================
    edges = cv2.Canny(gt_index, 0, 2)
    plt.plot(np.nonzero(edges)[1], np.nonzero(edges)[0], 'gd', markersize=2)
    edges2 = cv2.Canny(label_mass, 0, 2)
    plt.plot(np.nonzero(edges2)[1], np.nonzero(edges2)[0], 'yd', markersize=2)
    # ================================================================= scribbles ================================================================
    if flag == 0 and paint_scribbles == 1:
        plt.plot(np.nonzero(label_fore_scr)[1], np.nonzero(label_fore_scr)[0], 'o', markersize=3, alpha=0.8,
                 color='blue')
        plt.plot(np.nonzero(label_back_scr)[1], np.nonzero(label_back_scr)[0], 'o', markersize=2, alpha=0.5,
                 color='red')

    plt.axis('off')
    plt.savefig(
        'D:\Desktop\MONAILabel_datasets1\myTask07_BreastCrop\labels\logits\\flag_%s_direction_%s_scribbles_%s.png' % (
        flag, direction, paint_scribbles)
        , bbox_inches='tight', pad_inches=-0.1)
    plt.show()


if __name__ == '__main__':
    '''
    dicom->nii
    '''
    breast_dicom2nii()
    breast_dicom2nii_t2()
    breast_dicom2nii_dwi()
    breast_dicom2nii_ph0()
    '''
    重采样
    '''
    resample_sitk()
    '''
    校正
    '''
    correct_bias()
    '''
    根据质心位置裁剪
    '''
    crop_mass_area()
    crop_mass_area_t2_or_dwi()
    '''
    nii->h5（废弃）
    '''
    nii2h5_seg()
    nii2h5_classification()
    '''
    按照monai文件夹格式准备数据
    '''
    ready_for_monai_dataset()
    '''
    Breast_Training_029画对比图
    '''
    # center_point = [128 - 66, 128 - 61, 23]
    # fore_points = [[128 - 58, 128 - 48, 23]]
    # back_points = [[128 - 49, 128 - 56, 23]]
    # getResultPicture('proposed', 29, 2, 4, 3, 24, need_point=True, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('deepedit', 29, -1, -1, -1, 24, need_point=True, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('econet', 29, 1, 3, 2, 24, need_point=False, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('graphcuts', 29, 2, 4, 3, 24, need_point=False, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # CM = [512-401, 512-377, 41]  # -> (64, 64)
    # extrem_points = [
    #     [99, 124],
    #     [109, 124],
    #     # [119, 126],
    #     [114, 135],
    #     [117, 153],
    #     [108, 145]
    # ]
    # back_points = [
    #     [119, 145], [107, 151]
    # ]
    # getResultPictureForMI("Breast_Training_029_ph3", 41, extrem_points, back_points, CM)

    '''
        Breast_Training_022画对比图
    '''
    # center_point = [128 - 66, 128 - 64, 24]
    # fore_points = []
    # back_points = []
    # getResultPicture('proposed', 22, 2, 4, 3, 25, need_point=True, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('deepedit', 22, -1, -1, -1, 25, need_point=True, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('econet', 22, 1, 3, 2, 25, need_point=False, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('graphcuts', 22, 2, 4, 3, 25, need_point=False, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # CM = [512 - 102, 512 - 281, 50]  # -> (64, 64)
    # extrem_points = [
    #     [410, 223], [403, 231], [406, 237], [413, 233]
    # ]
    # back_points = [
    #     [404, 222]
    # ]
    # getResultPictureForMI("Breast_Training_022_ph3", 51, extrem_points, back_points, CM)

    '''
        Breast_Training_288画对比图
    '''
    # center_point = [128 - 62, 128 - 65, 26]
    # fore_points = [[128 - 69, 128 - 70, 26], [128 - 55, 128 - 59, 26]]
    # back_points = [[128 - 53, 128 - 71, 26]]
    # getResultPicture('proposed', 288, 2, 4, 3, 27, need_point=True, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('deepedit', 288, -1, -1, -1, 27, need_point=True, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('econet', 288, 1, 3, 2, 27, need_point=False, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # getResultPicture('graphcuts', 288, 2, 4, 3, 27, need_point=False, center_point=center_point, fore_points=fore_points, back_points=back_points)
    # CM = [512 - 380, 512 - 228, 88]  # -> (64, 64)
    # extrem_points = [
    #     (123, 264),
    #     (114, 273),
    #     (124, 286),
    #     (142, 303),
    #     (145, 287),
    #     (138, 278),
    #     (135, 271),
    #     (141, 299)
    # ]
    # back_points = [
    #     (129, 299), (142, 272), (120, 288)
    # ]
    # getResultPictureForMI("Breast_Training_288_ph3", 91, extrem_points, back_points, CM)

    # getScribbleData()
    '''
    消融实验画图
    '''
    # getResultPictureForAbation(flag=0, direction=0, i=21, label_number=3, fore_number=5, back_number=4, index=32)
    # getResultPictureForAbation(flag=0, direction=1, i=21, label_number=3, fore_number=5, back_number=4, index=62)
    # getResultPictureForAbation(flag=0, direction=2, i=21, label_number=3, fore_number=5, back_number=4, index=64)

    getResultPictureForAbation(flag=0, direction=0, i=21, label_number=3, fore_number=5, back_number=4, index=32, paint_scribbles=1)
    getResultPictureForAbation(flag=0, direction=1, i=21, label_number=3, fore_number=5, back_number=4, index=62, paint_scribbles=1)
    getResultPictureForAbation(flag=0, direction=2, i=21, label_number=3, fore_number=5, back_number=4, index=64, paint_scribbles=1)
    #
    # getResultPictureForAbation(flag=1, direction=0, i=21, label_number=3, fore_number=5, back_number=4, index=32)
    # getResultPictureForAbation(flag=1, direction=1, i=21, label_number=3, fore_number=5, back_number=4, index=62)
    # getResultPictureForAbation(flag=1, direction=2, i=21, label_number=3, fore_number=5, back_number=4, index=64)
