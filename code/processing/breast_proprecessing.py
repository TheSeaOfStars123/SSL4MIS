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

'''
    文件路径定义
'''
default_prefix = 'D:/Desktop/BREAST/BREAST/'
name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
default_directory = '/Volumes/Elements/breastMR_202003_sort'
fune_directory = '/Volumes/Elements/breast_dataset_patient'
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
    dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)

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
    name_mapping_df = pd.read_csv(name_mapping_path)
    # print(name_mapping_df.head())
    # print(name_mapping_df.columns)
    # print(name_mapping_df.index)
    i = 76
    slice = 0
    isAddMetaInfo = False
    # isAddMetaInfo = True
    isCopyLabelFile = True

    # dir_list = list_dir.get_file_list_by_name(default_directory)
    dir_list = ['Breast_MR_list.xlsx', 'ICC_40cases', '21562 YU XIN', '31084 SONG XIANG QING', '77336 SUN JIAN YING',
                '100248 MEI SHI QIN', '165299 CAI GUI E', '174948 SU JING HUA', '177914 ZHANG HUI RONG',
                '195347 LIU ANY', '225136 CAO MENG MENG', '322144 LIU HUI QIN', '346515 MENG CUI WEN',
                '642642 FANG YOU MEI', '665131 DUAN DONG HONG', '665237 YANG SU ZHEN', '694654 XUE YAN LI',
                '707418 WANG WEN FANG', '745072 YIN RUN ZHEN', '748196 ZHAO YU PING', '772030 CAI HOU PING',
                '788805 XIE GUI LAN', '800703 LIU JIE', '830377 LU LING XIA', '833326 SONG LI NING',
                '852004 HUANG SHAO HUA', '852185 LU YAO MEI', '852186 SHEN XIU E', '860003 FAN RUN QING',
                '862979 LV YUE YING', '863544 LIU LAN FEN', '879799 ZANG BEI NI', '883819 LIU XIU FEN',
                '898226 XING YU QIONG', '898226 XING YU QIONG', '900167 LI BING', '917889 CAO XIU LAN', '920960 MA JIE',
                '927644 SUN LI HONG', '927644 SUN LI HONG', '928817 SHI HONG KAI', '941546 WU YA LI', '941901 LIU JING',
                '942872 JIA LI', '946445 HONG YAN', '946581 WANG CUI MEI', '955462 YANG JIN XIA',
                '959673 ZHANG YUN XIA', '975944 WANG HUI TIAN', '976257 HUANG JIAN', '976314 XING RONG',
                '979070 LI WEN YING', '979070 LI WEN YING', '979070 LI WEN YING', '981932 XIANG SONG RONG',
                '981932 XIANG SONG RONG', '981932 XIANG SONG RONG', '983859 LIU JUN JIAN', '990856 KAN YA PING',
                '991340 BAO ZHEN HUA', '994864 REN XIU FEN', '995892 WANG WEI QIAN', '998649 ZUO SU JIE',
                '1000690 DONG CHUN MEI', '1003279 ZHAO YU', '1004230 XIE PING', '1005322 ZHANG QING HUI',
                '1006632 ZHANG SHU XIAN', '1007161 LIANG ZI PING', '1008065 YE TU WEI', '1008633 ZHANG XIU FEN',
                '1018912 WU LAN TUO YA', '1019175 LIU XUE QI', '1019317 ZHAO LI LI', '1019468 SHI HAI XIA',
                '1019639 QIU XUE YAN', '1019992 WANG DE MEI', '1021837 YAN SHU FANG', '1022980 LI RU LIAN',
                '1023545 LIU ZHONG XIA', '1024861 YANG GUI FEN', '1025940 CHENG YAN MING', '1028025 LIU FANG',
                '1029015 ZHANG ZHENG TIAN', '1031147 YANG WEN', '1032369 TAN MING QIAO', '1033418 HAN JIE',
                '1033496 ZHANG SHU LAN', '1035134 LIU XIU ZHEN', '1036891 WU LI MIN', '1037739 ZHAO DONG YAN',
                '1038340 SHUAI DAO YUE', '1040320 LAN YU PING', '1041077 LI XI XING', '1041388 ZHAO JUN MEI',
                '1044663 LI ZHI PING', '1046020 XIE YU LIAN', '2000306 LIN JUAN', '2000453 XU GUI LAN',
                '2000590 Sha Sha', '2000850 WANG AI HUA', '2000983 ZHAO XIU LAN', '2000986 WU ZHAO HUA',
                '2000999 Xiao Jian', '2001047 SI QIN', '2001186 WANG HUA', '2001426 Han Qing', '2001566 TANG SHUN',
                '2001612 HAO AI FANG', '2001711 WANG SHU JIE', '2001730 ZHANG AI QING', '2001741 Zha Jiao E',
                '2002000 NIE TENG TENG', '2002067 LI CUI PING', '2002132 Liu Jing Bo', '2002133 WANG LI CHUN',
                '2002231 HU YAN HUA', '2002243 Zhao Qin', '2002351 JIANG JIN', '2002392 MA HAI LIN',
                '2002401 ZHOU FU RONG', '2002430 ZHAO XI MIN', '2002558 XUE TAO', '2002559 LIU BO',
                '2002565 HOU YA YING', '2002681 WANG MING XIA', '2002720 HAN MEI', '2002899 Peng Xue Hua',
                '2003130 LIU YAN FEN', '2003202 LIANG XIAO LI', '2003279 WU LI YUN', '2003341 CHEN BI HUA',
                '2003398 Sun Yu Feng', '2003398 Sun Yu Feng', '2003612 Wu Hui Ying', '2003632 LI CHUN LAN',
                '2003858 Liu Gui Zhen', '2003956 He Hui Ling', '2004002 Xing Shu Min', '2004166 CHEN WEI',
                '2004212 ZHANG XIAO QING', '2004444 ZHAO HUAN WEN', '2004515 JIANG BAO YING', '2004547 ZHANG HAI XIA',
                '2004611 Lu Rui Hua', '2004629 ZHANG YA DONG', '2004683 YANG XIAO LING', '2004764 CUI JIN YING',
                '2004784 Li Di Hua', '2004986 DAI HONG', '2005079 MEN FENG XIA', '2005088 JIA JUN YING',
                '2005283 SUN LEI', '2005286 CAO LEI', '2005374 ZHU FENG YING', '2005644 ZHENG XIAO FANG',
                '2005769 WANG SHI PING', '2005825 XIAO YING', '2005859 YANG SU YING', '2005879 Zhou Xian Jiao',
                '2005881 GAO JIN QI', '2005962 ZHANG MEI YING', '2006005 SUN YAN JUN', '2006032 LI XIU QING',
                '2006047 Wang Xin Rong', '2006203 HAO JIA RONG', '2006406 YANG GUO FENG', '2006703 HE RUI',
                '2006871 WANG HUI FANG', '2006958 MIAO RONG', '2006983 Wu Ying Ni', '2007147 ZHANG XIU LAN',
                '2007264 Zhao Ying Hui', '2007342 LI CHUN GUANG', '2007342 LI CHUN GUANG', '2007398 Wang Rong Xiu',
                '2007445 REN BIN', '2007674 ZHANG YAN', '2007674 ZHANG YAN', '2007724 CHEN XIAO YAN',
                '2007865 WANG SHU YUN', '2008316 BAI HE', '2008362 FU XIAO NING', '2008479 HAN XIU JING',
                '2008484 YANG LING', '2008524 Ge Ri Le', '2008846 SONG JIE', '2009072 LIU XIU ZHEN',
                '2009073 MU FENG YUE', '2009094 WANG JUN YING', '2009185 HU HUANG YING', '2009436 BAI GUI LAN',
                '2009465 WANG XIU PING', '2009640 Wu Li Jun', '2009640 Wu Li Jun', '2009759 Fu Xiao Qi',
                '2009976 YANG JI XIU', '2010187 ZHANG YAO HUA', '2010299 GUO LI MING', '2010596 ZHANG CUI PING',
                '2010788 Zang Xiu Ju', '2010924 Lu Wang', '2010954 Wang Bao Ying', '2011476 BAI QUAN GANG',
                '2011606 WANG JUN E', '2011613 SONG LI HUA', '2011920 MENG CAI YUN', '2012140 WEN HONG YUN',
                '2012163 LUO JING HONG', '2012167 WU XIAO BO', '2012310 WU YUE YING', '2012446 Ma Rong',
                '2012466 TANG XIAO YAN', '2012466 TANG XIAO YAN', '2012475 GUO BING KUN', '2012562 HUANG JING',
                '2012568 ZHENG HAO LIAN', '2012651 Zhang Ran', '2012689 WANG CHUN MEI', '2012691 SUN LI YING',
                '2012811 LIU LIANG', '2013062 LIU JI JUN', '2013518 Xiao Yan Ping', '2013570 YANG CHUN FENG',
                '2013650 MENG YA QIU', '2013670 YOU XIAO BING', '2013827 Xiao Mao Xia', '2014018 WEI ZHI LI',
                '2014095 XU YA NAN', '2014187 LIU PING', '2014192 NIU YAN LING', '2014212 CHEN CHUN XIAN',
                '2014310 Zhao Shi Qin', '2014604 YU KE XIA', '2014905 FAN YU QIN', '2014940 TONG YUN FEI',
                '2014951 MIAO GUI RONG', '2015027 JIA MIN', '2015034 LI JIAN PING', '2015327 WANG AI MIN',
                '2015463 Min Qing Lan', '2015677 HE SAN NV', '2015868 Li Dong Xia', '2016225 SUN GUI LAN',
                '2016370 WAN QIONG', '2016398 LIU YING BIN', '2016525 LIU JING', '2016540 GAO YAN LING',
                '2016542 TAN HONG YAN', '2016728 SUN QIU PING', '2016910 TANG HUAN CHANG', '2017220 GUAN YA XIAN',
                '2017276 ZHANG XIU FEN', '2017307 TIAN LI LI', '2017307 TIAN LI LI', '2017354 ZHANG YAN JU',
                '2017389 XIA AI MING', '2018051 WANG GUI LING', '2018102 WANG LI JUN', '2018281 ZHAO SHAO YAN',
                '2018412 ZHAO HONG MEI', '2018521 SU HANG', '2018662 ZHANG SHAN NA', '2018678 Wang Jing',
                '2019229 LI WEI HONG', '2019533 FENG CHANG LI', '2019533 FENG CHANG LI', '2019812 LEI XIN',
                '2019930 HE SU XIANG', '2020144 GUO JING', '2020482 JIA YAN PING', '2020543 MENG ZHAO HONG',
                '2020545 LIANG XIAO YAN', '2020545 LIANG XIAO YAN', '2020662 XIONG SEN YAN', '2020707 WANG HONG YUN',
                '2021191 WANG SHU RONG', '2021466 YANG SHU JIANG', '2021466 YANG SHU JIANG', '2021715 GUO TING TING',
                '2022421 ZHAI YAN KUN', '2022421 ZHAI YAN KUN', '2022447 LI XIAO PING', '2022496 YUE QI',
                '2022555 DU CHENG ZHI', '2022697 Mo Hua Ying', '2022697 Mo Hua Ying', '2022750 LV SHU YUAN',
                '2022820 YE RUI XIA', '2022868 YANG YA YI', '2022956 TIAN XIA', '2022956 TIAN XIA',
                '2023128 XU YING QIU', '2023263 GUO FANG', '2023544 HAN HUI', '2023675 MA SHI QIN',
                '2023822 HOU YING HONG', '2024272 LIU XIAO MEI', '2024509 LU SHU HUA', '2024667 CHANG YA PING',
                '2024990 PENG JING HUA', '2025054 WANG LI XIANG', '2025076 DU JIN XIA', '2025160 ZENG SHU WEN',
                '2025331 FAN LI FANG', '2025606 XU CHUN HUA', '2025691 YANG HE', '2027534 TANG XIU YAN',
                '2028797 ZHAO YAN HONG', '2030177 LIU HONG XIA']

    for patient_name in dir_list[i + 2:]:
        i = i + 1
        nii_directory = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d' % i
        if not os.path.exists(nii_directory):
            os.makedirs(nii_directory)
        if isCopyLabelFile:
            label_nii_path = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData/Breast_Training_%03d/Breast_Training_%03d_seg.nii' % (
                i, i)
            # 首先将mass-label.nii复制到目标文件夹中
            shutil.copyfile(os.path.join(default_directory, patient_name, 'mass-label.nii'), label_nii_path)
        j = 1
        basic_ph_name_list = [
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
        for j in range(1, 6):
            phase_info = 'ph%d' % j
            new_name = 'Breast_Training_%03d_ph%d.nii' % (i, j)
            fune_dicom_directory = os.path.join(fune_directory, patient_name, phase_info)
            if isAddMetaInfo:
                slice = addFloderMetaInfo(original_dicom_directory, new_folder=fune_dicom_directory)
                dicom2nii(fune_dicom_directory, os.path.join(nii_directory, new_name))
            else:
                dicom2nii(original_dicom_directory, os.path.join(nii_directory, new_name))
            print(os.path.join(nii_directory, new_name) + ' Done!')
        # 每处理完一个病人的病例进行csv记录
        add_data = [{'Number': str(i),
                     'File_Path': basic_ph_name1,
                     'Patient_ID Patient_Name': patient_name,
                     'Breast_subject_ID': 'Breast_Training_%03d' % i,
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
            offsetX = 64
            offsetY = 64
            offsetZ = 32
            # 保存为nii文件
            # nib.Nifti1Image(voi_mri_data).to_filename('Breast_Training_002_ph2_voi.nii')
            # voi_mri_data.to_filename('Breast_Training_002_ph2_voi.nii')
            # nib.save(voi_mri_data, os.path.join(folder_path, 'Breast_Training_002_ph2_voi.nii'))
            # 对6个阶段
            for j in range(1, 6):
                mri_path = mri_default_path + "_ph" + str(j) + ".nii"
                mri_img = nib.load(mri_path)
                mri_data = mri_img.get_data()
                # 三维数组的切片 mri shape -> (512, 512, 92) voi mri shape -> (128, 128, 8)
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
                if z - offsetZ < 0:
                    delta = 0 - (z - offsetZ)
                    z = z + delta
                elif z + offsetZ > mri_data.shape[2]:
                    delta = z + offsetZ - mri_data.shape[2]
                    z = z - delta
                voi_mri_data = mri_data[(xshape - x) - offsetX:(xshape - x) + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
                voi_mri_data = np.flip(voi_mri_data, axis=0)
                # voi_mri_data = mri_data[x - offsetX:x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
                # voi_mri_data = mri_data[(-x) - offsetX:(-x) + offsetX, (-y) - offsetY:(-y) + offsetY, z - offsetZ:z + offsetZ]
                pair_img = nib.Nifti1Pair(voi_mri_data, np.eye(4))
                nib.save(pair_img, os.path.join(folder_path, 'Breast_Training_%03d_ph%d_voi_debug.nii' % (i, j)))

            # 对label三维数组同样切片
            voi_data = roi_data[x - offsetX:x + offsetX, y - offsetY:y + offsetY, z - offsetZ:z + offsetZ]
            pair_img = nib.Nifti1Pair(voi_data, np.eye(4))
            nib.save(pair_img, os.path.join(folder_path, 'Breast_Training_%03d_seg_voi_debug.nii' % (i)))

            # 输出切片后大小
            print("mri shape ->", mri_data.shape)
            print("roi shape ->", roi_data.shape)
            print("voi mri shape ->", voi_mri_data.shape)

        def list_or_tuple_to_string(list1):
            return '('+' '.join(str(e) for e in list1)+')'

        # 每处理完一个病人的病例进行csv记录
        # add_data = [{'Number': 'Breast_Training_%03d' % i, 'CM': list_or_tuple_to_string(CM), 'Shape:': list_or_tuple_to_string(mri_data.shape), 'labelSlice': list_or_tuple_to_string(label_slice_list), 'labelSlicesum':label_slice_sum}]
        # df = pd.DataFrame(add_data)
        # df.to_csv(output_label_info, index=None,
        #           mode='a', header=None)

'''
    将.nii转化为.h5文件
    输入训练集的原图和label，输出h5文件
    先选取作为示例
    一个h5中，包含一个ph4阶段image和label
'''
data_types = ['_ph4_voi_debug.nii']
def nii2h5():
    # 修改h5存放路径
    h5py_path = 'D:\Desktop\BREAST\BREAST\data'
    id_num = 0
    for id_ in sorted(os.listdir(root_path)):
        # 载入所有模态
        images = []
        for data_type in data_types:
            img_path = os.path.join(root_path, id_, id_ + data_type)
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = (img - img.min()) / (img.max() - img.min())
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        img = img[0, :, :]
        mask_path = os.path.join(root_path, id_, id_ + '_seg_voi_debug.nii')
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
        mask = np.clip(mask, 0, 1)
        mask = np.moveaxis(mask, (0, 1, 2), (2, 1, 0))
        if img.shape != mask.shape:
            print("Error")
        f = h5py.File(os.path.join(h5py_path, id_ + '_ph4_seg.h5'), 'w')
        f.create_dataset('image', data=img, compression="gzip")
        f.create_dataset('label', data=mask, compression="gzip")
        f.close()
        id_num += 1
    print("Converted total {} niis to h5 files".format(id_num))

if __name__ == '__main__':
    # crop_mass_area()
    nii2h5()
