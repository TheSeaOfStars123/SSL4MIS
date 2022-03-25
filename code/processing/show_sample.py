'''
  @ Date: 2022/3/24 10:29
  @ Author: Zhao YaChen
'''
# @Time : 2021/8/4 10:29 PM
# @Author : zyc
# @File : show_sample.py
# @Title :
# @Description :
import pydicom as dicom
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

def show_data(sample_img, sample_img2, sample_img3, sample_img4, mask_WT, index):
    # https://matplotlib.org/3.3.2/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
    # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    fig = plt.figure(figsize=(20, 10))

    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    flair = ax0.imshow(sample_img[:, :, index], cmap='bone')
    ax0.set_title("FLAIR", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(flair)

    #  Varying density along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    t1 = ax1.imshow(sample_img2[:, :, index], cmap='bone')
    ax1.set_title("T1", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1)

    #  Varying density along a streamline
    ax2 = fig.add_subplot(gs[0, 2])
    t2 = ax2.imshow(sample_img3[:, :, index], cmap='bone')
    ax2.set_title("T2", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t2)

    #  Varying density along a streamline
    ax3 = fig.add_subplot(gs[0, 3])
    t1ce = ax3.imshow(sample_img4[:, :, index], cmap='bone')
    ax3.set_title("T1 contrast", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1ce)

    #  Varying density along a streamline
    ax4 = fig.add_subplot(gs[1, 1:3])

    # ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
    l1 = ax4.imshow(mask_WT[:, :, index], cmap='summer', )


    ax4.set_title("", fontsize=20, weight='bold', y=-0.1)

    _ = [ax.set_axis_off() for ax in [ax0, ax1, ax2, ax3, ax4]]

    colors = [im.cmap(im.norm(1)) for im in [l1]]
    labels = ['Non-Enhancing tumor core']
    patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize='xx-large',
               title='Mask Labels', title_fontsize=18, edgecolor="black", facecolor='#c5c6c7')

    plt.suptitle("Multimodal Scans -  Data | Manually-segmented mask - Target", fontsize=20, weight='bold')

    fig.savefig("data_sample.png", format="png", pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("data_sample.svg", format="svg", pad_inches=0.2, transparent=False, bbox_inches='tight')




if __name__ == '__main__':
    # 尝试读取文件
    # file_path = '/Users/zyc/Desktop/breast_dataset_patient/31084 SONG XIANG QING/ph1/1.2.840.113619.2.388.57473.14116753.13368.1538914746.885.dcm'
    # file_path = '/Volumes/Elements/breastMR_202003_sort/77336 SUN JIAN YING/Ph1_dyn Ax Vibrant 5p/1.2.840.113619.2.388.57473.14116753.12824.1561900597.956.dcm'
    file_path = '/Users/zyc/Downloads/manifest-PyHQgfru6393647793776378748/ISPY1/ISPY1_1001/01-10-1985-690199-MR BREASTUNI UE-38479/42000.000000-PE Segmentation thresh70-88078/1-1.dcm'
    #  ds = dicom.dcmread(file_path, force=True)
    ds = dicom.read_file(file_path)
    print(ds)


    # sample_filename = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph1.nii'
    # sample_filename_mask = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_seg.nii'  # 512,512,108
    # sample_img = nib.load(sample_filename)
    # sample_img = np.asanyarray(sample_img.dataobj)
    # sample_img = np.rot90(sample_img)
    # sample_mask = nib.load(sample_filename_mask)
    # sample_mask = np.asanyarray(sample_mask.dataobj)
    # sample_mask = np.rot90(sample_mask)
    # print("img shape ->", sample_img.shape)
    # print("mask shape ->", sample_mask.shape)
    #
    #
    # sample_filename2 = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph2.nii'
    # sample_img2 = nib.load(sample_filename2)
    # sample_img2 = np.asanyarray(sample_img2.dataobj)
    # sample_img2 = np.rot90(sample_img2)
    #
    # sample_filename3 = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph3.nii'
    # sample_img3 = nib.load(sample_filename3)
    # sample_img3 = np.asanyarray(sample_img3.dataobj)
    # sample_img3 = np.rot90(sample_img3)
    #
    # sample_filename4 = '/Users/zyc/Desktop/breast-dataset-training-validation/Breast_TrainingData/Breast_Training_001/Breast_Training_001_ph4.nii'
    # sample_img4 = nib.load(sample_filename4)
    # sample_img4 = np.asanyarray(sample_img4.dataobj)
    # sample_img4 = np.rot90(sample_img4)
    #
    # mask_WT = sample_mask.copy()
    # show_data(sample_img, sample_img2, sample_img3, sample_img4, mask_WT, 78)