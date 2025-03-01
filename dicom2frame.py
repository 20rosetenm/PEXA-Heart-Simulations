import nibabel as nib
def load_and_split(path):
    input_scan = nib.load(path)
    input_scan_img = input_scan.get_fdata()
    input_scan_header = input_scan.header
    input_scan_frames = [input_scan_img[:,:,:,i] for i in range(input_scan_img.shape[3])]
    dim3d = input_scan_header['dim'].copy()
    dim3d[0] = 3
    dim3d[4] = 1
    new_header = input_scan_header.copy()
    new_header['dim'] = dim3d

    for frame in range(len(input_scan_frames)):
        new_nifti = nib.Nifti1Image(input_scan_frames[frame], input_scan.affine, new_header)
        filename = path.split('.')[0]
        nib.save(new_nifti, f'PEXA^9_Retro_CTA_LV+RV4A_0-50-5_frame_{frame}.nii.gz')
load_and_split("/mnt/c/Users/Nadin/OneDrive/Documents/Computational_Biomechanics_Lab/Repos/compbiomechproject/4D_CT/PEXA9 ExVivo CTA RV+LV+T3 RetroA/PEXA^9_CTA_RV+LV+TANK_3_RetroA.nii.gz")
# Step 1:use command in slack that anne sent to convert dicom to nifti (dcm2niix -z y -o . -f '%n_%d' -x i -y n .)
# Step 2: use this script and edit the above file names and directories to get the individual ED frame that you need
