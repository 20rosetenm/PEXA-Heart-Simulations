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
        nib.save(new_nifti, f'{filename}_frame{frame+1}.nii.gz')

for sub in subjects:
    for file in paths[sub]:
        print(f'process file {file}')
        load_and_split(file)
