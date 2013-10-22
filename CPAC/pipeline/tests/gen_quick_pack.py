"""
Here I'll take a CPAC sym_links path and just extract the relevant pieces for a quickpak yaml file.

For this example, I'll be using the ldopa output in part because it has two scans so it can test me more.

/home2/data/PreProc/LDOPA/sym_links/pipeline_0/_compcor_ncomponents_5_linear1.motion1.compcor1.CSF_0.98_GM_0.7_WM_0.98/s1_/scan_placebo_func


INPUT:  /home2/data/PreProc/LDOPA/sym_links/pipeline_0/_compcor_ncomponents_5_linear1.motion1.compcor1.CSF_0.98_GM_0.7_WM_0.98/s1_/scan_ldopa_func/func/bandpass_freqs_0.01.0.1/functional_mni.nii.gz


OUTPUT:            
functional_mni:
    placebo_func: '/home2/data/Originals/LDOPA/s10/placebo/func.nii.gz',
    ldopa_func: '/home2/data/Originals/LDOPA/s10/ldopa/func.nii.gz',
    
"""

import os
import yaml
from collections import OrderedDict
from os import path as op
from glob import glob


###
# Set Suffixes
###

anat_suffixes = {
    "anatomical_brain": "anatomical_brain.nii.gz", 
    "anatomical_reorient": "anatomical_reorient.nii.gz", 
    "anatomical_to_mni_nonlinear_xfm": "anatomical_to_mni_nonlinear_xfm.nii.gz", 
    "mni_normalized_anatomical": "mni_normalized_anatomical.nii.gz", 
    "ants_affine_xfm": "ants_affine_xfm.txt"
}

func_suffixes = {
    "preprocessed": "preprocessed.nii.gz", 
    "mean_functional": "mean_functional.nii.gz", 
    "functional_brain_mask": "functional_brain_mask.nii.gz", 
    "functional_nuisance_residuals": "functional_nuisance_residuals.nii.gz", 
    "functional_freq_filtered": "bandpass_*/functional_freq_filtered.nii.gz", 
    "functional_mni": "bandpass_*/functional_mni.nii.gz", # add something to check a different path if bandpass isn't there
    "functional_brain_mask_to_standard": "functional_brain_mask_to_standard.nii.gz"
}

reg_suffixes = {
    "functional_to_anat_linear_xfm": "functional_to_anat_linear_xfm.mat" 
#    "functional_to_mni_linear_xfm": "functional_to_mni_linear_xfm.mat"
}


###
# Set Paths
###

base_dir = "/home2/data/PreProc/LDOPA/sym_links/pipeline_0/_compcor_ncomponents_5_linear1.motion1.compcor1.CSF_0.98_GM_0.7_WM_0.98"

# Within the baseline directory, we might have something like this:
# `s1_/scan_ldopa_func/func/bandpass_freqs_0.01.0.1`

# See the subject's in this directory
sub_ids = os.listdir(base_dir)

# Output list that contains information on all the subjects for YAML file
resources = []

# Get the data in each subject's directory
for sub_id in sub_ids:
    print sub_id
    
    # Get path to the subject
    sub_dir = op.join(base_dir, sub_id)
    
    # Setup the resource dictionary
    keys            = ['subject_id'] + anat_suffixes.keys() + func_suffixes.keys() + reg_suffixes.keys()
    sub_resources   = OrderedDict.fromkeys(keys)
    
    # Add the subject id
    sub_resources['subject_id'] = sub_id
    
    # Add anatomical resources
    anat_dir = op.join(sub_dir, "scan")
    for key,path_suffix in anat_suffixes.iteritems():
        sub_resources[key] = op.join(anat_dir, path_suffix)
    
    # Add functional and registration resources
    # there might be multiple ones per scan
    scan_dirs = glob(op.join(sub_dir, "scan_*"))
    for scan_dir in scan_dirs:
        # Scan ID
        scan_id = op.basename(scan_dir)
        
        # Base Functional Path
        func_dir = op.join(scan_dir, "func")
        
        # Paths for each functional file
        for key,path_suffix in func_suffixes.iteritems():
            if sub_resources[key] is None:
                sub_resources[key] = {}
            file_path = glob(op.join(func_dir, path_suffix))
            if len(file_path) > 1:
                raise Exception("Currently only one file for '%s' is supported" % file_path)
            elif len(file_path) == 0:
                file_path = ""
            else:
                file_path = file_path[0]
            sub_resources[key][scan_id] = file_path
        
        # Base Registration Output Path
        reg_dir = op.join(scan_dir, "registration")
        
        # Paths for each registration related file
        for key,path_suffix in reg_suffixes.iteritems():
            if sub_resources[key] is None:
                sub_resources[key] = {}
            file_path = glob(op.join(reg_dir, path_suffix))
            if len(file_path) > 1:
                raise Exception("Currently only one file for '%s' is supported" % file_path)
            else:
                file_path = file_path[0]
            sub_resources[key][scan_id] = file_path
    
    resources.append(dict(sub_resources))

        
# Save resources into yaml file
with open('subject_list_quick_pack.yml', 'w') as yaml_file:
    yaml_file.write( yaml.dump(resources) )
