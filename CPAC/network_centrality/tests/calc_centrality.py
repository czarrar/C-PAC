#!/usr/bin/env python

import os, sys
import numpy as np
from numpy.testing import *

from nose.tools import ok_, eq_, raises, with_setup
from nose.plugins.attrib import attr    # http://nose.readthedocs.org/en/latest/plugins/attrib.html

import sys
sys.path.append("/Users/zarrar/Dropbox/Code/C-PAC")

from CPAC.network_centrality import calc_centrality
# from .. import calc_centrality

@attr('subject', 'centrality')
def test_calc_centrality():
    
    
    tmp1 = calc_centrality(infile, maskfile, [1,1], [1,1], 0, 0.01, 12)
    
    # datafile : string (nifti file)
    #     path to subject data file
    # template : string (nifti file)
    #     path to mask/parcellation unit
    # method_options : list (boolean)
    #     list of two booleans for binarize and weighted options respectively
    # weight_options : list (boolean)
    #     list of two booleans for binarize and weighted options respectively
    # option : an integer
    #     0 for probability p_value, 1 for sparsity threshold, 
    #     any other for threshold value
    # threshold : a float
    #     pvalue/sparsity_threshold/threshold value
    # allocated_memory : string
    #     amount of memory allocated to degree centrality
    
    
    
    
    
    
    config_file     = "files/config_fsl.yml"
    subject_infos   = common.gen_file_map(CPAC_OUTPUT)
    
    curdir = os.getcwd()
    os.chdir(__file__)
    
    conf             = common.load_configuration(config_file)
    old_subject_file = common.load_subject_list(conf.subjectListFile)
    new_subject_file = setup_group_subject_list(config_file, subject_infos)
    
    os.chdir(curdir)
    
    # All should be right here
    assert_equal(np.array(old_subject_file), np.array(new_subject_file))
    
    # Case where only one path is correct
    
    
    """
    I want to test a couple of cases:
    - If nothing changes
    - If the subject list actually has subjects with missing data, it detects that
        (here I can actually just modify the subject_infos or actually recall gen_file_map with a list of subject's desired)
    """
    
