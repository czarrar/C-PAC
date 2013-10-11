import os, sys
from numpy as np
from numpy.testing import *

from nose.tools import ok_, eq_, raises, with_setup
from nose.plugins.attrib import attr    # http://nose.readthedocs.org/en/latest/plugins/attrib.html

from .. import common

# this should be set elsewhere as some setting
CPAC_INPUT  = 
CPAC_OUTPUT = "/home/data/Projects/CPAC_Regression_Test/2013-08-19-20_v0-3-1/output"


@attr('configs', 'group')
def test_setup_group_list():
    # So setup_group_subject_list
    # will check conf.subjectListFile
    # and see overlap with subject_infos
    
    subject_infos = gen_file_map(CPAC_OUTPUT)
    
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
    
    
    
    