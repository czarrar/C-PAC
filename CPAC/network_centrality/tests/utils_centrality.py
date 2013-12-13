"""
This tests the centrality function in utils
"""

import os, sys
import numpy as np
from numpy.testing import *

from nose.tools import ok_, eq_, raises, with_setup
from nose.plugins.attrib import attr    # http://nose.readthedocs.org/en/latest/plugins/attrib.html

import sys
sys.path.insert(0, '/home2/data/Projects/CPAC_Regression_Test/nipype-installs/fcp-indi-nipype/running-install/lib/python2.7/site-packages')
sys.path.insert(1, "/home2/data/Projects/CPAC_Regression_Test/2013-05-30_cwas/C-PAC")
sys.path.append("/home/data/PublicProgram/epd-7.2-2-rh5-x86_64/lib/python2.7/site-packages")

from CPAC.network_centrality import centrality

def test_centrality_binarize():
    print "testing centrality binarize"
    
    method      = "binarize"
    nblock      = 20
    nvoxs       = 100
    r_value     = 0.2
    corr_matrix = np.random.random((nblock, nvoxs))
    
    ref  = np.sum(corr_matrix>r_value, axis=1)
    comp = centrality(corr_matrix, r_value, method)
    
    assert_equal(ref, comp)

def test_centrality_weighted():
    print "testing centrality weighted"
    
    method      = "weighted"
    nblock      = 20
    nvoxs       = 100
    r_value     = 0.2
    corr_matrix = np.random.random((nblock, nvoxs))
    
    ref  = np.sum(corr_matrix*(corr_matrix>r_value), axis=1)
    comp = centrality(corr_matrix, r_value, method)
    
    assert_equal(ref, comp)

