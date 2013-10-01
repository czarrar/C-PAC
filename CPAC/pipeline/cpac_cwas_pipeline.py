import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

import re
import os
import sys
import glob
from CPAC.utils import Configuration

def prep_cwas_workflow(c, subject_infos):
    from CPAC.cwas import create_cwas
    import numpy as np
    
    try:
        import mkl
        mkl.set_num_threads(c.cwasThreads)
    except ImportError:
        pass
    
    print 'Preparing CWAS workflow'
    #p_id, s_ids, scan_ids, s_paths = (list(tup) for tup in zip(*subject_infos))
    #print 'Subjects', s_ids
    
    # Read in list of subject functionals
    lines   = open(c.cwasFuncFiles).readlines()
    spaths  = [ l.strip().strip('"') for l in lines ]
    
    # Read in design/regressor file
    regressor = np.loadtxt(c.cwasRegressorFile)

    wf = pe.Workflow(name='cwas_workflow')
    wf.base_dir = c.workingDirectory
    
    cw = create_cwas()
    cw.inputs.inputspec.roi         = c.cwasROIFile
    cw.inputs.inputspec.subjects    = spaths
    cw.inputs.inputspec.regressor   = regressor
    cw.inputs.inputspec.cols        = c.cwasRegressorCols
    cw.inputs.inputspec.f_samples   = c.cwasFSamples
    cw.inputs.inputspec.strata      = c.cwasRegressorStrata # will stay None?
    cw.inputs.inputspec.parallel_nodes = c.cwasParallelNodes
    cw.inputs.inputspec.memory_limit = c.cwasMemory
    cw.inputs.inputspec.dtype       = c.cwasDtype
    
    ds = pe.Node(nio.DataSink(), name='cwas_sink')
    out_dir = os.path.dirname(s_paths[0]).replace(s_ids[0], 'cwas_results')
    ds.inputs.base_directory = out_dir
    ds.inputs.container = ''

    wf.connect(cw, 'outputspec.F_map',
               ds, 'F_map')
    wf.connect(cw, 'outputspec.p_map',
               ds, 'p_map')

    wf.run(plugin='MultiProc',
                         plugin_args={'n_procs': c.numCoresPerSubject})



def run(config, subject_infos):
    import re
    import commands
    commands.getoutput('source ~/.bashrc')
    import os
    import sys
    import pickle
    import yaml

    c = Configuration(yaml.load(open(os.path.realpath(config), 'r')))

    prep_cwas_workflow(c, pickle.load(open(subject_infos, 'r') ))


