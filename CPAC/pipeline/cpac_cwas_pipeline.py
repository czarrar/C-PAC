import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe

import re
import os
import sys
import glob
from CPAC.utils import Configuration

def cwas_workflow(c):
    from CPAC.cwas import create_cwas
    from string import Template
    import numpy as np
    import time
        
    if isinstance(c.cwas, dict):
        c.cwas = Configuration(c.cwas)
    
    try:
        import mkl
        mkl.set_num_threads(c.cwas.threads)
    except ImportError:
        pass
        
    # Auto-complete base
    fields = ["prior_mask_file", "file_with_functional_paths", "regressor_file"]
    for field in fields:
        s = Template(getattr(c.cwas, field))
        c.cwas.update(field, s.substitute(base=c.cwas.base))
    
    # Read in list of subject functionals
    lines   = open(c.cwas.file_with_functional_paths).readlines()
    spaths  = [ l.strip().strip('"') for l in lines ]
    
    # Read in design/regressor file
    regressor = np.loadtxt(c.cwas.regressor_file)
    
    # Load workflow
    wf = pe.Workflow(name='cwas_workflow')
    wf.base_dir = c.workingDirectory
    
    # Setup CWAS set of commands
    cw = create_cwas()
    cw.inputs.inputspec.roi         = c.cwas.prior_mask_file
    cw.inputs.inputspec.subjects    = spaths
    cw.inputs.inputspec.regressor   = regressor
    cw.inputs.inputspec.cols        = c.cwas.regressors_of_interest
    cw.inputs.inputspec.f_samples   = c.cwas.n_permutations
    cw.inputs.inputspec.strata      = c.cwas.strata
    cw.inputs.inputspec.parallel_nodes = c.cwas.parallel_nodes
    cw.inputs.inputspec.memory_limit = c.cwas.memory_limit
    cw.inputs.inputspec.dtype       = c.cwas.dtype
    
    # Output directory
    ds = pe.Node(nio.DataSink(), name='cwas_sink')
    ds.inputs.base_directory = os.path.join(c.outputDirectory, "cwas_results")
    ds.inputs.container = ''
    
    # Link F-stats and P-values
    wf.connect(cw, 'outputspec.F_map',
               ds, 'F_map')
    wf.connect(cw, 'outputspec.p_map',
               ds, 'p_map')
    
    # Run CWAS
    start   = time.time()
    wf.run(plugin='MultiProc',
                         plugin_args={'n_procs': c.numCoresPerSubject})
    end     = time.time()
    
    # Return time it took
    print 'It took', end-start, 'seconds.'
    return (end-start)

def prep_cwas_workflow(c, subject_infos):
    from CPAC.cwas import create_cwas
    import numpy as np
    
    try:
        import mkl
        mkl.set_num_threads(c.cwasThreads)
    except ImportError:
        pass
    
    print 'Preparing CWAS workflow'
    p_id, s_ids, scan_ids, s_paths = (list(tup) for tup in zip(*subject_infos))
    print 'Subjects', s_ids
    
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


