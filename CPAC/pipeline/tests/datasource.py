import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from CPAC.utils.datasource import create_func_datasource
from CPAC.pipeline.tests.cpac_pipeline import create_log_node

def create_anat_datasource(path, wf_name='anat_datasource'):
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as util
    
    wf = pe.Workflow(name=wf_name)
    
    inputnode = pe.Node(util.IdentityInterface(
                                fields=['subject', 'anat'],
                                mandatory_inputs=True),
                        name='inputnode')
    inputnode.anat = path
    
    outputnode = pe.Node(util.IdentityInterface(fields=['subject',
                                                     'anat' ]),
                         name='outputspec')
    
    wf.connect(inputnode, 'subject', outputnode, 'subject')
    wf.connect(inputnode, 'anat', outputnode, 'anat')
    
    return wf

def add_anat_resource(name, path, subject_id, strat, num_strat, extra=False):
    """
    Generates a datasource for an existing anatomical file
    and adds it to the list of strategies.
    
    Returns a tuple of strategy and number of strategies.
    """
    anatFlow = create_anat_datasource(path, '%s_gather_%d' % (name, num_strat))
    anatFlow.inputs.inputnode.subject = subject_id
    strat.update_resource_pool({key:(anatFlow, 'outputspec.anat')})
    
    if extra:
        strat.append_name(anatFlow.name)
        strat.set_leaf_properties(anatFlow, 'outputspec.anat')
        create_log_node(anatFlow, 'outputspec.anat', num_strat)

    return strat

def add_func_resource(name, paths, subject_id, strat, num_strat, extra=False):
    """
    Generates a datasource for an existing set of functional files
    and adds it to the list of strategies.
    
    Returns a tuple of strategy and number of strategies.
    """
    # We can read these files with this datasource function
    # The function will essentially iterate through each of key : value pairs in sub_dict
    funcFlow = create_func_datasource(paths, '%s_gather_%s' % (name, num_strat))
    # Note that the above function has the second variable as:
    # 'func_gather_%d' % num_strat
    # I'm not sure if this is significant.
    funcFlow.inputs.inputnode.subject = subject_id
    # Running funcFlow will have a wf with outputspec:
    # - subject
    # - scan (key for the scan)
    # - rest (actual data)
    
    strat.append_name(funcFlow.name)
    strat.set_leaf_properties(funcFlow, 'outputspec.rest')
    strat.update_resource_pool({key:(funcFlow, 'outputspec.rest')})
    
    create_log_node(funcFlow, 'outputspec.rest', num_strat)
    
    return strat
