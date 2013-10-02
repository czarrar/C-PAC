#!/usr/bin/env python

import argparse, yaml
import os, sys
from os import path
from CPAC.utils import Configuration
from CPAC.pipeline.cpac_cwas_pipeline import cwas_workflow

# Load command-line argument
parser = argparse.ArgumentParser(description='Conduct a CWAS using MDMR.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('--config', dest='config', required=True, 
                    type=argparse.FileType('r'), 
                    help='Yaml configuration file containing required options')

# Parse
args = parser.parse_args()

# These are all the options that should be in the config file
# TODO: check that these exist
# c.cwasThreads
# c.cwasFuncFiles
# c.cwasRegressorFile
# c.workingDirectory
# c.outputDirectory
# c.cwasROIFile
# c.cwasRegressorCols
# c.cwasFSamples
# c.cwasRegressorStrata
# c.cwasParallelNodes
# c.cwasMemory
# c.cwasDtype

# Load options
c = Configuration(yaml.load(args.config))

# Run CWAS
cwas_workflow(c)
