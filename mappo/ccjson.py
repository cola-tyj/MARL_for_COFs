import os
import simplejson
import numpy as np

from pycofbuilder.tools import smiles_to_xsmiles

import gemmi
from ase.cell import Cell


def from_cjson(path, file_name):
    '''
    Reads a ChemJSON file from a given path and file_name.
    '''
    file_name = os.path.join(path, file_name.split('.')[0] + '.cjson')

    with open(file_name, 'r') as file:
        cjson_data = simplejson.load(file)

    if 'properties' in cjson_data:
        properties = cjson_data['properties']
    
    return properties

def get_xmiles(path, file_name):
    properties = from_cjson(path, file_name)
    smiles = properties['smiles']
    xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles)
    #return properties['xsmiles']
    return xsmiles

def get_labels(path, file_name):
    properties = from_cjson(path, file_name)
    smiles = properties['smiles']
    xsmiles, xsmiles_label, composition, _labels = smiles_to_xsmiles(smiles)
    return _labels
