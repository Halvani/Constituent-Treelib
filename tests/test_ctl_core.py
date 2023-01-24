import torch
import spacy
import hashlib
from nltk import Tree

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


from constituent_treelib import ConstituentTree


def test_benepar_integration():
    nlp = ConstituentTree.create_pipeline(download_models=True) 
    assert 'benepar' in nlp.component_names


def test_bracketed_tree_string():
    sentence = "I love cakes and cookies!"
    nlp = ConstituentTree.create_pipeline(download_models=True) 
    tree = ConstituentTree(sentence, nlp)
    assert tree.to_bracketed_tree_string() == '(S (NP (PRP I)) (VP (VBP love) (NP (NNS cakes) (CC and) (NNS cookies))) (. !))'


def test_svg_export():
    file_name = "tree.svg"
    sentence = "Python is an awesome coding language!"
    nlp = ConstituentTree.create_pipeline(download_models=True)
    tree = ConstituentTree(sentence, nlp)    
    tree.export_tree(file_name)
    with open(file_name, 'rb') as file:
        md5_hash = hashlib.md5(file.read()).hexdigest()    
    assert md5_hash == "75cdbbeda69e84df53b6d07428e54244"
    os.remove(file_name)
