import os
import sys
import inspect
import hashlib
import pytest
import torch
import spacy

# Import CTL from the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from constituent_treelib import ConstituentTree, BracketedTree


class TestGlobal:
    nlp = ConstituentTree.create_pipeline(download_models=True)

    def test_spacy_basic_functionality(self):
        doc = self.nlp("Today was a good day!")
        ['NN', 'VBD', 'DT', 'JJ', 'NN', '.'] == [t.tag_ for t in doc]

    def test_benepar_integration(self):
        assert "benepar" in self.nlp.component_names

    def test_invalid_tree_exception(self):
        with pytest.raises(Exception):
            bracketed_tree_string = "(S (NP (PRP You)) (VP (MD must) (VP (VB construct) (NP (JJ additional) (NNS pylons)))) (. !"
            BracketedTree(bracketed_tree_string)

    def test_bracketed_tree_string(self):
        sentence = "I love cakes and cookies!"
        tree = ConstituentTree(sentence, self.nlp)
        assert tree.to_bracketed_tree_string() == "(S (NP (PRP I)) (VP (VBP love) (NP (NNS cakes) (CC and) (NNS cookies))) (. !))"

    def test_extracted_phrases(self):
        sentence = "Albert Einstein was a German-born theoretical physicist."
        tree = ConstituentTree(sentence, self.nlp)
        expected_output = {
            "NP": ["Albert Einstein", "a German - born theoretical physicist"],
            "S": ["Albert Einstein was a German - born theoretical physicist ."],
            "ADJP": ["German - born"],
            "VP": ["was a German - born theoretical physicist"]}
        assert tree.extract_all_phrases() == expected_output

    def test_svg_export(self):
        file_name = "tree.svg"
        sentence = "Python is an awesome coding language!"
        ConstituentTree(sentence, self.nlp).export_tree(file_name)
        with open(file_name, "rb") as file:
            md5_hash = hashlib.md5(file.read()).hexdigest()
        assert md5_hash == "75cdbbeda69e84df53b6d07428e54244"
        os.remove(file_name)
