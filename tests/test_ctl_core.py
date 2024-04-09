import os
import sys
import torch
import spacy
import pytest
import inspect
import hashlib
import unittest

# Import CTL from the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from constituent_treelib import ConstituentTree, BracketedTree, Language, Structure
from constituent_treelib.errors import *


class TestBracketedTree(unittest.TestCase):
    def test_error_none_bracketed_tree(self):
        with pytest.raises(NoneOrEmptyBracketedTreeError):
            BracketedTree(bracketed_tree_string=None)

    def test_error_empty_tree(self):
        with pytest.raises(NoneOrEmptyBracketedTreeError):
            BracketedTree(bracketed_tree_string="")

    def test_error_no_matching_closing_opening_parentheses(self):
        with pytest.raises(ParenthesesError):
            bracketed_tree_string = "(S (NP (PRP I)) (VP (VBP love) (NP (NNS cookies))) (. !)"
            BracketedTree(bracketed_tree_string)

    def test_error_too_many_closing_parentheses(self):
        with pytest.raises(ParenthesesError):
            bracketed_tree_string = "(S (NP (PRP I)) (VP (VBP love) (NP (NNS cakes) (CC and) (NNS cookies))) (. !)))"
            BracketedTree(bracketed_tree_string)

    def test_remove_postag_nodes(self):
        bracketed_tree_string = "(S (NP (PRP I)) (VP (VBP love) (NP (NNS cookies))) (. !))"
        bracketed_tree_string_without_postags = BracketedTree.remove_postag_nodes(bracketed_tree_string)
        assert bracketed_tree_string_without_postags == "(S (NP I) (VP love (NP cookies)) !)"

    def test_remove_token_leaves(self):
        bracketed_tree_string = "(S (NP (PRP I)) (VP (VBP love) (NP (NNS cookies))) (. !))"
        bracketed_tree = BracketedTree(bracketed_tree_string)
        bracketed_tree_string_without_token_leaves = BracketedTree.remove_token_leaves(bracketed_tree.nltk_tree)
        assert bracketed_tree_string_without_token_leaves == "(S (NP (PRP)) (VP (VBP) (NP (NNS))) (.))"


class TestConstituentTree(unittest.TestCase):
    nlp = None
    defect_nlp = None
    unnecessary_components = ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]

    def setUp(self):
        spacy_model_en_small = "en_core_web_sm"
        spacy_model_en_medium = "en_core_web_md"
        benepar_model_fr = "benepar_fr2"
        benepar_model_en_small = "benepar_en3"

        ConstituentTree.download(spacy_model_en_small, benepar_model_fr)
        ConstituentTree.download(spacy_model_en_medium, benepar_model_en_small)

        self.defect_nlp = spacy.load("en_core_web_sm", disable=self.unnecessary_components)
        self.defect_nlp.add_pipe("sentencizer")
        self.nlp = ConstituentTree.create_pipeline(download_models=False)

    def test_error_nlp_pipeline_none(self):
        with pytest.raises(NLPPipelineError):
            sentence = "I will not instantiate a ConstituentTree object without an nlp pipeline ever again."
            ConstituentTree(sentence)

    def test_error_nlp_pipeline_invalid_type(self):
        with pytest.raises(NLPPipelineError):
            sentence = "I will not instantiate a ConstituentTree object with an invalid nlp pipeline ever again."
            ConstituentTree(sentence, nlp="")

    def test_error_nlp_pipeline_without_benepar_component(self):
        with pytest.raises(NLPPipelineError):
            sentence = "I will not instantiate a ConstituentTree object with an invalid nlp pipeline ever again."
            ConstituentTree(sentence, nlp=self.defect_nlp)

    def test_error_nlp_pipeline_models_sentence_language_mismatch(self):
        with pytest.raises(LanguageError):
            sentence = "Huch, das war jetzt nicht gewollt."
            ConstituentTree(sentence, nlp=self.nlp)

    def test_spacy_pos_tagging(self):
        doc = self.nlp("Today was a good day!")
        assert ['NN', 'VBD', 'DT', 'JJ', 'NN', '.'] == [t.tag_ for t in doc]

    def test_error_sentence_none(self):
        with pytest.raises(SentenceError):
            sentence = None
            ConstituentTree(sentence, nlp=self.nlp)

    def test_tree_parsing(self):
        sentence = "I love cakes and cookies!"
        tree = ConstituentTree(sentence, self.nlp)
        bracketed_sentence = "(S (NP (PRP I)) (VP (VBP love) (NP (NNS cakes) (CC and) (NNS cookies))) (. !))"
        assert tree.to_bracketed_tree_string() == bracketed_sentence

    def test_create_tree_from_bracketed_string(self):
        bracketed_tree_string = "(S (NP (PRP You)) (VP (MD must) (VP (VB construct) (NP (JJ additional) " \
                                "(NNS pylons)))) (. !))"
        bracketed_tree = BracketedTree(bracketed_tree_string)
        tree_from_bracketed = ConstituentTree(sentence=bracketed_tree, nlp=self.nlp)
        sentence = "You must construct additional pylons !"
        assert tree_from_bracketed.leaves(tree_from_bracketed.nltk_tree) == sentence

    def test_error_multiple_sentences(self):
        with pytest.raises(SentenceError):
            sentence = "I love cakes and cookies! I don't like eggplants."
            ConstituentTree(sentence, nlp=self.nlp)

    def test_extracted_phrases(self):
        sentence = "Albert Einstein was a German-born theoretical physicist."
        tree = ConstituentTree(sentence, self.nlp)
        expected_output = {
            "NP": ["Albert Einstein", "a German - born theoretical physicist"],
            "S": ["Albert Einstein was a German - born theoretical physicist ."],
            "ADJP": ["German - born"],
            "VP": ["was a German - born theoretical physicist"]}
        assert tree.extract_all_phrases() == expected_output

    def test_extracted_phrases_tree_without_token_leaves(self):
        sentence = "Give it all you've got!"
        tree_without_token_leaves = ConstituentTree(sentence, self.nlp, structure=Structure.WithoutTokenLeaves)
        postag_phrases = {'S': ['VB PRP DT PRP VBP VBN .', 'PRP VBP VBN'],
                          'NP': ['DT PRP VBP VBN'],
                          'VP': ['VB PRP DT PRP VBP VBN', 'VBP VBN']}
        assert tree_without_token_leaves.extract_all_phrases() == postag_phrases

    def contraction_expansion(self):
        sentence = "I haven't the foggiest idea what you're talking about!"
        tree = ConstituentTree(sentence, self.nlp, expand_contractions=True)

        nc_text_only = tree.leaves(tree.nltk_tree, ConstituentTree.NodeContent.Text)
        nc_postag_only = tree.leaves(tree.nltk_tree, ConstituentTree.NodeContent.Pos)
        nc_combined = tree.leaves(tree.nltk_tree, ConstituentTree.NodeContent.Combined)

        true_text_only = "I have not the foggiest idea what you are talking about !"
        true_postag_only = "PRP VBP RB DT JJS NN WP PRP VBP VBG IN ."
        true_combined = "I_PRP have_VBP not_RB the_DT foggiest_JJS idea_NN what_WP " \
                        "you_PRP are_VBP talking_VBG about_IN !_."
        assert (nc_text_only == true_text_only and nc_postag_only == true_postag_only and nc_combined == true_combined)

    def test_tree_structure_without_token_leaves(self):
        sentence = "Let's test the improvements, shall we?"
        tree_without_token_leaves = ConstituentTree(sentence, self.nlp, structure=Structure.WithoutTokenLeaves)
        bracketed_string = "(SQ (S (VP VB (S (NP PRP) (VP VB (NP DT NNS))))) , MD (NP PRP) .)"
        assert tree_without_token_leaves.to_bracketed_tree_string() == bracketed_string

    def test_tree_structure_without_postag_nodes(self):
        sentence = "Let's test the improvements, shall we?"
        tree_without_postag_nodes = ConstituentTree(sentence, self.nlp, structure=Structure.WithoutPostagNodes)
        bracketed_string = "(SQ (S (VP Let (S (NP 's) (VP test (NP the improvements))))) , shall (NP we) ?)"
        assert tree_without_postag_nodes.to_bracketed_tree_string() == bracketed_string

    def test_svg_export(self):
        file_name = "tree.svg"
        sentence = "Python is an awesome coding language!"
        ConstituentTree(sentence, self.nlp).export_tree(file_name)
        with open(file_name, "rb") as file:
            md5_hash = hashlib.md5(file.read()).hexdigest()
        assert md5_hash == "d3e9fdbe78fee450f212d605584f3b2a"
        os.remove(file_name)
