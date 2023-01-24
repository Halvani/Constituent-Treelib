import torch
import spacy
import base64
import streamlit as st
from pathlib import Path
from constituent_treelib import ConstituentTree, BracketedTree


@st.experimental_singleton
def get_nlp():
	language = ConstituentTree.Language.English
	spacy_model_size = ConstituentTree.SpacyModelSize.Medium
	nlp = ConstituentTree.create_pipeline(language, spacy_model_size, download_models = False)
	return nlp


nlp = get_nlp()

st.title("*** Constituent Tree Playground ***")

text = st.text_input("Enter your sentence...", "There is a constituent behind the tree!")

tree = ConstituentTree(text, nlp)
tree.export_tree("temp.svg")
tree_svg = Path("temp.svg").read_text()
#tree_svg = tree_svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', count=1)
tree_svg = '<svg xmlns="http://www.w3.org/2000/svg"' + tree_svg[4:]


st.image(tree_svg)


tree = ConstituentTree(text, nlp)
tree.export_tree("temp.pdf")
tree_pdf = Path("temp.pdf").read_bytes()

st.download_button(
    label="Download as PDF",
    data=tree_pdf,
    file_name='tree.pdf'
)

st.download_button(
    label="Download as SVG",
    data=tree_svg,
    file_name='tree.svg'
)

all_phrases = tree.extract_all_phrases(min_words_in_phrases=1)

st.write("Extracted phrases...")
st.write(all_phrases)