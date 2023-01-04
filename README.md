# Constituent Treelib (CTL)
A lightweight Python library for constructing, processing, and visualizing constituent trees.


# Description
CTL allows you to easily construct a <a href="https://en.wikipedia.org/wiki/Constituent_(linguistics)">constituent tree</a> representation of sentences and to visualize them into various [formats](#Export_visualization). For this, CTL builds on top of <a href="https://github.com/nikitakit/self-attentive-parser">**benepar**</a> (*Berkeley Neural Parser*) as well as both well-known NLP frameworks <a href="https://spacy.io">**spaCy**</a> and <a href="https://github.com/nltk/nltk">**NLTK**</a>. Here, spaCy is used only for tokenization and sentence segmentation, while benepar performs the actual parsing of the sentences. NLTK, on the other hand, is needed as it provides the fundamental data structure for the parsed sentences. 

To gain a clearer picture of what a constituent tree looks like, let's consider the following example. Given the sentence *S = "The bridge was unfinished when it collapsed."*, CTL first parses *S* into a bracketed tree representation (in a <a href="https://catalog.ldc.upenn.edu/LDC99T42">*Penn tree-bank*</a> style):    
```
(S
  (NP (DT The) (NN bridge))
  (VP
    (VBD was)
    (ADJP (JJ unfinished))
    (SBAR (WHADVP (WRB when)) (S (NP (PRP it)) (VP (VBD collapsed)))))
  (. .))
```

This so-called *bracketed tree string* can then be visualized as a constituent tree in a desired format, here as an SVG file:

![(S
  (NP (DT The) (NN bridge))
  (VP
    (VBD was)
    (ADJP (JJ unfinished))
    (SBAR (WHADVP (WRB when)) (S (NP (PRP it)) (VP (VBD collapsed)))))
  (. .))](assets/images/sample_tree.svg)

This representation[^1] shows three aspects of the structure of *S*: 
- Linear order of the words and their part-of-speech in *S* (from left to right: ``The_DT``, ``bridge_NN``, ``was_VBD``, ...)
- Groupings of the words and their part-of-speech into phrases (from left to right: ``NP``, ``ADJP``, ``WHADVP``, ``NP``, ``VP``)
- Hierarchical structure of the phrases (from left to right: ``S``, ``VP``, ``SBAR``, ``S``)


# Applications
Constituent trees offer a wide range of applications, such as:
- Analysis and comparison of sentence structures between different languages for (computational) linguists 
- Extracting phrasal features for certain NLP tasks (e.g., <a href="https://aclanthology.org/W19-5203">Machine Translation</a>, <a href="https://ieeexplore.ieee.org/document/6693511">Information Extraction</a>, <a href="https://aclanthology.org/2020.tacl-1.22">Paraphrasing</a>, <a href="https://github.com/andreasvc/authident">Stylometry</a>, <a href="https://aclanthology.org/P12-2034">Deception Detection</a> or <a href="https://dl.acm.org/doi/10.1145/2482513.2482522">Natural Language Watermarking</a>)
- Use the resulting representations as a prestep to train <a href="https://distill.pub/2021/gnn-intro/">GNNs</a> for specific tasks (e.g., <a href="https://doi.org/10.1093/database/baac070">Chemical–Drug Relation Extraction</a> or <a href="https://aclanthology.org/2020.emnlp-main.322">Semantic Role Labeling</a>)


# Features
- Easy construction of constituent trees from raw sentences
- Multilingual (currently CTL supports [seven languages](#Available_models_and_languages))
- Convenient export of tree visualizations to various file [formats](#Export_visualization)
- Extraction of phrases according to given phrasal categories
- Extensively documented source code


# Installation
The easiest way to install CTL is to use pip, where you can choose between (1) the PyPI[^2] repository and (2) this repository. 

1 ```TODO```

2 ```pip install git+https://github.com/Halvani/constituent_treelib.git ```

The latter command will pull and install the latest commit from this repository as well as the required Python dependencies. Besides these, CTL also relies on the following two external tools to export the constructed constituent tree into various file formats:

1 To export the constituent tree into a PDF the <a href="https://github.com/wkhtmltopdf/wkhtmltopdf/blob/master/LICENSE">open-source</a> command line tool **<a href="https://wkhtmltopdf.org/downloads.html">wkhtmltopdf</a>** is required. Once downloaded and installed, the path to the wkhtmltopdf binary must be passed to the export function. 
 
2 To export the constituent tree into the file formats JPG, PNG, GIF, BMP, EPS, PSD, TIFF and YAML, the <a href="https://imagemagick.org/script/license.php">open-source</a> software suite 
**<a href="https://imagemagick.org/script/download.php#windows">ImageMagick</a>** is required.


# Quickstart
Below are some examples of how to get started with CTL.


## Creating an NLP pipeline
To instantiate a ``ConstituentTree`` object, CTL requires a spaCy-based NLP pipeline that incorporates a benepar component. Although you can set up this pipeline yourself, it is recommended (and more convenient) to let CTL do it for you automatically via the ``create_pipeline()`` method. Given the desired [language](#Available_models_and_languages), this method creates the NLP pipeline and also downloads[^3] the corresponding spaCy and benepar models, if requested. The following code shows an example of this: 
```python
from constituent_treelib.constituent_treelib import ConstituentTree

language = ConstituentTree.Language.English
spacy_model_size = ConstituentTree.SpacyModelSize.Medium

nlp = ConstituentTree.create_pipeline(language, spacy_model_size, download_models=True)
```

## Define a sentence
Now we can instantiate a ``ConstituentTree`` object and pass it the NLP pipeline along with a sentence, for example the memorable quote *"You must construct additional pylons"*[^4]. Rather than a raw sentence, ``ConstituentTree`` also accepts an already parsed  sentence in a bracketed tree notation, or alternatively in the form of an NLTK tree. The following example illustrates all three options:  
```python
from nltk import Tree

# Raw sentence
sentence = 'You must construct additional pylons!'

# Parsed sentence in a bracketed tree notation
'(S (NP (PRP You)) (VP (MD must) (VP (VB construct) (NP (JJ additional) (NNS pylons)))) (. !))'

# Parsed sentence in the form of an NLTK tree
sentence = Tree('S', [Tree('NP', [Tree('PRP', ['You'])]), Tree('VP', [Tree('MD', ['must']), Tree('VP', [Tree('VB', ['construct']), Tree('NP', [Tree('JJ', ['additional']), Tree('NNS', ['pylons'])])])]), Tree('.', ['!'])])

tree = ConstituentTree(sentence, nlp)
```

By default, punctuation marks are inserted at the end of a sentence directly under the **S** node. To avoid this behavior, one can also instantiate a ``ConstituentTree`` and instruct it to add an artificial root node (e.g. **TOP**) so that each punctuation mark at the end of a sentence is inserted as a separate branch under this root node. The following example shows this:
```python
tree = ConstituentTree(sentence, nlp, integrate_punctuation=False) 
print(tree)

>>>
 (TOP     # <--- 
  (S
    (NP (PRP You))
    (VP
      (MD must)
      (VP (VB construct) (NP (JJ additional) (NNS pylons))))
    (. !) # <--- 
  ))
```



## Extract phrases
Once we have created ``tree``, we can now extract phrases according to given phrasal categories e.g., verb phrases:  
```python
phrases = tree.extract_all_phrases()
print(phrases['VP'])

>>> ['must construct additional pylons', 'construct additional pylons']
```

As can be seen here, the second verb phrase is contained in the former. To avoid this, we can instruct the method to disregard nested phrases:  
```python
phrases = tree.extract_all_phrases(take_longest=True)
print(phrases['VP'])

>>> ['must construct additional pylons']
```


<a name="Export_visualization"></a>
## Export the tree
CTL offers you the possibility to export the constructed constituent tree into different file formats, which are listed below. Most of these formats represent a visualization of the tree, while the remaining file formats are for data exchange purposes. 

| Extension | Description | Output |
| --- | --- | --- |
| **PDF** | *Portable Document Format* | Vector graphic |
| **SVG** | *Scalable Vector Graphics* | Vector graphic |
| **EPS** | *Encapsulated PostScript* | Vector graphic |
| **JPG** | *Joint Photographic Experts Group* | Raster image |
| **PNG** | *Portable Network Graphics* | Raster image |
| **GIF** | *Graphics Interchange Format* | Raster image |
| **BMP** | *Bitmap* | Raster image |
| **PSD** | *Photoshop Document* | Raster image |
| **TIFF** | *Tagged Image File Format* | Raster image |
| **JSON** | *JavaScript Object Notation* | Data exchange format |
| **YAML** | *Yet Another Markup Language* | Data exchange format |
| **TXT** | *Plain-Text* | Pretty-print tree visualization |
| **TEX** | *LaTeX-Document* | LaTeX-typesetting of the tree |

The following example shows how to export the constituent tree into a PDF file:
```python
tree.export_tree(dest_filepath='my_tree.pdf', verbose=True)

>>> PDF-file successfully saved to: my_tree.pdf
```

In case of any raster image or vector graphic format, the resulting visualization will be cropped with respect to unnecessary margins. 




<a name="Available_models_and_languages"></a>
## Available models and languages
CTL currently supports seven languages: English, German, French, Polish, Swedish, Chinese and Korean. 

# License
The code and the jupyter notebook demo of CTL are released under the MIT License. See <a href="https://github.com/Halvani/constituent_treelib/blob/main/LICENSE">LICENSE</a> for further details.



# Citation
If you find this repository helpful, feel free to cite it in your paper or project: 
```bibtex
@misc{HalvaniConstituentTreelib:2023,
    title={{A Lightweight Python Library for Constructing, Processing, and Visualizing Constituent Trees}},
    author={Oren Halvani},
    year={2023},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/Halvani/constituent_treelib}}
}
```

[^1]: Note, if you are not familiar with the bracket labels of constituent trees, 
have a look at the following <a href="https://gist.github.com/nlothian/9240750">Gist</a> 
or alternatively <a href="http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html">this website</a>. 

[^2]: It's recommended to install CTL from <a href="https://pypi.org">PyPI</a> (*Python Package Index*). However, if you want to benefit from the latest update of CTL, you should use this repository instead, since I will only update PyPi at irregular intervals. 

[^3]: After the models have been downloaded, they are cached so that there are no redundant downloads when the method is called again. However, loading and initializing the spaCy and benepar models can take a while, so it makes sense to invoke the ``create_pipeline()`` method only once if you want to process multiple sentences.

[^4]: https://knowyourmeme.com/memes/you-must-construct-additional-pylons