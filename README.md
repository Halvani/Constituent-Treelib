# Constituent Treelib (CTL)
A lightweight Python library for constructing, processing, and visualizing constituent trees.


# Description
CTL allows you to easily construct a <a href="https://en.wikipedia.org/wiki/Constituent_(linguistics)">constituent tree</a> representation of sentences and to visualize them into various [formats](#Export_visualization). For this, CTL builds on top of <a href="https://github.com/nikitakit/self-attentive-parser">**benepar**</a> (*Berkeley Neural Parser*) as well as both well-known NLP frameworks <a href="https://spacy.io">**spaCy**</a> and <a href="https://github.com/nltk/nltk">**NLTK**</a>. Here, spaCy is used only for tokenization and sentence segmentation, while benepar performs the actual parsing of the sentences. NLTK, on the other hand, is needed as it provides the fundamental data structure for the parsed sentences. 

To gain a clearer picture of what a constituent tree looks like, let's consider the following example. Given the sentence *S* = ``The bridge was unfinished when it collapsed.``, CTL first parses *S* into a bracketed tree representation (in a <a href="https://catalog.ldc.upenn.edu/LDC99T42">*Penn tree-bank*</a> style) that leads to the following result:   
```
(S
  (NP (DT The) (NN bridge))
  (VP
    (VBD was)
    (ADJP (JJ unfinished))
    (SBAR (WHADVP (WRB when)) (S (NP (PRP it)) (VP (VBD collapsed)))))
  (. .))
```

This bracketed tree string can then be visualized as a constituent tree in a desired format, here as an SVG file:

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
- Analysis and comparison of sentence structures between different languages (e.g. English and German) for language learners
- Extracting phrasal features for certain NLP tasks (e.g., <a href="https://github.com/andreasvc/authident">stylometry</a>, <a href="https://ieeexplore.ieee.org/document/6693511">information extraction</a> or <a href="https://aclanthology.org/P12-2034">deception detection</a>)
- Use the resulting representations as a prestep to train <a href="https://distill.pub/2021/gnn-intro/">GNNs</a> for specific tasks (e.g., <a href="https://doi.org/10.1093/database/baac070">chemical–drug relation extraction</a> or <a href="https://aclanthology.org/2020.emnlp-main.322">semantic role labeling</a>)


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



## Creating an NLP pipeline
In order to construct a constituent tree, CTL relies on a spaCy-based NLP pipeline, which incorporates the *benepar* component. Although you can set up this pipeline yourself, it is recommended (and more convenient) to let CTL do it for you automatically via the ``create_pipeline()`` method. Given the desired language, the method creates the nlp pipeline and also downloads[^3] the corresponding spaCy and benepar models, if requested. The following code shows an example of this: 
```python
from constituent_treelib.constituent_treelib import ConstituentTree

language = ConstituentTree.Language.English
spacy_model_size = ConstituentTree.SpacyModelSize.Medium

nlp = ConstituentTree.create_pipeline(language, spacy_model_size=spacy_model_size, download_models=True)
```

Note that loading and initializing the models can take a while, so it makes sense to invoke ``create_pipeline()`` only once if you want to process multiple sentences. 


<a name="Available_models_and_languages"></a>
## Available models and languages
CTL currently supports seven languages: English, German, French, Polish, Swedish, Chinese and Korean. 



<a name="Export_visualization"></a>
## Export visualization
In order to export the visualization of the parsed constituent tree you can choose among various file formats (currently supported: [.png, .jpg, .gif, .bmp, .pdf, .svg, .txt, .tex, .ps]). 

### Export into PDF
To export the parsed tree into a PDF the open source (LGPLv3) command line tool **<a href="https://wkhtmltopdf.org/downloads.html">wkhtmltopdf</a>** is required. Once downloaded and installed, you need to pass CTL the respective path to the wkhtmltopdf binary.

### Export into JPG, PNG, GIF, BMP
To export the parsed tree into a rasterized image format, the <a href="https://imagemagick.org/script/license.php">open-source</a> software suite 
**<a href="https://imagemagick.org/script/download.php#windows">ImageMagick</a>** is required. 


# License
The code and the jupyter notebook demo of CTL are released under the MIT License. See <a href="https://github.com/Halvani/constituent_treelib/blob/main/LICENSE">LICENSE</a> for further details.



# Citation
TODO...



[^1]: Note, if you are not familiar with the bracket labels of constituent trees, 
have a look at the following <a href="https://gist.github.com/nlothian/9240750">Gist</a> 
or alternatively <a href="http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html">this website</a>. 

[^2]: It's recommended to install CTL from <a href="https://pypi.org">PyPI</a> (*Python Package Index*). However, if you want to benefit from the latest update of CTL, you should use this repository instead, since I will only update PyPi at irregular intervals.   

[^3]: After the models have been downloaded, they are cached so that there are no redundant downloads when the method is called again. 