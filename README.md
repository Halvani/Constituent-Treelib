# Constituent Treelib (CTL)
A lightweight Python library for processing and visualizing constituent trees.


# Description
CTL allows you to easily construct the <a>constituent tree</a> representation of sentences and to visualize the result into various formats. To parse sentences into constituent trees, CTL builds on top of the awesome **Berkeley Neural Parser** (or short: **benepar**) as well as both NLP frameworks spaCy and NLTK.   

To better understand what a constituent tree representation looks like, consider the following example.  Given the sentence *S* = ``The bridge was unfinished when it collapsed.``, CTL first constructs the corresponding constituent tree of *S* in a *Penn tree-bank* style    
```
(S
  (NP (DT The) (NN bridge))
  (VP
    (VBD was)
    (ADJP (JJ unfinished))
    (SBAR (WHADVP (WRB when)) (S (NP (PRP it)) (VP (VBD collapsed)))))
  (. .))
```

which can then be visualized into a desired format (here, plain-text):
```
                       S                                   
      _________________|_________________________________   
     |                          VP                       | 
     |           _______________|_____                   |  
     |          |      |             SBAR                | 
     |          |      |         _____|____              |  
     |          |      |        |          S             | 
     |          |      |        |      ____|______       |  
     NP         |     ADJP    WHADVP  NP          VP     | 
  ___|____      |      |        |     |           |      |  
 DT       NN   VBD     JJ      WRB   PRP         VBD     . 
 |        |     |      |        |     |           |      |  
The     bridge was unfinished  when   it      collapsed  . 
```

This representation[^1] shows three aspects of the structure of *S*: 
- Linear order of the words and their part-of-speech in *S* (from left to right: ``The_DT``, ``bridge_NN``, ``was_VBD``, ...)
- The groupings of the words and their part-of-speech into phrases (from left to right: ``NP``, ``ADJP``, ``WHADVP``, ``NP``, ``VP``)
- The hierarchical structure of the phrases (from left to right: ``S``, ``VP``, ``SBAR``, ``S``)

[^1]: Note, if you are not familiar with these so-called bracket labels, 
have a look at the following <a href="https://gist.github.com/nlothian/9240750">Gist</a> 
or alternatively <a href="http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html">this website</a>.






# Features
- Easy-to-use construction of a constituent tree from a given sentence. 
- Multilingual (currently seven languages are supported).
- Convenient export of tree visualization to various file formats (currently supported file formats [.png, .jpg, .gif, .bmp, .pdf, .svg, .txt, .tex, .ps]).
- Extraction of phrases according to given phrasal categories.


# Installation
The easiest way to install CTL is to use pip. Here, you can choose between (1) the PyPI[^2] repository and (2) this repository. 

1 ```TODO```

2 ```pip install git+https://github.com/Halvani/constituent_treelib.git ```

[^2]: It's recommended to install CTL from <a href="https://pypi.org">PyPI</a> (*Python Package Index*). However, if you want to benefit from the latest update of CTL, you should use this repository instead, since I will only update PyPi at irregular intervals.   



## Creating an NLP pipeline
In order to construct a constituent tree, CTL relies on a spaCy-based NLP pipeline, which incorporates the *benepar* component. Although you can set up this pipeline yourself, it is recommended (and more convenient) to let CTL do it for you automatically via the ``create_pipeline()`` method. 

Given the desired language, the method creates the nlp pipeline and also downloads the corresponding spaCy and benepar models, if requested. After the models have been downloaded, they are cached so that there are no redundant downloads when the method is called again.  

Note that loading and initializing the models can take a while, so it makes sense to invoke ``create_pipeline()`` only once if you want to process multiple sentences. 


## Available models and languages
CTL currently supports seven languages: English, German, French, Polish, Swedish, Chinese and Korean. 




## Export visualization
In order to export the visualization of the parsed constituent tree you can choose among various file formats (currently supported: [.png, .jpg, .gif, .bmp, .pdf, .svg, .txt, .tex, .ps]). 

### Export into PDF
To export the parsed tree into a PDF the open source (LGPLv3) command line tool **<a href="https://wkhtmltopdf.org/downloads.html">wkhtmltopdf</a>** is required. Once downloaded and installed, you need to pass CTL the respective path to the wkhtmltopdf binary.

### Export into JPG, PNG, GIF, BMP
To export the parsed tree into a rasterized image format, the open-source software suite 
**<a href="https://imagemagick.org/script/download.php#windows">ImageMagick</a>** is required. 



# License
The code and the jupyter notebook demo of CTL are released under the MIT License. See <a href="https://github.com/Halvani/constituent_treelib/blob/main/LICENSE">LICENSE</a> for further details.