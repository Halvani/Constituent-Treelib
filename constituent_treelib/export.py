import os
import re
import sys
import json
import pdfkit
from pathlib import Path
from nltk import Tree
from nltk.draw.tree import TreeView
from nltk.tree.prettyprinter import TreePrettyPrinter
from typing import Dict


def __to_dict(tree: Tree) -> Dict[str, str]:
    """Transforms the given tree into a nested dictionary.

    Args:
        tree: An nltk.Tree that should be transformed into a nested dictionary.

    Returns:
        A nested dictionary representation of the given tree.
    """
    return {tree.label(): [__to_dict(t) if isinstance(t, Tree) else t for t in tree]}


def export_figure(nltk_tree: Tree, destination_filepath: str, wkhtmltopdf_bin_filepath: str, verbose: bool,
                  dpi: int, tree_style_nltk: bool) -> None:
    """ Exports the constructed constituent tree in various file formats (currently supported:
    [.pdf, .svg, .ps, .png, .jpg, .gif, .bmp, .psd, .eps, .tiff, .txt, .tex, .json, .yaml].

    Args:
        nltk_tree: To parsed nltk.Tree that should be exported.

        destination_filepath: The file path to which the tree should be exported. In case of an image format,
        the resulting visualization will be cropped with respect to unnecessary margins.

        wkhtmltopdf_bin_filepath: To filepath to the rendering tool "wkhtmltopdf". Only required if the
        visualization of the constituent tree should be exported to a PDF file. If not already done, the tool
        wkhtmltopdf must first be downloaded and installed from https://wkhtmltopdf.org before the visualization
        can be exported.

        dpi: Specifies the desired resolution. A DPI value of 300 is considered a good standard for
        printable files.

        tree_style_nltk: If set to True, the classic NLTK style will be used to visualize the nltk.Tree.

        verbose: If set to True, a short message about whether the output file creation was
        successful is displayed.
    """

    # Convert the nltk.Tree into an SVG representation.
    svg_obj = nltk_tree._repr_svg_()

    if tree_style_nltk:
        svg_obj = TreePrettyPrinter(nltk_tree).svg(nodecolor='black', leafcolor='black', funccolor='black')

    extension = Path(destination_filepath).suffix.lower()
    try:
        if extension == '.ps':
            TreeView(nltk_tree)._cframe.print_to_file(destination_filepath)

        elif extension == '.txt':
            tree_as_text = TreePrettyPrinter(nltk_tree).text()
            Path(destination_filepath).write_text(data=tree_as_text, encoding='utf-8')

        elif extension == '.tex':
            # Build a minimal working tex file.
            tex_head = '\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{qtree}\n'
            tex_body = '\\begin{document}\n' + nltk_tree.pformat_latex_qtree() + '\n\\end{document}'
            tex_file = f'{tex_head}{tex_body}'

            Path(destination_filepath).write_text(data=tex_file, encoding='utf-8')

        elif extension == '.json':
            tree_dic = __to_dict(nltk_tree)
            json_str = json.dumps(tree_dic, indent=1)
            Path(destination_filepath).write_text(data=json_str, encoding='utf-8')

        elif extension == '.svg':
            Path(destination_filepath).write_text(svg_obj, encoding='utf-8')

        elif extension in ['.jpg', '.png', '.gif', '.bmp', '.eps', '.psd', '.tiff', '.yaml']:
            # In case that ImageMagick is not installed, the Python binding "wand" raises an exception and returns the
            # appropriate URL for the version of ImageMagick that matches the current operating system.
            # After downloading and installing ImageMagick, "wand" can be used to export the visualization.
            # Take a look at the docs https://docs.wand-py.org for further information.
            from wand.api import library
            from wand.image import Image

            with Image(blob=svg_obj.encode('utf-8'), resolution=dpi) as image:
                image.save(filename=destination_filepath)

        elif extension == '.pdf':
            msg_invalid_wkhtmltopdf_path = 'A valid path to the wkhtmltopdf binary must be provided in order to export ' \
                                           'the parsed nltk.Tree into a pdf file.'

            # The path to the wkhtmltopdf binary was not specified (None).
            if wkhtmltopdf_bin_filepath is None:
                # In case of a Windows OS, an attempt is made to locate the path of the wkhtmltopdf binary by looking up
                # the default installation directory ("Program Files/wkhtmltopdf")
                if sys.platform == 'win32':
                    wkhtmltopdf_bin_filepath = Path(os.environ.get('ProgramFiles'),
                                                    'wkhtmltopdf',
                                                    'bin',
                                                    'wkhtmltopdf.exe')

                    # Check if the default path to the wkhtmltopdf binary actually exists.
                    if not Path(wkhtmltopdf_bin_filepath).exists():
                        raise AssertionError(
                            'The wkhtmltopdf binary (e.g., wkhtmltopdf.exe on a Windows OS) could not be found '
                            'under the default installation directory ("Program Files/wkhtmltopdf"). '
                            'If not installed yet, you should download it first from https://wkhtmltopdf.org. '
                            'Note, you only need to install the program itself and provide this method the path where '
                            'it can be found. Setting environment variables is not required.')

                # If the current OS is not Windows, the path to the wkhtmltopdf binary must be specified manually.
                else:
                    raise AssertionError(msg_invalid_wkhtmltopdf_path)

            # In case that a path to the wkhtmltopdf binary has been provided, check if it is valid.
            else:
                if len(wkhtmltopdf_bin_filepath.strip()) == 0 or not Path(wkhtmltopdf_bin_filepath).exists():
                    raise AssertionError(msg_invalid_wkhtmltopdf_path)

            # Determine visible size of the SVG object.
            height = int(re.search(r"height=\"([0-9]+)", svg_obj).groups(1)[0])
            width = int(re.search(r"width=\"([0-9]+)", svg_obj).groups(1)[0])
            height = str((height + 10) * dpi / 96) + "px"
            width = str(width * dpi / 96) + "px"

            # Default options for the rendering process. All options of wkhtmltopdf can be looked up under'
            # https://wkhtmltopdf.org/usage/wkhtmltopdf.txt
            options = {
                'dpi': str(dpi),
                'page-size': 'Letter',
                'margin-top': '0mm',
                'margin-bottom': '0mm',
                'margin-left': '0mm',
                'margin-right': '0mm',
                'page-width': width,
                'page-height': height,
                'encoding': "UTF-8",
                'disable-smart-shrinking': None
            }

            config = pdfkit.configuration(wkhtmltopdf=str(wkhtmltopdf_bin_filepath))
            pdfkit.from_string(svg_obj, output_path=destination_filepath, configuration=config, options=options)
        else:
            raise ValueError(
                "Currently, only the following file formats are supported "
                "[.pdf, .svg, .ps, .png, .jpg, .gif, .bmp, .psd, .eps, .tiff, .txt, .tex, .json, .yaml]")

    except Exception as e:
        raise Exception(f'The specified {extension[1:].upper()}-file could not be saved. Error: {e}')

    if verbose:
        print(f'{extension[1:].upper()}-file successfully saved to: {destination_filepath}')
