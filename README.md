# Style Transfer

## Task 5: Style transfer of famous artists using CNN network
+ Description: Compose images in the style of another image
+ Network: based on VGG19
+ [Tensorflow tutorial][tensorflow-tutorial]
+ [Image database][wikimedia]
+ Remarks: Test (and if necessary train) on your own images and styles of at 
least 2 different artists. Check the influence of using different number of
intermediate layers and in different location of network structure.

## Project structure
### Theory:
+ 1 title page 
+ 1 table of content page
+ 10 - 20 pages - about: 
    + purpose of the work
    + description of the method and description of its implementation
    + obtained results and short summary
+ 1 bibliography page - at least 5 positions.
### Program
+ Code with comments. It has to contain all the necessary elements needed to 
run it, e.g. list of used libraries and images.
### Presentation
+ 15 minute Presentation in January.

The order of the presentation in accordance with the date of handing over the 
work. The maximum number of presentations during project classes is 3.

### Grading
+ Theory: 0 - 4 points,
+ Program: 0 - 4 points,
+ Presentation and questions: -2 to +2 points

The project deadline is 24.12.2020 (theory + program).
Each week of delay: -1 point,

Final Neural Network subject grad:
0.7 * Test grade + 0.3 * Project grade

## Using latex
+ Download [TexPortable][tex-portable] for Windows
+ Use [LaTeX Workshop][latex-vscode] for [Visual Studio Code][vscode]

## (From the latex template) Handling References when submitting to arXiv.org
The most convenient way to manage references is using an external BibTeX file and pointing to it from the main file. 
However, this requires running the [bibtex](http://www.bibtex.org/) tool to "compile" the `.bib` file and create `.bbl` file containing "bibitems" that can be directly inserted in the main tex file. 
However, unfortunately the arXiv Tex environment ([Tex Live](https://www.tug.org/texlive/)) do not do that. 
So easiest way when submitting to arXiv is to create a single self-contained .tex file that contains the references.
This can be done by running the BibTeX command on your machine and insert the content of the generated `.bbl` file into the `.tex` file and commenting out the `\bibliography{references}` that point to the external references file.

Below are the commands that should be run in the project folder:
1. Run `$ latex template`
2. Run `$ bibtex template`
3. A `template.bbl` file will be generated (make sure it is there)
4. Copy the `template.bbl` file content to `template.tex` into the `\begin{thebibliography}` command.
5. Comment out the `\bibliography{references}` command in `template.tex`.
6. You ready to submit to arXiv.org.

[tex-portable]: https://1drv.ms/u/s!AqZvnCxLmXx9hNYhYFKVQA4d-E9HGw?e=lgzL9B  "TexPortable"
[latex-vscode]: https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop "LaTeX Workshop"
[vscode]: https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop "Visual Studio Code"
[tensorflow-tutorial]: https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb "Tensorflow Tutorial"
[wikimedia]: https://commons.wikimedia.org/wiki/Main_Page "Wikimedia"