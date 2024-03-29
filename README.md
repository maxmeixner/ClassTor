# ClassTor
A python tool for Classification of ligand conformations based on Torsion angles

ClassTor is a simple python tool for analysis and classification of sampled conformations of a ligand-sized molecule based on dihedral (torsion) angles. It consists of two python modules (classtor_spectrum.py, classtor_classification.py) and an executable analysis script (ClassTor.py), that uses these modules.

# Usage
```./ClassTor.py AngNum [OPTION] ```

**Mandatory argument:**\
*AngNum* &nbsp;&nbsp;&nbsp;&nbsp; integer number of torsion angles, must be the first argument. The corresponding\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; amount of individual two-column files with the dihedral angle value of a torsion per\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; frame must be labelled ```X_angles.dat```, where ```X``` is the letter of a labeled torsion starting\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; with ```a```. 0 < *AngNum* <= 26

**Optional arguments:**\
preparation:\
-pa *STR* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; path to location of torsion angle files (default: current directory ```./```)

-x &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when specified, the last line of every torsion angle file is taken as a reference value and is\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; only included for comparison calculations to that reference structure, not for the generation\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; of each torsion angle spectrum or classification\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: not specified)

spectrum:\
-gk *INT* &nbsp;&nbsp;&nbsp;&nbsp; width of the gaussian kernel used for convoluting the frequency of each torsion angle value\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when generating the corresponding spectrum\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: 15)

-t *INT* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of adjacent data points considered for the detection of relative extrema (minima\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and maxima) and the characterization of the course at the spectrum borders (around -180&deg;\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and +180&deg;)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: 20)

classification:\
-f *STR* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; string of letters separated by whitespace specifying a subset of torsions used for\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; classification, e.g. when only the most flexible torsions should be considered.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If not specified, all *AngNum* torsions are considered\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: not specified)

-m &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when specified, the class members of each class are added to the classification output file\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *summary_classification.log* to a separate line under the corresponding class\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: not specified)

subset extraction - ClassTor's perturbational approach:\
-ss *INT* &nbsp;&nbsp;&nbsp;&nbsp; size of the conformer subset extracted after classification\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(default: 10)

-sr *INT* &nbsp;&nbsp;&nbsp;&nbsp; specifies the first reference classifier for the perturbational subset extraction:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0 < *INT* <= highest class-ID: the classifier of the specified class-ID is used (first reference\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is always included in the subset)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0: the average classifier is determined out of all classes obtained after classification and used\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; as a first reference (default; will be included in the subset if the average classifier matches\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; an obtained classifier of the classification)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -1: the classifier of a random class (0 < *INT* <= highest class-ID) is used

-sr *STR* &nbsp;&nbsp;&nbsp;&nbsp;specifies the order in which all classifiers from the classification are compared to the first\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; reference (*-sr*) in the first selection round, options: random, topdown, reverse\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "random": shuffle the list of classifiers to randomize the order (default)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "topdown": order is from highest to lowest populated class\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "reverse": order is from lowest to highest populated class

clustering:\
-c &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when specified, hierarchical agglomerative clustering is performed with centroid structures\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; after classification. This is an alternative approach of generating a subset of conformers with\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; size *-ss* from the classification results (default: not specified) 

# Dependencies
Python 2.7.9\
numpy 1.16.6\
scikit-learn 0.20.4\
scipy 1.2.3\
matplotlib 2.2.5

# Installation
Download the analysis script (ClassTor.py) and the two module files (classtor_spectrum.py, classtor_classification.py). Place the module files in your python path. If unsure about the path, you can find out as follows:\
```pyhthon```\
```import sys```\
```print sys.path```\
The output is a list of paths included in the python path. The module files are usually located in a directory called .../python2.7/site-packages/. 

# Citing ClassTor
Meixner, M., Zachmann, M., Metzler, S., Scheerer, J., Zacharias, M., & Antes, I. (2022). Dynamic Docking of Macrocycles in Bound and Unbound Protein Structures with DynaDock. *Journal of Chemical Information and Modeling*, *62*(14), 3426-3441.

# License
ClassTor Copyright (C) 2022 Maximilian Meixner

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
