# ClassTor
A python tool for Classification of Torsion angles

ClassTor is a simple python tool for analysis and classification of sampled conformations of a ligand-sized molecule based on dihedral (torsion) angles. It consists of two python modules (classtor_spectrum.py, classtor_classification.py) and an executable analysis script (ClassTor.py), that uses these modules.

# Usage
```./ClassTor.py AngNum [OPTION] ```

**Mandatory argument:**\
*AngNum* &nbsp;&nbsp;&nbsp;&nbsp; integer number of torsion angles, must be the first argument 

**Optional arguments:**\
preparation:\
-pa *STR* &nbsp;&nbsp;&nbsp;&nbsp; path to location of torsion angle files (default: current directory ```./```)

-x &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when specified, the last line of every torsion angle file is taken as a reference value and is\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; only included for comparison calculations to that reference structure, not for the generation\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; of each torsion angle spectrum or classification\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: not specified)

spectrum:\
-gk *INT* &nbsp;&nbsp;&nbsp;&nbsp; width of the gaussian kernel used for convoluting the frequency of each torsion angle value when generating the\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; corresponding spectrum\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: 15)

-t *INT* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of adjacent data points considered for the detection of relative extrema (minima and maxima) and the\
;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; characterization of the course at the spectrum borders (around -180&deg; and +180&deg;)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: 20)

classification:\
-f *STR* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; string of letters separated by whitespace specifying a subset of torsions used for classification, e.g. when only the most\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; flexible torsions should be considered. If not specified, all *AngNum* torsions are considered\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: not specified)

-m &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when specified, the class members of each class are added to the classification output file *summary_classification.log* to\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a separate line under the corresponding class\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (default: not specified)

clustering:\
-c &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; when specified, an additional hierarchical agglomerative clustering of all classification centroid structures will be performed\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;after classification\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(default: not specified)

subset extraction - ClassTors perturbational approach:\
-ss *INT* &nbsp;&nbsp;&nbsp;&nbsp; size of the conformer subset extracted after classification\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(default: 10)

-sr *INT* &nbsp;&nbsp;&nbsp;&nbsp; specifies the first reference classifier for the perturbational subset extraction:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0 < *INT* <= highest class-ID: the classifier of the specified class-ID is used (first reference is always included in the subset)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0: the average classifier is determined out of all classes obtained after classification and used as a first reference (default; \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; will be included in the subset if the average classifier matches an obtained classifier of the classification)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -1: the classifier of a random class (0 < *INT* <= highest class-ID) is used

-sr *STR* &nbsp;&nbsp;&nbsp;&nbsp;specifies the order in which all classifiers from the classification are compared to the first reference (*-sr*) in the\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; first selection round, options: random, topdown, reverse\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "random": shuffle the list of classifiers to randomize the order (default)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "topdown": order is from highest to lowest populated class\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "reverse": order is from lowest to highest populated class

# Getting started
:red_circle: info about download and putting the modules in pythonpath to make them importable\
explain requirements for ClassTor.py (a_angles.dat from e.g. cpptraj) :red_circle:


# Citing ClassTor
tba

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
