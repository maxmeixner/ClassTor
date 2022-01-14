#!/usr/bin/python

import numpy as np
import argparse
from collections import OrderedDict
import os
import classtor_spectrum as cs
import classtor_classification as cc

#########################################################################################
"""
                   ClassTor - Classification of Torsion angle values

    ClassTor.py
    Copyright (C) 2022 Maximilian Meixner

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
"""
#########################################################################################

# PARSE and PREP
parser = argparse.ArgumentParser(prog='ClassTor (Classification of Torsion angle values)\n',
                                 description='generating gaussian-convoluted torsional distribution spectra'
                                             'and performing meaningful torsional classification '
                                             'according to the sampled torsion angle values',
                                 epilog='carefully check the courses of the generated spectra'
                                        ' and modify the shape if neccessary [-gk]. '
                                        'Compare the spectrum with the information retrieved from the characterization'
                                        ' and adjust if desired [-t]')
# required argument
parser.add_argument("AngNum", help="number of torsions, max: 26 if starting from 'a'", type=int)
# optional arguments
# preparation:
parser.add_argument("-pa", dest="AnglePath", help="path to location of torsion angle files, default = ./", default="./",
                    type=str)
parser.add_argument("-x", dest="Xray", help="enable if last line in every _angles.dat is a reference value "
                                            "for distance calculations (e.g. the torsion angle value "
                                            "of the bound xray structure, default is off",
                    action='store_true')
# spectrum:
parser.add_argument("-gk", dest="GaussKern", help="width of gaussian kernel, default = 15", default=15, type=int)
parser.add_argument("-t", dest="Threshold", help="threshold value defining a number of neighboring points considered"
                                                 " for detection of local extrema (order argument of function "
                                                 "argrelextrema) and for assessing the course around the spectrum "
                                                 " borders, default = 20",
                    default=20, type=int)
# classification:
parser.add_argument("-f", dest="Flexible", help="labels of torsions used for classification separated by whitespace; "
                                            "If not specified, all AngNum torsions are considered for classification",
                    nargs='+', type=str)
parser.add_argument("-m", dest="Member", help="if specified, the member frames of each class will be printed under the"
                                              "corresponding class in the output file summary_classification.log, "
					      "default is not specified",
                    action='store_true')
# extraction of diverse subset of conformers:
parser.add_argument("-ss", dest="SubSize", help="size of diverse subset of conformers, default = 10",
                    default=10, type=int)
parser.add_argument("-sr", dest="SubRef", help="SubsetReference, specify which classifier after classification should "
                                            "be used as first reference for pairwise comparison to all other "
                                               "classifiers in order to create a diverse subset of conformers, "
                                            "-1: random, 0: average, 1: classifier of class 1 (highest populated class)"
                                               ", 2: classifier of class 2, ..., "
                                            "default = 0 (average)",
                    default=0, type=int)
parser.add_argument("-so", dest="SubOrder", help="order of classifiers for pairwise comparison to first reference "
                                                 "classifier (see -sr) when generating a diverse subset of conformers"
                                                 "default = random, options: random (shuffle), "
                                                 "topdown (from highest to lowest populated class),"
                                                 "reverse (from lowest to highest populated class)",
                    default="random", type=str, choices=['random', 'topdown', 'reverse'])
# clustering:
parser.add_argument("-c", dest="Clustering", help="if specified, an additional hierarchical agglomerative clustering "
						  "will be performed after classification with the centroid stuctures "
						  "of the previous classification, default is not specified",
		    action='store_true')
args = parser.parse_args()
#########################################################################################

# prep angle files
all_torsion_angles = OrderedDict()  # 'a' : [angle-value, angle-value, ...]
all_torsion_angles_xray = OrderedDict()
xray_torsion_angles = OrderedDict()
# consecutive list of args.AngNum characters starting from 'a' (= torsion labels)
for label in map(chr, range(97, (97 + args.AngNum))):
    all_torsion_angles[label] = []
    # read torsion angle value files
    with open('%s%s_angles.dat' % (args.AnglePath, label), 'r') as F:
        for line in F:
            all_torsion_angles[label].append(line.split())
    del all_torsion_angles[label][0]  # delete header
    for element in range(len(all_torsion_angles[label])):  # convert all torsion angle values to floats
        all_torsion_angles[label][element][1] = float(all_torsion_angles[label][element][1])
        del all_torsion_angles[label][element][0]  # delete frame number
    # if last frame is xray angles, remove form all_torsion_angles but keep a copy in all_torsion_angles_xray
    if args.Xray:
        # copy the list
        all_torsion_angles_xray[label] = list(all_torsion_angles[label])
        # convert to np.array
        all_torsion_angles_xray[label] = np.array([round(item) for sublist in all_torsion_angles_xray[label]
                                                   for item in sublist])
        # delete xray value in original dictionary
        del all_torsion_angles[label][-1]
    # convert to np.array of rounded values
    all_torsion_angles[label] = np.array([round(item) for sublist in all_torsion_angles[label] for item in sublist])

# if a subset of flexible torsions is specified via [-f]
if args.Flexible:
    flexible_torsion_angles = OrderedDict()
    for label in args.Flexible:
        flexible_torsion_angles[label] = all_torsion_angles[label]

    # use flex torsions for classification -> target_torsions
    target_torsions = flexible_torsion_angles
    number_torsions = len(args.Flexible)

else:
    target_torsions = all_torsion_angles
    number_torsions = args.AngNum
first_letter = target_torsions.keys()[0]

"""
File format info:
   torsion angle file structure a_angles.dat:      = torsion a
   frame       angle value                         = header
   1   (int)  -11.35  (float)                      = element 0
   2           20.55                               = element 1 ...

   all_torsions = {'a': [(1, -11.35), (2, 20.55), ...], 'b': [element0, element1, ...], ..., 'z': [...]}

"""
#########################################################################################

# OUTPUT FILE I: SUMMARY_SPECTRUM.LOG
summary_spectrum = open('summary_spectrum_gk%d_t%d.log' % (args.GaussKern, args.Threshold), 'w')
summary_spectrum.write("Characterization of distribution spectra: \n")
spectrumline = '{:8} {:5} {:4} {:30} {:9} {:20} {inf}' + '\n'  # formatting
summary_spectrum.write(spectrumline.format('torsion', 'bins', 'bin', 'bin definition [from, to]', 'midpoint', 'status',
                                           inf='information'))
summary_spectrum.write(spectrumline.format('--------', '-----', '----', '------------------------------', '---------',
                                           '--------------------', inf='-----------'))
# store crucial info in dictionaries on the fly
dict_all_bin_def = OrderedDict()
dict_midpoints = OrderedDict()
range_covered = OrderedDict()
equal_heights = OrderedDict()
# TORSION ANGLE DISTRIBUTION SPECTRA of ALL LABELED TORSIONS (no matter if args.Flexible is specified)
for torsion in all_torsion_angles:
    distribution_spectrum = cs.prep_gauss(all_torsion_angles[torsion].tolist(), args.GaussKern)
    # OUTPUT FILES II: DISTRIBUTION SPECTRUM
    gaussian = open('gaussian_%s' % torsion, 'w')  # convoluted spectrum
    for element in distribution_spectrum:
       gaussian.write(str(element[0]) + '\t' + str(element[1]) + '\n')
    gaussian.close()
    #########################################################################################
    
    # CHARACTERIZATION OF DISTRIBUTION SPECTRUM
    characterization = cs.SpectrumAnalysis(torsion, distribution_spectrum, args.Threshold)
    # add bin definition, midpoints and flexibility info to dictionaries for flexscore / classification / silhouette
    dict_all_bin_def[torsion] = characterization.bin_def
    dict_midpoints[torsion] = characterization.midpoints
    range_covered[torsion] = characterization.std_dev_midpoints
    equal_heights[torsion] = characterization.std_dev_midpoint_heights
    # OUTPUT FILE I: characterization first entry
    summary_spectrum.write(spectrumline.format(torsion,
                                               str(characterization.bin_number),
                                               str(characterization.bin_list[0]),
                                               str(characterization.bin_def[0]),
                                               str(characterization.midpoints[0][1]),
                                               str(characterization.status),
                                               inf=str(characterization.info)))
    # OUTPUT FILE I: characterization remaining entries
    for bin_label in range(1, len(characterization.bin_list)):
        summary_spectrum.write(spectrumline.format(" ",
                                                   " ",
                                                   str(characterization.bin_list[bin_label]),
                                                   str(characterization.bin_def[bin_label]),
                                                   str(characterization.midpoints[bin_label][1]),
                                                   " ",
                                                   inf=" "))
# FLEXIBILITY SCORING
summary_spectrum.write("\nFlexibility ranking per number of bins: \n")
flexline = '{:6} {:6} {:8} {:10}' + '\n'
summary_spectrum.write(flexline.format('bins', 'rank', 'torsion', 'flexscore'))
summary_spectrum.write(flexline.format('------', '------', '--------', '----------'))
# get flexibility information from classtor_spectrum
flex_info = [(torsion, len(dict_all_bin_def[torsion].keys()), range_covered[torsion], equal_heights[torsion])
              for torsion in dict_all_bin_def]
# ranking torsions with the same amount of bins according to range- and population score
# starting from highest to lowest amount of bins
flex_scores = cs.flex_score(flex_info)
for bin_number in flex_scores:
    # OUTPUT FILE I: flexibility first entry
    summary_spectrum.write(flexline.format(str(bin_number),
                                           str(1),
                                           str(flex_scores[bin_number][0][0]),
                                           str(flex_scores[bin_number][0][1])))
    # OUTPUT FILE I: flexibility remaining entries
    for count in range(1, len(flex_scores[bin_number])):
        summary_spectrum.write(flexline.format(" ",
                                               str(count+1),
                                               str(flex_scores[bin_number][count][0]),
                                               str(flex_scores[bin_number][count][1])))
#########################################################################################

# CLASSIFICATION OF TORSION ANGLES
# choose only flexible torsion for classification if specified via [-f]
if args.Flexible:
    dict_bin_def = OrderedDict()
    for torsion in args.Flexible:
        dict_bin_def[torsion] = dict_all_bin_def[torsion]
else:
    dict_bin_def = dict_all_bin_def

classification = cc.DoClassification(dict_bin_def, target_torsions, dict_midpoints, number_torsions)

# silhouette_coefficient for classification results, take always all labeled torsionis even if -f is given,
# then only classification.class_labels is different
silhouette = cc.silhouette_coefficient(range(1, len(all_torsion_angles['a']) + 1),
                                                         all_torsion_angles,
                                                         dict_all_bin_def,
                                                         classification.class_labels,
                                                         args.AngNum)
# Diverse Subset perturbational approach
diverse_subset = cc.DiverseSubset(zip(range(1,len(classification.class_centroids)+1),
                                      classification.classifiers,
                                      classification.class_centroids),
                                  args.SubRef,
                                  args.SubOrder,
                                  args.SubSize,
                                  all_torsion_angles)
# STDOUT classification
print "first", args.SubSize, "class centroids:", classification.class_centroids[:args.SubSize]
print args.SubSize, "most diverse centroids:", diverse_subset.diverse_centroids
for info in diverse_subset.diverse_info:
    print info
# OUTPUT FILE III: SUMMARY_CLASSIFICATION.LOG
if args.Flexible:
    summary_classification = open('summary_classification_gk%d_t%d_%s.log'
                                  % (args.GaussKern, args.Threshold, ''.join(args.Flexible)), 'w')
else:
    summary_classification = open('summary_classification_gk%d_t%d.log' % (args.GaussKern, args.Threshold), 'w')
classifier_length = (len(str(classification.classifiers[0]))+14)

#########################################################################################

# CLUSTERING OF CLASSIFICATION CENTROIDS (if -c is specified, the centroids of the 
# previous classification results are used for hierarchical clusteringi and the 
# respective info is added to the same output file III: summary_classification.log)
if args.Clustering:
    clustering = cc.DoClustering(dict_bin_def, target_torsions, dict_midpoints, args.SubSize, number_torsions)
    print args.SubSize, "clustered class centroids:", clustering.cluster_centroids
    #################### content OUTPUT FILE III ########################################
    classline = '{:7} {:7} {:{Flength}} {:7} {:7} {:7} {:11}' + '\n'
    summary_classification.write(classline.format('class', 'size', 'classifier', 'centr', 'frac[%]', 'cluster',
                                              'clust-centr', Flength=classifier_length))
else: 
    classline = '{:7} {:7} {:{Flength}} {:7} {:7} ' + '\n'
    summary_classification.write(classline.format('class', 'size', 'classifier', 'centr', 'frac[%]',
                                              Flength=classifier_length))
num = 1
for ID in classification.classifiers:
    if args.Clustering:

	# control variable distinguishing cluster-member vs. cluster-centroid
	is_centroid = False # class-centroid is only cluster-member, not its centroid

	# set control variable True if class-centroid matches a cluster-centroid and save that clusID
        for clusID in clustering.clusters:
	    if classification.classes[str(ID)][2] == clustering.clusters[clusID][2]:
		clusnum = clusID 
		is_centroid = True
		break

	# class-centroid is also the cluster-centroid
	if is_centroid: 
	    summary_classification.write(
                    classline.format(str(num),
                    str(classification.classes[str(ID)][1]),
                    str(ID.tolist()),
                    str(classification.classes[str(ID)][2]),
                    str(round(classification.classes[str(ID)][1] / float(len(target_torsions[first_letter]))*100, 2)),
                    str(clusnum),
		    str("c"), # mark this class-centroid as the centroid of its cluster
			Flength=classifier_length))

	# class-centroid is not the cluster-centroid
	else: 
	    # identify the cluster which the current class-centroid is a member of and save that clusID
	    for clusID in clustering.clusters:
		if classification.classes[str(ID)][2] in clustering.clusters[clusID][0]:
		    clusnum = clusID
	    summary_classification.write(
                    classline.format(str(num),
                    str(classification.classes[str(ID)][1]),
                    str(ID.tolist()),
                    str(classification.classes[str(ID)][2]),
                    str(round(classification.classes[str(ID)][1] / float(len(target_torsions[first_letter]))*100, 2)),
                    str(clusnum),
		    str(" "), # not the centroid of the cluster
			Flength=classifier_length))
    else:
	summary_classification.write(
                classline.format(str(num),
                    str(classification.classes[str(ID)][1]),
                    str(ID.tolist()),
                    str(classification.classes[str(ID)][2]),
                    str(round(classification.classes[str(ID)][1] / float(len(target_torsions[first_letter]))*100, 2)),
                        Flength=classifier_length))
    if args.Member:
	summary_classification.write('memberframes: ' + str(classification.classes[str(ID)][0]) + '\n')
    num += 1
    
summary_classification.close()

#########################################################################################

# CLASSIFICATION done, the rest is additional output-file writing (heatmaps, gnuplot)
# OUTPUT FILE IV: distance matrix for gnuplot
# prepare centroids and file names (+headers) according to if args.Xray and / or args.Flexible are specified
if args.Xray:
    # xray + first SubSize-class centroids
    class_list_cent = [len(all_torsion_angles_xray[all_torsion_angles_xray.keys()[0]])] + \
                            classification.class_centroids[:args.SubSize]
    class_list = cc.PairDistMatrix_Angle(class_list_cent, all_torsion_angles_xray)
    # xray + SubSize-cluster centroids
    if args.Clustering:
        clus_list_cent = [len(all_torsion_angles_xray[all_torsion_angles_xray.keys()[0]])] + \
			    clustering.cluster_centroids[:args.SubSize]
        clus_list = cc.PairDistMatrix_Angle(clus_list_cent, all_torsion_angles_xray)
    # xray + SubSize most diverse centroids
    div_list_cent = [len(all_torsion_angles_xray[all_torsion_angles_xray.keys()[0]])] + \
                            diverse_subset.diverse_centroids
    div_list = cc.PairDistMatrix_Angle(div_list_cent, all_torsion_angles_xray)
else:
    # first SubSize-class centroids
    class_list_cent = classification.class_centroids[:args.SubSize]
    class_list = cc.PairDistMatrix_Angle(class_list_cent, all_torsion_angles)
    # SubSize clustered classification centroids
    if args.Clustering:
        clus_list_cent = clustering.cluster_centroids[:args.SubSize]
        clus_list = cc.PairDistMatrix_Angle(clus_list_cent, all_torsion_angles)
    # SubSize most diverse centroids
    div_list_cent = diverse_subset.diverse_centroids
    div_list = cc.PairDistMatrix_Angle(div_list_cent, all_torsion_angles)
# naming heatmap and RMS2TA out files
if args.Flexible:
    if args.Clustering:
        in_name = 'RMS2TA_%s_%dclass_%dclus_%ddiverse.gnu' % (''.join(args.Flexible), 
							      args.SubSize, args.SubSize, args.SubSize)
        in_heat = 'heatmap_%s_%dclass_%dclus_%ddiverse.in' % (''.join(args.Flexible), 
							      args.SubSize, args.SubSize, args.SubSize)
    else:
        in_name = 'RMS2TA_%s_%dclass_%ddiverse.gnu' % (''.join(args.Flexible), args.SubSize, args.SubSize)
        in_heat = 'heatmap_%s_%dclass_%ddiverse.in' % (''.join(args.Flexible), args.SubSize, args.SubSize)
else:
    if args.Clustering:
        in_name = 'RMS2TA_%dclass_%dclus_%ddiverse.gnu' % (args.SubSize, args.SubSize, args.SubSize)
	in_heat = 'heatmap_%dclass_%dclus_%ddiverse.in' % (args.SubSize, args.SubSize, args.SubSize)
    else:
        in_name = 'RMS2TA_%dclass_%ddiverse.gnu' % (args.SubSize, args.SubSize)
        in_heat = 'heatmap_%dclass_%ddiverse.in' % (args.SubSize, args.SubSize)
gnu_out = open(in_name, 'w')
heatmap = open(in_heat, 'w')
# header
gnu_out.write("# class-centroid list:" + str(class_list_cent) + '\n')
if args.Clustering:
    gnu_out.write("# cluster-centroid list:" + str(clus_list_cent) + '\n')
gnu_out.write("# div-centroid list:" + str(div_list_cent) + '\n')
gnu_out.write("# column1: x-centr, column2: y-centr, column3&4(&5): rms-dist x-centr. to y-centr. \n")
gnu_out.write("# column3: highest populated class centroids \n")
if args.Clustering:
    gnu_out.write("# column4: clustering centroids \n"
                  "# column5: most diverse class centroids \n")
else:
    gnu_out.write("# column4: most diverse class centroids \n")
# content of output file:
for line in range(len(class_list)):
    for column in range(len(class_list[line])):
	if args.Clustering:
            gnu_out.write(str(line + 1) + '\t' + str(column + 1) + '\t' + str(class_list[line][column]) + '\t' +
                                                                          str(clus_list[line][column]) + '\t' +
								          str(div_list[line][column]) + '\n')
        else:
            gnu_out.write(str(line + 1) + '\t' + str(column + 1) + '\t' + str(class_list[line][column]) + '\t' +
								          str(div_list[line][column]) + '\n')
    gnu_out.write('\n')

# OUTPUT FILE V: gnuplot input, heatmap
heatmap.write(
    "#set term png size 1280,1280 crop font calibri 32 enhanced \n" +
    "#set output 'RMS2A.png' \n" +
    "set size square \n" +
    "set multiplot \n" +
    "set pm3d map interpolate 0,0 \n" +
    "unset key \n" +
    "\n" +
    "set xrange[1:" + str(len(class_list[0])) + "] \n" +
    "set yrange[1:" + str(len(class_list[0])) + "] \n" +
    "#unset xtics \n" +
    "#unset ytics \n" +
    "set border linewidth 1.5 \n")
if args.Xray:
    heatmap.write(  # black lines separating the reference in heatmap for clarity + x/y-tics
        "set arrow from first 1.5,1 to first 1.5," + str(len(class_list[0])) +
        " nohead front linecolor 'black' linewidth 1.5 \n" +
        "set arrow from first 1,1.5 to first " + str(len(class_list[0])) +
        ",1.5 nohead front linecolor 'black' linewidth 1.5 \n" +
        "set xtics('x' 1, '1' 2, '2' 3, '3' 4, '4' 5, '5' 6, '6' 7, '7' 8, '8' 9, '9' 10, '10' 11) \n" +
        "set ytics('x' 1, '1' 2, '2' 3, '3' 4, '4' 5, '5' 6, '6' 7, '7' 8, '8' 9, '9' 10, '10' 11) \n")
heatmap.write(
    "\n" +
    "set cbrange [0:180] \n" +
    "set palette defined (0 'blue', 90 'white', 180 'red') \n" +
    "set colorbox \n" +
    "set tics font 'arial,8,bold' scale 0.5 \n" +
    "set cbtics out nomirror \n" +
    "#set clabel 'pRMS_{TA} [deg]' offset -3.9,12 rotate by 0 \n" +
    "\n" +
    "# 'using 1:2:3' plots highest populated class centroids \n"
    "# if clustering was performed: \n"
    "# 'using 1:2:4' plots clustering centroids, 'using 1:2:5' plots diverse centroids \n"
    "# else: 'using 1:2:4' plots diverse centroids \n"
    "splot '" + os.getcwd() + "/" + in_name + "' using 1:2:3 \n" +
    "unset multiplot \n" +
    "quit \n")

gnu_out.close()
heatmap.close()
#########################################################################################
