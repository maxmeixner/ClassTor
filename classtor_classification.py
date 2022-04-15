#!/usr/bin/env python

import numpy as np
import collections
from sklearn.metrics import silhouette_score as SS

"""
#########################################################################################
                   ClassTor - Classification of Torsion angle values
			Copyright (C) 2022 Maximilian Meixner
#########################################################################################
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published 
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
#########################################################################################
"""

class DoClassification(object):

    def __init__(self, dict_bin_def, dict_all_torsions, dict_all_midpoints, torsions):
        # inputs
        self.n_torsions = torsions
        self.bin_defs = dict_bin_def
        self.data = dict_all_torsions
        self.midpoints = dict_all_midpoints

        # directly derivable from input
        self.frame_number = self.data[self.data.keys()[-1]].size
        self.identifiers = self.ang2bin()
        self.classifiers = []

        # generate one output dictionary (self.classes) and save info (memberframes [0], population [1], centroid [2])
        # key = classifier (classID)
        # sorted according to descending population
        self.classes = self.AllClassPop({})
        self.AllCentroids(self.classes)

        # array of class labels per sample sorted according to frame number
        self.class_labels = self.AllClassLabels(self.classes)
        # list of population of each class sorted according to descending population of classes
        self.class_populations = [self.classes[classID][1] for classID in self.classes]
        # list of centroids sorted according to population
        self.class_centroids = [self.classes[classID][2] for classID in self.classes]

        # self.distance_to_centroid = {}
        # self.distance_within_cluster = {}

    # convert combination of torsion angles to identifier per frame
    def ang2bin(self):
        identifiers = []
        # give each frame an identifier = combination of bin labels instead of torsion angle values
        for frame in range(self.frame_number):
            identifiers.append([])
            for letter in self.data:
                last_bin_label = self.bin_defs[letter].keys()[-1]
                for bin_label in self.bin_defs[letter]:
                    # if bin 0 spreads across borders: when key = 0 and feature is a list of lists, check both entries
                    if bin_label == 0 and any(isinstance(el, list) for el in self.bin_defs[letter][bin_label]):
                        if self.bin_defs[letter][bin_label][0][0] <= self.data[letter][frame] \
                                < self.bin_defs[letter][bin_label][0][1]:
                            identifiers[frame].append(bin_label)
                            break
                        # include upper bin border for last bin (<=)
                        elif self.bin_defs[letter][bin_label][1][0] <= self.data[letter][frame] \
                                <= self.bin_defs[letter][bin_label][1][1]:
                            identifiers[frame].append(bin_label)
                            break
                    else:
                        # each bin defined by one range only
                        if bin_label == last_bin_label:
                            # include upper bin border for last bin
                            if self.bin_defs[letter][bin_label][0] <= self.data[letter][frame] \
                                    <= self.bin_defs[letter][bin_label][1]:
                                identifiers[frame].append(bin_label)
                                break
                        elif self.bin_defs[letter][bin_label][0] <= self.data[letter][frame] \
                                    < self.bin_defs[letter][bin_label][1]:
                            identifiers[frame].append(bin_label)
                            break
        return np.array(identifiers)

    # determine the total number of classes as the number of unique identifiers and the corresponding class members
    # starting from 'frame 1' = index + 1
    def AllClassPop(self, class_dict):
        unsorted_classifiers = []
        for ID in self.identifiers:
            if str(ID) not in class_dict:
                unsorted_classifiers.append(ID)
                # list of lists (first element, list of frames)
                class_dict[str(ID)] = [[index + 1 for index in range(len(self.identifiers)) if
                                    self.identifiers[index].tolist() == ID.tolist()]]
        # add population to each class
        for classID in class_dict:
            class_dict[classID].append(len(class_dict[classID][0]))
        # save unordered populations separately
        populations = [class_dict[classID][1] for classID in class_dict]
        # order according to descending population
        tmp = sorted(zip(populations, class_dict.keys()), key=lambda x: x[0], reverse=True)
        # save ordered classifiers
        for element in tmp:
            for classifier in unsorted_classifiers:
                if element[1] == str(classifier):
                    self.classifiers.append(classifier)

        sorted_class_dict = sorted(class_dict.items(), key=lambda x: x[1][1], reverse=True)
        return collections.OrderedDict(sorted_class_dict)

    # determine the centroid = representative
    # (frame with the shortest average-distance to all bin midpoints of respective classifier) of each class
    def AllCentroids(self, class_dict):
        for classID in self.classifiers:
            # get torsion angle values of each class member and convert to unit circle-coordinates
            target_angles = get_target_angles(class_dict[str(classID)][0], self.data, self.bin_defs)

            """Letter-Label-Pairs: get letters attached to corresponding bin_label of classID
            (will be needed to match the correct midpoint, since both letters (dict keys) and labels (dict values)
            have to be referenced in self.midpoints)
            e.g. torsions = a b c, classID = [0 0 2] -> LLP = [(a, 0), (b, 0), (c, 2)]"""
            LLPs = zip([letter for letter in self.bin_defs], classID.tolist())

            """get torsion angle values of midpoints, e.g.
            first LLP[0] = letter a
            first LLP[1] = bin label 0 = matches list index in (bin label, midpoints-value)-pairs in midpoints
            self.midpoints[LLP[0]][LLP[1]][1] = midpoint angle value"""
            target_midpoints = np.array([deg2rad(self.midpoints[LLP[0]][LLP[1]][1]) for LLP in LLPs])

            # calculate rms-distances between all frames of class and corresponding midpoints
            distance_to_midpoints = [rms(np.linalg.norm((target - target_midpoints), axis=1))
                                     for target in target_angles]
            # select the centroid = representative structure
            smallest = min(distance_to_midpoints)
            centroid_index = distance_to_midpoints.index(smallest)
            centroid_frame = class_dict[str(classID)][0][centroid_index]
            class_dict[str(classID)].append(centroid_frame)

        return class_dict

    # create array of cluster integer per frame, same as scikit_clustering.labels_ in DoClustering
    def AllClassLabels(self, class_dict):
        class_frames_labels = []
        # list of tuples [([0] = classIDs, [1] = class integer value), ... ]
        classIDs_ints = zip(class_dict.keys(), range(len(class_dict.keys())))
        # pair every frame with the corresponding class integer value of the class they belong to
        for tuple in classIDs_ints:
            for frame in class_dict[tuple[0]][0]:
                class_frames_labels.append((frame, tuple[1]))
        # sort according to frame ID and only keep the class integer value
        sorted_class_frames_labels = sorted(class_frames_labels, key=lambda x: x[0])
        class_labels = [tuple[1] for tuple in sorted_class_frames_labels]

        return np.array(class_labels)



# child class for subsequent hierarchical clustering
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy as SCH # linkage, dendrogram, cut_tree


class DoClustering(DoClassification):

    def __init__(self, dict_bin_def, dict_all_torsions, dict_all_midpoints, subset_size, torsions):
        # inherit variables and methods from parent class, faster way to do this?
        super(DoClustering, self).__init__(dict_bin_def=dict_bin_def, dict_all_torsions=dict_all_torsions,
                                           dict_all_midpoints=dict_all_midpoints, torsions=torsions)
        # inputs
        self.n_clusters = subset_size

        # derivable variables
        self.target_angles = get_target_angles(self.class_centroids, self.data, self.bin_defs)
        """format target_angles for n samples and 3 flexible torsions a b c (array of array of array):
                                [[[x1a  y1a]
                                  [x1b  y1b]        sample 1
                                  [x1c  y1c]]
                                 [[x2a  y2a]
                                  [x1b  y2b]        sample 2
                                  [x2c  y2c]]
                                      ...    ]"""

        self.cluster_feature_matrix = target2features(self.target_angles, self.n_torsions)
        """format feature_matrix of corresponding target_angles (list of lists, n_samples n_features 
        = input format for sklearn.cluster):
        [[x1a, y1a, x1b, y1b, x1c, y1c], [x2a, y2a, x2b, y2b, x2c, y2c], ...]"""

        # output clustering results as one dictionary, similar to classes in DoClassification
        self.clusters = collections.OrderedDict()
        self.AggloClust(self.clusters)
        self.ClustCentroids(self.clusters)

        self.cluster_centroids = [self.clusters[clusterID][2] for clusterID in self.clusters]

    def AggloClust(self, cluster_dict):
        # cluster settings SCIPY 1.2.3
        scipy_clustering = SCH.linkage(self.cluster_feature_matrix, method='ward', metric='euclidean', optimal_ordering=True)
        # height where to cut the dendrogram in order to result in the desired amount of clusters
        cut_tree_at = (scipy_clustering[-self.n_clusters, 2] + scipy_clustering[-(self.n_clusters - 1), 2]) / 2
        # cluster ID of each sample: [0, 0, 2, 1, 4, 2, ...]
        clustIDs = np.concatenate(SCH.cut_tree(scipy_clustering, n_clusters=self.n_clusters))

        # extract members of each cluster and save corresponding frames in dictionary
        # sorted according to population, original cluster indices (SCH.cut_tree) might change
        populations_unsorted = []
        clusters_indices = []
        class_centroids = []
        for cluster in range(self.n_clusters):  # X clusters: 0 - (X-1)
            # array of indices belonging to current cluster
            clusters_indices.append(np.where(clustIDs == cluster)[0])
            # centroids from classification that belong to current cluster
            class_centroids.append([self.class_centroids[index] for index in clusters_indices[cluster]])
            # population of current cluster
            populations_unsorted.append(len(np.where(clustIDs == cluster)[0]))

        # sort cluster indices according to population, use this list of tuples to call sorted order of cluster indices
        cluster_population_sorted = sorted(zip(range(self.n_clusters), populations_unsorted),
                                           key=lambda x: x[1], reverse=True)

        # store cluster in dictionary with corresponding members (=class_centroid frames) and population
        # in order of decreasing population
        for index in range(len(cluster_population_sorted)):
            cluster_dict[index] = [class_centroids[cluster_population_sorted[index][0]],
                                   cluster_population_sorted[index][1]]

        return cluster_dict

    def ClustCentroids(self, cluster_dict):
        # get converted tors. angle values of each member of current cluster
        for cluster_index in cluster_dict:
            target_angles = get_target_angles(cluster_dict[cluster_index][0], self.data, self.bin_defs)

            # calculate pairwise rms-distance between all cluster members
            pairwise_distances = []
            for reference in target_angles:
                pairwise_distances.append(sum([rms(np.linalg.norm((target - reference), axis=1))
                                         for target in target_angles]))

            # determine cluster centroid as frame with minimum distance to all other cluster members
            smallest = min(pairwise_distances)
            clustcentroid_index = pairwise_distances.index(smallest)
            clustcentroid_frame = cluster_dict[cluster_index][0][clustcentroid_index]
            cluster_dict[cluster_index].append(clustcentroid_frame)

        return cluster_dict


import random


# create a diverse subset of structures using a perturbational approach
class DiverseSubset():

    def __init__(self, input_list, first_ref, order, subset_size, tors_angles):

        if len(input_list) < subset_size:
            raise SystemExit('The size of the desired subset (-ss) is larger than '
                             'the number of classes after classification, '
                             'decrease -ss or change classification settings')

        else:
            # diverse_info = STDOUT
            self.diverse_info = ["Perturbational approach:"]
            self.diverse_subset = self.PerturbationalSelection(input_list, first_ref, order, subset_size, tors_angles)
            self.diverse_centroids = [item[2] for item in self.diverse_subset]

    def PerturbationalSelection(self, input_list, first_ref, order, subset_size, tors_angles):
        # input_list = [(ID, identifier, centroid), ...]
        all_classifiers = [item[1].tolist() for item in input_list]

        # assign reference (first_ref = SubRef)
        # random reference
        if first_ref == -1:
            reference = random.choice(input_list)
        # average reference
        elif first_ref == 0:
            avg_classifier = []
            # add 1 to all bin labels (to avoid zeros)
            for item in range(len(all_classifiers)):
                for number in range(len(all_classifiers[item])):
                    all_classifiers[item][number] = all_classifiers[item][number] + 1
            # calculate average and subtract 1
            for number in range(len(all_classifiers[0])):
                avg_classifier.append(int(round(float((sum([element[number] for element in all_classifiers])/float(len(all_classifiers))-1)))))
            # subtract 1 to obtain original bin labels in all classifiers
            for item in range(len(all_classifiers)):
                for number in range(len(all_classifiers[item])):
                    all_classifiers[item][number] = all_classifiers[item][number] - 1
            reference = (0.0, np.array(avg_classifier), 0.0)
            # add to results if exists
            if reference[1].tolist() in all_classifiers:
                for item in range(len(input_list)):
                    if np.all(input_list[item][1] == reference[1]):
                        self.diverse_info.append("first reference classifier exists: " + str(input_list[item]))
                        break
            else:
                self.diverse_info.append("first reference classifier is virtual: " + str(reference[1].tolist()))
        # specific reference from list of classifiers
        else:
            reference = input_list[first_ref - 1]

        # assign order
        # random order
        if order == "random":
            random.shuffle(input_list)
        # reverse order
        elif order == "reverse":
            input_list.reverse()
        # topdown order = do not change order of input_list = proceed without modifications
        results = []
        # selection, start with highest number of perturbations and decrease until resulting subset >= subset_size
        n_perturbations = len(all_classifiers[0])  # all bins have to be different for start
        while len(results) < subset_size:
            # set results (back) to start
            if first_ref != 0:
                # reference = specific or random (real) one, then it should be included as the first result
                results = [reference]
            else:
                # if reference = average: only include if it exists among classifiers
                if reference[1].tolist() in all_classifiers:
                    for item in range(len(input_list)):
                        if np.all(input_list[item][1] == reference[1]):
                            results = [input_list[item]]
                else:
                    results = []
            # perform first round
            for item in input_list:
                results.append(check2elements(reference, item, n_perturbations))
            results = [real for real in results if real is not None]
            # perform consecutive rounds
            count = 1
            while count <= len(results):
                for item in results[count+1:]:
                    ref = results[count]
                    if check2elements(ref, item, n_perturbations) is None:
                        results.remove(item)
                    else:
                        pass
                count += 1
            if len(results) >= subset_size:
                pass
            else:
                n_perturbations -= 1
        # calc pairwise distances and order whole subset according to diversity
        div_centroids = [item[2] for item in results]
        div_distances = PairDistMatrix_Angle(div_centroids, tors_angles)
        div_distances_sum = [(div_distances[count], round(sum(div_distances[count]), 3), div_centroids[count])
                             for count in range(len(div_distances))]
        div_sorted = sorted(div_distances_sum, key=lambda x: x[1], reverse=True)
        div_sorted_centroids = [item[2] for item in div_sorted]
        # truncate to subset_size in case actual subset size > desired subset_size and
        # recalculate pairdist in sorted order
        final_centroids = div_sorted_centroids[:subset_size]
        final_pairdist = PairDistMatrix_Angle(final_centroids, tors_angles)
        final_outlist = [(final_pairdist[item], round(sum(final_pairdist[item])), final_centroids[item])
                         for item in range(len(final_centroids))]
        self.diverse_info.append(str(n_perturbations) + " perturbations created a subset of " + str(len(results)) +
                                  " structures, taking the " + str(subset_size) + " most diverse")

        return final_outlist


# convert from degrees to radian and replace by sin-cos pair = coordinates on the unit circle
def deg2rad(d):
    r = d * np.pi / 180
    return np.array([np.cos(r), np.sin(r)])


# calculate root-mean-square value, RMS:
def rms(x):
    return np.sqrt(x.dot(x) / x.size)


# return array of tors. angle values converted to pairs of coordinates on unit circle of desired frames
# given as input_list; data = torsion angle value dictionary, keys = torsion labels
def get_target_angles(input_list, data, bin_defs):
    return np.array([[deg2rad(data[letter][frame - 1])
                                            for letter in bin_defs] for frame in input_list])


# Convert the given target angles to the correct format of a feature matrix (necessary for silhouette_score)
def target2features(target_angles, torsions):
    # transform from array to feature matrix of the form [n_samples, n_features]
    feature_matrix = []
    for center in range(len(target_angles.tolist())):
        feature_matrix.append([])
        for torsion in range(torsions):
            for element in target_angles[center][torsion]:
                feature_matrix[center].append(element)

    return feature_matrix


# calculate the silhouette coefficient with scikit-learn 0.20
def silhouette_coefficient(list_of_frames, input_data, input_bin_defs, labels, input_torsions):
    # from list_of_frames get the corresponding target_angles
    # and transform them to a feature_matrix in coordinates on the unit circle
    feature_matrix = target2features(get_target_angles(list_of_frames, input_data, input_bin_defs), input_torsions)
    # calculate the silhouette score
    return SS(feature_matrix, labels)


# compare two identifiers for their perturbations
def check2elements(element1, element2, limit):
    # each element = (ID, classifier, centroid)
    perturbations = []
    for bin in range(len(element1[1])):
        if element1[1][bin] == element2[1][bin]:
            perturbations.append(False)
        else:
            perturbations.append(True)
    if perturbations.count(True) >= limit:
        return element2
    else:
        return


# calculate a pairwise distance matrix in torsion angle space
# therefore introduce 'target scan' here to assure smallest distance between two angle values
# = taking care of periodicity problem without translating to coordinates on unit circle
def PairDistMatrix_Angle(list_of_frames, dict_torsions):
    dist_matrix = []
    list_of_letters = dict_torsions.keys()
    target_angles = np.array([[dict_torsions[letter][frame - 1] for letter in list_of_letters]
                              for frame in list_of_frames])

    for reference in range(len(target_angles)):
        dist_matrix.append([])
        for target in range(len(target_angles)):
            distances = []
            # perform target_scan
            for torsion in range(len(target_angles[target])):
                target_scan = [abs((target_angles[target][torsion]-360) - target_angles[reference][torsion]),
                               abs(target_angles[target][torsion] - target_angles[reference][torsion]),
                               abs((target_angles[target][torsion]+360) - target_angles[reference][torsion])]
                shortest = min(target_scan)
                distances.append(shortest**2)
            dist_matrix[reference].append(round(np.sqrt(1/float(len(target_angles[target])) * sum(distances)), 2))

    return dist_matrix
