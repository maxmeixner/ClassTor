#!/usr/bin/env python

import numpy as np
from scipy.signal import argrelextrema

#########################################################################################


# generate the distribution spectrum / histogram of a single torsion
def prep_gauss(angle_values, kernel_width):
    # count occurrences
    occurrences = []
    for integer in range(-180, 181):
        occurrences.append(angle_values.count(integer))

    # data set
    x_values = np.arange(0, 361, 1)  # spectrum mirrored with np.arange(-180,181,1)
    y_values = np.array(occurrences)

    # convert full width at half maximum (kernel_width)
    sigma = kernel_width / np.sqrt(8 * np.log(2))

    # smooth
    smoothed_values = np.zeros(y_values.shape)
    for x_position in x_values:
        kernel = np.exp(-(x_values - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_values[x_position] = sum(y_values * kernel)

    # distribution spectrum
    return zip(range(-180, 181), smoothed_values)

#########################################################################################


# analyze the distribution spectrum of a single torsion
class SpectrumAnalysis(object):

    def __init__(self, torsion_angle, data_points, extrema_order):
        # inputs
        self.torsion = torsion_angle
        self.data = data_points
        self.smoothed_values = [element[1] for element in data_points]
        self.extrema_order = extrema_order
        self.info = []

        # directly detectable from raw input, do not change order
        self.minima = self.__FindMinima(self.smoothed_values, self.extrema_order)
        self.maxima = self.__FindMaxima(self.smoothed_values, self.extrema_order)
        self.status = self.StatusInfo()
        self.bin_def = self.DefineBins(self.minima, self.status)
        self.bin_list = [bin_label for bin_label in self.bin_def]
        self.bin_number = len(self.bin_list)

        # necessary adjustments based on previously detected status, do not change order
        self.fitted_maxima = self.__FitMax(self.bin_def, self.maxima, self.status)
        self.counted_maxima = self.__CountFittedMaxima(self.fitted_maxima, self.bin_list)
        self.midpoints = self.__AdjustMidpoints()

        # flexibility info
        self.std_dev_midpoints = np.std(self.__ShiftMids([mid[1] for mid in self.midpoints]))
        self.std_dev_midpoint_heights = np.std(np.array([self.smoothed_values[mid[1] + 180] for mid in self.midpoints]))

        # outputs
        self.results = self.ResultsFunc()


    def ResultsFunc(self):

        return [self.torsion, self.status, self.bin_number, self.bin_list, self.bin_def, self.midpoints, self.info]

    # find minima
    @staticmethod
    def __FindMinima(smoothed_values, extrema_order):

        minima_unflat = np.array(argrelextrema(np.array(smoothed_values), np.less, order=extrema_order))
        # shift to range [-180, 180], flatten list of lists
        minima = [j - 180 for minima_unflat[0] in minima_unflat for j in minima_unflat[0]]

        # important here: borders (-180,180) are considered minima, independent of 'status' of the spectrum
        minima.insert(0, -180)
        minima.append(180)

        return minima

    # find maxima
    @staticmethod
    def __FindMaxima(smoothed_values, extrema_order):

        maxima_unflat = np.array(argrelextrema(np.array(smoothed_values), np.greater, order=extrema_order))
        maxima = [j - 180 for maxima_unflat[0] in maxima_unflat for j in maxima_unflat[0]]

        return maxima

    # create bin definition
    def DefineBins(self, minima, status):
        # bin_def = {bin: [bin_border1, bin_border2]}
        # = {0: [-180, min1], 1: [min1, min2], ..., n: [minX, 180]
        bin_def = {}
        for bin_label in range(len(minima) - 1):
            bin_def[bin_label] = []
            bin_def[bin_label].append(minima[bin_label])
            bin_def[bin_label].append(minima[bin_label + 1])

        # merge first '[0]' and last '[bin_def.keys()[-1]]' bin for non-clear spectra
        # delete 'last' bin bin_def
        if status[1] != 'clear':
            bin_def[0] = [bin_def[0], bin_def[bin_def.keys()[-1]]]
            del bin_def[bin_def.keys()[-1]]
            self.info.append(str("Bin 0 spreads across spectrum borders"))

        return bin_def

    # get status info
    def StatusInfo(self):
        
        # perform break-check
        breakcheck = self.__checkforbreaks(self.smoothed_values, self.maxima, self.extrema_order)

        if breakcheck == True:
            status = ('open', 'break')
        elif breakcheck == None:
            status = ('open', 'shift')
        elif breakcheck == "Limit":
            status = ('closed', 'limit')
        else:
            status = ('closed', 'clear')

        return status

    # return the maximum with higher occurrence out of two of the same bin
    def __KeepOneMax(self, first_maximum, second_maximum):
        # get the occurance at both maxima (occurance = y-value of spectrum)
        first_maximum_yvalue = [target[1] for target in self.data if target[0] == first_maximum]
        second_maximum_yvalue = [target[1] for target in self.data if target[0] == second_maximum]

        if first_maximum_yvalue > second_maximum_yvalue:
            return first_maximum
        elif second_maximum_yvalue > first_maximum_yvalue:
            return second_maximum
        # if both maxima happen to have the same occurrence, take the middle value between both maxima
        else:
            return (first_maximum + second_maximum)/2

    # return the (corrected) maximum of each bin
    # correction is only applied when:
    # exactly one bin shows either 2 maxima or no maximum (can sometimes happen)
    # IMPORTANT for 'limit' and 'break': maxima of bin 0 AND self.counted_maxima have to be corrected
    # BEFORE entering this function (see first criteria here = self.counted_maxima, this is done in __AdjustMidpoints())
    def __CorrectMax(self, given_maxima_list):

        # each bin shows 1 maximum, nothing to adjust
        if sum(self.counted_maxima) == self.bin_number:

            return given_maxima_list

        # exactly one bin shows 2 maxima
        elif sum(self.counted_maxima) == self.bin_number + 1:
            # identify target bin
            target_bin = self.counted_maxima.index(2)
            # bins that were fine (showing only 1 maximum)
            remaining_maxima = [element for element in given_maxima_list if element[0] != target_bin]
            # bin that shows two maxima
            target_maxima = [element for element in given_maxima_list if element[0] == target_bin]
            # keep maximum with higher occurrence
            final_maximum = self.__KeepOneMax(target_maxima[0][1], target_maxima[1][1])
            # add to those that were fine
            remaining_maxima.append((target_bin, final_maximum))
            # add remark to info
            self.info.append(str("manually deleted a lower populated maximum for bin " + str(target_bin)
                                 + " of torsion " + str(self.torsion) + ", only kept maximum at " + str(final_maximum)))

            return sorted(remaining_maxima, key=lambda corr: corr[0])

        # exactly one bin lacks its maximum
        elif sum(self.counted_maxima) == self.bin_number - 1:
            # identify target bin
            target_bin = self.counted_maxima.index(0)
            # calculate new maximum ( = middle between corresponding bin borders)
            manual_max = (self.bin_def[target_bin][1] + self.bin_def[target_bin][0]) / 2
            # add to those that were fine
            given_maxima_list.append((target_bin, manual_max))
            # add remark to info
            self.info.append(str("manually added a maximum at " + str(manual_max) + " for bin " + str(target_bin) +
                                 " of torsion " + str(self.torsion)))

            return sorted(given_maxima_list, key=lambda corr: corr[0])

        else:
            raise SystemExit("Midpoint-Assignment Error! \
            Attempt of correcting local maxima failed for torsion {}, \
            more than one bin either lacks a maximum or shows more than one maximum or a combination of both, \
            observe the distribution spectrum and try changing its shape via -gk and/or \
            change the range of data points taken for examination \
            of local minima and maxima via -t".format(self.torsion))

    # examine which maximum fits which bin [(bin, maximum), (0, -155), (1, -4), ...]
    @staticmethod
    def __FitMax(bin_def, maxima, status):
        fitted_maxima = []
        for bin_label in bin_def:
            for maximum in maxima:
                # range of bin 0 is split across spectrum borders for non-clear spectra
                if status[1] != 'clear':
                    if bin_label == 0:
                        if bin_def[bin_label][0][0] < maximum < bin_def[bin_label][0][1] or \
                                bin_def[bin_label][1][0] < maximum < bin_def[bin_label][1][1]:
                            fitted_maxima.append((bin_label, maximum))
                    else:
                        if bin_def[bin_label][0] < maximum < bin_def[bin_label][1]:
                            fitted_maxima.append((bin_label, maximum))
                else:
                    if bin_def[bin_label][0] < maximum < bin_def[bin_label][1]:
                        fitted_maxima.append((bin_label, maximum))

        return fitted_maxima

    # count bins in fitted_maxima to determine if there are bins with more or less than 1 maximum
    @staticmethod
    def __CountFittedMaxima(fitted_maxima, bin_list):
        # get the bin labels for bins that were assigned maxima
        fitted_maxima_labels = [label[0] for label in fitted_maxima]
        # count which bin has how many maxima, e.g. [2, 1, 1]
        # meaning bin0 has 2 maxima, bin1 and bin2 1 maximum each (index = bin, integer = maxima-count)
        counted_maxima = []
        for bin_label in bin_list:
            counted_maxima.append(fitted_maxima_labels.count(bin_label))

        return counted_maxima

    # define midpoints for each bin (= maximum if previously detected, handled in self.__CorrectMax())
    def __AdjustMidpoints(self):
        # CLEAR, SHIFT
        if self.status[1] == 'clear' or self.status[1] == 'shift':

            return self.__CorrectMax(self.fitted_maxima)

        # LIMIT
        elif self.status[1] == 'limit':
            # correct maxima of bin 0 in any case,
            # same procedure as in self.__CorrectMax() for specified bin 0 instead of target_bin
            remaining_maxima = [element for element in self.fitted_maxima if element[0] != 0]
            target_maxima = [element for element in self.fitted_maxima if element[0] == 0]
            final_maximum = self.__KeepOneMax(target_maxima[0][1], target_maxima[1][1])
            remaining_maxima.insert(0, (0, final_maximum))
            # set maximum count for bin 0 to 1 since it has been corrected
            self.counted_maxima[0] = 1

            return self.__CorrectMax(remaining_maxima)

        # BREAK
        elif self.status[1] == 'break':
            # add 180 as maximum of bin 0 in any case to self.fitted_maxima as it was not detected as such
            self.fitted_maxima.insert(0, (0, 180))
            # and change corresponding counted maximum value:
            self.counted_maxima[0] = 1

            return self.__CorrectMax(self.fitted_maxima)

    # shift given midpoints relative to the first one so that minimum distances between midpoints are assured
    @staticmethod
    def __ShiftMids(list_of_midpoints):
        # take first one as reference
        shift_midpoints = [list_of_midpoints[0]]
        # check if shifting all others brings them closer to reference, old: target scan
        for mid in list_of_midpoints[1:]:
            base = abs(mid - shift_midpoints[0])
            shift_up = abs((mid + 360) - shift_midpoints[0])
            shift_down = abs((mid - 360) - shift_midpoints[0])
            if min([base, shift_up, shift_down]) == base:
                shift_midpoints.append(mid)
            elif min([base, shift_up, shift_down]) == shift_up:
                shift_midpoints.append(mid + 360)
            elif min([base, shift_up, shift_down]) == shift_down:
                shift_midpoints.append(mid - 360)

        return np.array(shift_midpoints)

    ######
    # BREAK-CHECK

    # check beginning of spectrum course for min/max
    @staticmethod
    def __checkbeg(smoothed_values):
        stop = len(smoothed_values) / 2  # middle of spectrum
        shortlist = smoothed_values[:stop]  # first half of spectrum
        beginning = shortlist[0]  # first value of spectrum
        index = 1
        while shortlist[index] == beginning:  # go through all items as long as their value == first value
            index += 1
        else:  # found the first different value
            if shortlist[index] > beginning:
                return "minimum"  # course at border shows minimum behavior
            elif shortlist[index] < beginning:
                return "maximum"  # course at border shows maximum behavior

    # check end of spectrum for min/max
    def __checkend(self, smoothed_values):
        rev_list = smoothed_values[::-1]
        return self.__checkbeg(rev_list)

    # break-check: inspection of spectrum borders defining status
    # CLOSED = minima behavior at spectrum borders or no values at borders (clear or limit)
    # OPEN = maxima behavior at spectrum borders (break) or one minimum- and one maximum-like border (shift)
    def __checkforbreaks(self, smoothed_values, maxima, extrema_order):

        last = int(len(smoothed_values) - 1)

        # no values at borders: CLOSED (clear)
        if smoothed_values[0] == 0 and smoothed_values[last] == 0:
            return False

        # course at borders tend to 'minima': CLOSED (clear or limit)
        elif smoothed_values[0] < smoothed_values[1] and smoothed_values[last] < smoothed_values[last - 1]:
            if maxima[0] <= (-180 + extrema_order) and maxima[-1] >= (180 - extrema_order):
                return "Limit"
            else:
                return False

        # course at borders tend to 'maxima': OPEN (break)
        elif smoothed_values[0] > smoothed_values[1] and smoothed_values[last] > smoothed_values[last - 1]:
            return True

        # course at one border tends to 'minimum' and 'maximum' on the opposite border: OPEN (shift)
        elif smoothed_values[0] > smoothed_values[1] and smoothed_values[last] < smoothed_values[last - 1] or \
             smoothed_values[0] < smoothed_values[1] and smoothed_values[last] > smoothed_values[last - 1]:
            return None

        # if comparing the first and last two values is not clear (when they are the same value), check more values
        else:
            # extended course at borders tend to 'minima': CLOSED (clear or limit)
            if self.__checkbeg(smoothed_values) == "minimum" and self.__checkend(smoothed_values) == "minimum":
                if maxima[0] <= (-180 + extrema_order) and maxima[-1] >= (180 - extrema_order):
                    return "Limit"
                else:
                    return False
                
            # extended course at borders tend to 'maxima': OPEN (break)
            elif self.__checkbeg(smoothed_values) == "maximum" and self.__checkend(smoothed_values) == "maximum":
                return True

            # extended course at one border tends to 'minimum' and 'maximum' at opposite border: OPEN (shift)
            elif self.__checkbeg(smoothed_values) == "minimum" and self.__checkend(smoothed_values) == "maximum" or \
                  self.__checkbeg(smoothed_values) == "maximum" and self.__checkend(smoothed_values) == "minimum":
                return None
#########################################################################################

from collections import OrderedDict


# calculate flexibility score and rank torsions (with equal number of bins) accordingly
def flex_score(flex_info_list):
    flex_score_dict = OrderedDict()
    # start with highest bin number
    bin_count = max([element[1] for element in flex_info_list])
    while bin_count > 1:
        # extract all torsions with 'bin_count' amount of bins
        equal_bin_torsions = [element for element in flex_info_list if element[1] == bin_count]
        # sort them according to range covered, as well as to population
        sorted_range = sorted(equal_bin_torsions, key=lambda x: x[2], reverse=True)
        sorted_population = sorted(equal_bin_torsions, key=lambda x: x[3])

        # perform ranking, assign the right score values
        ranking = []
        for count in range(len(equal_bin_torsions)):
            # get range score
            range_score = range(len(equal_bin_torsions), 0, -1)[count]
            for count2 in range(len(sorted_population)):
                # match torsion label of second sorted list
                if sorted_population[count2][0] == sorted_range[count][0]:
                    # get population score
                    pop_score = range(len(equal_bin_torsions), 0, -1)[count2]
                    pop = sorted_population[count2][3]
            # combine both scores
            # (add a fraction of actual std.dev. value of midpoint heights (pop) to avoid similar scores)
            score = round(range_score * (pop_score + (1 / float(1 + pop))), 4)
            ranking.append((sorted_range[count][0], score))
        # add to dictionary
        flex_score_dict[bin_count] = sorted(ranking, key=lambda x: x[1], reverse=True)

        bin_count -= 1

    return flex_score_dict
