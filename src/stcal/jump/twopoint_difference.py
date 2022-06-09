import logging
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def find_crs(dataa, group_dq, read_noise, normal_rej_thresh,
             two_diff_rej_thresh, three_diff_rej_thresh, nframes,
             flag_4_neighbors, max_jump_to_flag_neighbors,
             min_jump_to_flag_neighbors, dqflags, copy_arrs=True):

    """
    Find CRs/Jumps in each integration within the input data array. The input
    data array is assumed to be in units of electrons, i.e. already multiplied
    by the gain. We also assume that the read noise is in units of electrons.
    We also assume that there are at least three groups in the integrations.
    This was checked by jump_step before this routine is called.

    Parameters
    ----------
    dataa: float, 4D array (num_ints, num_groups, num_rows,  num_cols)
        input ramp data

    group_dq : int, 4D array
        group DQ flags

    read_noise : float, 2D array
        The read noise of each pixel

    normal_rej_thresh : float
        cosmic ray sigma rejection threshold

    two_diff_rej_thresh : float
        cosmic ray sigma rejection threshold for ramps having 3 groups

    three_diff_rej_thresh : float
        cosmic ray sigma rejection threshold for ramps having 4 groups

    nframes : int
        The number of frames that are included in the group average

    flag_4_neighbors : bool
        if set to True (default is True), it will cause the four perpendicular
        neighbors of all detected jumps to also be flagged as a jump.

    max_jump_to_flag_neighbors : float
        value in units of sigma that sets the upper limit for flagging of
        neighbors. Any jump above this cutoff will not have its neighbors
        flagged.

    min_jump_to_flag_neighbors : float
        value in units of sigma that sets the lower limit for flagging of
        neighbors (marginal detections). Any primary jump below this value will
        not have its neighbors flagged.

    copy_arrs : bool
        Flag for making internal copies of the arrays so the input isn't modified,
        defaults to True.

    Returns
    -------
    gdq : int, 4D array
        group DQ array with reset flags

    row_below_gdq : int, 3D array (num_ints, num_groups, num_cols)
        pixels below current row also to be flagged as a CR

    row_above_gdq : int, 3D array (num_ints, num_groups, num_cols)
        pixels above current row also to be flagged as a CR

    """

    # copy data and group DQ array
    if copy_arrs:
        dataa = dataa.copy()
        gdq = group_dq.copy()
    else:
        gdq = group_dq

    # Get data characteristics
    nints, ngroups, nrows, ncols = dataa.shape
    ndiffs = ngroups - 1

    # get readnoise, squared
    read_noise_2 = read_noise**2
    print("New Code Fast Version")
    # create arrays for output
    row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    row_below_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)

    # get dq flags for saturated, donotuse, jump
    sat_flag = dqflags["SATURATED"]
    dnu_flag = dqflags["DO_NOT_USE"]
    jump_flag = dqflags["JUMP_DET"]

    for integ in range(nints):

        log.info(f'Working on integration {integ + 1}:')

        # get data, gdq for this integration
        dat = dataa[integ]
        gdq_integ = gdq[integ]

        # set 'saturated' or 'do not use' pixels to nan in data
        dat[np.where(np.bitwise_and(gdq_integ, sat_flag))] = np.nan
        dat[np.where(np.bitwise_and(gdq_integ, dnu_flag))] = np.nan

        # calculate the differences between adjacent groups (first diffs)
        # use mask on data, so the results will have sat/donotuse groups masked
        first_diffs = np.diff(dat, axis=0)

        # calc. the median of first_diffs for each pixel along the group axis
        first_diffs_masked = np.ma.masked_array(first_diffs, mask=np.isnan(first_diffs))
        largest_diffs = np.expand_dims(np.argmax(first_diffs_masked, axis=0), axis=0)
        np.put_along_axis(first_diffs_masked.mask, largest_diffs, True, axis=0)
        median_diffs = np.ma.median(first_diffs_masked, axis=0)

        # calculate sigma for each pixel
        sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / nframes)

        # reset sigma so pxels with 0 readnoise are not flagged as jumps
        sigma[np.where(sigma == 0.)] = np.nan

        # compute 'ratio' for each group. this is the value that will be
        # compared to 'threshold' to classify jumps. subtract the median of
        # first_diffs from first_diffs, take the abs. value and divide by sigma.
        ratio = np.abs(first_diffs - median_diffs[np.newaxis, :, :]) / sigma[np.newaxis, :, :]
        masked_ratio = np.ma.masked_greater(ratio, normal_rej_thresh)
        # repeat the determination of the median after the removal of the first pass has flagged jumps
        #masked_diffs = np.ma.masked_where(np.ma.getmask(masked_ratio), first_diffs)
        #masked_median_diffs = np.ma.median(masked_diffs, axis=0)
        #masked_sigma = np.sqrt(np.abs(masked_median_diffs) + read_noise_2 / nframes)
        #masked_ratio2 = np.abs(masked_diffs - masked_median_diffs[np.newaxis, :, :])/ \
        #    sigma[np.newaxis, :, :]
        #final_masked_ratio = np.ma.masked_greater(masked_ratio2, normal_rej_thresh)
        jump_mask = masked_ratio.mask

        jump_mask[np.bitwise_and(jump_mask, gdq[integ, 1:, :, :] == sat_flag)] = False
        jump_mask[np.bitwise_and(jump_mask, gdq[integ, 1:, :, :] == dnu_flag)] = False
        gdq[integ, 1:, :, :] = np.bitwise_or(gdq[integ, 1:, :, :], jump_mask * dqflags["JUMP_DET"])

        if flag_4_neighbors:  # iterate over each 'jump' pixel
            cr_group, cr_row, cr_col = np.where(np.bitwise_and(gdq[integ], jump_flag))

            for j in range(len(cr_group)):

                ratio_this_pix = ratio[cr_group[j] - 1, cr_row[j], cr_col[j]]

                # Jumps must be in a certain range to have neighbors flagged
                if ratio_this_pix < max_jump_to_flag_neighbors and \
                        ratio_this_pix > min_jump_to_flag_neighbors:
                    group = cr_group[j]
                    row = cr_row[j]
                    col = cr_col[j]

                    # This section saves flagged neighbors that are above or
                    # below the current range of row. If this method
                    # running in a single process, the row above and below are
                    # not used. If it is running in multiprocessing mode, then
                    # the rows above and below need to be returned to
                    # find_jumps to use when it reconstructs the full group dq
                    # array from the slices.

                    # Only flag adjacent pixels if they do not already have the
                    # 'SATURATION' or 'DONOTUSE' flag set
                    if row != 0:
                        if (gdq[integ, group, row - 1, col] & sat_flag) == 0:
                            if (gdq[integ, group, row - 1, col] & dnu_flag) == 0:
                                gdq[integ, group, row - 1, col] =\
                                    np.bitwise_or(gdq[integ, group, row - 1, col], jump_flag)
                    else:
                        row_below_gdq[integ, cr_group[j], cr_col[j]] = jump_flag

                    if row != nrows - 1:
                        if (gdq[integ, group, row + 1, col] & sat_flag) == 0:
                            if (gdq[integ, group, row + 1, col] & dnu_flag) == 0:
                                gdq[integ, group, row + 1, col] = \
                                    np.bitwise_or(gdq[integ, group, row + 1, col], jump_flag)
                    else:
                        row_above_gdq[integ, cr_group[j], cr_col[j]] = jump_flag

                    # Here we are just checking that we don't flag neighbors of
                    # jumps that are off the detector.
                    if cr_col[j] != 0:
                        if (gdq[integ, group, row, col - 1] & sat_flag) == 0:
                            if (gdq[integ, group, row, col - 1] & dnu_flag) == 0:
                                gdq[integ, group, row, col - 1] =\
                                    np.bitwise_or(gdq[integ, group, row, col - 1], jump_flag)

                    if cr_col[j] != ncols - 1:
                        if (gdq[integ, group, row, col + 1] & sat_flag) == 0:
                            if (gdq[integ, group, row, col + 1] & dnu_flag) == 0:
                                gdq[integ, group, row, col + 1] =\
                                    np.bitwise_or(gdq[integ, group, row, col + 1], jump_flag)
    return gdq, row_below_gdq, row_above_gdq


def calc_med_first_diffs(first_diffs):

    """ Calculate the median of `first diffs` along the group axis.

        If there 4+ usable groups (e.g not flagged as saturated, donotuse,
        or a previously clipped CR), then the group with largest absoulte
        first difference will be clipped and the median of the remianing groups
        will be returned. If there are exactly 3 usable groups, the median of
        those three groups will be returned without any clipping. Finally, if
        there are two usable groups, the group with the smallest absolute
        difference will be returned.

        Parameters
        -----------
        first_diffs : array, float
            array containing the first differences of adjacent groups
            for a single integration. Can be 3d or 1d (for a single pix)

        Returns
        -------
        median_diffs : float or array, float
            If the input is a single pixel, a float containing the median for
            the groups in that pixel will be returned. If the input is a 3d
            array of several pixels, a 2d array with the median for each pixel
            will be returned.
        """

    if first_diffs.ndim == 1:  # in the case where input is a single pixel

        num_usable_groups = len(first_diffs) - np.sum(np.isnan(first_diffs), axis=0)
        if num_usable_groups >= 4:  # if 4+, clip largest and return median
            mask = np.ones_like(first_diffs).astype(bool)
            mask[np.nanargmax(np.abs(first_diffs))] = False  # clip the diff with the largest abs value
            return np.nanmedian(first_diffs[mask])
        elif num_usable_groups == 3:  # if 3, no clipping just return median
            return(np.nanmedian(first_diffs))
        elif num_usable_groups == 2:  # if 2, return diff with minimum abs
            return(first_diffs[np.nanargmin(np.abs(first_diffs))])
        else:
            return(np.nan)

    # if input is multi-dimensional

    ngroups, nrows, ncols = first_diffs.shape
    num_usable_groups = ngroups - np.sum(np.isnan(first_diffs), axis=0)
    median_diffs = np.zeros((nrows, ncols))  # empty array to store median for each pix

    # process groups with >=4 usable groups
    row4, col4 = np.where(num_usable_groups >= 4)  # locations of >= 4 usable group pixels
    if len(row4) > 0:
        four_slice = first_diffs[:, row4, col4]
        four_slice[np.nanargmax(np.abs(four_slice), axis=0),
                   np.arange(four_slice.shape[1])] = np.nan  # mask largest group in slice
        median_diffs[row4, col4] = np.nanmedian(four_slice, axis=0)  # add median to return arr for these pix

    # process groups with 3 usable groups
    row3, col3 = np.where(num_usable_groups == 3)  # locations of >= 4 usable group pixels
    if len(row3) > 0:
        three_slice = first_diffs[:, row3, col3]
        median_diffs[row3, col3] = np.nanmedian(three_slice, axis=0)  # add median to return arr for these pix

    # process groups with 2 usable groups
    row2, col2 = np.where(num_usable_groups == 2)  # locations of >= 4 usable group pixels
    if len(row2) > 0:
        two_slice = first_diffs[:, row2, col2]
        two_slice[np.nanargmax(np.abs(two_slice), axis=0),
                  np.arange(two_slice.shape[1])] = np.nan  # mask larger abs. val
        median_diffs[row2, col2] = np.nanmin(two_slice, axis=0)  # add med. to return arr

    # set the medians all groups with less than 2 usable groups to nan to skip further
    # calculations for these pixels
    row_none, col_none = np.where(num_usable_groups < 2)
    median_diffs[row_none, col_none] = np.nan

    return(median_diffs)
