#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A lightweight privacy method based on time shuffling designed for sharing \
power profiles based data in open forums.

Conceptulised in the Energy and Power Group.
Energy and Power Group is part of the Department of Engineering Science,
University of Oxford.

A profile is split into periods with datapoints scrambled within each period.
The mean value of each period is maintained between the scrambled data and
the raw profile. For example, if the original
profile was a HH profile of 48 datapoints and periods was chosen as 8,
the profile would be shuffled within 8 periods, each containing 6
datapoints representing 3 hours. Each of those periods in the scrambled
profile will have the same mean value as the equivalent period in the
original profile. Therefore, if the scrambled profile is shared openly,
only 3 hourly data can be interpreted accurately.

@author: Scot Wheeler, scot.wheeler@eng.ox.ac.uk

Copyright (c) 2021 Scot Wheeler

"""

__version__ = '0.0.4'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def single_scramble(profile, periods=None, freq=None, source_freq=None,
                    seed=42, df_colname="consumption"):
    """
    Split profile into p periods and scrambles the datapoints within each\
    period.

    The number of periods determines the level of granularity that can be
    correctly interpreted without unscrambling. For example, if the original
    profile was a HH profile of 48 datapoints and periods was chosen as 8,
    the profile would be shuffled within 8 periods, each containing 6
    datapoints representing 3 hours. Each of those periods in the scrambled
    profile will have the same mean value as the equivalent period in the
    original profile. Therefore, if the scrambled profile is shared openly,
    only 3 hourly data can be interpreted accurately.

    Todo: test freq and source_freq functionality

    Parameters
    ----------
    profile : np.Array or pd.Series
        An array like object containing a time based profile. Must have a
        datetime index if using the freq argument.
    periods : int, optional
        The number of periods to split the profile into, before scramling the
        order of datapoint within each of these periods. Periods must be a
        divisor of the total number of datapoints in the profile.
        Either this, or freq must be supplied.
        The default is None.
    freq : str, optional
        Frequency string of the desired final profile granularity over which
        the profile can be openly interpreted. e.g. '3H' for 3 hourly. Either
        this, or periods must be supplied. This has not been tested in depth.
        The default is None.
    source_freq : str, optional
        Frequency string of the original profile. This is required if the
        original profile does not contain a datetime index, or the frequency
        cannot be determined from it. Used in combination with freq arg.
        The default is None.
    seed : int, optional
        A random seed value used to determine the order by which data points
        are shuffled. This must be known to successfully unshuffle the profile.
        The default is 42.
    df_colname : str, int, float, optional
        The name of the column to use as the profile if a dataframe with
        multiple columns is passed as the profile.
        The default is "consumption".

    Returns
    -------
    np.Array, pd.Series or pd.DataFrame
        The scrambled profile as defined by the number of periods. Any
        datetime index is returned unshuffled.
    """
    if not isinstance(seed, int):
        raise Exception("Seed argument must be an int")
    if isinstance(profile, pd.Series):
        series = True
        dataframe = False
        raw_profile = profile.copy()
        profile = profile.to_numpy()
    elif isinstance(profile, pd.DataFrame):
        series = False
        dataframe = True
        raw_profile = profile.copy()
        profile = profile[df_colname].to_numpy()
    else:
        series = False
        dataframe = False

    T = len(profile)

    if isinstance(freq, str):
        # if freq defined, try using instead of periods
        try:
            # if profile has a datetime index
            source_freq = pd.infer_freq(profile.index)
        except:
            print("""
                  Could not determine source_freq, check if profile has
                  datetime index or use the source_freq argument
                  """)

    # if both freq and source_freq are defined, use instead of periods arg
    if (isinstance(freq, str) and isinstance(source_freq, str)):
        periods = T / (pd.to_timedelta(freq)/pd.to_timedelta(source_freq))

    # check that periods is a factor of the total number of periods T
    if (len(profile) % periods) != 0:
        raise Exception(str(periods) + """
                      periods arg not a divisor of total length of profile
                      """)

    # sections
    s = int(len(profile)/periods)

    # reshape array to seperate out sections
    p = profile.reshape((-1, s))
    p2 = p.copy()

    # create a scrambled key
    key = list(range(s))
    rng = np.random.default_rng(seed)
    rng.shuffle(key)

    for j in range(periods):
        for i in range(s):
            p2[j][i] = p[j][key[i]]

    p3 = p2.flatten()

    if series:
        p4 = raw_profile.copy()
        p4.iloc[:] = p3
        return p4
    if dataframe:
        p4 = raw_profile.copy()
        p4[df_colname] = p3
        return p4
    return p3


def multi_scramble(profile, multiperiods, seeds=None, **kwargs):
    """
    Apply profile scrambling at multiple granularity levels.

    This can be unscrambled at increasing levels of granularity depending
    on how many of the periods and seed keys are shared.

    Parameters
    ----------
    profile : np.Array or pd.Series
        An array like object containing a time based profile. Must have a
        datetime index if using the freq argument
    multiperiods : list
        A list of integers representing the different periods to iteratively
        split the profile into and shuffle within. Should be decreasing and
        should be factors of the previous period and original profile length.
        e.g. for a profile with 48 HH datapoints, [8,2,1] would be valid and
        correspond to granularities of 3 hourly, 12 hourly and daily which
        need to be unscrambled in the reverse order.
    seeds : list, optional
        A list of integers which are used to seed the random profile shuffle
        within each period. The default is None.
    **kwargs : TYPE
        Arguments to pass through to single_scramble e.g. df_colname.


    Returns
    -------
    profile : np.Array, pd.Series or pd.DataFrame
        The mulit-level scrambled profile as defined by the number of periods.
        Any datetime index is returned unshuffled.
    seeds : list
        A list of seeds used as security keys. Required to unscramble.

    """
    # check if multiperiods are factors of the previous
    for i, periods in enumerate(multiperiods):
        if i == 0:
            if (len(profile) % periods) != 0:
                raise Exception(str(periods) + """
                      The first periods arg is not a divisor of total length
                      of profile
                      """)
        else:
            if (multiperiods[i-1] % multiperiods[i]) != 0:
                raise Exception(str(multiperiods[i])
                                + " is not a divisor of "
                                + str(multiperiods[i-1]))
    # check if seeds defined, if not, generate random seeds.
    if isinstance(seeds, type(None)):
        seeds = []
        for periods in multiperiods:
            rng = np.random.default_rng()
            seed = rng.integers(12345)
            seeds.append(int(seed))
    else:  # check if supplied seeds are ints
        if len(seeds) != len(multiperiods):
            raise Exception("Number of seeds must equal number of\
                            multiperiods")
        for seed in seeds:
            if not isinstance(seed, int):
                raise Exception("Seed argument must be an int")

    for periods, seed in zip(multiperiods, seeds):
        profile = single_scramble(profile, periods=periods, seed=seed,
                                  **kwargs)

    return profile, seeds


def single_unscramble(scrambled, periods, seed,
                      df_colname="consumption"):
    """
    Unscramble a single level of a scrambled profile.

    Parameters
    ----------
    scrambled : np.Array or pd.Series
        An array like object containing a time based profile which has
        previously been scrambled.
    periods : int,
        The number of periods used to scramble the profile to this level.
    seed : int,
        The seed used during the scramble process at the associated level.
    df_colname : str, int, float, optional
        The name of the column to use as the profile if a dataframe with
        multiple columns is passed as the profile.
        The default is "consumption".


    Returns
    -------
    np.Array, pd.Series or pd.DataFrame
        The unscrambled profile as defined by the number of periods and seed.
        Any datetime index is returned unshuffled.

    """
    if not isinstance(seed, int):
        raise Exception("Seed argument must be an int")

    if isinstance(scrambled, pd.Series):
        series = True
        dataframe = False
        raw_profile = scrambled.copy()
        scrambled = scrambled.to_numpy()
    elif isinstance(scrambled, pd.DataFrame):
        series = False
        dataframe = True
        raw_profile = scrambled.copy()
        scrambled = scrambled[df_colname].to_numpy()
    else:
        series = False
        dataframe = False

    # check that each freq is valid
    if (len(scrambled) % periods) != 0:
        raise Exception(
            str(periods) + """ periods arg not a divisor of total length of\
            profile""")

    s = int(len(scrambled)/periods)

    # reshape array to seperate out sections
    p = scrambled.reshape((-1, s))
    p2 = p.copy()

    # create a scrambled key
    key = list(range(s))
    rng = np.random.default_rng(seed)
    rng.shuffle(key)

    for j in range(periods):
        for i in range(s):
            p2[j][key[i]] = p[j][i]

    p3 = p2.flatten()
    if series:
        p4 = raw_profile.copy()
        p4.iloc[:] = p3
        return p4
    if dataframe:
        p4 = raw_profile.copy()
        p4[df_colname] = p3
        return p4
    return p3


def multi_unscramble(scrambled, multiperiods, seeds, **kwargs):
    """
    Apply profile unscrambling at multiple granularity levels.

    The correct multiperiods and seeds are required in the same order
    as defined when the profifle was scrambled.
    To partially unscramble, remove periods from the left of the list.
    e.g. a profile scrambled with periods [8,2,1] can be unlocked to the
    level 2 stage (8 periods) using just [2,1] with corresponding seed values.

    Parameters
    ----------
    scrambled : np.Array, pd.Series or pd.DataFrame
        The mulit-level scrambled profil to unscramble.
    multiperiods : list
        List of periods (or subset thereof from smallest value) used in the
        original scrambling.
    seeds : list
        The seeds (or keys) used to shuffle the profile. Should be the same
        length as multiperiods.
    **kwargs : TYPE
        Additional arguments to pass through to single_unscramble.


    Returns
    -------
    profile : np.Array, pd.Series or pd.DataFrame
        The unscrambled profile to the level defined by the number of
        multiperiods.

    """
    # check if multiperiods are factors of the previous:
    for i, periods in enumerate(multiperiods):
        if i == 0:
            if (len(scrambled) % periods) != 0:
                raise Exception(str(periods) + """
                      The first periods arg is not a divisor of total length
                      of profile
                      """)
        else:
            if (multiperiods[i-1] % multiperiods[i]) != 0:
                raise Exception(str(multiperiods[i])
                                + " is not a divisor of "
                                + str(multiperiods[i-1]))

    # check if length of seeds ==  length of multiperiods:
    if len(seeds) != len(multiperiods):
        raise Exception("Number of seeds must equal number of\
                        multiperiods")

    # check if seeds are ints
    for seed in seeds:
        if not isinstance(seed, int):
            raise Exception("Seed argument must be an int")

    profile = scrambled.copy()
    for period, seed in reversed(list(zip(multiperiods, seeds))):
        profile = single_unscramble(profile, period, seed, **kwargs).copy()

    return profile


def resample_profile(profile, periods):
    """
    Resamples a profile to have the number of periods defined by periods.

    Uses the mean of the period. The end point of the period is used as the
    index.

    Parameters
    ----------
    profile : np.Array, pd.Series or pd.DataFrame
        An array like object containing a time based profile.
    periods : int
        The number of periods to resample the profile to.

    Returns
    -------
    np.Array, pd.Series or pd.DataFrame
        Resampled profile. Closed and labelled to right of period.
        If array passed as profile, returns 2d array including a new index
        to help with plotting against orginal.
    """
    # check that each periods is valid
    if (len(profile) % periods) != 0:
        raise Exception(
            str(periods) + " not a valid periods, it must be a factor of\
            length of profile")

    if isinstance(profile, pd.Series) or isinstance(profile, pd.DataFrame):
        source_freq = pd.infer_freq(profile.index)
        new_freq = pd.to_timedelta(source_freq) * (profile.shape[0] / periods)
        resampled = profile.resample(new_freq, closed='right',
                                     label='right').mean()
        return resampled
    else:
        T = len(t)
        s = int(len(profile)/periods)
        # reshape array to seperate out sections
        p = profile.reshape((-1, s))
        new = p.mean(axis=1)
        new_t = np.linspace(((T/periods)-1), (T-1), int(periods))
    return np.array([new_t, new])


# %%
if __name__ == "__main__":
    # %% demo data
    t = np.linspace(1, 48, 48)
    T = len(t)

    profile = np.array(range(1, 49))

    profile_8mean = resample_profile(profile, 8)
    profile_2mean = resample_profile(profile, 2)

    eight = single_scramble(profile, periods=8)
    eight_breaks = np.linspace(((T/8)+0.5), ((48 - (T/8))+0.5), num=7)
    eight_mean = resample_profile(eight, 8)
    eight_two = single_scramble(eight, periods=2)
    two_mean = resample_profile(eight_two, 2)

    full_scrambled, keys = multi_scramble(profile, [8, 2, 1])

    unscrambled = multi_unscramble(full_scrambled, [8, 2, 1], keys)

    # %% demo plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(7, 8))
    p1 = ax1.bar(x=t, height=profile, alpha=0.66)
    p1a = ax1.step(np.append(profile_8mean[0], 48+5)-4.5,
                   np.append(profile_8mean[1], profile_8mean[1][-1]),
                   where='post', color='red')
    p1b = ax1.step(np.append(profile_2mean[0], 48+23)-22.5,
                   np.append(profile_2mean[1], profile_2mean[1][-1]),
                   where='post', color='green')
    p2 = ax2.bar(x=t, height=eight, alpha=0.66)

    p2a = ax2.step(np.append(eight_mean[0], 48+5)-4.5,
                   np.append(eight_mean[1], eight_mean[1][-1]),
                   where='post', color='red')
    p3 = ax3.bar(x=t, height=eight_two, alpha=0.66)
    p3a = ax3.step(np.append(two_mean[0], 48+23)-22.5,
                   np.append(two_mean[1], two_mean[1][-1]),
                   where='post', color='green')
    p4 = ax4.bar(x=t, height=full_scrambled, alpha=0.66)

    for xc in eight_breaks:
        ax2.axvline(x=xc, color='k')
    ax3.axvline(x=24.5, color='k')
    ax4.axhline(y=full_scrambled.mean(), color='purple')
    ax1.axhline(y=profile.mean(), color='purple')

    ax1.set_ylim(top=60)
    ax2.set_ylim(top=60)
    ax3.set_ylim(top=60)
    ax1.bar_label(p1, fontsize='x-small', rotation=70)
    ax2.bar_label(p2, fontsize='x-small', rotation=70)
    ax3.bar_label(p3, fontsize='x-small', rotation=70)

    ax1.set_title("Level 1: Raw data. 30 minute resolution.")
    ax2.set_title("Level 2: 8 periods. Can be correctly interpretted at \
3 hour resolution.")
    ax3.set_title("Level 3: 2 periods. Can be correctly interpretted at \
12 hourly resolution.")
    ax4.set_title("Level 4: 1 periods. Can be correctly interpretted at \
24 hourly resolution only.")

    fig.tight_layout()

    fig.show()
