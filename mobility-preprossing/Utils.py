import glob
from datetime import datetime, timedelta
import pandas as pd
import os


class Utils:
    @staticmethod
    def find_points(source_longitude, source_latitude, destination_longitude, destination_latitude,
                    start_time, end_time, motion=True):
        # if start time and end time are the same, return nothing
        if start_time == end_time:
            return None

        # create the result list
        intermediates = []

        # find differences in seconds
        steps = int(end_time - start_time)

        # if motion is true, find all points between two points, otherwise the cab is motionless.
        if motion:
            # find the number of steps vertically and horizontally
            lat_step = (destination_latitude - source_latitude) / (steps + 1)
            lon_step = (destination_longitude - source_longitude) / (steps + 1)

            # add all the movements in the defined period
            for i in range(steps + 1):
                intermediates.append([source_latitude + i * lat_step,
                                      source_longitude + i * lon_step,
                                      datetime.fromtimestamp(start_time) + timedelta(seconds=i)])
        else:
            # assume that the cab is motionless
            for i in range(steps + 1):
                intermediates.append([source_latitude,
                                      source_longitude,
                                      datetime.fromtimestamp(start_time) + timedelta(seconds=i)])

        return intermediates

    @staticmethod
    def find_start_time(path):
        times = []

        # read every file in Cabspotting data set
        for file in glob.glob(path):
            dataset = pd.read_csv(os.getcwd() + '/' + file,
                                  delim_whitespace=True,
                                  header=None,
                                  names=range(4))

            times.append(dataset.iloc[-1][3])

        return min(times)

    @staticmethod
    def find_stop_time(path):
        times = []

        # read every file in Cabspotting data set
        for file in glob.glob(path):
            dataset = pd.read_csv(os.getcwd() + '/' + file,
                                  delim_whitespace=True,
                                  header=None,
                                  names=range(4))

            times.append(dataset.iloc[0][3])

        return max(times)
