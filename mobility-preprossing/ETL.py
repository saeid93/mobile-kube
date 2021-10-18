import os
import pandas as pd
from datetime import datetime, timedelta
from Utils import Utils


class ETL:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.final_frame = pd.DataFrame(columns=["latitude",
                                                 "longitude",
                                                 "timestamp"])

    def extract(self):
        self.data = pd.read_csv(self.file_path,
                                delim_whitespace=True,
                                header=None,
                                names=range(4))

    def transform(self, first, last, interval):
        # create first line of each
        first_line_df = pd.DataFrame(
            {
                0: [self.data.iloc[-1][0]],
                1: [self.data.iloc[-1][1]],
                2: pd.Series([self.data.iloc[-1][2]], dtype='int32'),
                3: pd.Series([first], dtype='int32')
            }
        )

        # create last line of each
        end_line_df = pd.DataFrame(
            {
                0: [self.data.iloc[0][0]],
                1: [self.data.iloc[0][1]],
                2: pd.Series([self.data.iloc[0][2]], dtype='int32'),
                3: pd.Series([last], dtype='int32')
            }
        )

        # concat with first line (because dataset is in the reverse order we attach the start time in the end)
        df = pd.concat([self.data, first_line_df], ignore_index=True)

        # concat with last line (because dataset is in the reverse order we attach the end time at the beginning)
        df = pd.concat([end_line_df, df], ignore_index=True)

        """Phase two: prepare intermediate data frames"""
        # delete last row
        starts = df.tail(-1)

        # reverse dataframe
        starts = starts.sort_index(ascending=False)

        # rename the columns
        starts.rename(columns={0: 'src_latitude',
                               1: 'src_longitude',
                               2: 'src_fare',
                               3: 'src_timestamp'},
                      inplace=True)

        # reset indexing
        starts = starts.reset_index()

        # delete index column
        del starts['index']

        # delete first row
        stops = df.head(-1)

        # reverse dataframe
        stops = stops.sort_index(ascending=False)

        # rename the columns
        stops.rename(columns={0: 'dst_latitude',
                              1: 'dst_longitude',
                              2: 'dst_fare',
                              3: 'dst_timestamp'},
                     inplace=True)

        # reset index
        stops = stops.reset_index()

        # delete index
        del stops['index']

        # concatenate them
        rows = pd.concat([starts, stops], axis=1)

        """Phase three: find all intermediate points and attach them to the final data frame"""

        # keep the track of timestamp during point calculation
        time_marker = datetime.fromtimestamp(first)
        for _, row in rows.iterrows():
            if datetime.fromtimestamp(row['src_timestamp']).day != datetime.fromtimestamp(row['dst_timestamp']).day:
                results = Utils.find_points(
                    row['src_longitude'],
                    row['src_latitude'],
                    row['dst_longitude'],
                    row['dst_latitude'],
                    row['src_timestamp'],
                    row['dst_timestamp'],
                    motion=False)

            else:
                results = Utils.find_points(
                    row['src_longitude'],
                    row['src_latitude'],
                    row['dst_longitude'],
                    row['dst_latitude'],
                    row['src_timestamp'],
                    row['dst_timestamp'],
                    motion=True)

            if results is not None:
                # filter the data
                delta = timedelta(seconds=interval)
                frame = []

                for result in results:
                    if result[2] == time_marker:
                        frame.append(result)
                        time_marker += delta

                # filter the results based on delta time
                self.final_frame = pd.concat(
                    [
                        self.final_frame,
                        pd.DataFrame(
                            frame,
                            columns=[
                                "latitude",
                                "longitude",
                                "timestamp"
                            ]
                        )
                    ]
                )

    def load(self, output_path):
        self.final_frame.to_csv(os.path.join(output_path, os.path.basename(self.file_path)), encoding='utf-8', index=False)
