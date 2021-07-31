import click
import glob
import os
import requests
import zipfile
from Utils import Utils
from ETL import ETL
import pandas as pd
from multiprocessing import Pool


def parallel_etl(first, last, interval, path):
    etl = ETL(path)
    etl.extract()
    etl.transform(first, last, interval)
    etl.load()


@click.command()
@click.option('--dataset', '-d', default="data/*.txt", help="Directory of Cabspotting data set", show_default=True,
              type=str)
@click.option('--get', '-g', help="Get data set from the internet", type=bool, default=False, show_default=True)
@click.option('--url', '-u', help="The url of Cabspotting data set", default="", type=str, show_default=True)
@click.option('--interval', '-i', help="Enter the intervals between two points in seconds", default=100, type=int,
              show_default=True)
@click.option('--processes', '-p', help="number of processes", default=1, type=int, show_default=True)
def main(dataset, get, url, interval, processes):
    # download the data set from the internet
    if get is True:
        # download data set from the internet
        request = requests.get(url)
        with open('./data.zip', 'wb') as file:
            file.write(request.content)

        # unzip file
        with zipfile.ZipFile('./data.zip', 'r') as zip_file:
            zip_file.extractall('./')

    # create the target directory for clean data
    if not os.path.exists('./cleaned_dataset'):
        os.makedirs('./cleaned_dataset')

    # find the first and the last movement among all cabs in the data set
    first = Utils.find_start_time(dataset)
    last = Utils.find_stop_time(dataset)

    files = [os.getcwd() + '/' + file for file in glob.glob(dataset)]

    # create arguments in order to make it suitable for using in parallel
    paralleled_arguments = tuple(zip(
        [first] * len(files),
        [last] * len(files),
        [interval] * len(files),
        files
    ))

    # Transform dataset in parallel
    with Pool(processes=processes) as pool:
        pool.starmap(parallel_etl, paralleled_arguments)


if __name__ == '__main__':
    main()

