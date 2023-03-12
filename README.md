# COVID-HMM
This is a version of our COVID forecaster, which will implement a HMM.

# Run it
This program implements the following libraries:
* `argparse` 
* `numpy`
* `os`
* `pandas`
* `matplotlib`
* `hmmlearn`
* `datetime`

To run the code, use the following command:

`python3 src/main.py -d <directory> -c <country> -t <start_date> <end_date>`

Non-optional arguments:
* `-d | --directory`: the path to the directory containing COVID-19 data
* `-c | --country`: name of a valid country
* `-t | --date_range`: two dates, in the form YYYY-MM-DD YYYY-MM-DD
  * example: `-t 2022-01-09 2023-01-09` is a valid way to use this flag
  
Optional arguments:
* `-p | --province`: province in specified country
* `-a | --agg_days`: aggregate value (number of days between data points)
  * example: `-a 7` would make the model train on data which contains weekly data, instead of daily
