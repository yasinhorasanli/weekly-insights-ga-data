# weekly-insights-ga-data

Project that analyzes google analytics data and generates weekly insights.
This repository contains a Python script (main.py) designed to analyze weekly metrics from a Google Analytics dataset. The script performs week-over-week analysis and segmentation analysis, with additional support for logging and OpenAI API integration.

## Prerequisites

- Python 3.6+
- Required Python packages (listed in requirements.txt):
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
    - openai

You can install these packages using:
```
pip install -r requirements.txt
```

## Setup
**1. Clone the Repository:**

```
git clone https://github.com/yasinhorasanli/weekly-insights-ga-data.git

cd weekly-insights-ga-data
```

**2. Add OpenAI API Key:**

Store your OpenAI API key in a file located at `./api_keys/openai_api_key.txt`.

**3. Run the Script:**

Execute the main.py script and give the date when asked:

```
python src/main.py
Specify the date in DD-MM-YYYY format: <date-you-want-get-insight>
```

or give the date as a starting argument in DD-MM-YYYY format

```
python src/main.py <date-you-want-get-insight>
```
