import logging
from pprint import pformat
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from openai import OpenAI
import os

openai_api_key = open("./api_keys/openai_api_key.txt").read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key
model = OpenAI()

numerical_metrics = ['NewUsers', 'TotalUsers', 'ScreenPageViews', 'Sessions', 'EngagementRate', 'UserEngagementDuration']
categorical_features  = ['Country', 'DeviceCategory', 'LandingPage', 'SessionMedium']

def get_logger(path):
    global logger
    logging.basicConfig(filename=path, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger()



def get_weekly_changes(df: pd.DataFrame, week_num: int):

    df = df.sort_values(by=['date'])
   
    df['week'] = df['date'].dt.isocalendar().week
    mask = (df['week'] <= week_num)
    masked_df = df.loc[mask]

    #for metric in metric_list:
    #    weekly_data = masked_df.groupby('week')[metric].sum().reset_index()

    weekly_data = masked_df.groupby('week')[numerical_metrics].sum().reset_index()

    # Calculate week-over-week percentage change
    for metric in numerical_metrics:
        weekly_data['weekly_change_{}'.format(metric)] = weekly_data[metric].pct_change() * 100

    return weekly_data


def segmentation_analysis(df: pd.DataFrame, week_num=None):

    if week_num is not None:
        df = df.sort_values(by=['date'])
   
        df['week'] = df['date'].dt.isocalendar().week
        mask = (df['week'] == week_num-1)
        masked_df = df.loc[mask]
        df = masked_df

    column_segmentation_dict = {}
    for column in categorical_features:
        segmentation = df.groupby(column).agg({
            'NewUsers': 'mean',
            'TotalUsers': 'mean',
            'ScreenPageViews': 'mean', 
            'Sessions': 'mean', 
            'EngagementRate': 'mean',
            'UserEngagementDuration': 'mean'
        })

        print("Segmentation by {}:".format(column))
        print(segmentation)
        column_segmentation_dict[column] = segmentation

    return column_segmentation_dict


def find_highest_change(overall_map, last_week_map) -> dict:
    results = {}
    
    for segment in overall_map:
        overall_segmentation = overall_map[segment]
        last_week_segmentation = last_week_map[segment]

        # Calculate percentage change between overall and last week
        change = ((last_week_segmentation - overall_segmentation) / overall_segmentation) * 100

        # Find the metric with the maximum absolute change for each sub-segment
        for metric in change.columns:
            max_change_segment = change[metric].abs().idxmax()
            max_change_value = change.loc[max_change_segment, metric]

            # Check if we need to update the result for this segment
            if segment not in results or abs(max_change_value) > abs(results[segment]['metric_value']):
                results[segment] = {
                    'sub_segment': max_change_segment,
                    'metric': metric,
                    'metric_value': max_change_value
                }

    return results

def anomaly_detection(df: pd.DataFrame, week_num: int) -> pd.DataFrame:
    # features for anomaly detection
    features = df[numerical_metrics]

    # fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.001)  
    df['anomaly'] = iso_forest.fit_predict(features)

    # identify the anomalies
    anomalies_ml = df[df['anomaly'] == -1]
    print("Anomalies Detected by Isolation Forest:")
    print(anomalies_ml[['date', 'Country', 'NewUsers', 'Sessions', 'EngagementRate', 'UserEngagementDuration', 
                        'TotalUsers', 'ScreenPageViews', 'DeviceCategory','LandingPage', 'SessionMedium']])
    
    anomalies_ml['week'] = anomalies_ml['date'].dt.isocalendar().week
    mask = (anomalies_ml['week'] == week_num-1)
    masked_df = anomalies_ml.loc[mask]

    return masked_df

def add_explanations_to_insights(insights):
    concat_insights = '\n'.join(insights)
    logger.info(concat_insights)
    #prompt = f"Explain why the following metric change might be important for a business: {insight}"
    prompts = [{"role": "user", "content": concat_insights}]
    batchInstruction = {
        "role": "system",
        "content": f"""
        You will be provided with a text containing insights generated using a business's Google Analytics data. Your task is to:
        1. Group the related the insights together and keep them under maximum 5 bullet points.
        2. Explain why the following metric change or anomaly might be important for a business.
        3. Add these explanations under the bullet points.

        **Important Instructions:**
        - This will be sent to CEO of that company, so keep it in mind.
        - Do not add any additional data or comment other than explanation.
        """
    }

    #**Example Input:**
    #["Course 1.", "Course 2.", "Course 3."]

    #**Example Output:**
    #["Converted Text 1.", "Converted Text 2.", "Converted Text 3."]
    prompts.append(batchInstruction)
    response = model.chat.completions.create(
        model="gpt-4o", messages=prompts, temperature=0.7, max_tokens=400, top_p=1
    )
    insights_w_explanation = response.choices[0].message.content

    return insights_w_explanation


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'], format='%a, %d %b %Y %H:%M:%S GMT')
    # remaining_columns = ['Country', 'DeviceCategory', 'LandingPage', 'SessionMedium', 'date']
    df[numerical_metrics] = df[numerical_metrics].apply(pd.to_numeric, errors='coerce')
    return df


def load_data(path) -> pd.DataFrame:
    with open(path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    logger.info(df.head())
    df = preprocess_data(df)
    return df

def main() -> None:

    get_logger("./log/main.log")
    logger.info("################################################### WEEKLY INSIGHT MODULE IS STARTING ###################################################")

    df = load_data('./data/assessment_data.json')

    if len(sys.argv) > 1:
        today = sys.argv[1]
    else:
        print("Specify the date in DD-MM-YYYY format: ")
        today = input()

    #date_str = '29/12/2017' # The date - 29 Dec 2017
    format_str = '%d-%m-%Y' # The format
    datetime_obj = datetime.strptime(today, format_str)
    logger.info("Today's date: {}".format(datetime_obj.date()))

    week_num = datetime.date(datetime_obj).isocalendar()[1]
    logger.info("Week of year: {}".format(week_num))

    mask = (df['date'] < datetime_obj)
    df = df.loc[mask]

    ########## CHANGES & TREND ANALYSIS ##########
    weekly_changes_df = get_weekly_changes(df, week_num)


    ##########  SEGMENTATION ANALYSIS   ##########
    overall_segmentation_dict = segmentation_analysis(df)
    last_week_segmentation_dict  = segmentation_analysis(df, week_num)

    # the segment with the highest average change
    highest_change_results = find_highest_change(overall_segmentation_dict, last_week_segmentation_dict)

    ##########    ANOMALY DETECTION     ##########
    anomaly_df = anomaly_detection(df, week_num)
    
    most_repetitive_elements = {}
    for feature in categorical_features:
        most_frequent_value = anomaly_df[feature].mode()[0]
        frequency = (anomaly_df[feature].value_counts(normalize=True) * 100).loc[most_frequent_value]
        most_repetitive_elements[feature] = (most_frequent_value, frequency)

    #Example
    # top_country = overall_segmentation_dict['Country']['NewUsers'].idxmax()
    # print(f"{top_country} has the highest average number of new users: {overall_segmentation_dict['Country'].loc[top_country, 'NewUsers']:.2f}.")
    #Example
    # top_country = last_week_segmentation_dict['Country']['NewUsers'].idxmax()
    # print(f"{top_country} has the highest average number of new users: {last_week_segmentation_dict['Country'].loc[top_country, 'NewUsers']:.2f}.")    
    


    ##########    INSIGHT GENERATION     ##########
    weekly_insights = []
    for metric in numerical_metrics:

        # Changes and trend
        change_for_metric = weekly_changes_df['weekly_change_{}'.format(metric)]
        max_week_row = change_for_metric.loc[weekly_changes_df['week'].idxmax()]
        if max_week_row > 20:
            weekly_insights.append(f"Significant increase in {metric} this week: {max_week_row:.2f}% growth compared to the previous week.")
        elif max_week_row < -20:
            weekly_insights.append(f"Significant decrease in {metric} this week: {max_week_row:.2f}% decrease compared to the previous week.")


    # segmentation analysis
    for segment, result in highest_change_results.items():
        weekly_insights.append((f"{result['sub_segment']} as {segment} has the highest average change in {result['metric']}: {result['metric_value']:.2f}%"))



    weekly_insights.append("Possible anomaly case count: {}".format(anomaly_df.shape[0]))
    # Display the results
    for feature, (value, frequency) in most_repetitive_elements.items():
        weekly_insights.append(f"Among the possible anomaly cases, the most repetitive element in '{feature}' is: {value} with a frequency of {frequency:.2f}%")


    logger.info(pformat(weekly_insights))

    insights_w_exp = add_explanations_to_insights(weekly_insights)

    logger.info(pformat(insights_w_exp))
    logger.info(insights_w_exp)
    
    print(insights_w_exp)


if __name__ == "__main__":
    main()
