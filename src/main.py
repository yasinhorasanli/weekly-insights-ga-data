import logging
from pprint import pformat
import os
import pandas as pd
from datetime import datetime
import sys
import json
from openai import OpenAI
import os

from util import categorical_features, numerical_metrics
from util import (
    get_weekly_changes,
    segmentation_analysis,
    find_highest_change,
    anomaly_detection,
    rule_based_anomaly_detection,
    add_explanations_to_insights,
)


openai_api_key = open("./api_keys/openai_api_key.txt").read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key
model = OpenAI()


def get_logger(path):
    global logger
    logging.basicConfig(
        filename=path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], format="%a, %d %b %Y %H:%M:%S GMT")
    df[numerical_metrics] = df[numerical_metrics].apply(pd.to_numeric, errors="coerce")
    return df


def load_data(path) -> pd.DataFrame:
    with open(path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    # logger.info(df.head())
    df = preprocess_data(df)
    return df


def main() -> None:

    get_logger("./log/main.log")
    logger.info(
        "################################################### WEEKLY INSIGHT MODULE IS STARTING ###################################################"
    )

    df = load_data("./data/assessment_data.json")

    if len(sys.argv) > 1:
        today = sys.argv[1]
    else:
        print("Specify the date in DD-MM-YYYY format: ")
        today = input()

    format_str = "%d-%m-%Y"  # The format
    datetime_obj = datetime.strptime(today, format_str)
    logger.info("Today's date: {}".format(datetime_obj.date()))

    week_num = datetime.date(datetime_obj).isocalendar()[1]
    logger.info("Week of year: {}".format(week_num))

    mask = df["date"] < datetime_obj
    df = df.loc[mask]

    ########## CHANGES & TREND ANALYSIS ##########
    weekly_changes_df = get_weekly_changes(df, week_num)

    ##########  SEGMENTATION ANALYSIS   ##########
    overall_segmentation_dict = segmentation_analysis(df)
    last_week_segmentation_dict = segmentation_analysis(df, week_num)
    # the segment with the highest average change
    highest_change_results = find_highest_change(
        overall_segmentation_dict, last_week_segmentation_dict
    )

    ##########    ANOMALY DETECTION     ##########
    anomaly_df = anomaly_detection(df, week_num)

    rule_based_anomaly_df = rule_based_anomaly_detection(df, week_num)

    logger.info(pformat(rule_based_anomaly_df))

    most_repetitive_elements = {}
    if anomaly_df.shape[0] > 0:
        for feature in categorical_features:
            most_frequent_value = anomaly_df[feature].mode()[0]
            frequency = (anomaly_df[feature].value_counts(normalize=True) * 100).loc[
                most_frequent_value
            ]
            most_repetitive_elements[feature] = (most_frequent_value, frequency)

    ##########    INSIGHT GENERATION     ##########
    weekly_insights = []

    # Changes and trend
    for metric in numerical_metrics:
        change_for_metric = weekly_changes_df["weekly_change_{}".format(metric)]
        max_week_row = change_for_metric.loc[weekly_changes_df["week"].idxmax()]
        if max_week_row > 20:
            weekly_insights.append(
                f"Significant increase in {metric} this week: {max_week_row:.2f}% growth compared to the previous week."
            )
        elif max_week_row < -20:
            weekly_insights.append(
                f"Significant decrease in {metric} this week: {max_week_row:.2f}% decrease compared to the previous week."
            )

    # segmentation analysis
    for segment, result in highest_change_results.items():
        weekly_insights.append(
            (
                f"{result['sub_segment']} as {segment} has the highest average change in {result['metric']}: {result['metric_value']:.2f}%"
            )
        )

    # isolation tree anomaly detection
    if anomaly_df.shape[0] > 0:
        # weekly_insights.append("Possible anomaly case count: {}".format(anomaly_df.shape[0]))
        for feature, (value, frequency) in most_repetitive_elements.items():
            weekly_insights.append(
                f"Among the possible anomaly cases, the most repetitive element in '{feature}' is: {value} with a frequency of {frequency:.2f}%"
            )

    # rule-based anomaly detection
    if rule_based_anomaly_df.shape[0] > 0:
        # weekly_insights.append(f"Rule-based anomaly case count: {rule_based_anomaly_df.shape[0]}")
        for _, row in rule_based_anomaly_df.iterrows():
            insight = (
                f"Anomaly detected: '{row['anomaly_type']}' on {row['date'].strftime('%Y-%m-%d')} "
                f"for landing page '{row['LandingPage']}' using {row['DeviceCategory']}. "
            )
            weekly_insights.append(insight)

    logger.info(weekly_insights)

    insights_w_exp = add_explanations_to_insights(model, weekly_insights)

    logger.info(insights_w_exp)
    print(insights_w_exp)


if __name__ == "__main__":
    main()
