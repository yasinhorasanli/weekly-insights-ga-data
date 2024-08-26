import pandas as pd
from sklearn.ensemble import IsolationForest

numerical_metrics = [
    "NewUsers",
    "TotalUsers",
    "ScreenPageViews",
    "Sessions",
    "EngagementRate",
    "UserEngagementDuration",
]
categorical_features = ["Country", "DeviceCategory", "LandingPage", "SessionMedium"]


def get_weekly_changes(df: pd.DataFrame, week_num: int):

    df = df.sort_values(by=["date"])

    df["week"] = df["date"].dt.isocalendar().week
    mask = df["week"] <= week_num
    masked_df = df.loc[mask]

    weekly_data = masked_df.groupby("week")[numerical_metrics].sum().reset_index()

    # Calculate week-over-week percentage change
    for metric in numerical_metrics:
        weekly_data["weekly_change_{}".format(metric)] = (
            weekly_data[metric].pct_change() * 100
        )

    return weekly_data


def segmentation_analysis(df: pd.DataFrame, week_num=None):

    # if the analysis done for only a week
    if week_num is not None:
        df = df.sort_values(by=["date"])

        df["week"] = df["date"].dt.isocalendar().week
        mask = df["week"] == week_num - 1
        masked_df = df.loc[mask]
        df = masked_df

    column_segmentation_dict = {}
    for column in categorical_features:
        segmentation = df.groupby(column).agg(
            {
                "NewUsers": "mean",
                "TotalUsers": "mean",
                "ScreenPageViews": "mean",
                "Sessions": "mean",
                "EngagementRate": "mean",
                "UserEngagementDuration": "mean",
            }
        )

        column_segmentation_dict[column] = segmentation

    return column_segmentation_dict


def find_highest_change(overall_map, last_week_map) -> dict:
    results = {}

    for segment in overall_map:
        overall_segmentation = overall_map[segment]
        last_week_segmentation = last_week_map[segment]

        # Calculate percentage change between overall and last week
        change = (
            (last_week_segmentation - overall_segmentation) / overall_segmentation
        ) * 100

        # Find the metric with the maximum absolute change for each sub-segment
        for metric in change.columns:
            max_change_segment = change[metric].abs().idxmax()
            max_change_value = change.loc[max_change_segment, metric]

            # Check if we need to update the result for this segment
            if segment not in results or abs(max_change_value) > abs(
                results[segment]["metric_value"]
            ):
                results[segment] = {
                    "sub_segment": max_change_segment,
                    "metric": metric,
                    "metric_value": max_change_value,
                }

    return results


def anomaly_detection(df: pd.DataFrame, week_num: int) -> pd.DataFrame:

    features = df[numerical_metrics]
    # fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.001)
    df["anomaly"] = iso_forest.fit_predict(features)

    # identify the anomalies
    anomalies_ml = df[df["anomaly"] == -1].copy()
    anomalies_ml["week"] = anomalies_ml["date"].dt.isocalendar().week
    mask = anomalies_ml["week"] == week_num - 1
    masked_df = anomalies_ml.loc[mask]

    return masked_df


def rule_based_anomaly_detection(df: pd.DataFrame, week_num: int) -> pd.DataFrame:
    # Filter the DataFrame for the specified week
    df_week = df[df["date"].dt.isocalendar().week == week_num].copy()

    df_week["anomaly_type"] = ""

    df_week.loc[df_week["Sessions"] > (2 * df_week["TotalUsers"]), "anomaly_type"] = (
        "Sessions are much greater than Total Users"
    )
    df_week.loc[df_week["NewUsers"] > df_week["TotalUsers"], "anomaly_type"] = (
        "New Users are greater than Total Users"
    )
    df_week.loc[df_week["TotalUsers"] > df_week["Sessions"], "anomaly_type"] = (
        "Total Users are greater than Sessions"
    )

    # Filter only the rows that has anomaly_type filled
    anomalies = df_week[df_week["anomaly_type"] != ""][
        [
            "date",
            "Country",
            "LandingPage",
            "DeviceCategory",
            "Sessions",
            "TotalUsers",
            "NewUsers",
            "EngagementRate",
            "anomaly_type",
        ]
    ]

    anomalies = anomalies[
        (anomalies["Country"] != "(not set)")
        & (anomalies["LandingPage"] != "(not set)")
        & (anomalies["DeviceCategory"] != "(not set)")
    ]

    return anomalies


def add_explanations_to_insights(model, insights):
    concat_insights = "\n".join(insights)
    prompts = [{"role": "user", "content": concat_insights}]
    batchInstruction = {
        "role": "system",
        "content": f"""
        You will be provided with a text containing insights generated using a business's Google Analytics data. Your task is to:
        1. Group the related the insights together and keep them under maximum five bullet points.
        2. Explain why the following metric change or anomaly might be important for a business.
        3. Add these explanations under the bullet points.

        **Important Instructions:**
        - This will be sent to CEO of that company, so keep it in mind.
        - Try to keep the word count less than 600.
        - Do not add any additional data or comment other than explanation.
        - It should only consist of maximum five points list. 
        - Do not add anything other than this numbered list.
        """,
    }
    prompts.append(batchInstruction)
    response = model.chat.completions.create(
        model="gpt-4o", messages=prompts, temperature=0.7, max_tokens=650, top_p=1
    )
    insights_w_explanation = response.choices[0].message.content

    return insights_w_explanation
