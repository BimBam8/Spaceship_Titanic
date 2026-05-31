from evidently import Report
from evidently.presets import DataSummaryPreset
import pandas as pd



def run_quality_report(df: pd.DataFrame, output_path: str = "reports/data_quality.html"):
    report = Report(metrics=[DataSummaryPreset()])
    try:
        report.run(current_data=df, reference_data=None).save_html(output_path)
        print(f"Report saved to {output_path}")
    except():
        print("report run failed, report not saved")

if __name__=="__main__":
    df = pd.read_csv("data/train.csv")
    # run_quality_report(df)
    print(df.head(10))