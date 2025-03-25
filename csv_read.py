import pandas as pd
from pathlib import Path

def load_annotated_cases(out_path):
    """
    Reads the annotated_cases.csv file and returns its contents as a DataFrame.

    Args:
        out_path (str): The path to the folder containing annotated_cases.csv.

    Returns:
        pd.DataFrame: A DataFrame containing the annotated cases.
    """
    output_file = Path(out_path) / "annotated_cases.csv"

    if not output_file.exists():
        print(f"No annotated_cases.csv found at {output_file}")
        return pd.DataFrame()

    df = pd.read_csv(output_file, delimiter='|', encoding='utf-8')
    print(f"Loaded {len(df)} annotated cases from {output_file}")

    return df

# Example usage:
cases_df = load_annotated_cases("C:\\Users\\andrey\\Desktop")

# Preview the first few rows
print(cases_df.iloc[5])
