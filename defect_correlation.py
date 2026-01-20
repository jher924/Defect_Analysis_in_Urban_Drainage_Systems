import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_defect_type_correlation(df):
    """
    --- Plot correlation matrix between defect types based on their presence in the same pipe ---

    Parameters:
    df: pandas DataFrame containing defect data. Must include the columns:
    'Pipe_ID' and 'Defect_code'.
    Each row represents a defect instance.
    """

    # --- Check if DataFrame exists and is not empty ---
    if df is None or df.empty:
        raise ValueError("⚠️ The DataFrame 'df' is empty or does not exist.")

    # --- Check if required columns exist ---
    required_cols = ['Pipe_ID', 'Defect_code']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the DataFrame.")

    # --- Order defect types by frequency ---
    order_defects = df['Defect_code'].value_counts().index.tolist()

    # --- Create a presence matrix of defects per pipe ---
    count_defects = pd.crosstab(df['Pipe_ID'], df['Defect_code'])
    presence_defects = (count_defects > 0).astype(int)

    # --- Compute correlation matrix ---
    corr_matrix = presence_defects.corr(method='pearson')
    corr_matrix = corr_matrix.loc[order_defects, order_defects]

    # --- Plot heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.ylabel("Defect type")
    plt.xlabel("Defect type")
    plt.tight_layout()
    plt.show()