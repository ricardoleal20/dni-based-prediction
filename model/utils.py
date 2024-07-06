"""
Different utilities for the project
"""
import io
# External imports
import pandas as pd

def df_info_to_latex(df: pd.DataFrame, max_cols: int = 4) -> list[str]:
    """Convert the DF `.info()` method to a LaTex table"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Process the lines of the information
    lines = info_str.split('\n')
    latex_strs = []
    latex_str = ""

    current_table_cols = []
    header_line = ""
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) > 2 and parts[1].isdigit():
                row = f"{' '.join(parts[:-2])} & {' '.join(parts[-2:])} \\\\\n"
            else:
                row = f"{' '.join(parts)} \\\\\n"
            if len(current_table_cols) < max_cols:
                current_table_cols.append(row)
            if len(current_table_cols) == max_cols:
                header_line = current_table_cols[0]
                latex_str += "\\begin{tabular}{ll}\n\\hline\n"
                latex_str += header_line
                for r in current_table_cols[1:]:
                    latex_str += r
                latex_str += "\\hline\n\\end{tabular}\n\n"
                latex_strs.append(latex_str)
                current_table_cols = []
                latex_str = ""
    if current_table_cols:
        header_line = current_table_cols[0]
        latex_str += "\\begin{tabular}{ll}\n\\hline\n"
        latex_str += header_line
        for r in current_table_cols[1:]:
            latex_str += r
        latex_str += "\\hline\n\\end{tabular}\n\n"
        latex_strs.append(latex_str)

    return latex_strs

def df_describe_to_latex(df: pd.DataFrame, max_cols: int =4) -> list[str]:
    """Convert the DF `.describe()` method to a LaTex table."""
    describe_df = df.describe()
    latex_strs = []
    n_cols = len(describe_df.columns)

    for start_col in range(0, n_cols, max_cols):
        sub_df = describe_df.iloc[:, start_col:start_col+max_cols]
        latex_str = sub_df.to_latex()
        latex_strs.append(latex_str)

    return latex_strs


def write_to_file(texts: list[str]) -> None:
    """Write text to a file"""
    with open("latex.txt", "w", encoding="utf-8") as file:
        for text in texts:
            file.write(text + "\n\n")
