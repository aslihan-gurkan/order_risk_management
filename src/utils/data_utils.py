from typing import List, Tuple

import pandas as pd


def grab_col_names(
    dataframe: pd.DataFrame,
    cat_th: int = 10,
    car_th: int = 20,
) -> Tuple[List[str], List[str], List[str], List[str]]:

    cat_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype == "O"
        or pd.api.types.is_string_dtype(dataframe[col])
        or isinstance(dataframe[col].dtype, pd.CategoricalDtype)
    ]

    num_cols = [
        col for col in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    num_but_cat = [
        col for col in num_cols
        if dataframe[col].nunique(dropna=True) < cat_th
    ]

    cat_but_car = [
        col for col in cat_cols
        if dataframe[col].nunique(dropna=True) > car_th
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car, num_but_cat


def missing_values_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    result = pd.DataFrame({
        "variable": na_columns,
        "missing_count": [dataframe[col].isnull().sum() for col in na_columns],
        "missing_ratio": [dataframe[col].isnull().mean() for col in na_columns],
    }).sort_values("missing_ratio", ascending=False)

    return result


def outlier_thresholds(
    dataframe: pd.DataFrame,
    col_name: str,
    q1: float = 0.05,
    q3: float = 0.95,
):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def check_outlier(dataframe: pd.DataFrame, col_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(dataframe[col_name]):
        return False

    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[
        (dataframe[col_name] > up_limit) |
        (dataframe[col_name] < low_limit)
    ]

    return outliers.any(axis=None)


def target_summary_with_cat(
    dataframe: pd.DataFrame,
    target: str,
    categorical_col: str,
    min_count: int = 50,
) -> pd.DataFrame:

    result = (
        dataframe
        .groupby(categorical_col, dropna=False)
        .agg(
            order_count=(target, "count"),
            problematic_count=(target, "sum"),
            problematic_rate=(target, "mean"),
        )
        .reset_index()
        .query("order_count >= @min_count")
        .sort_values("problematic_rate", ascending=False)
    )

    return result


def target_summary_with_num(
    dataframe: pd.DataFrame,
    target: str,
    numerical_col: str,
) -> pd.DataFrame:

    result = (
        dataframe
        .groupby(target)
        .agg(
            count=(numerical_col, "count"),
            mean=(numerical_col, "mean"),
            median=(numerical_col, "median"),
            min=(numerical_col, "min"),
            max=(numerical_col, "max"),
        )
        .reset_index()
    )

    result.insert(0, "variable", numerical_col)

    return result


def rare_analyser(
    dataframe: pd.DataFrame,
    target: str,
    cat_cols: List[str],
    max_unique: int = 50,
) -> pd.DataFrame:

    rows = []

    for col in cat_cols:
        if col == target or dataframe[col].nunique(dropna=True) > max_unique:
            continue

        summary = (
            dataframe
            .groupby(col, dropna=False)
            .agg(
                count=(target, "count"),
                ratio=(target, lambda x: len(x) / len(dataframe)),
                target_mean=(target, "mean"),
            )
            .reset_index()
        )

        summary.insert(0, "variable", col)
        summary = summary.rename(columns={col: "category"})
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)