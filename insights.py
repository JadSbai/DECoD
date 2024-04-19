import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from clinical_utils import (get_variables_description, create_induced_variables,
                            InductorVariable, Correlation)


class BirthInsights:
    def __init__(self, data):
        self.df = data
        self.cont_vars = ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_AGE_NC', 'APGAR_1_NC', 'APGAR_2_NC']
        self.description = get_variables_description()
        self.categorize_rows()
        self.induced_variables = create_induced_variables()
        self.target_insights = self.create_target_insights()
        self.other_insights = self.create_subpopulation_insights()
        print(self.target_insights['MAT_SMOKING_NC'][0].results)
        print(self.target_insights['MAT_SMOKING_NC'][0].description)

        print(self.other_insights['BIRTH_WEIGHT_NC'][1].results)

    def categorize_rows(self):
        # TODO: Add region subpopulations

        CRITERIA = {
            "MS": lambda row: row['MAT_SMOKING_NC'] != np.nan and row['MAT_SMOKING_NC'] == 1,
            "BT": lambda row: row['BIRTH_ORDER_NC'] != np.nan and row['BIRTH_ORDER_NC'] == 1,
            "PS": lambda row: row['PREV_STILLBIRTH_NC'] != np.nan and row['PREV_STILLBIRTH_NC'] == 1,
        }

        # Function to categorize each row
        def categorize_row(row):
            categories = [name for name, condition in CRITERIA.items() if condition(row)]
            return ','.join(categories) if categories else 'None'

        # Apply the function to each row in the dataframe
        self.df['subpopulation'] = self.df.apply(categorize_row, axis=1)

    def extract_insights(self, inductor, induced, data):
        opposite = (inductor in self.cont_vars and induced not in self.cont_vars) or (
                inductor not in self.cont_vars and induced in self.cont_vars)
        if opposite:
            binned_var = inductor if inductor in self.cont_vars else induced
            category_var = induced if inductor in self.cont_vars else inductor
            analysis_df = data.copy()

            # First, calculate the quantile bins and then create labels from these bins
            quantiles = pd.qcut(analysis_df[binned_var], q=3, duplicates='drop', retbins=True)
            bin_edges = quantiles[1]  # This extracts the bin edges
            labels = [f"{np.round(bin_edges[i], 2)} to {np.round(bin_edges[i + 1], 2)}" for i in
                      range(len(bin_edges) - 1)]

            # Now apply qcut with the generated labels
            analysis_df.loc[:, f'{binned_var}_bins'] = pd.qcut(analysis_df[binned_var], q=3, labels=labels,
                                                               duplicates='drop')

            # Calculate statistics within bins
            grouped = analysis_df.groupby(category_var)[binned_var]
            stats = grouped.agg(
                ['mean', 'median', 'std', 'min', 'max', lambda x: x.quantile(0.10), lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.75), lambda x: x.quantile(0.90)])
            stats.rename(columns={'<lambda_0>': '10th Percentile', '<lambda_1>': '25th Percentile',
                                  '<lambda_2>': '75th Percentile', '<lambda_3>': '90th Percentile'},
                         inplace=True)

            # Aggregate data within bins
            grouped = analysis_df.groupby([f'{binned_var}_bins', category_var])[binned_var]
            binned_stats = grouped.size().unstack(fill_value=0)
            binned_stats_normalized = binned_stats.div(binned_stats.sum(axis=1), axis=0)
            top_bin = binned_stats_normalized.idxmax().to_dict()  # Get the bin with the highest proportion of data for each category of the target variable

            results = {'Binned Statistics': binned_stats_normalized.to_dict(orient='index'),
                       'Stats': stats.to_dict(orient='index')}

            description = f"Statistical analysis for {binned_var} across categories of {category_var} within specified ranges shows: \n{stats}"

            description += f'Binned statistics for {binned_var} across categories of {category_var} show:\n{binned_stats_normalized} \n'

            description += f'Notable trends include {binned_var} values in the range {top_bin} showing a higher prevalence of specific {category_var} categories.'

        elif inductor in self.cont_vars:
            # Both variables are continuous
            # Calculate Pearson correlation
            analysis_df = data.copy()
            correlation, _ = pearsonr(analysis_df.dropna(subset=[induced, inductor])[induced],
                                      analysis_df.dropna(subset=[induced, inductor])[inductor])

            # Fit a linear regression model
            model = LinearRegression().fit(analysis_df[[inductor]].dropna(), analysis_df[induced].dropna())
            slope = model.coef_[0]
            intercept = model.intercept_

            results = {
                'Correlation': correlation,
                'Regression Coefficients': {'Slope': slope, 'Intercept': intercept}
            }
            description = (f"The correlation between {induced} and {inductor} is {correlation:.2f}. "
                           f"A linear regression model fitting {inductor} to predict {induced} "
                           f"has a slope of {slope:.2f} and an intercept of {intercept:.2f}.")

        else:
            # Compute proportion of categories for categorical variables grouped by target category
            proportions = (data.groupby([induced, inductor]).size() / data.groupby(
                induced).size()).unstack().fillna(0)
            results = {'Proportions': proportions.to_dict()}
            description = f"For each category of {induced}, the proportions of different categories in {inductor} are:\n{proportions}"

        corr = Correlation(inductor, induced, results, description)
        return corr

    def create_target_insights(self):
        results = {}

        for target in self.induced_variables:
            correlated_vars = target.correlated_variables
            relevant_cols = [target.name] + correlated_vars
            analysis_df = self.df.dropna(subset=relevant_cols)
            results[target.name] = []

            for var in correlated_vars:
                corr = self.extract_insights(var, target.name, analysis_df)
                results[target.name].append(corr)

        return results

    def create_subpopulation_insights(self):
        known_pairs = [(InductorVariable.PS, InductorVariable.BW), (InductorVariable.BT, InductorVariable.BW)]
        results = {}
        for pair in known_pairs:
            inductor, induced = pair
            if results.get(induced.value) is None:
                results[induced.value] = []

            analysis_df = self.df.dropna(subset=[inductor.value, induced.value])

            corr = self.extract_insights(inductor.value, induced.value, analysis_df)
            results[induced.value].append(corr)
        return results

    def get_subpopulation_insights(self, subs):
        sub_insights = []
        for sub in subs.split(','):
            if sub == 'BT':
                sub_insights.extend(self.other_insights[InductorVariable.BT])
            if sub == 'PS':
                sub_insights.extend(self.other_insights[InductorVariable.PS])
        return sub_insights

    def get_full_insights(self, row, target):
        return [insight for insight in self.target_insights[target] if pd.notna(row[insight.inductor.value])] + \
            [insight for insight in self.get_subpopulation_insights(row['subpopulation']) if
             pd.notna(row[insight.inductor.value])]
