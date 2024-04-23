import numpy as np
import pandas as pd
import re
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from clinical_utils import (get_variables_description, create_induced_variables,
                            InductorVariable, Correlation, InducedVariableName)


def get_known_correlations(target):
    known_correlations = []
    if target == InducedVariableName.MS.value:
        corr = Correlation(InductorVariable.BW, InductorVariable.MS, None, 'This is a known correlation: Smoking during pregnancy is known to be correlated with lower birth weight.')
        known_correlations.append(corr)
    return known_correlations



def filter_text_by_target(text, target):
    # Initialize the filtered text with the header (assume the first section up to the first binned statistics mention is the header)
    header_end_index = text.find("Binned statistics for")
    filtered_text = text[:header_end_index].strip()

    # Extract the binned statistics portion by finding the section starting from "Binned statistics for"
    binned_stats_start_index = text.find("Binned statistics for")
    if binned_stats_start_index == -1:
        return filtered_text  # Return early if no binned stats are found

    binned_stats_section = text[binned_stats_start_index:]

    # Split the text into bin descriptions, ensuring to capture the header of the binned statistics section
    bins_descriptions = re.split(r'\n(?=\d+\.\d+ to \d+\.\d+:)', binned_stats_section)  # Splits each new bin line

    # Add the header for the binned statistics to filtered_text
    filtered_text += '\n\n' + bins_descriptions[0]  # Include the binned statistics header

    for description in bins_descriptions[1:]:  # Skip the first entry as it's the header
        # Regex to find weight ranges within each bin description
        weight_range_pattern = re.compile(r"(\d+\.\d+) to (\d+\.\d+):")
        match = weight_range_pattern.search(description)
        if match:
            lower_bound = float(match.group(1))
            upper_bound = float(match.group(2))
            # Check if the target value is within the bin range
            if lower_bound <= target <= upper_bound:
                filtered_text += '\n' + description  # Append only matching bin descriptions

    return filtered_text


def filter_text_by_cat(text, target_cat, cat_name):
    # Define the regex pattern to identify and preserve relevant lines
    # Matches lines with specific category details
    pattern = re.compile(fr'\s+- {cat_name} {target_cat}: \d+\.\d+')

    # Split the text into sections based on category headers
    category_sections = text.split('\n- Category ')

    # Initialize result text with the header (first line)
    result_text = category_sections[0].strip() + '\n'

    for section in category_sections[1:]:
        if section.strip() == "":  # Skip empty sections
            continue

        # Extract the category number and the rest of the section content
        category_number = section.split(":")[0].strip()
        section_content = section[len(category_number) + 1:].strip()

        # Append the category header back to results
        result_text += f"- Category {category_number}:\n"

        # Find all matching proportions and rebuild the section if matches are found
        matches = pattern.findall(section)
        if matches:
            for match in matches:
                # Remove leading spaces for cleaner formatting
                result_text += match.strip() + '\n'
        else:
            # If no matches, remove the category section
            result_text = result_text.rsplit('\n', 1)[0]  # Remove the last line which is the category header

    return result_text


class BirthInsights:
    def __init__(self, data):
        self.df = data
        self.cont_vars = ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_AGE_NC', 'APGAR_1_NC', 'APGAR_2_NC']
        self.description = get_variables_description()
        self.categorize_rows()
        self.induced_variables = create_induced_variables()
        self.target_insights = self.create_target_insights()
        self.other_insights = self.create_subpopulation_insights()
        # print(self.target_insights['MAT_SMOKING_NC'][0].results)
        # print(self.target_insights['MAT_SMOKING_NC'][0].description)

    def categorize_rows(self):
        # TODO: Add region subpopulations

        CRITERIA = {
            "MS": lambda row: row['MAT_SMOKING_NC'] != np.nan and row['MAT_SMOKING_NC'] == 1,
            "BT": lambda row: row['BIRTH_ORDER_NC'] != np.nan and row['BIRTH_ORDER_NC'] == 2,
            "PS": lambda row: row['PREV_STILLBIRTH_NC'] != np.nan and row['PREV_STILLBIRTH_NC'] == 1,
        }

        def categorize_row(row):
            categories = [name for name, condition in CRITERIA.items() if condition(row)]
            return ','.join(categories) if categories else 'None'

        self.df['subpopulation'] = self.df.apply(categorize_row, axis=1)

    def extract_insights(self, inductor, induced, data):
        opposite = (inductor in self.cont_vars and induced not in self.cont_vars) or (
                inductor not in self.cont_vars and induced in self.cont_vars)
        # Calculate and store proportions of each target category

        if opposite:
            binned_var = inductor if inductor in self.cont_vars else induced
            category_var = induced if inductor in self.cont_vars else inductor
            analysis_df = data.copy()
            total_counts = analysis_df[category_var].value_counts(normalize=True)

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

            description = f"Statistical analysis for {binned_var} across categories of {category_var}:\n"
            for index, row in stats.iterrows():
                description += f"Category {index}:\n"
                description += f"- Proportion (w.r.t total): {total_counts[index]:.2f} \n"
                description += f"- Mean: {row['mean']:.2f}\n"
                description += f"- Standard Deviation: {row['std']:.2f}\n"
                description += f"- Median: {row['median']:.2f}\n"
                # description += f"- Minimum: {row['min']:.2f}\n"
                # description += f"- Maximum: {row['max']:.2f}\n"
                # description += f"- 10th Percentile: {row['10th Percentile']:.2f}\n"
                # description += f"- 25th Percentile: {row['25th Percentile']:.2f}\n"
                # description += f"- 75th Percentile: {row['75th Percentile']:.2f}\n"
                # description += f"- 90th Percentile: {row['90th Percentile']:.2f}\n\n"

            # Aggregate data within bins
            grouped = analysis_df.groupby([f'{binned_var}_bins', category_var])[binned_var]
            binned_stats = grouped.size().unstack(fill_value=0)
            binned_stats_normalized = binned_stats.div(binned_stats.sum(axis=1), axis=0)
            top_bin = binned_stats_normalized.idxmax().to_dict()  # Get the bin with the highest proportion of data for each category of the target variable

            results = {'Binned Statistics': binned_stats_normalized.to_dict(orient='index'),
                       'Stats': stats.to_dict(orient='index')}

            description += f'Binned statistics for {binned_var} across categories of {category_var}:\n'

            for (bin_label, sub_df) in binned_stats_normalized.iterrows():
                categories_descriptions = ", ".join(f"Category {cat}: {prop:.2f}" for cat, prop in sub_df.items())
                description += f"{bin_label}: {categories_descriptions}\n"

            description += "Bins with highest proportion per category:\n"
            for category, bin_label in top_bin.items():
                description += f"- Category {category} mostly in {bin_label}\n"

        elif inductor in self.cont_vars:
            analysis_df = data.copy()
            correlation, _ = pearsonr(analysis_df.dropna(subset=[induced, inductor])[induced],
                                      analysis_df.dropna(subset=[induced, inductor])[inductor])

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
            induced_counts = data[induced].value_counts(normalize=True)
            inductor_counts = data[inductor].value_counts(normalize=True)

            proportions = (data.groupby([induced, inductor]).size() / data.groupby(
                induced).size()).unstack().fillna(0)
            results = {'Proportions': proportions.to_dict()}
            description = f"Proportion statistics for each category of {induced} by {inductor}:\n"

            for category in proportions.index:
                description += f"- Category {category} (total proportion is {induced_counts[category]:.2f}):\n"
                for sub_category, value in proportions.loc[category].items():
                    description += f"  - {inductor} {sub_category}: {value:.2f}\n"
            # description = f"For each category of {induced}, here are the statistics:\n\n"
            # for category in proportions.index:
            #     description += f"- Category {category}:\n"
            #     description += f"  - Proportion of {category} with respect to total: {induced_counts[category]:.2f} \n"
            #     description += f"  - Proportions of each category of {inductor}:\n"
            #     for sub_category, value in proportions.loc[category].items():
            #         description += f"    - Proportion of {inductor} category {sub_category}: {value:.2f}\n"
            #         # description += f"    - Proportion of {inductor} category {sub_category} with respect to total: {inductor_counts[sub_category]:.2f}\n"
            #     description += "\n"

        corr = Correlation(inductor, induced, results, description)
        return corr

    def create_target_insights(self):
        results = {}

        for target in self.induced_variables:
            correlated_vars = target.correlated_variables
            results[target.name] = []

            for var in correlated_vars:
                relevant_cols = [target.name] + [var]
                analysis_df = self.df.dropna(subset=relevant_cols)
                corr = self.extract_insights(var, target.name, analysis_df)
                results[target.name].append(corr)

        return results

    def create_subpopulation_insights(self):
        known_pairs = [(InductorVariable.PS, InductorVariable.BW), (InductorVariable.BT, InductorVariable.BW)]
        results = {}
        for pair in known_pairs:
            inductor, induced = pair
            if results.get(inductor.value) is None:
                results[inductor.value] = []

            analysis_df = self.df.dropna(subset=[inductor.value, induced.value])

            corr = self.extract_insights(inductor.value, induced.value, analysis_df)
            results[inductor.value].append(corr)
        return results

    def get_subpopulation_insights(self, subs):
        sub_insights = []
        for sub in subs.split(','):
            if sub == 'BT':
                sub_insights.extend(self.other_insights[InductorVariable.BT.value])
            if sub == 'PS':
                sub_insights.extend(self.other_insights[InductorVariable.PS.value])
        return sub_insights

    def get_full_insights(self, row, target):
        target_insights = [self.refine_insight(insight, row) for insight in self.target_insights[target] if pd.notna(row[insight.inductor])]
        sub_insights = [self.refine_insight(insight, row) for insight in self.get_subpopulation_insights(row['subpopulation']) if pd.notna(row[insight.inductor])]
        known_correlations = [corr.description for corr in get_known_correlations(target)]
        return target_insights, sub_insights, known_correlations

    def refine_insight(self, insight, row):
        if insight.inductor in self.cont_vars and insight.induced not in self.cont_vars:
            insight.description = filter_text_by_target(insight.description, row[insight.inductor])
        elif insight.inductor not in self.cont_vars and insight.induced in self.cont_vars:
            insight.description = filter_text_by_target(insight.description, row[insight.induced])
        elif insight.inductor not in self.cont_vars and insight.induced not in self.cont_vars:
            insight.description = filter_text_by_cat(insight.description, row[insight.inductor], insight.inductor)
        return insight.description



