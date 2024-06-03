import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import pickle
import gower
from impute.utils import get_variables_description, load_trace
from LLMs.llm import LLM


def imputation_prompt(target, row, descriptions, selected_vars, selected_insights, neighbors):
    """Generate a prompt for imputing missing values based on provided insights and nearest neighbors."""
    messages = []

    user_input = f"""You are tasked with analyzing clinical data to impute the missing value for the target variable {target} in a birth health record based on provided statistical insights, through detailed analysis and logical reasoning. Some selected variables have been found to be more correlated to the target variable {target} than the others. You will use these specific insights and will analyze the nearest neighbors to the patient record that contains the missing values."""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "This sounds good. Can you provide the birth record that contains the missing values?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""### Birth Record Data:
    This is the original patient row, which contains missing values. This will be your point of reference whenever you try to draw conclusions.
    {row}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Thank you. Can you briefly explain what the variables mean in the birth record?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""### Description of the Data:
    Use the following descriptions to make sense of the values in the birth record and the insights provided.
    {descriptions}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = f"Thank you. Can you now tell me which variables were selected that are correlated to the target {target}?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""### Selected Variables:
    {selected_vars}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Thank you. Can you now provide insights regarding how these selected variables are distributed with respect to the target variable {target} in the dataset?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""### Selected Variable Insights:
    This section provides statistical insights of the selected variables based on their specific value in the birth record with respect to the target variable, within the WHOLE DATASET:
    {"\n".join(selected_insights)}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Thank you. Can you provide the nearest neighbors to the birth record so I can have an idea of a likely value?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""These are the 10 closest records to the birth record. Use them and compare them with the target record to see if you can draw interesting conclusions:
    {neighbors[0][0]}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Thank you. Is there anything I should pay attention to or guidance I should follow to decide on the imputed value?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = """
    Considering all these insights, your task is to:
    1. Analyze the specific values of the birth record.
    2. Assess whether the mom breastfed her baby (1) or didn't (0), given the insights provided.
    3. Use the statistical insights to understand patterns and relate them to the specific values of the birth record.
    4. Analyze the nearest neighbors.
    5. Construct a concise and succinct logical reasoning for your assessment, using the specific values from the birth record.
    6. Do not go through each insight one by one. Instead, build a high-level understanding and extract the most relevant elements.
    7. Report concisely your output inside a JSON object with the following format:
    {{
    'suggested_value': '0 or 1',
    'reasoning': 'Detailed reasoning based on the data and insights provided.'
    }}
    I insist, you need to return your output as a JSON object with the format:
    {{
    'suggested_value': '0 or 1',
    'reasoning': 'Detailed reasoning based on the data and insights provided.'
    }}"""
    messages.append({'role': 'user', 'content': user_input})

    return messages


def correction_prompt(target, row, minority, selected_vars, descriptions, neighbors, intermediate):
    """Generate a prompt for correcting imputation based on provided insights and multiple imputation scenarios."""
    messages = []

    user_input = f"""You are tasked with analyzing clinical data to decide whether or not a patient likely smokes by deciding on the final imputed value of the target variable {target} in a health record based on provided statistical insights, through detailed and logical reasoning. In other words, you are acting as an ensemble method. You will be provided with 3 alternative imputation values."""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Can you provide the patient record?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""{row}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Thank you. Can you briefly explain what the variables mean in the patient record?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""### Description of the Data:
    Use the following descriptions to make sense of the values in the patient record.
    {descriptions}"""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Thank you. Can you provide the 3 alternative imputation values?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""### Alternative Imputation Values:
    These values represent the outcomes from the 3 imputation strategies mentioned in the imputation process:
    Here is the minority choice is {minority}. As you can see, 2 columns are saying {1 - minority} and only 1 column says {minority}. So you will have to decide whether the minority choice of {minority} is correct or not."""
    messages.append({'role': 'user', 'content': user_input})

    assistant_prompt = "Ok, I get it. Is there anything I should pay attention to when deciding whether the minority choice of {minority} is right or not?"
    messages.append({'role': 'assistant', 'content': assistant_prompt})

    user_input = f"""Yes, you need to check the following key indicators before deciding whether the minority choice is right or not:
    - Indicator 1: people who smoke often are less than 30 in age
    - Indicator 2: people who smoke often give birth to babies who weigh less than 3000 grams
    - Indicator 3: people who smoke often have a low gestational age that is less than 39

    Finally, Indicator 4 is: how many people smoke amongst the 10 closest and 10 farthest records to the patient record in each of the 3 imputed scenarios:
    {target}_imputed_full:
    - Number of smokers amongst closest: {neighbors[0][0]}
    - Number of smokers amongst farthest: {neighbors[0][1]}
    {target}_imputed_full_selected:
    - Number of smokers amongst closest: {neighbors[1][0]}
    - Number of smokers amongst farthest: {neighbors[1][1]}
    {target}_imputed_cluster:
    - Number of smokers amongst closest: {neighbors[2][0]}
    - Number of smokers amongst farthest: {neighbors[2][1]}

    You can consider that this indicator supports that the patient is smoking if, in at least 2 of the scenarios, the number of smokers in the closest records is superior to the number of smokers in the farthest records. Otherwise, the person is likely not a smoker.

    I insist, if 2 or more of the 4 indicators support that the person smokes, you SHOULD say that the person smokes. Otherwise, you should say that the person doesn't smoke.

    Considering these insights, your task is to:
    1. Analyze the provided intermediate values and the specific values of the patient record.
    2. Assess whether the patient smokes (1) or doesn't (0), given the insights provided.
    4. If 2 or more of the 4 indicators provide support that the person smokes, this is enough evidence to say that the person smokes, otherwise the person doesn't smoke.
    5. Construct a concise and succinct logical reasoning for your assessment, using the specific values from the patient record.
    6. Report concisely your output inside a JSON object with the following format:
    {{
    'suggested_value': '0 or 1',
    'reasoning': 'Detailed reasoning based on the data and insights provided.'
    }}"""
    messages.append({'role': 'user', 'content': user_input})

    return messages


class LLMImputer:
    def __init__(self, target, trace):
        """Initialize the LLMImputer with necessary variables and data."""
        self.cont_vars = ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_AGE_NC', 'APGAR_1_NC', 'APGAR_2_NC',
                          'PREV_STILLBIRTH_NC', 'PREV_LIVE_BIRTHS_NC', 'TOT_BIRTH_NUM_NC', 'BIRTH_ORDER_NC']
        self.trace = load_trace(trace)
        self.target = target
        self.imputed_values = pd.DataFrame(dict(self.trace[self.target]['values'])).round().astype(int)
        self.missing_data = pd.read_csv('./results/CSVs/missing_data.csv')
        self.missing_data.set_index('Unnamed: 0', inplace=True)
        self.descriptions = get_variables_description(self.missing_data)
        self.mask = self.missing_data.isna()[self.target]
        self.cluster_imputed = pd.DataFrame(dict(self.trace[self.target]['clustering_imputed']))
        self.llm = LLM(model_name='phi3', seed=0)

    def clean_LSOA(self, imputed):
        """Clean LSOA data for imputed values."""
        unique_codes = self.missing_data['LSOA_CD_BIRTH_NC'].dropna().unique().astype(int)
        closest = []
        for val in imputed['LSOA_CD_BIRTH_NC']:
            differences = np.abs(unique_codes - val)
            closest_index = np.argmin(differences)
            closest.append(unique_codes[closest_index])
        imputed['LSOA_CD_BIRTH_NC'] = closest
        return imputed

    def llm_ensemble(self):
        """Run ensemble method using LLM for imputation."""
        num_exp = 10000
        df = self.imputed_values.copy()
        chosen_indices = np.random.choice(list(df.index), size=num_exp, replace=False)
        outputs = []
        for index in chosen_indices:
            imputed_values = self.imputed_values.loc[index].to_dict()
            truth = imputed_values[f'truth_{self.target}']
            combined = imputed_values[f'combined_{self.target}']
            prompt = self.get_correction_prompt(index, 1)
            raw_output = self.llm.impute(prompt)
            output = {'output': raw_output, 'truth': truth, 'combined': combined}
            outputs.append(output)
        with open('./results/llm_ensemble.pkl', 'wb') as file:
            pickle.dump(outputs, file)

    def explainable_impute(self):
        """Run explainable imputation using LLM."""
        num_exp = 1000
        copy = self.imputed_values.copy()
        chosen_indices = np.random.choice(list(copy.index), size=num_exp, replace=False)
        outputs = []
        for index in chosen_indices:
            imputed_values = self.imputed_values.loc[index].to_dict()
            truth = imputed_values[f'truth_{self.target}']
            combined = imputed_values[f'combined_{self.target}']
            prompt = self.get_imputation_prompt(index)
            raw_output = self.llm.impute(prompt)
            output = {'output': raw_output, 'truth': truth, 'combined': combined}
            outputs.append(output)
        with open('./results/llm_breastfed_impute_wrong.pkl', 'wb') as file:
            pickle.dump(outputs, file)

    def get_correction_prompt(self, index, minority):
        """Generate correction prompt based on imputation values and nearest neighbors."""
        trace = self.trace[self.target]
        selected_variables = trace['selected_vars']
        cluster_label = trace['clusters'][index]
        imputed_values = self.imputed_values.loc[index].to_dict()
        imputed_values.pop(f'truth_{self.target}')
        imputed_values.pop(f'combined_{self.target}')
        original_row = self.missing_data.loc[index]
        clustered_imputed = self.imputed_values.copy()
        clusters_series = pd.Series(trace['clusters'])
        clustered_imputed[self.target] = self.missing_data[self.target].copy()
        clustered_imputed['CLUSTER'] = clusters_series
        cluster_data = clustered_imputed[clustered_imputed['CLUSTER'] == cluster_label]
        cluster_data = cluster_data.drop(columns=['CLUSTER'])
        neighbors = self.find_k_nearest_neighbors(original_row, selected_variables, cluster_data)
        prompt = correction_prompt(self.target, original_row.to_dict(), minority, selected_variables, self.descriptions,
                                   neighbors, imputed_values)
        return prompt

    def get_imputation_prompt(self, index):
        """Generate imputation prompt based on insights and nearest neighbors."""
        trace = self.trace[self.target]
        original_row = self.missing_data.loc[index]
        selected_variables = trace['selected_vars']
        neighbors = self.find_k_nearest_neighbors(original_row, selected_variables, modes=['selec'])
        selec_insights, _, _ = self.generate_insights(original_row.to_dict(), selected_variables)
        prompt = imputation_prompt(self.target, original_row, self.descriptions, selected_variables, selec_insights,
                                   neighbors)
        return prompt

    def find_k_nearest_neighbors(self, row, selec=None, cluster_data=None, modes=['full', 'selec', 'cluster'],
                                 k=10):
        """Find k nearest neighbors based on Gower distance."""
        k_neighbors = []
        for mode in modes:
            print(f'Mode {mode}')
            if mode == 'selec':
                cols_to_keep = [col for col in selec if pd.notna(row[col]) and col != 'LSOA_CD_BIRTH_NC'] + [
                    self.target]
                df = self.missing_data[cols_to_keep].dropna()
            elif mode == 'full':
                cols_to_keep = [col for col in list(self.missing_data.columns) if
                                pd.notna(row[col]) and col != 'LSOA_CD_BIRTH_NC'] + [self.target]
                df = self.missing_data[cols_to_keep].dropna()
            else:
                cols_to_keep = [col for col in list(cluster_data.columns) if
                                pd.notna(row[col]) and col != 'LSOA_CD_BIRTH_NC'] + [self.target]
                df = cluster_data[cols_to_keep].dropna()
            distances = gower.gower_matrix(df.drop(columns=[self.target]).to_numpy(),
                                           row.loc[cols_to_keep].drop(self.target).to_numpy().reshape(1, -1))
            df['dist'] = distances[:, 0]
            nearest = df.nsmallest(k, 'dist')
            farthest = df.nlargest(k, 'dist')
            k_neighbors.append((nearest[self.target].sum(), farthest[self.target].sum()))
        return k_neighbors

    def get_distribution(self, data, target):
        """Get distribution summary of a target variable."""
        description = f"Distribution Summary of {target}: \n\n"
        if target in self.cont_vars:
            description += f"Mean: {data[target].mean():.2f}, Std Dev: {data[target].std():.2f}, \n"
            description += f"Median: {data[target].median():.2f}, \n"
            description += f"Minimum: {data[target].min():.2f}\n"
            description += f"Max: {data[target].max():.2f}\n"
        else:
            counts = data[target].value_counts(normalize=True)
            description += ", ".join(
                [f"Proportion of category {index}: {proportion:.2f}" for index, proportion in counts.items()])
        return description

    def generate_insights(self, row, selected_vars, cluster_label=None, clustered_data=None):
        """Generate statistical insights for the selected variables."""
        excluded = ['LSOA_CD_BIRTH_NC', f'{self.target}_missing']
        selected_to_analyse = [var for var in selected_vars if
                               var not in excluded and (var.endswith('missing') or pd.notna(row[var]))]
        selected_insights = []
        target = self.target

        missing_data = self.missing_data.copy()
        conditions = (missing_data['GEST_AGE_NC'] <= 50) & (missing_data['BIRTH_WEIGHT_NC'] >= 400) & (
                    missing_data['BIRTH_WEIGHT_NC'] <= 8000)
        missing_data = missing_data[conditions]

        for var in selected_to_analyse:
            data = missing_data.copy().dropna(subset=[var, target])
            selected_insights.append(self.extract_insights(var, self.target, data, row))

        others_to_analyse = [var for var in list(self.missing_data.columns) if
                             var not in excluded and not var.endswith('missing')]
        others_insights = []
        for var in others_to_analyse:
            data = missing_data.copy().dropna(subset=[var, target])
            others_insights.append(self.extract_insights(var, self.target, data, row))

        cluster_insights = []
        if cluster_label is not None:
            cluster_data = clustered_data[clustered_data['CLUSTER'] == cluster_label]
            cluster_data = cluster_data.drop(columns=['CLUSTER'])
            nan_indices = cluster_data[cluster_data[self.target].isna()].index
            target_missingness = len(nan_indices) / len(cluster_data)

            cluster_vars_to_analyse = [var for var in list(cluster_data.columns) if
                                       var not in excluded and (var.endswith('missing') or pd.notna(row[var]))]
            small = target_missingness > 0.9 or len(cluster_data) < 150
            for var in cluster_vars_to_analyse:
                if small:
                    data = cluster_data.copy().dropna(subset=[var])
                    insight = self.get_distribution(var)
                else:
                    data = cluster_data.copy().dropna(subset=[var, target])
                    insight = self.extract_insights(var, self.target, data, row)
                cluster_insights.append(insight)

        return selected_insights, others_insights, cluster_insights

    def extract_insights(self, inductor, induced, data, row):
        """Extract detailed insights for the given variables."""
        opposite = (inductor in self.cont_vars and induced not in self.cont_vars) or (
                    inductor not in self.cont_vars and induced in self.cont_vars)
        inductor_value = row[inductor]

        if opposite:
            if inductor in self.cont_vars:
                binned_var = inductor
                category_var = induced
                analysis_df = data.copy()
                quantiles = pd.qcut(analysis_df[binned_var], q=3, duplicates='drop', retbins=True)
                bin_edges = quantiles[1]
                if len(bin_edges) < 3:
                    bin_edges = np.linspace(analysis_df[binned_var].min(), analysis_df[binned_var].max(), num=4)
                labels = [f"{np.round(bin_edges[i], 2)} to {np.round(bin_edges[i + 1], 2)}" for i in
                          range(len(bin_edges) - 1)]
                analysis_df[f'{binned_var}_bins'] = pd.cut(analysis_df[binned_var], bins=bin_edges, labels=labels,
                                                           include_lowest=True)
                specific_bin_label = pd.cut([inductor_value], bins=bin_edges, labels=labels, include_lowest=True)[0]
                stats = analysis_df.groupby(category_var)[binned_var].agg(
                    ['mean', 'std', 'min', 'max', '10th Percentile', '25th Percentile', '75th Percentile',
                     '90th Percentile'])
                stats.columns = ['mean', 'std', 'min', 'max', '10th Percentile', '25th Percentile', '75th Percentile',
                                 '90th Percentile']
                description = f"Statistical analysis for {binned_var} across categories of {category_var}:\n"
                for cat in stats.index:
                    description += f"{cat}:\n"
                    description += f" - Mean: {stats['mean'][cat]:.2f}\n"
                    description += f" - Std Dev: {stats['std'][cat]:.2f}\n"
                    description += f" - Min: {stats['min'][cat]:.2f}\n"
                    description += f" - Max: {stats['max'][cat]:.2f}\n"
                    description += f" - 10th Percentile: {stats['10th Percentile'][cat]:.2f}\n"
                    description += f" - 25th Percentile: {stats['25th Percentile'][cat]:.2f}\n"
                    description += f" - 75th Percentile: {stats['75th Percentile'][cat]:.2f}\n"
                    description += f" - 90th Percentile: {stats['90th Percentile'][cat]:.2f}\n"
                description += "\n"
                grouped = analysis_df.groupby([f'{binned_var}_bins', category_var]).size().unstack(fill_value=0)
                binned_stats_normalized = grouped.div(grouped.sum(axis=1), axis=0)
                specific_bin_stats = binned_stats_normalized.loc[specific_bin_label]
                description += f"Distribution of categories of {induced} within bin '{specific_bin_label}' of {inductor}:\n"
                description += "\n".join([f"Category {cat}: {prop:.2f}" for cat, prop in specific_bin_stats.items()])
                description += "\n\n"
            else:
                binned_var = induced
                category_var = inductor
                analysis_df = data.copy()
                quantiles = pd.qcut(analysis_df[binned_var], q=3, duplicates='drop', retbins=True)
                bin_edges = quantiles[1]
                labels = [f"{np.round(bin_edges[i], 2)} to {np.round(bin_edges[i + 1], 2)}" for i in
                          range(len(bin_edges) - 1)]
                analysis_df[f'{binned_var}_bins'] = pd.qcut(analysis_df[binned_var], q=3, labels=labels,
                                                            duplicates='drop')
                specific_category_value = inductor_value
                description = f"Statistical analysis for {category_var} = {specific_category_value} across bins of {binned_var}:\n"
                stats = analysis_df.groupby(f'{binned_var}_bins', observed=True)[category_var].value_counts(
                    normalize=True).unstack(fill_value=0)
                specific_stats = stats[specific_category_value]
                for bin_label in specific_stats.index:
                    description += f" - Bin '{bin_label}': {specific_stats[bin_label]:.2f}\n"
                description += "\n"
        elif inductor in self.cont_vars:
            analysis_df = data.copy()
            correlation, _ = pearsonr(analysis_df.dropna(subset=[induced, inductor])[induced],
                                      analysis_df.dropna(subset=[induced, inductor])[inductor])
            model = LinearRegression().fit(analysis_df[[inductor]].dropna(), analysis_df[induced].dropna())
            description = f"The correlation between {induced} and {inductor} is {correlation:.2f}. "
            description += f"For the specific value of {inductor} ({inductor_value}), the linear regression prediction of {induced} is: {model.predict([[inductor_value]])[0]:.2f}\n"
            description += "\n"
        else:
            proportions = (data.groupby([induced, inductor]).size() / data.groupby(inductor).size()).unstack()
            specific_proportions = proportions.loc[:, inductor_value]
            description = f"Proportion statistics for category {inductor_value} of {inductor} by category of {induced}:\n"
            description += ", ".join([f"Category {cat}: {value:.2f}" for cat, value in specific_proportions.items()])
            description += "\n\n"

        return description

