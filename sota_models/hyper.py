import numpy as np
from hyperimpute.plugins.imputers import Imputers
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, mean_absolute_error, \
    mean_squared_error


def plot_model_comparisons(results):
    # Determine the number of unique columns and metrics to set up the plot dimensions
    columns = set()
    for model_data in results.values():
        for col in model_data.keys():
            columns.add(col)

    # Convert columns to list to maintain order in plotting
    columns = list(columns)

    # Loop over each column to create a subplot for each metric
    for column in columns:
        metrics = set()
        for model_data in results.values():
            if column in model_data:
                metrics.update(model_data[column].keys())

        metrics = list(metrics)
        num_metrics = len(metrics)

        # Create a figure for each column
        fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5), constrained_layout=True)
        fig.suptitle(f'Performance Comparison for Column: {column}', fontsize=16)

        if num_metrics == 1:
            axs = [axs]  # Make it iterable

        # Plot each metric in a subplot
        for i, metric in enumerate(metrics):
            ax = axs[i]
            model_names = []
            means = []
            errors = []

            # Collect data for each model
            for model, model_data in results.items():
                if column in model_data and metric in model_data[column]:
                    model_names.append(model)
                    means.append(model_data[column][metric]['mean'])
                    errors.append(model_data[column][metric]['std'])

            # Error bar plot
            x_pos = np.arange(len(model_names))
            ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.7, capsize=10)
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.set_title(f'Metric: {metric}')

    plt.show()


class HyperImputer:
    def __init__(self, missing_data, true_data, method):
        self.missing_data = missing_data.drop(columns=['MAT_REGION_NC'])
        self.true_data = true_data.drop(columns=['MAT_REGION_NC'])
        self.mask = self.get_mask()
        self.method = method
        self.imputers = Imputers()
        self.targets = self.get_targets()
        self.cont_vars = ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_AGE_NC', 'APGAR_1_NC', 'APGAR_2_NC']
        self.class_threshold = self.get_class_threshold(self.true_data)
        self.args = self.get_args(self.class_threshold)
        self.plugin = Imputers().get(method, **self.args)
        self.imputed = None

    def get_targets(self):
        new_missing_data = self.missing_data.drop(columns=['subpopulation'])
        columns_with_na = new_missing_data.columns[new_missing_data.isna().any()].tolist()
        print("Columns with missing values: ", columns_with_na)
        return columns_with_na

    def get_class_threshold(self, data):
        max_unique = 0  # Initialize the maximum count to zero
        for column in data.columns:
            if column not in self.cont_vars:
                unique_count = data[column].nunique()
                if unique_count > max_unique:
                    max_unique = unique_count
        print("Class threshold: ", max_unique)
        return max_unique + 1

    def get_args(self, class_threshold=2):
        if self.method == "hyperimpute":
            return {
                "class_threshold": class_threshold,
                "n_inner_iter": 40,
                "select_model_by_column": True,
                "select_model_by_iteration": True,
                "select_lazy": True,
                "select_patience": 5,
                "random_state": 0,
                "optimizer": "hyperband",
                "baseline_imputer": 0,
            }
        else:
            return {}

    def explain(self):
        trace = self.parse_and_save_trace()
        print("Models selected for each imputed variable: ", trace)

    def impute(self):
        out = self.plugin.fit_transform(self.missing_data.copy())
        out.columns = self.missing_data.columns
        clean_out = self.clean_imputed(out)
        self.imputed = clean_out
        return clean_out

    def clean_imputed(self, imputed):
        for col in imputed.columns:
            if col not in self.cont_vars:
                imputed[col] = imputed[col].round().astype(int)
        return imputed

    def list_imputers(self):
        imputers = self.imputers.list()
        print("Imputers available: ", imputers)
        return imputers

    def get_models(self):
        return self.plugin.models()

    def save_performance(self):
        self.imputed.to_csv(f'./imputed/{self.method}_imputed.csv')

    def get_mask(self):
        mask = self.missing_data.isna()
        return mask

    def subpopulation_impute(self):
        # TODO: Fix the convergence error
        # TODO: Extract explainability information.
        # TODO: Hyperparameter tuning: Type of optimizer, baseline imputers, model pool...

        all_subpops = set()
        sublists = self.missing_data['subpopulation'].apply(
            lambda x: x.split(',') if pd.notna(x) else [])
        for sublist in sublists:
            all_subpops.update(sublist)

        subpopulation_results = {}
        imputed_indices = set()  # To track which rows have been successfully imputed

        min_size = 3000  # Minimum number of rows to fit a model

        for subpop in all_subpops:
            if subpop:
                indices = self.missing_data['subpopulation'].apply(lambda x: subpop in x.split(',') if pd.notna(x) else False)
                sub_data = self.missing_data[indices].copy()
                sub_data.drop(columns=['subpopulation'], inplace=True)
                if len(sub_data) >= min_size:
                    print(f'Length of subpopulation {subpop}: ', len(sub_data))
                    class_threshold = self.get_class_threshold(sub_data)
                    args = self.get_args(class_threshold)
                    self.plugin = self.imputers.get(self.method, **args)
                    imputed_data = self.plugin.fit_transform(sub_data.copy())
                    imputed_data.columns = sub_data.columns
                    cleaned_imputed = self.clean_imputed(imputed_data)
                    subpopulation_results[subpop] = cleaned_imputed
                    imputed_indices.update(sub_data.index)

        # remaining_indices = set(self.missing_data.index) - imputed_indices
        # remaining_data = self.missing_data.loc[list(remaining_indices)].copy()
        #
        # if not remaining_data.empty:
        #     if len(remaining_data) < min_size:
        #         remaining_data = self.missing_data.copy()  # Use whole dataset if not enough remaining data

        self.missing_data = self.missing_data.drop(columns=['subpopulation'])
        class_threshold = self.get_class_threshold(self.missing_data)
        args = self.get_args(class_threshold)
        self.plugin = self.imputers.get(self.method, **args)
        general_imputed = self.plugin.fit_transform(self.missing_data.copy())
        general_imputed.columns = self.missing_data.columns
        clean_general_imputed = self.clean_imputed(general_imputed)
        subpopulation_results['general'] = clean_general_imputed

        self.imputed = clean_general_imputed
        print("Full Imputation performance: ", self.compute_imputation_metrics())
        self.plot_performance()
        # Weigh and combine results
        self.imputed = self.weight_and_combine(subpopulation_results)
        # TODO: Final check that all missing data has indeed been imputed
        print("Combined Imputation performance: ", self.compute_imputation_metrics())
        self.plot_performance()
        self.save_performance()  # Saving final imputed data

    def weight_and_combine(self, subpopulation_results):
        # Initialize a DataFrame with original data to preserve non-imputed entries
        final_imputed = self.missing_data.copy()

        # Collect votes or values for each entry based on subpopulation results
        aggregation = {col: {i: [] for i in self.missing_data.index} for col in self.targets}
        cat_aggregation = {col: {i: {} for i in self.missing_data.index} for col in self.targets}

        weights = {subpop: len(results) for subpop, results in subpopulation_results.items() if subpop != 'general'}
        total_weights = sum(weights.values())
        general_weight = min(weights.values()) / 2

        for subpop, results in subpopulation_results.items():
            weight = weights.get(subpop,
                                 general_weight) / total_weights  # Normalize weights, general imputation has general weight
            for col in self.targets:
                is_continuous = col in self.cont_vars  # Check if the variable is continuous
                for i in results.index:
                    is_missing = pd.isna(self.missing_data.loc[i, col])
                    val = results.loc[i, col]
                    if pd.notna(val) and is_missing:  # Ensure the value is not NaN
                        if is_continuous:
                            aggregation[col][i].append((val, weight))
                        else:
                            if val not in cat_aggregation[col][i].keys():
                                cat_aggregation[col][i][val] = weight
                            else:
                                cat_aggregation[col][i][val] += weight

        # Apply the majority vote or weighted mean for each missing entry
        for col in self.targets:
            is_continuous = col in self.cont_vars  # Check if the variable is continuous
            for i in self.missing_data.index:
                if pd.isna(self.missing_data.loc[i, col]):
                    if is_continuous and len(aggregation[col][i]) > 0:
                        weighted_sum = sum(val * weight for val, weight in aggregation[col][i])
                        total_weight = sum(weight for _, weight in aggregation[col][i])
                        final_imputed.loc[i, col] = weighted_sum / total_weight
                    elif len(cat_aggregation[col][i].keys()) > 0:
                        final_imputed.loc[i, col] = max(cat_aggregation[col][i], key=cat_aggregation[col][i].get)

        return final_imputed

    def benchmark_models(self, target_models):
        all_metrics = {}
        for model in target_models:
            self.method = model
            all_metrics[model] = []
            for seed in range(3):
                self.args = self.get_args(self.class_threshold)
                self.args['random_state'] = seed
                self.plugin = Imputers().get(model, **self.args)
                self.impute()
                self.save_performance()
                metrics = self.compute_imputation_metrics()
                all_metrics[model].append(metrics)

        # To store the results with averages and standard deviations
        results = {}

        # Process each model's collected metrics
        for model, metrics_list in all_metrics.items():
            results[model] = {}
            # Initialize a dictionary to store data for each column
            columns_data = {}

            # Process each seed's results
            for metrics in metrics_list:
                for column, metrics_dict in metrics.items():
                    if column not in columns_data:
                        columns_data[column] = {metric: [] for metric in metrics_dict.keys()}
                    # Append each metric value in the respective list
                    for metric, value in metrics_dict.items():
                        columns_data[column][metric].append(value)

            # Calculate mean and standard deviation for each metric in each column
            for column, metrics_dict in columns_data.items():
                results[model][column] = {}
                for metric, values in metrics_dict.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    results[model][column][metric] = {'mean': mean_val, 'std': std_val}

        plot_model_comparisons(results)

        with open('benchmark_results.pkl', 'wb') as file:
            pickle.dump(results, file)
        return results

    def parse_and_save_trace(self):
        trace = self.plugin.trace()
        models = trace['models']
        last_model_used = {}
        for variable, model_list in models.items():
            last_model_used[variable] = model_list[-1]
        with open('trace.pkl', 'wb') as file:
            pickle.dump(last_model_used, file)
        return last_model_used

    def compute_imputation_metrics(self, beta=2.0):
        metrics = {}
        for col in self.targets:
            mask = self.mask[col]
            is_cat = col not in self.cont_vars

            true_col = self.true_data[col][mask]
            imputed_col = self.imputed[col][mask]
            ws_score = wasserstein_distance(np.asarray(true_col), np.asarray(imputed_col))

            if is_cat:
                accuracy = accuracy_score(true_col, imputed_col)
                precision = precision_score(true_col, imputed_col, average='macro', zero_division=0)
                recall = recall_score(true_col, imputed_col, average='macro', zero_division=0)
                fbeta = fbeta_score(true_col, imputed_col, average='macro', zero_division=0, beta=beta)

                metrics[col] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'fbeta': fbeta,
                    'ws_score': ws_score
                }

            else:
                mae = mean_absolute_error(true_col, imputed_col)
                rmse = mean_squared_error(true_col, imputed_col)
                metrics[col] = {'mae': mae, 'rmse': rmse, 'ws_score': ws_score}

        return metrics

    def plot_performance(self):
        # Check for categorical variables
        for var in self.targets:
            if var not in self.cont_vars:
                fig, ax = plt.subplots()
                # Calculate value counts for both imputed and true data
                true_counts = self.true_data[var][self.mask[var]].value_counts().sort_index()
                imputed_counts = self.imputed[var][self.mask[var]].value_counts().sort_index()

                # Create a DataFrame to hold counts for easier plotting
                df_counts = pd.DataFrame({'True': true_counts, 'Imputed': imputed_counts})
                df_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Comparison of {var} Counts')
                ax.set_xlabel(var)
                ax.set_ylabel('Counts')
                plt.show()

            else:
                fig, ax = plt.subplots()
                true_values = self.true_data[var][self.mask[var]]
                imputed_values = self.imputed[var][self.mask[var]]

                ax.plot(true_values.index, true_values, label='True', marker='o', linestyle='-')
                ax.plot(imputed_values.index, imputed_values, label='Imputed', marker='x', linestyle='--')
                ax.set_title(f'Comparison of {var} Values (Imputed Only)')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                plt.show()

# TODO: Add AUROC metric for binary classification. Have to differentiate between binary and categoricial. And ths requires also getting the probabilities of the classes. Look into the hyperimpute code.
