import numpy as np
from hyperimpute.plugins.imputers import Imputers
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, fbeta_score, mean_absolute_error, davies_bouldin_score, \
    calinski_harabasz_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import RandomUnderSampler
import gower
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, \
    RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import logging


def get_n_components(variances):
    """Calculate number of components to reach 95% variance."""
    cum_var = 0
    n = 0
    for i, variance in enumerate(variances):
        cum_var += variance
        if cum_var >= 0.95:
            n = i + 1
            break
    return n


def plot_k_distance_graph(X, k):
    """Plot k-distance graph for determining optimal number of clusters."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances[:, k - 1], axis=0)
    plt.plot(distances)
    plt.show()


def visualise_var_across_clusters(var, data):
    """Visualize variable distribution across clusters."""
    means = data.groupby('CLUSTER')[var].mean()
    print(f'Means of {var} for each cluster:', means)
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='CLUSTER', y=var, data=data)
    plt.title(f'{var} distribution per cluster')
    plt.show()


class HyperImputer:
    def __init__(self, missing_data, true_data, encoders, method, is_deprivation, is_mnar=False, selec='LASSO',
                 clust='important', random_state=42):
        self.random_state = random_state
        self.missing_data = missing_data.copy()
        self.missing_data_mnar = self.missing_data.copy()
        self.true_data = true_data.copy()
        self.encoders = encoders
        self.mask = self.get_mask()
        self.method = method
        self.imputers = Imputers()
        self.targets = ['BIRTH_WEIGHT_NC', 'BREASTFEED_BIRTH_FLG_NC', 'SMOKE_NC', 'GEST_AGE_NC', 'MAT_AGE_NC']
        self.cont_vars = ['BIRTH_WEIGHT_NC', 'GEST_AGE_NC', 'MAT_AGE_NC', 'APGAR_1_NC', 'APGAR_2_NC',
                          'PREV_STILLBIRTH_NC', 'PREV_LIVE_BIRTHS_NC', 'TOT_BIRTH_NUM_NC']
        self.imputed = None
        self.selec = selec
        self.mnarn = is_mnar
        self.clust = clust
        self.isdep = is_deprivation

    def get_args(self, size, class_threshold=3):
        """Get arguments for hyperimpute method."""
        if self.method == "hyperimpute":
            if size <= 1000:
                classifier_seeds = ['random_forest', 'logistic_regression']
                regressor_seeds = ['random_forest_regressor', 'linear_regression']
            else:
                classifier_seeds = ['xgboost', 'catboost']
                regressor_seeds = ['xgboost_regressor', 'catboost_regressor']
                # if torch.cuda.is_available():
                #     classifier_seeds.append("neural_nets")
                #     regressor_seeds.append("neural_nets_regression")

            return {
                "class_threshold": class_threshold,
                "n_inner_iter": 100,
                "select_model_by_column": True,
                "select_model_by_iteration": True,
                "select_lazy": True,
                "select_patience": 5,
                "classifier_seed": classifier_seeds,
                "regression_seed": regressor_seeds,
                "random_state": self.random_state,
                "optimizer": "hyperband",
                "baseline_imputer": 1,
            }
        else:
            return {"random_state": self.random_state}

    def deal_with_MNAR(self, data):
        """Handle Missing Not At Random data."""
        mnar_data = data.copy()
        for col in data.columns:
            mnar_data[f'{col}_missing'] = data[col].isna().astype(int)
        return mnar_data

    def find_optimal_clusters(self, X, method='KMeans', cat_cols=None):
        """Find optimal number of clusters using elbow method."""
        k_range = range(1, 11)
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='auto', random_state=self.random_state)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        inertias = np.array(inertias)

        # Normalize the inertia values to ensure a proper scale
        norm_inertias = (inertias - inertias.min()) / (inertias.max() - inertias.min())

        # Get coordinates of all points
        n_points = len(norm_inertias)
        all_coords = np.vstack((range(n_points), norm_inertias)).T

        # Get coordinates of the first point and the last point
        first_point = all_coords[0]
        last_point = all_coords[-1]

        # Calculate the line equation from first to last
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

        # Calculate the distance from each point to the line
        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel

        # Distance to line
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

        # The point with the maximum distance to the line is the elbow
        elbow_index = np.argmax(dist_to_line)
        optimal_k = k_range[elbow_index] + 1

        return optimal_k

    def impute(self, missing_data=None):
        """Perform data imputation."""
        data = missing_data.copy() if missing_data is not None else self.missing_data.copy()
        args = self.get_args(len(data))
        self.plugin = self.imputers.get(self.method, **args)
        out = self.plugin.fit_transform(data)
        out = out.set_index(data.index)
        out.columns = data.columns
        # clean_out = self.clean_imputed(out)
        # self.raw_imputed = out
        # if self.method == 'hyperimpute':
        #     self.parse_and_save_trace(title)
        self.imputed = out
        return out

    def benchmark_models(self, target_models):
        """Benchmark different imputation models."""
        # selectors=['LASSO', 'gbm']
        # clustering=['important', 'reduced']
        for model in target_models:
            logging.basicConfig(
                filename='debug_log.log',
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.method = model
            all_combinations = [('LASSO', 'reduced', False)]
            logging.info(f'Imputing with model {self.method}')
            for combination in all_combinations:
                select, clust, mnar = combination
                for seed in range(1):
                    logging.info(f'Imputing with seed {seed}')
                    self.random_state = seed
                    self.mnarn = mnar
                    self.clust = clust
                    self.selec = select
                    self.subpopulation_impute(clust, select)

    def clean_LSOA(self, imputed):
        """Clean LSOA codes in imputed data."""
        unique_codes = self.missing_data['LSOA_CD_BIRTH_NC'].dropna().unique().astype(int)
        closest = []
        for val in imputed['LSOA_CD_BIRTH_NC']:
            differences = np.abs(unique_codes - val)
            closest_index = np.argmin(differences)
            closest.append(unique_codes[closest_index])
        imputed['LSOA_CD_BIRTH_NC'] = closest
        return imputed

    def convert_LSOA(self):
        """Convert LSOA codes to original format."""
        self.imputed['LSOA_CD_BIRTH_NC'] = self.imputed['LSOA_CD_BIRTH_NC'].map(self.encoders['LSOA_CD_BIRTH_NC'])
        self.true_data['LSOA_CD_BIRTH_NC'] = self.true_data['LSOA_CD_BIRTH_NC'].map(self.encoders['LSOA_CD_BIRTH_NC'])

    def clean_imputed(self, imputed):
        """Clean imputed data."""
        for col in imputed.columns:
            if col not in self.cont_vars:
                imputed[col] = imputed[col].round().astype(int)
        if 'LSOA_CD_BIRTH_NC' in imputed.columns:
            imputed = self.clean_LSOA(imputed)
        return imputed

    def list_imputers(self):
        """List available imputers."""
        imputers = self.imputers.list()
        print("Imputers available: ", imputers)
        return imputers

    def save_performance(self, trace=None, custom_LSOA=False, custom_title=None):
        """Save imputed data and performance trace."""
        if custom_LSOA is True:
            self.imputed.to_csv('./results/csvs/custom_LSOA_imputed.csv')
        elif custom_title is not None:
            self.imputed.to_csv(f'./results/csvs/{custom_title}.csv')
        else:
            self.imputed.to_csv(
                f'./results/csvs/{self.method}_imputed{"_deprived" if self.isdep else ""}_{self.selec}_{self.clust}{"_MNAR" if self.mnarn else ""}_{self.random_state}.csv')

        if trace is not None:
            title = f'{custom_title}.pkl' if custom_title is not None else f'./results/tree_{self.method}_target_trace{"_deprived" if self.isdep else ""}_{self.selec}_{self.clust}'
            with open(title, 'wb') as file:
                pickle.dump(trace, file)

    def get_mask(self):
        """Get mask for missing data."""
        mask = self.missing_data.isna()
        return mask

    def get_best_folds(self, X, y, sub_model=None):
        """Get best number of folds for cross-validation."""
        folds = range(2, 11)
        mean_scores = []
        for k in folds:
            if sub_model is None:
                model = ElasticNetCV(cv=k, random_state=self.random_state)
            else:
                model = RFECV(estimator=sub_model, cv=k)
            scores = cross_val_score(model, X, y, cv=KFold(n_splits=k, shuffle=True))
            mean_scores.append(np.mean(scores))

        print('Best score on optimal fold: ', np.max(mean_scores))
        return folds[np.argmax(mean_scores)]

    def extract_lasso(self, data, target):
        """Extract features using LASSO."""
        scaler = RobustScaler()
        copy = data.copy()
        df_scaled = pd.DataFrame(scaler.fit_transform(copy), columns=data.columns)

        X = df_scaled.drop(target, axis=1)
        y = df_scaled[target]

        best_folds = 5
        # best_folds = self.get_best_folds(X, y)

        elastic = ElasticNetCV(cv=best_folds, random_state=self.random_state).fit(X, y)
        coef = np.abs(elastic.coef_)

        final_features = X.columns[coef > 0.]
        rankings = tuple(zip(list(X.columns), coef))
        return list(final_features), rankings

    def grid_search(self, X, y, target, method='gbm'):
        """Perform grid search for hyperparameter tuning."""
        if method == 'rf':
            model = RandomForestClassifier(
                random_state=self.random_state) if target not in self.cont_vars else RandomForestRegressor(
                random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 20, 50, 100]
            }
        elif method == 'gbm':
            model = GradientBoostingClassifier(
                random_state=self.random_state) if target not in self.cont_vars else GradientBoostingRegressor(
                random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.05, 0.01, 0.05, 0.1],
                'max_depth': [5, 10, 20, 50, 100]
            }

        scoring = 'f1_macro' if target not in self.cont_vars else 'neg_mean_absolute_error'
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scoring, n_jobs=-1)
        grid.fit(X, y)

        return grid.best_params_

    def extract_rfe(self, data, target, method='rf'):
        """Extract features using Recursive Feature Elimination (RFE)."""
        X = data.drop(target, axis=1)
        y = data[target]
        logging.info(f'Started Fitting RF')
        # best_params = self.grid_search(X, y, target, method)
        best_params = {
            'n_estimators': 100,
            'learning_rate': 0.01,
            'max_depth': 20
        }

        if method == 'gbm':
            if target in self.cont_vars:
                model = GradientBoostingRegressor(random_state=self.random_state,
                                                  learning_rate=best_params['learning_rate'],
                                                  n_estimators=best_params['n_estimators'],
                                                  max_depth=best_params['max_depth'])
            else:
                model = GradientBoostingClassifier(random_state=self.random_state,
                                                   learning_rate=best_params['learning_rate'],
                                                   n_estimators=best_params['n_estimators'],
                                                   max_depth=best_params['max_depth'])
        elif method == 'rf':
            if target in self.cont_vars:
                model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'], random_state=self.random_state)
            else:
                model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                               max_depth=best_params['max_depth'], random_state=self.random_state)

        folds = 5
        # folds = self.get_best_folds(X, y, sub_model=model)
        rfe = RFECV(estimator=model, cv=folds)
        rfe.fit(X, y)
        logging.info('Finished Fitting')

        rankings = tuple(zip(list(X.columns), rfe.ranking_))
        selected_features = list(X.columns[rfe.ranking_ == 1])
        return selected_features, rankings

    def extract_key_variables(self, data, target, method='LASSO'):
        """Extract key variables using selected method (LASSO or RFE)."""
        if self.mnarn:
            copy = data.copy().fillna(-1)
        else:
            copy = data.copy().dropna()

        if method == 'LASSO':
            features, rankings = self.extract_lasso(copy, target)
        else:
            features, rankings = self.extract_rfe(copy, target, method=method)

        return features, rankings

    def convert_cat_cols(self, data):
        """Convert categorical columns to numerical format."""
        string_cols = ['LHB_CD_BIRTH_NC', 'CHILD_SEX_NC', 'CHILD_ETHNIC_GRP_NC', 'LSOA_CD_BIRTH_NC']
        copy = data.copy()
        cat_cols = []
        for col in string_cols:
            if col in copy.columns:
                cat_cols.append(col)
                copy[col] = copy[col].map(self.encoders[col])
        temp = copy.drop(columns=cat_cols)
        scaler = RobustScaler()
        scaled = pd.DataFrame(scaler.fit_transform(temp), columns=temp.columns)
        for col in cat_cols:
            scaled[col] = copy[col].copy()
        return scaled

    def get_clusters(self, data, target, cluster_method='reduced', selection='LASSO'):
        """Get clusters from the data using specified method and selection technique."""
        important_features, rankings = self.extract_key_variables(data, target, selection)
        missing_ratio = data.isnull().mean()
        filtered_features = [feature for feature in important_features if missing_ratio[feature] <= 0.10]
        full_filtered = [col for col in data.columns if missing_ratio[col] <= 0.10]
        X_important = data.copy()[filtered_features]
        X_normal = data.copy()[full_filtered]
        if target in list(X_normal.columns):
            X_normal = X_normal.drop(columns=[target])

        if len(filtered_features) < 3:
            logging.info(f'Important features: {important_features}')
            logging.info(f'Filtered features: {filtered_features}')

        logging.info('Imputing for the PCA')
        X_normal_imputed = self.impute(X_normal)
        X_important_imputed = self.impute(X_important)

        if cluster_method == 'important':
            chosen = X_important_imputed
            temp_imputed = X_important_imputed
            features = important_features
            pca_features = None
        else:
            to_reduce = X_normal_imputed if cluster_method == 'reduced' else X_important_imputed
            temp_imputed = to_reduce
            scaler = RobustScaler()
            X_imputed = pd.DataFrame(scaler.fit_transform(to_reduce), columns=to_reduce.columns)
            reducer = PCA(n_components=0.95, random_state=self.random_state)
            chosen = reducer.fit_transform(X_imputed)
            print(reducer.components_.shape)
            features = list(X_normal.columns) if cluster_method == 'reduced' else important_features
            pca_features = list(to_reduce.columns)

        num_clusters = self.find_optimal_clusters(chosen, cluster_method)
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=self.random_state)
        clusters = kmeans.fit_predict(chosen)
        score1 = davies_bouldin_score(chosen, clusters)
        score2 = calinski_harabasz_score(chosen, clusters)

        clustered_data = data.copy()
        clustered_data['CLUSTER'] = clusters

        return clustered_data, temp_imputed, features, pca_features, rankings, (score1, score2)

    def introduce_missingness(self, data, target, seed):
        """Introduce artificial missingness in the data for simulation purposes."""
        np.random.seed(seed)
        actual_missing_ratio = self.missing_data[target].isna().mean()
        if actual_missing_ratio < 0.1:
            actual_missing_ratio = 0.1
        n_missing = int(actual_missing_ratio * len(data))
        missing_indices = np.random.choice(data.index, n_missing, replace=False)
        data.loc[missing_indices, target] = np.nan
        return data, missing_indices

    def custom_clean(self, imputed, target):
        """Clean imputed data by rounding and handling special cases."""
        imputed[target] = imputed[target].round().astype(int)
        if target == 'LSOA_CD_BIRTH_NC':
            imputed = self.clean_LSOA(imputed)
        return imputed

    def run_imputations(self, data, target, clustering, selection, is_train):
        """Run different imputation methods on the data."""
        imputations = pd.DataFrame()
        logging.info('Started full Impute')
        imputations[f'{target}_imputed_full'] = self.impute(data)[target]
        new_data = self.deal_with_MNAR(data) if self.mnarn else data
        logging.info('Started selected Impute')
        imputations[f'{target}_imputed_full_selected'] = self.full_selected_impute(new_data, target, selection)[target]
        logging.info('Started clustered Impute')
        cluster_imputed, temp_imputed, clusters, features, pca_features, rankings, scores = self.cluster_impute(
            new_data, target, clustering, selection)
        imputations[f'{target}_imputed_cluster'] = cluster_imputed[target]
        return imputations, temp_imputed, clusters, features, pca_features, rankings, scores

    def deal_with_all_nan_target(self, target, sub_data, nan_indices, current_imputations):
        """Handle cases where the target variable is completely missing."""
        data = sub_data.copy()
        total = len(data)
        desired = int(total * 0.9)
        current = len(nan_indices)
        num_to_fill = current - desired
        np.random.seed(self.random_state)
        fill_indices = np.random.choice(nan_indices, size=num_to_fill, replace=False)
        data.loc[fill_indices, target] = current_imputations.loc[fill_indices].mean(axis=1)
        return data

    def cluster_impute(self, data, target, cluster_method, selection, current_imputations):
        """Perform cluster-based imputation."""
        min_size = 150
        clustered_data, temp_imputed, selected_vars, pca_vars, rankings, scores = self.get_clusters(data, target,
                                                                                                    cluster_method,
                                                                                                    selection)
        all_clusters = clustered_data['CLUSTER'].unique()
        imputed = data.copy()
        for cluster in all_clusters:
            logging.info(f'Imputing cluster {cluster}')
            cluster_mask = clustered_data['CLUSTER'] == cluster
            sub_data = clustered_data[cluster_mask].copy()
            sub_data = sub_data[selected_vars + [target]]
            nan_indices = sub_data[sub_data[target].isna()].index
            target_missingness = len(nan_indices) / len(sub_data)
            if target_missingness > 0.9:
                sub_data = self.deal_with_all_nan_target(target, sub_data, nan_indices, current_imputations)
                logging.info(
                    f'Target has been partially pre-imputed because it had a missingness of {target_missingness}.')
                nan_indices = sub_data[sub_data[target].isna()].index
                logging.info(f'And now has a missingness of {len(nan_indices) / len(sub_data)}.')
            all_nan_cols = sub_data.columns[sub_data.isna().all().tolist()]
            sub_data = sub_data.drop(columns=all_nan_cols)
            if len(all_nan_cols) > 0:
                logging.info(f'The following columns have been dropped due to being all NaN: {all_nan_cols}')
            num_missing = sub_data[target].isna().sum()
            logging.info(f'The cluster contains {num_missing} missing values.')
            sub_imputed = None
            if num_missing > 0:
                if len(sub_data) > min_size:
                    logging.info(f'Length of cluster {cluster}: {len(sub_data)}')
                    sub_imputed = self.impute(sub_data)
                else:
                    logging.info(f'Cluster {cluster} with size {len(sub_data)} is too small, using KNN instead.')
                    sub_imputed = self.knn_impute(sub_data)
            if sub_imputed is not None:
                imputed.loc[cluster_mask, target] = sub_imputed[target].values

        self.imputed = imputed
        return imputed, temp_imputed.to_dict(orient='dict'), clustered_data.to_dict(orient='dict')[
            'CLUSTER'], selected_vars, pca_vars, rankings, scores

    def full_selected_impute(self, data, target, selection):
        """Perform full imputation on selected features."""
        important_features, _ = self.extract_key_variables(data, target, method=selection)
        imputed = data.copy()
        full_sub_data = imputed[important_features + [target]]
        full_sub_imputed = self.impute(full_sub_data)
        return full_sub_imputed

    def custom_metrics(self, predicted, truth, target):
        """Calculate custom metrics for evaluation."""
        if target not in self.cont_vars:
            accuracy = accuracy_score(truth, predicted)
            fbeta = fbeta_score(truth, predicted, average='macro', zero_division=0, beta=1)
            return {
                'accuracy': accuracy,
                'fbeta': fbeta,
            }
        else:
            mae = mean_absolute_error(truth, predicted)
            return {'mae': mae}

    def custom_imputation(self, target):
        """Custom imputation process for a target variable."""
        logging.basicConfig(
            filename=f'benchmark_{target}_log.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        target_trace = {}
        missing_mask = self.missing_data[target].isna()
        data = self.missing_data.copy()
        imputed, temp_imputed, clusters, features, pca_features, rankings, scores = self.run_imputations(data, target,
                                                                                                         'reduced',
                                                                                                         'LASSO',
                                                                                                         is_train=False)
        imputed = imputed.loc[missing_mask]

        truth = self.true_data.loc[missing_mask, target]
        for_ref = imputed.copy()
        for_ref[f'truth_{target}'] = truth

        target_trace[target] = {'clusters': clusters, 'clustering_imputed': temp_imputed, 'selected_vars': features,
                                'pca_features': pca_features, 'values': for_ref.to_dict(orient='dict'),
                                'cluster_scores': scores, 'feature_rankings': rankings}
        self.imputed = imputed
        self.save_performance(trace=target_trace, custom_title='standard_LSOA_impute')

    def simple_subpop_impute(self, clustering, selection):
        """Simple subpopulation imputation process."""
        full_imputed_data = self.missing_data.copy()
        target_trace = {}
        for target in self.targets:
            logging.info(f'Imputing target variable: {target}')
            missing_mask = self.missing_data[target].isna()
            data = self.missing_data.copy()
            imputed, temp_imputed, clusters, features, pca_features, rankings, scores = self.run_imputations(data,
                                                                                                             target,
                                                                                                             clustering,
                                                                                                             selection,
                                                                                                             is_train=False)
            imputed = imputed.loc[missing_mask]

            truth = self.true_data.loc[missing_mask, target]
            for_ref = imputed.copy()
            for_ref[f'truth_{target}'] = truth

            target_trace[target] = {'clusters': clusters, 'clustering_imputed': temp_imputed, 'selected_vars': features,
                                    'pca_features': pca_features, 'values': for_ref.to_dict(orient='dict'),
                                    'cluster_scores': scores, 'feature_rankings': rankings}

        self.imputed = full_imputed_data
        self.save_performance(trace=target_trace)

    def subpopulation_impute(self, clustering, selection, ensemble='tree'):
        """Subpopulation imputation with ensemble method."""
        full_imputed_data = self.missing_data.copy()
        target_trace = {}
        n_sim = 3
        for target in self.targets:
            models = []
            test_datas = []
            for seed in range(n_sim):
                logging.info(
                    f'Model {self.method}: {seed + 1} training iteration for {target} with params: {"deprived-" if self.isdep else ""}{self.selec}-{self.clust}-{"mnar" if self.mnarn else ""}')
                train_data = self.missing_data[self.missing_data[target].notna()].copy()
                train_with_missing, train_missing_indices = self.introduce_missingness(train_data.copy(), target, seed)

                imputed_train, _, _, _, _, _, _ = self.run_imputations(train_with_missing, target, clustering,
                                                                       selection, is_train=True)
                imputed_train[target] = train_data[target]
                imputed_train = imputed_train.loc[train_missing_indices]

                fit_data, test_data = train_test_split(imputed_train, test_size=0.2, random_state=seed)

                if ensemble == 'tree':
                    if target in self.cont_vars:
                        model = DecisionTreeRegressor(max_depth=15, min_samples_split=10, min_samples_leaf=5,
                                                      random_state=self.random_state)
                    else:
                        model = DecisionTreeClassifier(max_depth=15, min_samples_split=10, min_samples_leaf=5,
                                                       random_state=self.random_state)
                else:
                    model = LogisticRegression(max_iter=3000,
                                               random_state=self.random_state) if target not in self.cont_vars else LinearRegression()

                X = fit_data.drop(columns=[target])
                y = fit_data[target]

                if target not in self.cont_vars:
                    rus = RandomUnderSampler()
                    X_r, y_r = rus.fit_resample(X, y)
                    model.fit(X_r, y_r)
                else:
                    model.fit(X, y)

                models.append(model)
                test_datas.append(test_data)

            perf_matrix = np.zeros((n_sim, n_sim))
            for i, model in enumerate(models):
                for j, test_data in enumerate(test_datas):
                    combined_test = model.predict(test_data.drop(columns=[target]))
                    metrics = self.custom_metrics(combined_test, test_data[target], target)
                    if target in self.cont_vars:
                        perf = metrics['mae']
                    else:
                        perf = metrics['accuracy']
                    perf_matrix[i][j] = perf

            average_perfs = perf_matrix.mean(axis=1)
            best_index = np.argmax(average_perfs) if target not in self.cont_vars else np.argmin(average_perfs)
            final_model = models[best_index]

            missing_mask = self.missing_data[target].isna()
            data = self.missing_data.copy()
            imputed, temp_imputed, clusters, features, pca_features, rankings, scores = self.run_imputations(data,
                                                                                                             target,
                                                                                                             clustering,
                                                                                                             selection,
                                                                                                             is_train=False)
            imputed = imputed.loc[missing_mask]

            truth = self.true_data.loc[missing_mask, target]
            for_ref = imputed.copy()
            for_ref[f'truth_{target}'] = truth

            combined_imputed = final_model.predict(imputed)
            # probas = final_model.predict_proba(imputed) if target not in self.cont_vars else None
            for_ref[f'combined_{target}'] = combined_imputed

            full_imputed_data.loc[missing_mask, target] = combined_imputed
            full_imputed_data['CLUSTER'] = clusters

            with open(f'{target}_tree.pkl', 'wb') as file:
                pickle.dump(final_model, file)

            target_trace[target] = {'clusters': clusters, 'clustering_imputed': temp_imputed, 'selected_vars': features,
                                    'pca_features': pca_features, 'values': for_ref.to_dict(orient='dict'),
                                    'cluster_scores': scores, 'feature_rankings': rankings,
                                    'tree_features': final_model.feature_importances_ if hasattr(final_model,
                                                                                                 'feature_importances_') else None}

        self.imputed = full_imputed_data
        self.save_performance(trace=target_trace)

        return full_imputed_data, target_trace

    def get_lhb_map(self):
        """Generate a mapping from LHB to LSOA."""
        lhb_to_lsoa = {}
        for index, row in self.missing_data.iterrows():
            lhb = row['LHB_CD_BIRTH_NC']
            lsoa = row['LSOA_CD_BIRTH_NC']
            if pd.notna(lhb) and pd.notna(lsoa):
                if lhb in lhb_to_lsoa:
                    lhb_to_lsoa[lhb].add(lsoa)
                else:
                    lhb_to_lsoa[lhb] = {lsoa}
        lhb_to_lsoa = {k: list(v) for k, v in lhb_to_lsoa.items()}
        return lhb_to_lsoa

    def custom_LSOA_impute(self):
        """Perform custom LSOA imputation."""
        logging.basicConfig(
            filename='custom_LSOA_log.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.info('Started custom LSOA imputation')
        lhb_to_lsoa = self.get_lhb_map()
        imputed = self.missing_data.copy()
        known_lsoa = self.missing_data.dropna(subset=['LSOA_CD_BIRTH_NC'])
        missing_lsoa = self.missing_data[self.missing_data['LSOA_CD_BIRTH_NC'].isnull()]
        counter = 0
        total = len(missing_lsoa)

        for index, missing_row in missing_lsoa.iterrows():
            counter += 1
            logging.info(f'Imputing the {counter}th row out of {total}')
            if pd.notna(missing_row['LHB_CD_BIRTH_NC']) and missing_row['LHB_CD_BIRTH_NC'] in lhb_to_lsoa:
                lsoas_for_lhb = lhb_to_lsoa[missing_row['LHB_CD_BIRTH_NC']]
                subset_known_lsoa = known_lsoa[known_lsoa['LSOA_CD_BIRTH_NC'].isin(lsoas_for_lhb)]
            else:
                subset_known_lsoa = known_lsoa  # Use all known LSOAs if LHB is NaN or not in map

            if not subset_known_lsoa.empty:
                lsoa_distances = {}
                for lsoa, group in subset_known_lsoa.groupby('LSOA_CD_BIRTH_NC'):
                    miss_row = pd.DataFrame(missing_row.drop(['LSOA_CD_BIRTH_NC'])).transpose()
                    cols_with_nan = miss_row.columns[miss_row.isna().all()].tolist()
                    cols_with_nan += ['LSOA_CD_BIRTH_NC']
                    miss_row = miss_row.drop(columns=cols_with_nan)
                    clean_grp = group.drop(columns=cols_with_nan)
                    row_gower = gower.gower_matrix(miss_row, clean_grp)
                    lsoa_distances[lsoa] = np.mean(row_gower)
                best_lsoa = min(lsoa_distances, key=lsoa_distances.get)

                imputed.at[index, 'LSOA_CD_BIRTH_NC'] = best_lsoa

        self.imputed = imputed
        self.save_performance(custom_LSOA=True)
        with open('./results/custom_LSOA_dict.pkl', 'wb') as file:
            pickle.dump(imputed.to_dict(orient='dict'), file)
        return imputed
