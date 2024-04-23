import numpy as np
import pandas as pd


class ClinicalSynthetizer:
    def __init__(self, n):
        self.n = n
        np.random.seed(42)
        self.cont_distributions = {
            'BIRTH_WEIGHT_NC': {'mean': 3366, 'std': 753},
            'GEST_AGE_NC': {'mean': 39.05, 'std': 2.13},
            'MAT_AGE_NC': {'mean': 29.30, 'std': 5.61},
            'APGAR_1_NC': {'mean': 8.60, 'std': 1.93},
            'APGAR_2_NC': {'mean': 9.63, 'std': 1.67},
        }

        self.df = pd.DataFrame({
            # 'BIRTH_WEIGHT_NC': weight_col,
            # 'GEST_AGE_NC': gest_age_col,
            # 'MAT_AGE_NC': mat_age_col,
            # 'APGAR_1_NC': apgar_1_col,
            # 'APGAR_2_NC': apgar_2_col,
            # 'CHILD_SEX_NC': gender_col,
            # 'MAT_SMOKING_NC': smoke_col,
            # 'CHILD_ETHNIC_GROUP_NC': ethnic_col,
            # 'LABOUR_ONSET_NC': labour_col,
            # 'BREASTFEED_BIRTH_FLG_NC': breastfeed_col,
            # 'BIRTH_ORDER_NC': birth_order_col,
            # 'PREV_STILLBIRTH_NC': prev_stillbirths_col,
            # 'PREV_LIVEBIRTHS_NC': prev_livebirths_col,
            # 'BREASTFEED_8_WKS_FLG_NC': breastfeed_8w_col,
            # 'MAT_REGION_NC': region_col
        })

    def generate(self):
        mat_age_col = np.random.normal(loc=self.cont_distributions['MAT_AGE_NC']['mean'],
                                       scale=self.cont_distributions['MAT_AGE_NC']['std'], size=self.n)

        # Generate categorical variables
        gender = [0, 1]
        smoke = [0, 1]
        ethnic = [0, 1]
        labour = [1, 2, 3, 4]
        breastfeed = [0, 1]
        birth_order = [1, 2]
        prev_stillbirths = [0, 1]
        prev_livebirths = [0, 1, 2, 3, 4, 5]
        region = range(1917)

        # Assume means and standard deviations
        mean_birth_weight = self.cont_distributions['BIRTH_WEIGHT_NC']['mean']  # average birth weight in kg
        std_birth_weight = self.cont_distributions['BIRTH_WEIGHT_NC']['std']  # standard deviation
        mean_gestational_age = self.cont_distributions['GEST_AGE_NC']['mean']  # average gestational weeks
        std_gestational_age = self.cont_distributions['GEST_AGE_NC']['std']  # standard deviation

        mean_apgar1 = self.cont_distributions['APGAR_1_NC']['mean']
        std_apgar1 = self.cont_distributions['APGAR_1_NC']['std']
        mean_apgar2 = self.cont_distributions['APGAR_2_NC']['mean']
        std_apgar2 = self.cont_distributions['APGAR_2_NC']['std']

        apgar_corr = 0.77
        apgar_cov = apgar_corr * std_apgar1 * std_apgar2

        # Correlation between birth weight and gestational age
        correlation = 0.61  # this is a guess; you'll want to set this based on your analysis or literature
        covariance = correlation * std_birth_weight * std_gestational_age

        # Covariance matrix
        cov_matrix = np.array([
            [std_birth_weight ** 2, covariance],
            [covariance, std_gestational_age ** 2]
        ])

        apgar_cov_matrix = np.array([
            [std_apgar1 ** 2, apgar_cov],
            [apgar_cov, std_apgar2 ** 2]
        ])

        apgar1, apgar2 = np.random.multivariate_normal(
            [mean_apgar1, mean_apgar2], apgar_cov_matrix, self.n).T

        # Generate correlated data
        birth_weight, gestational_age = np.random.multivariate_normal(
            [mean_birth_weight, mean_gestational_age], cov_matrix, self.n).T

        smoke_col = np.random.choice(smoke, size=self.n, p=[0.8, 0.2])
        breastfeed_col = np.random.choice(breastfeed, size=self.n, p=[0.45, 0.55])

        self.df['BIRTH_WEIGHT_NC'] = np.round(birth_weight).clip(500, 5000)
        self.df['GEST_AGE_NC'] = np.round(gestational_age).clip(25, 50)
        self.df['MAT_SMOKING_NC'] = smoke_col
        self.df['BREASTFEED_BIRTH_FLG_NC'] = breastfeed_col
        self.df['BIRTH_ORDER_NC'] = np.random.choice(birth_order, size=self.n, p=[0.98, 0.02])
        self.df['APGAR_1_NC'] = np.round(apgar1).clip(0, 10)
        self.df['APGAR_2_NC'] = np.round(apgar2).clip(0, 10)
        self.df['MAT_AGE_NC'] = np.round(mat_age_col).clip(15, 50)

        # Mean and standard deviation adjustments
        mean_adjustment_smoking = {
            0: 0,  # Non-smokers: no adjustment
            1: -500  # Smokers: reduce mean by 0.5 kg
        }

        for key, adjustment in mean_adjustment_smoking.items():
            self.df.loc[self.df['MAT_SMOKING_NC'] == key, 'BIRTH_WEIGHT_NC'] += adjustment

        # Adjust birth weight for birth order
        weight_birth_order = {1: 0, 2: -1000}  # adjustment values
        gest_birth_order = {1: 0, 2: -4}  # adjustment values

        for key, adjustment in weight_birth_order.items():
            self.df.loc[self.df['BIRTH_ORDER_NC'] == key, 'BIRTH_WEIGHT_NC'] += adjustment

        for key, adjustment in gest_birth_order.items():
            self.df.loc[self.df['BIRTH_ORDER_NC'] == key, 'GEST_AGE_NC'] += adjustment

        self.df['BREASTFEED_8_WKS_FLG_NC'] = self.df['BREASTFEED_BIRTH_FLG_NC'].apply(
            lambda x: np.random.choice([0, 1], p=[0.38, 0.62] if x == 1 else [0.9, 0.1]))

        gender_col = np.random.choice(gender, size=self.n, p=[0.5, 0.5])
        ethnic_col = np.random.choice(ethnic, size=self.n, p=[0.64, 0.36])
        labour_col = np.random.choice(labour, size=self.n, p=[0.60, 0.25, 0.1, 0.05])
        prev_stillbirths_col = np.random.choice(prev_stillbirths, size=self.n, p=[0.99, 0.01])
        prev_livebirths_col = np.random.choice(prev_livebirths, size=self.n, p=[0.43, 0.36, 0.13, 0.05, 0.02, 0.01])
        region_col = np.random.choice(region, size=self.n)

        self.df['CHILD_ETHNIC_GROUP_NC'] = ethnic_col
        self.df['CHILD_SEX_NC'] = gender_col
        self.df['LABOUR_ONSET_NC'] = labour_col
        self.df['PREV_STILLBIRTH_NC'] = prev_stillbirths_col
        self.df['PREV_LIVEBIRTHS_NC'] = prev_livebirths_col
        self.df['MAT_REGION_NC'] = region_col

        self.df.to_csv('./datasets/clinical/birth_data.csv', index=False)
