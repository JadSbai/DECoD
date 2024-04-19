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

    def generate(self):
        # Generate continuous variables
        weight_col = np.random.normal(loc=self.cont_distributions['BIRTH_WEIGHT_NC']['mean'],
                                      scale=self.cont_distributions['BIRTH_WEIGHT_NC']['std'], size=self.n)
        gest_age_col = np.random.normal(loc=self.cont_distributions['GEST_AGE_NC']['mean'],
                                        scale=self.cont_distributions['GEST_AGE_NC']['std'], size=self.n)
        mat_age_col = np.random.normal(loc=self.cont_distributions['MAT_AGE_NC']['mean'],
                                       scale=self.cont_distributions['MAT_AGE_NC']['std'], size=self.n)
        apgar_1_col = np.random.normal(loc=self.cont_distributions['APGAR_1_NC']['mean'],
                                       scale=self.cont_distributions['APGAR_1_NC']['std'], size=self.n)
        apgar_2_col = np.random.normal(loc=self.cont_distributions['APGAR_2_NC']['mean'],
                                       scale=self.cont_distributions['APGAR_2_NC']['std'], size=self.n)

        # Generate categorical variables
        gender = [0, 1]
        smoke = [0, 1]
        ethnic = [0, 1]
        labour = [1, 2, 3, 4]
        breastfeed = [0, 1]
        birth_order = [1, 2]
        prev_stillbirths = [0, 1]
        prev_livebirths = [0, 1, 2, 3, 4, 5]
        breastfeed_8w = [0, 1]
        tot_birth = [1, 2]
        region = range(1917)

        gender_col = np.random.choice(gender, size=self.n, p=[0.5, 0.5])
        smoke_col = np.random.choice(smoke, size=self.n, p=[0.8, 0.2])
        ethnic_col = np.random.choice(ethnic, size=self.n, p=[0.64, 0.36])
        labour_col = np.random.choice(labour, size=self.n, p=[0.60, 0.25, 0.1, 0.05])
        breastfeed_col = np.random.choice(breastfeed, size=self.n, p=[0.55, 0.45])
        birth_order_col = np.random.choice(birth_order, size=self.n, p=[0.98, 0.02])
        prev_stillbirths_col = np.random.choice(prev_stillbirths, size=self.n, p=[0.99, 0.01])
        prev_livebirths_col = np.random.choice(prev_livebirths, size=self.n, p=[0.43, 0.36, 0.13, 0.05, 0.02, 0.01])
        breastfeed_8w_col = np.random.choice(breastfeed_8w, size=self.n, p=[0.6, 0.4])
        region_col = np.random.choice(region, size=self.n)
        tot_birth_col = np.random.choice(tot_birth, size=self.n, p=[0.98, 0.02])

        # Create DataFrame
        df = pd.DataFrame({
            'BIRTH_WEIGHT_NC': weight_col,
            'GEST_AGE_NC': gest_age_col,
            'MAT_AGE_NC': mat_age_col,
            'APGAR_1_NC': apgar_1_col,
            'APGAR_2_NC': apgar_2_col,
            'TOT_BIRTH_NUM_NC': tot_birth_col,
            'CHILD_SEX_NC': gender_col,
            'MAT_SMOKING_NC': smoke_col,
            'CHILD_ETHNIC_GROUP_NC': ethnic_col,
            'LABOUR_ONSET_NC': labour_col,
            'BREASTFEED_BIRTH_FLG_NC': breastfeed_col,
            'BIRTH_ORDER_NC': birth_order_col,
            'PREV_STILLBIRTH_NC': prev_stillbirths_col,
            'PREV_LIVEBIRTHS_NC': prev_livebirths_col,
            'BREASTFEED_8_WKS_FLG_NC': breastfeed_8w_col,
            'MAT_REGION_NC': region_col
        })

        df["BIRTH_WEIGHT_NC"] = df["BIRTH_WEIGHT_NC"].round()
        df["GEST_AGE_NC"] = df["GEST_AGE_NC"].round()
        df["MAT_AGE_NC"] = df["MAT_AGE_NC"].round()
        df["APGAR_1_NC"] = df["APGAR_1_NC"].round()
        df["APGAR_2_NC"] = df["APGAR_2_NC"].round()

        df.to_csv('./datasets/clinical/birth_data.csv', index=False)
