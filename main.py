import numpy as np
import pandas as pd
# from LLMs.transformers_imputer import TransformersImputer
# from LLMs.imputation_manager import ImputationManager
from sota_models.miss_forest import MissForestImputer
from utils import plot_imputation_metrics, remove_values_to_threshold
from sota_models.gain.gain_imputer import GAINImputer
from data_tasks.generation.clinical_data_synthetizer import ClinicalSynthetizer
from insights import BirthInsights


def LLM_impute():
    print("let's go")
    description = "The dataset comprises of records from both international and domestic students in an international university in Japan. This dataset is used to examine the mental health conditions and help-seeking behaviors of international and domestic students in a multicultural environment. All column names are self explanatory except for the column called “DepSev” which indicates the severity of depressive disorder reported based on the following criteria: Minimal depression (Min), Mild depression (Mild), Moderate depression (Mod), Moderately severe depression (ModSev), Severe depression (Sev). The column called “Suicide” indicates whether students have suicidal Ideation in the last 2 weeks or not.  "
    prior = "Students who tend to have a higher depression severity are more likely to have had suicidal ideation than those with low depressive severity."
    original_data = pd.read_csv('datasets/japan/new_japan_mental.csv')
    missing_data = pd.read_csv('datasets/japan/missing_japan_mental.csv')

    model_id = model_ids.mistral

    imputation_manager = ImputationManager(dataset=original_data, missing_dataset=missing_data,
                                           dataset_description=description, clinician_prior=prior)

    imputer = TransformersImputer(model_id=model_id, manager=imputation_manager)
    output = imputer.impute_data()
    imputation_manager.compute_error(output)


def classical_impute():
    original_data = pd.read_csv('datasets/japan/new_japan_mental.csv')
    missing_data = pd.read_csv('datasets/japan/missing_japan_mental.csv')
    df_or = original_data.copy()
    for c in df_or.columns:
        random_index = np.random.choice(df_or.index, size=75)
        df_or.loc[random_index, c] = np.nan
    imputer = MissForestImputer(original_data, df_or)
    imputer.fit()
    metrics = imputer.compute_error()
    plot_imputation_metrics(metrics)
    imputer.plot_categorical_comparisons()


def gain_impute():
    imputer = GAINImputer(data_name='japan_mental')
    imputer.impute()
    imputer.compute_error()


def synthetize():
    print('Started!')
    synth = ClinicalSynthetizer(1000)
    synth.generate()
    missing_percentage = {
        # 'BIRTH_WEIGHT_NC': 0.2,
        # 'GEST_AGE_NC': 0.2,
        # 'MAT_AGE_NC': 0.02,
        'MAT_SMOKING_NC': 40,
        'BREASTFEED_BIRTH_FLG_NC': 80,
        # 'MAT_REGION_NC': 5
    }

    columns_to_modify = [
        # 'BIRTH_WEIGHT_NC',
        # 'GEST_AGE_NC',
        # 'MAT_AGE_NC',
        'MAT_SMOKING_NC',
        'BREASTFEED_BIRTH_FLG_NC',
        # 'MAT_REGION_NC'
    ]
    remove_values_to_threshold('datasets/clinical/birth_data.csv', columns_to_modify, missing_percentage,
                               'datasets/clinical/birth_data_missing.csv')
    print('Finished!')


def insights():
    insights = BirthInsights(pd.read_csv('datasets/clinical/birth_data_missing.csv'))


if __name__ == '__main__':
    insights()
    # synthetize()
    # gain_impute()
    # LLM_impute()
    # download_model("osunlp/TableLlama")
    # print('starting')
    # vllm = VLLMImputer()
    # vllm.generate()
