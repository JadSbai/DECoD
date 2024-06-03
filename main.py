from utils import remove_values_to_threshold
from data_tasks.generation.clinical_data_synthetizer import ClinicalSynthetizer
import pandas as pd
import numpy as np
from impute.data_centric_impute import HyperImputer
from sklearn.preprocessing import OrdinalEncoder
import os
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import mean_absolute_error

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def synthesize():
    synth = ClinicalSynthetizer(1000)
    synth.generate()
    missing_percentage = {
        'BIRTH_WEIGHT_NC': 20,
        'MAT_SMOKING_NC': 40,
        'BREASTFEED_BIRTH_FLG_NC': 80,
    }

    columns_to_modify = [
        'BIRTH_WEIGHT_NC',
        'MAT_SMOKING_NC',
        'BREASTFEED_BIRTH_FLG_NC',
    ]
    remove_values_to_threshold('datasets/fake_clinical/birth_data_small.csv', columns_to_modify, missing_percentage,
                               'datasets/fake_clinical/birth_data_missing_small.csv')
    print('Finished!')


def prep_data(original_dataset, comp_dataset_file, dep_file, is_dep=False):
    full_df = pd.read_csv(comp_dataset_file)
    old_df = pd.read_csv(original_dataset)
    deprivation_df = pd.read_csv(dep_file)

    full_df['LSOA_CD_BIRTH_NC'] = old_df['LSOA_CD_BIRTH_NC'].copy()
    full_df['BREASTFEED_BIRTH_FLG_NC'] = old_df['BREASTFEED_BIRTH_FLG_NC'].copy()
    full_df['BIRTH_WEIGHT_NC'] = old_df['BIRTH_WEIGHT_NC'].astype('float64') * 1000
    full_df['GEST_AGE_NC'] = old_df['GEST_AGE_NC'].copy()
    full_df['MAT_AGE_NC'] = old_df['MAT_AGE_NC'].copy()

    df_true = full_df.copy()
    df_true = df_true[
        ['SMOKE_NC_fill', 'MAT_AGE_NC_fill', 'GEST_AGE_NC_fill', 'BIRTH_WEIGHT_NC_fill', 'BREASTFEED_BIRTH_NC_fill',
         'LSOA_CD_BIRTH_NC_fill']]
    df_true['SMOKE_NC_fill'] = df_true['SMOKE_NC_fill'].replace(9.0, np.nan)
    df_true.dropna(inplace=True)
    df_true.rename(
        columns={'SMOKE_NC_fill': 'SMOKE_NC', 'MAT_AGE_NC_fill': 'MAT_AGE_NC', 'GEST_AGE_NC_fill': 'GEST_AGE_NC',
                 'BIRTH_WEIGHT_NC_fill': 'BIRTH_WEIGHT_NC'}, inplace=True)

    deprivation_df.rename(columns={'Code': 'LSOA_CD_BIRTH_NC', 'Decile': 'DEP_SCORE'}, inplace=True)
    df = pd.merge(full_df, deprivation_df[['LSOA_CD_BIRTH_NC', 'DEP_SCORE']], on='LSOA_CD_BIRTH_NC', how='left')

    columns_to_drop = [col for col in full_df.columns if not col.endswith('NC')] + ['ALF_MTCH_PCT_NC', 'CHILD_ALF_PE',
                                                                                    'ALF_STS_NC', 'MAT_ALF_PE_NC',
                                                                                    'MAT_ALF_STS_NC',
                                                                                    'MAT_ALF_MTCH_PCT_NC',
                                                                                    'MAT_SMOKING_NC']
    df_missing = df.drop(columns=columns_to_drop)

    df_missing = df_missing.loc[df_true.index]
    df_missing['BREASTFEED_BIRTH_FLG_NC'] = df_missing['BREASTFEED_BIRTH_FLG_NC'].replace(9.0, np.nan)
    df_missing['CHILD_ETHNIC_GRP_NC'] = df_missing['CHILD_ETHNIC_GRP_NC'].replace('z', np.nan)

    missing_copy = df_missing.copy()
    true_copy = df_true.copy()
    to_encode = ['LHB_DEP_NC', 'CHILD_SEX_NC', 'CHILD_ETHNIC_GRP_NC', 'LSOA_CD_BIRTH_NC']
    final_df_missing, final_true_df, encoders = encode_cat_cols(missing_copy, true_copy, to_encode)

    target = 'BIRTH_WEIGHT_NC'
    mask = old_df[target].notna()
    true = old_df[target][mask]
    maybe_true = full_df['BIRTH_WEIGHT_NC_fill'].dropna()[mask]
    true = true.loc[maybe_true.index]
    print(len(maybe_true))
    print(len(true))

    print(mean_absolute_error(true, maybe_true))
    # agreements = (true == maybe_true).sum()
    # agreements = agreements / len(true)
    # print(agreements)

    # conf_low, conf_high = proportion_confint(agreements, len(true), method='wilson')
    # print(conf_low)
    # print(conf_high)

    return final_true_df, final_df_missing, encoders


def encode_cat_cols(df, true_df, cols):
    encoders = {}
    for col in cols:
        ordinal = OrdinalEncoder(min_frequency=80, handle_unknown='use_encoded_value', unknown_value=np.nan)
        if col == 'LSOA_CD_BIRTH_NC':
            ordinal.fit(true_df[[col]])
            missing_encoded = pd.DataFrame(ordinal.transform(df[[col]]), index=df.index, columns=[col])
            true_encoded = pd.DataFrame(ordinal.transform(true_df[[col]]), index=true_df.index, columns=[col])
            df[col] = missing_encoded[col].copy()
            true_df[col] = true_encoded[col].copy()
        else:
            missing_encoded = pd.DataFrame(ordinal.fit_transform(df[[col]]), index=df.index, columns=[col])
            df[col] = missing_encoded[col].copy()
        categories = ordinal.categories_[0]
        decoding_map = {i: categories[i] for i in range(len(categories))}
        encoders[col] = decoding_map

    return df, true_df, encoders





if __name__ == '__main__':
    dataset_file = './datasets/key.csv'
    old_dataset = './datasets/JS_NCCH.csv'
    deprivation_file = './datasets/decile_scores.csv'

    true_data, missing_data, encoders = prep_data(old_dataset, dataset_file, deprivation_file)

    # Uncomment the following lines to run the hyperimputation
    # imputer = HyperImpute(missing_data, true_data, encoders, "hyperimpute", False)
    # Imputers available: ['miracle', 'ice', 'gain', 'nop', 'softimpute', 'EM', 'sklearn_ice', 'most_frequent', 'sklearn_missforest', 'sinkhorn', 'miwae', 'mean', 'mice', 'hyperimpute', 'median', 'missforest']
    # imputer.benchmark_models(['hyperimpute'])
    # imputer.custom_imputation('LSOA_CD_BIRTH_NC')
    # imputer.custom_LSOA_impute()
