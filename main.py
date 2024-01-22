import numpy as np
from sdv.single_table import CTGANSynthesizer
import data_tasks.generation.data_loader as dl
import data_tasks.generation.data_synthetizer as ds
import pandas as pd
from LLMs.hosted_llm_imputer import HostedMentalHealthImputer
from LLMs.imputation_manager import ImputationManager
from sota_models.miss_forest import MissForestImputer
from utils import plot_imputation_metrics
from sota_models.gain.gain_imputer import GAINImputer


def synthesize():
    # synthesizer = create_new_synthetizer('japan_mental')
    # synthesizer = load_saved_synthetizer('japan_mental')
    # # synthesizer.sample_lol(50)
    # real_data = pd.read_csv('datasets/japan_mental.csv')
    # synthetic_data = pd.read_csv('datasets/synthetic_japan_mental.csv')
    # evaluator = de.DataEvaluator(real_data, synthetic_data, synthesizer.get_metadata())
    # evaluator.pairwise_plot('Suicide', 'Religion')
    # to_remove = ['Phone', 'Japanese', 'English', 'Stay', 'Age', 'ToSC','APD','AHome','APH','Afear','ACS','AGuilt','AMiscell','ToAS','Partner','Friends','Parents','Relative','Profess', 'Phone','Doctor','Reli','Alone','Others','Internet','Partner_bi','Friends_bi','Parents_bi','Relative_bi','Professional_bi','Phone_bi','Doctor_bi','religion_bi','Alone_bi','Others_bi','Internet_bi']
    # to_remove = ['inter_dom', 'Region', 'Age_cate', 'Stay_Cate', 'Japanese_cate', 'English_cate', 'Intimate', 'Dep',
    #              'DepType', 'ToDep']
    pass


def create_new_synthetizer(synthesizer_name):
    synthesizer_name = 'japan_mental'
    loader = dl.DataLoader(synthesizer_name)
    return ds.DataSynthetizer(synthesizer_name, loader)


def load_saved_synthetizer(synthesizer_name):
    loaded_synthesizer = CTGANSynthesizer.load(
        filepath=f'data_tasks/saved_synthetizers/{synthesizer_name}_synthetizer.pkl'
    )
    print('loaded')
    return ds.DataSynthetizer(synthesizer_name, loaded_synthetizer=loaded_synthesizer)


def LLM_impute():
    mistral_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    phixtral_id = "mlabonne/phixtral-4x2_8"

    description = "The dataset comprises of records from both international and domestic students in an international university in Japan. This dataset is used to examine the mental health conditions and help-seeking behaviors of international and domestic students in a multicultural environment. All column names are self explanatory except for the column called “DepSev” which indicates the severity of depressive disorder reported based on the following criteria: Minimal depression (Min), Mild depression (Mild), Moderate depression (Mod), Moderately severe depression (ModSev), Severe depression (Sev). The column called “Suicide” indicates whether students have suicidal Ideation in the last 2 weeks or not.  "
    prior = "Students who tend to have a higher depression severity are more likely to have had suicidal ideation than those with low depressive severity."
    original_data = pd.read_csv('datasets/new_japan_mental.csv')
    missing_data = pd.read_csv('datasets/missing_japan_mental.csv')

    imputation_manager = ImputationManager(dataset=original_data, missing_dataset=missing_data,
                                           dataset_description=description, clinician_prior=prior)

    imputer = HostedMentalHealthImputer(model_id=mistral_id, manager=imputation_manager)
    output = imputer.impute_data()
    imputation_manager.compute_error(output)


def classical_impute():
    original_data = pd.read_csv('datasets/new_japan_mental.csv')
    missing_data = pd.read_csv('datasets/missing_japan_mental.csv')
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



if __name__ == '__main__':
    # synthesize()
    gain_impute()