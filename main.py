from sdv.single_table import CTGANSynthesizer
import data_generation.data_loader as dl
import data_generation.data_synthetizer as ds
import data_generation.data_evaluator as de
import pandas as pd

from data_generation.data_cleaner import remove_columns_from_csv, remove_values_to_threshold


def create_new_synthetizer(synthesizer_name):
    synthesizer_name = 'japan_mental'
    loader = dl.DataLoader(synthesizer_name)
    return ds.DataSynthetizer(synthesizer_name, loader)


def load_saved_synthetizer(synthesizer_name):
    loaded_synthesizer = CTGANSynthesizer.load(
        filepath=f'data_generation/saved_synthetizers/{synthesizer_name}_synthetizer.pkl'
    )
    print('loaded')
    return ds.DataSynthetizer(synthesizer_name, loaded_synthetizer=loaded_synthesizer)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # synthesizer = create_new_synthetizer('japan_mental')
    # synthesizer = load_saved_synthetizer('japan_mental')
    # # synthesizer.sample_lol(50)
    # real_data = pd.read_csv('datasets/japan_mental.csv')
    # synthetic_data = pd.read_csv('datasets/synthetic_japan_mental.csv')
    # evaluator = de.DataEvaluator(real_data, synthetic_data, synthesizer.get_metadata())
    # evaluator.pairwise_plot('Suicide', 'Religion')
    # to_remove = ['Phone', 'Japanese', 'English', 'Stay', 'Age', 'ToSC','APD','AHome','APH','Afear','ACS','AGuilt','AMiscell','ToAS','Partner','Friends','Parents','Relative','Profess', 'Phone','Doctor','Reli','Alone','Others','Internet','Partner_bi','Friends_bi','Parents_bi','Relative_bi','Professional_bi','Phone_bi','Doctor_bi','religion_bi','Alone_bi','Others_bi','Internet_bi']
    # remove_columns_from_csv('datasets/old_japan_mental.csv', to_remove, 'datasets/japan_mental.csv')
    remove_values_to_threshold('datasets/japan_mental.csv', ['Intimate', 'DepType', 'DepSev'], 70, 'datasets/missing_japan_mental.csv')
