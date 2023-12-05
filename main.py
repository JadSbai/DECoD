from sdv.single_table import CTGANSynthesizer
import data_generation.data_loader as dl
import data_generation.data_synthetizer as ds
import data_generation.data_evaluator as de
import pandas as pd


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
    # synthesizer = create_new_synthetizer('japan_mental')
    synthesizer = load_saved_synthetizer('japan_mental')
    # synthesizer.sample_lol(50)
    real_data = pd.read_csv('datasets/japan_mental.csv')
    synthetic_data = pd.read_csv('datasets/synthetic_japan_mental.csv')
    evaluator = de.DataEvaluator(real_data, synthetic_data, synthesizer.get_metadata())
    evaluator.pairwise_plot('Suicide', 'Religion')
