from sdv.single_table import CTGANSynthesizer


class DataSynthetizer:
    def __init__(self, name, data_loader=None, loaded_synthetizer=None):
        if loaded_synthetizer is not None:
            self.name = name
            self.synthesizer = loaded_synthetizer
        else:
            self.name = name
            self.data_loader = data_loader
            self.synthesizer = CTGANSynthesizer(data_loader.metadata, verbose=True)
            self.synthesizer.fit(self.data_loader.dataset)
            self.synthesizer.save(
                filepath=f'data_generation/saved_synthetizers/{name}_synthetizer.pkl'
            )

    def sample_data(self, n):
        return self.synthesizer.sample(num_rows=n, output_file_path=f'datasets/synthetic_{self.name}.csv')

    def get_metadata(self):
        return self.synthesizer.get_metadata()

    def get_parameters(self):
        return self.synthesizer.get_parameters()

    def get_loss_values(self):
        return self.synthesizer.get_loss_values()
