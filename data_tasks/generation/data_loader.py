from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata


class DataLoader:
    def __init__(self, filename):
        self.dataset = None
        self.load_dataset(filename)
        self.metadata = None
        self.define_metadata(filename)

    def load_dataset(self, name):
        datasets = load_csvs(folder_name='datasets/')
        self.dataset = datasets[name]

    def define_metadata(self, filename):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_csv(filepath=f'datasets/{filename}.csv')
        self.metadata.validate()

    def visualize_metadata(self):
        self.metadata.visualize(
            show_table_details='full',
            output_filepath='my_metadata.png'
        )

    def print_metadata(self):
        print(self.metadata.to_dict())

    def update_metadata(self):
        # TODO: Update the metadata with the new data
        self.metadata.validate()
        pass

    def validate_data(self):
        self.metadata.validate_data(data=self.dataset)

