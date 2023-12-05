from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot


class DataEvaluator:
    def __init__(self, real_data, synthetic_data, metadata):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata

    def diagnostic(self):
        diagnostic = run_diagnostic(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata)
        diagnostic.get_details(property_name='Data Validity')
        diagnostic.get_details(property_name='Data Structure')

    def quality(self):
        quality_report = evaluate_quality(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata)
        quality_report.get_details(property_name='Column Shapes')
        quality_report.get_details(property_name='Column Pair Trends')

    def column_plot(self, name):
        fig = get_column_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata,
            column_name=name
        )
        fig.show()

    def pairwise_plot(self, first, second):
        fig = get_column_pair_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=self.metadata,
            column_names=[first, second],
        )
        fig.show()
