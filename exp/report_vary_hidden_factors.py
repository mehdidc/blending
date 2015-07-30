from bokeh.io import vplot
from bokeh.models import ColumnDataSource, DataRange1d, Plot, LinearAxis, Grid, Circle, HoverTool, BoxSelectTool
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, NumberFormatter, StringEditor, IntEditor, NumberEditor, SelectEditor
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.browserlib import view
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_file, show, vplot

from collections import defaultdict

def save_reports(reports):
    output_file("vary_hidden_factors.html")
    

    datasets = set(r["dataset"] for r in reports)

    for d in datasets:
        R = filter(lambda r:r["dataset"]==d, reports)
        R = sorted(R, key=lambda r:r["latent_size"])

        for r in R:
            pass

if __name__ == "__main__":

    from lightexperiments.light import Light
    light = Light()
    light.launch()
    reports = light.db.find({"tags": ["decorrelation" "vary_hidden_factors"]})
    save_reports(list(reports))
    light.close()
