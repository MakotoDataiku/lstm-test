# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
models = dataiku.Folder("DQ1im0k6")
models_info = models.get_info()
output = dataiku.Dataset("output")
output_df = output.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

output_scored_2_df = output_df # For this sample code, simply copy input to output


# Write recipe outputs
output_scored_2 = dataiku.Dataset("output_scored_2")
output_scored_2.write_with_schema(output_scored_2_df)
