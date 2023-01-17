from dash import html
from dash import dash_table
import os
from dash import dcc
import csv
import pandas as pd
import numpy as np

# def generate_table(dataframe, max_rows=10, max_columns=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns[:max_columns]])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns[:max_columns]
#             ]) for i in range(min(len(dataframe), max_rows))
#         ]),
#     ], className='data_table')

def generate_table(dataframe, session_id, flag=1, down_result=1):
    if flag == 1:
        c_list = dataframe.values.tolist()[0]
        dataframe.columns = c_list
        dataframe.drop(dataframe.index[[0, 0]], inplace=True)

    ter_Div = dash_table.DataTable(
        # id='dash-table',
        data=dataframe.to_dict('records'),
        columns=[{'name': column, 'id': column} for column in dataframe.columns],
        virtualization=True,
        style_as_list_view=True,
        page_size=10,
        style_header={'font-weight': 'bold', 'text-align': 'center'},
        # style_data={'font-family': 'Times New Roman', 'font-weight': 700,  'text-align': 'center'},
        style_cell={'font-family': 'Times New Roman', 'text-align': 'center'},
        style_data_conditional=[
            {
                'if': {
                    'row_index': 'odd'
                }, 'background-color': '#cfd8dc'
            }
        ]
    )
    if down_result == 1:
        dataframe.to_csv("download\\" + str(session_id) + "r.csv")
        return html.Div(ter_Div)
    elif down_result == 3:
        dataframe.to_csv("download\\" + str(session_id) + "t.csv")
        return html.Div(ter_Div)
    else:
        return html.Div(ter_Div)



def parameter_process(param, default_value):
    if param is None or param == '':
        return default_value
    return int(param)
