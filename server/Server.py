import base64
import io
import os
import pathlib
import uuid
from collections import defaultdict

import dash
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import dcc
from dash import html
import numpy as np
import csv
import pandas as pd
from dash.dependencies import Input, Output, State
from deap import creator
from sklearn.datasets import load_boston
import dash_uploader as du
from flask import send_from_directory

from gp_utils import gp_predict, ps_tree_predict, evolutionary_forest_predict, test_predict, scores_difference
from Server_Function import generate_table, parameter_process
from moea.feature_selection import feature_selection_utils
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance, feature_append
from sklearn.model_selection import train_test_split

app = dash.Dash(
    __name__,
    title='PlatEML',
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

server = app.server
app.config.suppress_callback_exceptions = True
global_dict = defaultdict(dict)

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

du.configure_upload(app, folder='temp')

# data
df_training_data = []
df_result_data = []

translate = {
    "chinese":
        {
            "Parameter_Control_Panel": "参数控制面板",
            "Tournament_Size": "锦标赛规模",
            "Operators": "运算符",
            "Multi_objective_Operators": "多目标算子",
            "Training_Data": "训练数据",
            "Language": "语言",
            "Task_Type": "任务类型",
            "Interpretable_Modeling": "可解释建模",
            "Evolutionary_Forest": "演化森林",
            "Feature_Selection": "特征选择",
            "Evolutionary_Algebra": "演化代数",
            "Population_Size": "种群大小",
            "Training_Progress": "训练进度",
            "Start_up": "启动",
            "Training_Results": "训练结果",
            "Test_Results": "测试结果",
            "Visualization_Display": "可视化展示",
            "Download_Training_Results": "下载训练结果",
            "Decision_Trees": "决策树",
            "Feature_Visualization": "特征可视化",
            "Feature": "特征",
            "Importance": "重要性",
            "Use_the_feature": "使用以上特征训练其它模型",
            "Model_Information": "模型信息",
            "Display_Feature_Number": "展示特征编号",
            "Pareto_Frontier_Chart": "帕累托前沿图",
            "Interpretable_Model": "可解释模型",
            "Average_Error": "平均误差",
            "Expression": "表达式",
            "Error_numeric": "错误: 存在非数值类型或空值(注意上传数据应该无索引)",
            "Variable": "变量",
            "Target_Variable": "目标变量",
            "Number": "编号",
            "Error_feature": "错误: 与训练数据特征数量不相同",
            "Download_Test_Results": "下载测试结果",
            "Effect_of_Feature_Construction": "特征构建的效果图",
            "Loading_New_Features": "载入新特征",
            "Error_no_ef": "错误: 请先使用演化森林训练出新的特征"
        },
    "english":
        {
            "Parameter_Control_Panel": "Parameter Control Panel",
            "Tournament_Size": "Tournament Size",
            "Operators": "Operators",
            "Multi_objective_Operators": "Multi-objective Operators",
            "Training_Data": "Training Data",
            "Language": "Language",
            "Task_Type": "Task Type",
            "Interpretable_Modeling": "Interpretable Modeling",
            "Evolutionary_Forest": "Evolutionary Forest",
            "Feature_Selection": "Feature Selection",
            "Evolutionary_Algebra": "Evolutionary Algebra",
            "Population_Size": "Population Size",
            "Training_Progress": "Training Progress",
            "Start_up": "Start up",
            "Training_Results": "Training Results",
            "Test_Results": "Test Results",
            "Visualization_Display": "Visualization Display",
            "Download_Training_Results": "Download Training Results",
            "Decision_Trees": "Decision Trees",
            "Feature_Visualization": "Feature Visualization",
            "Feature": "Feature",
            "Importance": "Importance",
            "Use_the_feature": "Use the above features to train other models",
            "Model_Information": "Model Information",
            "Display_Feature_Number": "Display Feature Number",
            "Pareto_Frontier_Chart": "Pareto Frontier Chart",
            "Interpretable_Model": "Interpretable Model",
            "Average_Error": "Average Error",
            "Expression": "Expression",
            "Error_numeric": "Error: Non-numeric or null value exists (note that uploaded data should have no index)",
            "Variable": "Var",
            "Target_Variable": "Target Var",
            "Number": "Num",
            "Error_feature": "Error: Not the same number of features as the training data",
            "Download_Test_Results": "Download Test Results",
            "Effect_of_Feature_Construction": "Effect of Feature Construction",
            "Loading_New_Features": "Loading New Features",
            "Error_no_ef": "Error: Please train new features using evolutionary forest first"
        }
}

def description_card(session_id):
    """
    控制面板描述卡
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("ECNU"),
            html.H3(children="可解释机器学习建模系统", style={'text-align': 'left'}),
            html.Div(
                id="intro",
                children=translate[global_dict[session_id]['language']]['Parameter_Control_Panel'],
            ),
        ],
    )


operator_list = ['+', '-', '*', '/']

selector_list = ['NSGA2', 'MOEA/D', 'RM-MEDA', 'NSGA3', 'C-TAEA']

@app.callback(Output('option_bar', 'children'),
              [Input('task-type', 'value')],
              State('session-id', 'children')
)
def get_option_bar(task, session_id):
    print('Task', task)
    if task is None or task == '' or task == 'interpretable-modeling':
        return [
            html.P(children=translate[global_dict[session_id]['language']]['Tournament_Size'], style={'font-weight': 'bold'}),
            html.Div(dcc.Input(id="tournament_size", value='5', type="number")),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Operators'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="operator-select",
                options=[{"label": i, "value": i} for i in operator_list],
                value=operator_list,
                multi=True,
            ),
            html.Div(dcc.Input(id="moea-operator-select", value=''), style={"display": "none"}),
        ]
    elif task == 'evolutionary-forest':
        return [
            html.Div(dcc.Input(id="tournament_size", value=''), style={"display": "none"}),
            html.Div(dcc.Input(id="operator-select", value=''), style={"display": "none"}),
            html.Div(dcc.Input(id="moea-operator-select", value=''), style={"display": "none"}),
        ]
    elif task == 'PS-Tree':
        ps_tree_operator_list = ['NSGA2', 'NSGA3', 'Lexicase', 'IBEA', 'SPEA2']
        return [
            html.Div(dcc.Input(id="tournament_size", value=''), style={"display": "none"}),
            html.Div(dcc.Input(id="operator-select", value=''), style={"display": "none"}),
            html.P(children=translate[global_dict[session_id]['language']]['Multi_objective_Operators'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="moea-operator-select",
                options=[{"label": i, "value": i} for i in ps_tree_operator_list],
                value=selector_list[0],
            ),
        ]
    else:
        return [
            html.Div(dcc.Input(id="tournament_size", value=''), style={"display": "none"}),
            html.Div(dcc.Input(id="operator-select", value=''), style={"display": "none"}),
            html.P(children=translate[global_dict[session_id]['language']]['Multi_objective_Operators'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="moea-operator-select",
                options=[{"label": i, "value": i} for i in selector_list],
                value=selector_list[0],
            ),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Evolutionary_Forest'], style={'font-weight': 'bold'}),
            html.Div(
                children=html.Button(id="upload-feature",
                                     children=translate[global_dict[session_id]['language']]['Loading_New_Features'], n_clicks=0),
            ),
        ]



def generate_control_card(session_id):
    """
    :return: A Div containing control options.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P(children=translate[global_dict[session_id]['language']]['Training_Data']),
            # html.Div(id='data_hidden'),
            du.Upload(id='upload_data', text='Upload(.csv)',
                      text_completed='Completed: ', cancel_button=True,
                       pause_button=True, filetypes=['csv'],
                      default_style={'height': '50px', 'display': 'flex',
                                     'justify-content': 'center', 'align-items': 'center' },
                      upload_id=session_id
                      ),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Language']),
            html.Div(
                dbc.Container(
                    dbc.Row(
                    [
                        dbc.Col(width=6, children=html.Button(id="chinese", children="中文", n_clicks=0)),
                        dbc.Col(width=6, children=html.Button(id="english", children="English", n_clicks=0))
                    ]),
                )
            ),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Task_Type']),
            html.Div(
                dcc.Dropdown(
                    id="task-type",
                    options=[
                        {
                            'label': html.Div(children=translate[global_dict[session_id]['language']]['Interpretable_Modeling']),
                            'value': 'interpretable-modeling'
                        },
                        {
                            'label': 'PS-Tree',
                            'value': 'PS-Tree'
                        },
                        {
                            'label': html.Div(children=translate[global_dict[session_id]['language']]['Evolutionary_Forest']),
                            'value': 'evolutionary-forest'
                        },
                        {
                            'label': html.Div(children=translate[global_dict[session_id]['language']]['Feature_Selection']),
                            'value': 'feature-selection'
                        }
                    ],
                    value='interpretable-modeling',
                    clearable=False,
                )
            ),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Evolutionary_Algebra']),
            html.Div(dcc.Input(id="generation", value='3', type="number")),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Population_Size']),
            html.Div(dcc.Input(id="pop_size", value='20', type="number")),
            html.Br(),
            html.Div(id='option_bar'),
            html.Br(),
            html.P(children=translate[global_dict[session_id]['language']]['Training_Progress']),
            dbc.Progress(id="progress", striped=True),
            dcc.Interval(id="interval", interval=250, n_intervals=0),
            html.Br(),
            html.Div(
                id="start-btn-outer",
                children=html.Button(id="start-btn", children=translate[global_dict[session_id]['language']]['Start_up'], n_clicks=0),
            ),
            html.Div(id="language-type")
        ],
    )

def server_layout():
    # global layout
    session_id = str(uuid.uuid4())
    print('session_id', session_id)
    global_dict[session_id]['language'] = 'english'
    return html.Div([ html.Div(session_id, id='session-id', style={'display': 'none'}), dcc.Location(id='redirect-url'), html.Div(id="app-container")])
@app.callback(
    [
        Output('app-container', 'children'),
    ],
    [
        Input('redirect-url', 'pathname'),
    ],
    [
        State('session-id', 'children'),
    ])
def server_layout_page(pathname, session_id):
    main_page = [html.Div(
                # Banner
                [html.Div(
                    id="banner",
                    className="banner",
                    children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
                ),
                # Left column
                html.Div(
                    id="left-column",
                    # className="four columns",
                    className="three columns",
                    children=[description_card(session_id), generate_control_card(session_id)]
                             + [
                                 html.Div(
                                     ["initial child"], id="output-clientside", style={"display": "none"}
                                 )
                             ],
                ),
                # Right column
                html.Div(
                    id="right-column",
                    className="nine columns",
                    children=[
                        html.Div(
                            id="training_data_card",
                            children=[
                                html.B(children=translate[global_dict[session_id]['language']]['Training_Data']),
                                html.Hr(),
                                html.Div(id='training-state'),
                            ],
                        ),
                        html.Div(
                            id="result_card",
                            children=[
                                html.B(children=translate[global_dict[session_id]['language']]['Training_Results']),
                                html.Hr(),
                                html.Div(id='result-state'),
                                html.Div(id='download-result',
                                         style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                            ],
                        ),
                        html.Div(
                            id="test_card",
                            children=[
                                html.B(children=translate[global_dict[session_id]['language']]['Test_Results']),
                                html.Hr(),
                                html.Div(id='test-state'),
                                html.Div(id='download-test',
                                         style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                                html.Div(id='upload-test')
                            ],
                        ),
                        html.Div(
                            id="model_card",
                            children=[
                                html.B(children=translate[global_dict[session_id]['language']]['Visualization_Display']),
                                html.Hr(),
                                html.Div(id='feature-table'),
                                html.Div(id='model-state', style={'display': 'flex'}),
                            ]
                        ),
                        html.Br()
                    ],
                ),
        ])]
    if pathname == '/chinese' or pathname == '/' or pathname == '/english':
        return main_page
    else:
        return [html.P("404")]

app.layout = server_layout()

@app.callback(
    [
        # Output('redirect-url-container', 'children'),
        Output('redirect-url', 'href'),
        Output('chinese', 'n_clicks'),
        Output('english', 'n_clicks'),
    ],
    [
        Input('chinese', 'n_clicks'),
        Input('english', 'n_clicks'),
    ],
    [
        State('session-id', 'children')
    ]
)
def jump_to_page(a_n_clicks, b_n_clicks, session_id):
    # ctx = dash.callback_context
    # if ctx.triggered[0]['prop_id'] == 'chinese.n_clicks':
    #     return dcc.Location(id='url', href='/chinese')
    if a_n_clicks > 0:
        global_dict[session_id]['language'] = 'chinese'
        return '/chinese', 0, 0
    # elif ctx.triggered[0]['prop_id'] == 'english.n_clicks':
        # return dcc.Location(id='url', href='/english')
    elif b_n_clicks > 0:
        global_dict[session_id]['language'] = 'english'
        return '/english', 0, 0
    return dash.no_update

infix_map = {
    'ARG': 'X',
    'subtract': 'sub',
    'multiply': 'mul',
    'analytical_quotient': 'div',
}

@app.callback([
    Output('result-state', 'children'),
    Output('download-result', 'children'),
    Output('model-state', 'children'),
    Output('upload-test', 'children'),
],
    [
        Input('start-btn', 'n_clicks'),
    ],
    [
        State('session-id', 'children'),
        State('generation', 'value'),
        State('pop_size', 'value'),
        State('tournament_size', 'value'),
        State('operator-select', 'value'),
        State('moea-operator-select', 'value'),
        State('task-type', 'value'),
    ])
def update_output(n_clicks, session_id, generation, pop_size, tournament_size, operator_select,
                  moea_operator_select, task_type):
    print('output session_id', session_id)
    if n_clicks > 0:
        state = [0]
        global_dict[session_id]['state'] = state
        if task_type in ['interpretable-modeling', 'evolutionary-forest', 'PS-Tree']:
            pop_size = parameter_process(pop_size, 20)
            generation = parameter_process(generation, 20)
            tournament_size = parameter_process(tournament_size, 5)
            global_data = global_dict[session_id]['global_data']

            if hasattr(creator, "FitnessMin"):
                delattr(creator, "FitnessMin")
            if hasattr(creator, "Individual"):
                delattr(creator, "Individual")
            if task_type == 'interpretable-modeling':  # gplearn
                data, base64_datas, train_infos, gp = gp_predict(pop_size, generation, tournament_size, global_data,
                                                             operator_select, state, global_dict[session_id]['language'])
            elif task_type == 'PS-Tree':
                data, base64_datas, train_infos, gp = ps_tree_predict(pop_size, generation, tournament_size, global_data,
                                                                  moea_operator_select, state, global_dict[session_id]['language'])
            elif task_type == 'evolutionary-forest':
                data, base64_datas, train_infos, gp = evolutionary_forest_predict(pop_size, generation, tournament_size,
                                                                              global_data, moea_operator_select, state, global_dict[session_id]['language'])
            training_infos = {
                'data': data,
                'base64_datas': base64_datas,
                'train_infos': train_infos,
                'gp': gp
            }
            global_dict[session_id]['training_infos'] = training_infos

            ter_upload = html.Div(du.Upload(id='upload_testdata', text='Upload(.csv)',
                      text_completed='Completed: ', cancel_button=True,
                       pause_button=True, filetypes=['csv'],
                      default_style={'height': '50px', 'display': 'flex',
                                     'justify-content': 'center', 'align-items': 'center' },
                      upload_id=session_id
                      ))

            for i in range(len(base64_datas)):
                base64_datas[i] = base64_datas[i].decode("utf-8")
            if task_type == 'PS-Tree':
                return (generate_table(data.T, session_id), html.Div([html.Li(html.A(children=translate[global_dict[session_id]['language']]['Download_Training_Results'], href=f'/download/{str(session_id)}r.csv', target='_blank'))]),
                        [html.Div(style={'width': '100%'},
                             children=[
                                 html.H4(children=translate[global_dict[session_id]['language']]['Decision_Trees'], style={'textAlign': 'center', 'font-size': '2.0rem'}),
                                 html.Img(src='data:image/png;base64,{}'.format(base64_datas[0]),
                                          style={'display': 'block', 'margin-left': 'auto',
                                                 'margin-right': 'auto',
                                                 'overflow-x': 'scroll', 'overflow-y': 'scroll',
                                                 'max-height': '400px',
                                                 'max-width': '100%'})])], ter_upload)
            elif task_type == 'evolutionary-forest':
                r = global_dict[session_id]['training_infos']['gp']
                code_importance_dict = get_feature_importance(r, simple_version=False)
                names, importance = list(code_importance_dict.keys()), list(code_importance_dict.values())
                feature_importance = np.array(importance)

                data_dict = {'feature_names': names, 'feature_importance': feature_importance}
                fi_df = pd.DataFrame(data_dict)
                fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
                global_dict[session_id]['fi_df'] = fi_df

                feature_names = [a.split(':')[1] for a in names]
                for i in infix_map:
                    feature_names = [a.replace(i, infix_map[i]) for a in feature_names]
                data_dict1 = {'feature_names': feature_names, 'feature_importance': feature_importance}
                fi_df1 = pd.DataFrame(data_dict1)
                fi_df1.sort_values(by=['feature_importance'], ascending=False, inplace=True)
                global_dict[session_id]['feature_table'] = fi_df1  #存下特征及其重要性的dataframe
                ef_importance = fi_df1['feature_importance'].values.tolist()
                ef_names = fi_df1['feature_names'].values.tolist()

                return (generate_table(data.T, session_id), html.Div([html.Li(html.A(children=translate[global_dict[session_id]['language']]['Download_Training_Results'], href=f'/download/{str(session_id)}r.csv', target='_blank'))]),
                        [html.Div(style={'width': '100%'},
                             children=[
                                 html.H4(children=translate[global_dict[session_id]['language']]['Evolutionary_Forest'], style={'textAlign': 'center', 'font-size': '2.0rem'}),
                                 html.Img(src='data:image/png;base64,{}'.format(base64_datas[0]),
                                          style={'display': 'block', 'margin-left': 'auto',
                                                 'margin-right': 'auto',
                                                 'overflow-x': 'scroll', 'overflow-y': 'scroll',
                                                 'max-height': '400px',
                                                 'max-width': '100%'}),
                                 html.H4(children=translate[global_dict[session_id]['language']]['Feature_Visualization'], style={'textAlign': 'center', 'font-size': '2.0rem'}),
                                 html.Div
                                 (
                                         dbc.Checklist(
                                             id='check-list-input',
                                             options=[
                                                 {'label': f"{translate[global_dict[session_id]['language']]['Feature']}: {ef_names[i]} ------ {translate[global_dict[session_id]['language']]['Importance']}: {round(ef_importance[i], 3)}",
                                                  'value': i}
                                                 for i in range(0, len(feature_names))
                                             ],
                                             value=np.arange(0, int(len(feature_names)/3)),
                                             style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto',
                                                'overflow-y': 'scroll', 'max-height': '400px', 'max-width': '100%',
                                                'border-style': 'solid', 'padding-left': '15px', 'textAlign': 'center',
                                                'border-width': 'thin', 'padding-top': '10px', 'padding-bottom': '10px',
                                                'border-color': '#f0f0f4',
                                                    },

                                         ),
                                 ),
                                 html.Br(),
                                 html.Div(html.Button(id="restart-btn", children=translate[global_dict[session_id]['language']]['Use_the_feature'], n_clicks=0), style={'textAlign': 'center'}),
                                 html.Div(id="feature-png"),

                                 html.Br(),
                             ])], ter_upload)
            else:
                return (generate_table(data.T, session_id), html.Div([html.Li(html.A(children=translate[global_dict[session_id]['language']]['Download_Training_Results'], href=f'/download/{str(session_id)}r.csv', target='_blank'))]),
                    [
                        html.Div(id='model_image', style={'width': '75%'}),
                        html.Div(
                            children=[html.H6(children=translate[global_dict[session_id]['language']]['Model_Information'], style={'font-size': '1.6rem'}),
                                      html.P(children=translate[global_dict[session_id]['language']]['Display_Feature_Number']),
                                      dcc.Dropdown(options=[{"label": i, "value": i} for i in range(len(base64_datas))],
                                                   id='model_id_selection',
                                                   value=0),
                                      html.Div(id='model_info')]
                            , style={
                                'width': '25%'
                            }
                        )], ter_upload
                )
        else:
            print('Feature selection task!')
            global_data = global_dict[session_id]['global_data']
            if 'new_feature' in global_dict[session_id]:
                global_data = global_dict[session_id]['new_feature']
            pf_data, pf_figure = feature_selection_utils(pop_size, generation, global_data, moea_operator_select,
                                                         state, global_dict[session_id]['language'])
            print('Task finished!')
            training_infos = {
                'data': pf_data,
                'base64_datas': pf_figure,
            }
            global_dict[session_id]['training_infos'] = training_infos
            return (generate_table(pf_data, session_id, 2), html.Div([html.Li(html.A(children=translate[global_dict[session_id]['language']]['Download_Training_Results'], href=f'/download/{str(session_id)}r.csv', target='_blank'))]),
                   [
                       html.Div(
                children=[
                            html.H4(children=translate[global_dict[session_id]['language']]['Pareto_Frontier_Chart'], style={'textAlign': 'center', 'font-size': '2.0rem'}),
                            html.Img(src='data:image/png;base64,{}'.format(pf_figure.decode("utf-8")),
                                    style={'display': 'block', 'margin-left': 'auto',
                                          'margin-right': 'auto',
                                          'overflow-x': 'scroll', 'overflow-y': 'scroll',
                                          'max-height': '400px',
                                          'max-width': '100%'})],
                style={
                    'margin-left': 'auto',
                    'margin-right': 'auto',
                }
            )], '')
    else:
        return '', '', '', ''

def render_image(model_id, session_id):
    # render the image of gp
    training_infos = global_dict[session_id]['training_infos']
    data, base64_datas, train_infos = training_infos['data'], training_infos['base64_datas'], \
                                      training_infos['train_infos']
    return [html.H4(children=translate[global_dict[session_id]['language']]['Interpretable_Model'], style={'textAlign': 'center', 'font-size': '2.0rem'}),
            html.Img(src='data:image/png;base64,{}'.format(base64_datas[model_id]),
                     style={'display': 'block', 'margin-left': 'auto',
                            'margin-right': 'auto',
                            'overflow-x': 'scroll', 'overflow-y': 'scroll',
                            'max-height': '400px',
                            'max-width': '100%'})]


def render_info(model_id, session_id):
    training_infos = global_dict[session_id]['training_infos']
    # render loss information
    data, base64_datas, train_infos = training_infos['data'], training_infos['base64_datas'], \
                                      training_infos['train_infos']
    return [
        f"{translate[global_dict[session_id]['language']]['Average_Error']}:",
        html.Br(),
        train_infos[model_id]['mse'],
        html.Br(),
        f"{translate[global_dict[session_id]['language']]['Expression']}:",
        html.Br(),
        train_infos[model_id]['tree'],
        # html.Div(style={'padding-bottom':'5px'})
    ]


@app.callback([
    Output('model_image', 'children'),
    Output('model_info', 'children'),
],
    [Input('model_id_selection', 'value')],
    [State('session-id', 'children')]
)
def model_selection(model_id, session_id):
    if model_id is None:
        return render_image(0, session_id), render_info(0, session_id)
    else:
        return render_image(model_id, session_id), render_info(model_id, session_id)


@app.callback(Output('training-state', 'children'),
              [Input('upload_data', 'isCompleted')],
              [State('upload_data', 'fileNames'),
               State('session-id', 'children')])
def update_output(isCompleted, fileNames, session_id):
    print('data loader session_id', session_id)
    if isCompleted:
        # content_type, content_string = content.split(',')
        # df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')), header=None, delimiter='\t')

        df = pd.read_csv("temp\\" + str(session_id) + "\\" + fileNames[0], header=None)

        # 判断是否合法
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True) # 替换空字符串为缺失值
        empty_data = df.isnull().any() # 是否存在缺失值
        number_data = df.dtypes # 数据类型
        for i in range(0, df.shape[1]):
            if (empty_data[i] == True) or (number_data[i] != "float64" and number_data[i] != "int64"):
                return html.Div(children=translate[global_dict[session_id]['language']]['Error_numeric'])
    else:
        print(os.getcwd())

        #boston数据集
        # data_url_boston = "http://lib.stat.cmu.edu/datasets/boston"
        # raw_df_boston = pd.read_csv(data_url_boston, sep="\s+", skiprows=22, header= None)
        # data_boston = np.hstack([raw_df_boston.values[::2, :], raw_df_boston.values[1::2, :2]])
        # target_boston = raw_df_boston.values[1::2, 2]
        # x, y = data_boston, target_boston
        x, y = load_boston(return_X_y=True)
        df = pd.DataFrame(np.concatenate([x, np.reshape(y, (-1, 1))], axis=1))

    global_dict[session_id]['global_data'] = df
    df.columns = [f"{translate[global_dict[session_id]['language']]['Variable']}" + str(c) if i < len(df.columns) - 1 else f"{translate[global_dict[session_id]['language']]['Target_Variable']}" for i, c in enumerate(df.columns)]
    df.insert(0, f"{translate[global_dict[session_id]['language']]['Number']}", np.arange(0, len(df)))
    return generate_table(df, 0, 2, 2)

@app.callback([Output('test-state', 'children'),
               Output('download-test', 'children'),
               ],
              [Input('upload_testdata', 'isCompleted')],
              [State('upload_testdata', 'fileNames'),
               State('session-id', 'children')])
def update_test(isCompleted, fileNames, session_id):
    if isCompleted:
        df = pd.read_csv("temp\\" + str(session_id) + "\\" + fileNames[0], header=None)
        # 判断是否合法
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True) # 替换空字符串为缺失值
        empty_data = df.isnull().any() # 是否存在缺失值
        number_data = df.dtypes # 数据类型
        for i in range(0, df.shape[1]):
            if (empty_data[i] == True) or (number_data[i] != "float64" and number_data[i] != "int64"):
                return html.Div(children=translate[global_dict[session_id]['language']]['Error_numeric']), ''
        global_dict[session_id]['global_test'] = df
        flag = 0
        if global_dict[session_id]['global_data'].shape[1] == df.shape[1]:
            flag = 2
        elif global_dict[session_id]['global_data'].shape[1] == df.shape[1]+1:
            flag = 1
        else:
            return html.Div(children=translate[global_dict[session_id]['language']]['Error_feature']), ''
        data = test_predict(df, global_dict[session_id]['training_infos']['gp'], flag, session_id, global_dict[session_id]['language'])
        #每次新上传数据，删掉之前ef训练的特征
        if 'feature_table' in global_dict[session_id]:
            global_dict[session_id].pop('feature_table')
        if 'new_feature' in global_dict[session_id]:
            global_dict[session_id].pop('new_feature')

        return generate_table(data.T, session_id, 1, 3), html.Div([html.Li(html.A(children=translate[global_dict[session_id]['language']]['Download_Test_Results'], href=f'/download/{str(session_id)}t.csv', target='_blank'))])
    else:
        return '', ''

@app.callback([Output("progress", "value"), Output("progress", "label")],
              [Input("interval", "n_intervals")],
              State("session-id", 'children'))
def advance_progress(n, session_id):
    if not session_id in global_dict or not 'state' in global_dict[session_id]:
        return 0, '0'
    state = global_dict[session_id]['state']
    progress = int(min(state[0], 100))
    return progress, f"{progress}.00%" if progress >= 5 else ""

@app.server.route('/download/<file>')
def download(file):
    return send_from_directory('download', file)

@app.callback(
    Output('feature-png', 'children'),
    [
        Input('restart-btn', 'n_clicks'),
        # Input('check-list-input', 'value')
    ],
    [
        State('check-list-input', 'value'),
        State('session-id', 'children')
    ]
)
def check_list_output(clicks, value, session_id):
    if clicks > 0:
        fi_df = global_dict[session_id]['fi_df']
        r = global_dict[session_id]['training_infos']['gp']
        feature_dataframe = fi_df.iloc[list(value)]
        x = global_dict[session_id]['global_data'].iloc[:, :-1]
        y = global_dict[session_id]['global_data'].iloc[:, -1]
        x1, x2, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        x_train, x_test = r.x_scaler.transform(x1), r.x_scaler.transform(x2)
        code_importance_dict = feature_dataframe['feature_names'].values.tolist()
        new_train = feature_append(r, x_train, code_importance_dict[:len(code_importance_dict)],
                                   only_new_features=True)
        new_test = feature_append(r, x_test, code_importance_dict[:len(code_importance_dict)],
                                   only_new_features=True)
        scores_difference(x_train, x_test, new_train, new_test, y_train, y_test, session_id)

        with open("temp/" + str(session_id) + ".png", 'rb') as img_obj:
            base64_data1 = base64.b64encode(img_obj.read())
        base64_data1 = base64_data1.decode("utf-8")
        return  [html.H4(children=translate[global_dict[session_id]['language']]['Effect_of_Feature_Construction'], style={'textAlign': 'center', 'font-size': '2.0rem'}),
                 html.Img(src='data:image/png;base64,{}'.format(base64_data1),
                          style={'display': 'block', 'margin-left': 'auto',
                                 'margin-right': 'auto',
                                 'max-height': '400px',
                                 'max-width': '100%'})]
    return ''

@app.callback(
    [
        Output('feature-table', 'children'),
        # Output('model-state', 'children'),
    ],
    [
        Input('upload-feature', 'n_clicks'),
        # Input('check-list-input', 'value')
    ],
    [
        State('session-id', 'children')
    ]
)
def feature_table_display(clicks, session_id):
    if clicks > 0:
        if 'feature_table' not in global_dict[session_id]:
            return [html.Div(children=translate[global_dict[session_id]['language']]['Error_no_ef'])]
        else:
            df = global_dict[session_id]['feature_table']
            df.insert(0, "Num", np.arange(0, len(df)))
            df.columns = [f"{translate[global_dict[session_id]['language']]['Number']}",
                          f"{translate[global_dict[session_id]['language']]['Feature']}",
                          f"{translate[global_dict[session_id]['language']]['Importance']}"]

            #以下是为了得到新的特征值集合
            fi_df = global_dict[session_id]['fi_df']
            r = global_dict[session_id]['training_infos']['gp']
            x = global_dict[session_id]['global_data'].iloc[:, :-1]
            y = global_dict[session_id]['global_data'].iloc[:, -1]
            xx = r.x_scaler.transform(x)
            code_importance_dict = fi_df['feature_names'].values.tolist()
            new_x = feature_append(r, xx, code_importance_dict[:len(code_importance_dict)],
                                       only_new_features=True)
            new_xx = pd.DataFrame(new_x)
            new_feature = pd.concat([new_xx, y], axis=1, join='outer')
            global_dict[session_id]['new_feature'] = new_feature
            return [html.Div([html.H4(children=translate[global_dict[session_id]['language']]['Feature_Visualization'],
                    style={'textAlign': 'center', 'font-size': '2.0rem'}),
                    html.Div(generate_table(df, session_id, 2, 2))])]
    return ['']

if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server()
    # app.run_server(host='0.0.0.0')
