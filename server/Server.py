import base64
import io
import os
import pathlib
import uuid
from collections import defaultdict

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from deap import creator
from sklearn.datasets import load_boston

from gp_utils import gp_predict, ps_tree_predict, evolutionary_forest_predict
from Server_Function import generate_table, parameter_process
from moea.feature_selection import feature_selection_utils

app = dash.Dash(
    __name__,
    title='可解释机器学习系统',
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server
app.config.suppress_callback_exceptions = True
global_dict = defaultdict(dict)

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()


def description_card():
    """
    控制面板描述卡
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("ECNU"),
            html.H3("可解释机器学习建模系统", style={'text-align': 'left'}),
            html.Div(
                id="intro",
                children="参数控制面板",
            ),
        ],
    )


operator_list = ['+', '-', '*', '/']

selector_list = ['NSGA2', 'MOEA/D', 'RM-MEDA', 'NSGA3', 'C-TAEA']


@app.callback(Output('option_bar', 'children'),
              [Input('task-type', 'value')])
def get_option_bar(task):
    print('Task', task)
    if task is None or task == '' or task == 'interpretable-modeling':
        return [
            html.P("锦标赛规模", style={'font-weight': 'bold'}),
            html.Div(dcc.Input(id="tournament_size", value='5', type="number")),
            html.Br(),
            html.P("运算符", style={'font-weight': 'bold'}),
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
            html.P("多目标算子", style={'font-weight': 'bold'}),
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
            html.P("多目标算子", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="moea-operator-select",
                options=[{"label": i, "value": i} for i in selector_list],
                value=selector_list[0],
            ),
        ]


def generate_control_card():
    """
    :return: A Div containing control options.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P('训练数据'),
            html.Div(id='data_hidden'),
            dcc.Upload(html.Button('上传文件'), id='upload_data'),
            html.Br(),
            html.P('任务类型'),
            html.Div(
                dcc.Dropdown(
                    id="task-type",
                    options=[
                        {
                            'label': '可解释建模',
                            'value': 'interpretable-modeling'
                        },
                        {
                            'label': 'PS-Tree',
                            'value': 'PS-Tree'
                        },
                        {
                            'label': '演化森林',
                            'value': 'evolutionary-forest'
                        },
                        {
                            'label': '特征选择',
                            'value': 'feature-selection'
                        }
                    ],
                    value='interpretable-modeling',
                    clearable=False,
                )
            ),
            html.Br(),
            html.P('演化代数'),
            html.Div(dcc.Input(id="generation", value='3', type="number")),
            html.Br(),
            html.P("种群大小"),
            html.Div(dcc.Input(id="pop_size", value='20', type="number")),
            html.Br(),
            html.Div(id='option_bar'),
            html.Br(),
            html.P("训练进度"),
            dbc.Progress(id="progress", value=0),
            dcc.Interval(id="interval", interval=250, n_intervals=0),
            html.Br(),
            html.Div(
                id="start-btn-outer",
                children=html.Button(id="start-btn", children="启动", n_clicks=0),
            ),
        ],
    )


def server_layout():
    # global layout
    session_id = str(uuid.uuid4())
    print('session_id', session_id)
    return html.Div(
        id="app-container",
        children=[
            # Banner
            html.Div(
                id="banner",
                className="banner",
                children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
            ),
            # Left column
            html.Div(
                id="left-column",
                # className="four columns",
                className="three columns",
                children=[description_card(), generate_control_card()]
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
                    ),
                    html.Div(
                        id="result_card",
                        children=[
                            html.B("训练结果"),
                            html.Hr(),
                            html.Div(id='result-state'),
                        ],
                    ),
                    html.Div(
                        id="model_card",
                        children=[
                            html.B("可视化展示"),
                            html.Hr(),
                            html.Div(id='model-state', style={'display': 'flex'}),
                        ]
                    ),
                    html.Br()
                ],
            ),
            html.Div(session_id, id='session-id', style={'display': 'none'}),
        ],
    )


app.layout = server_layout()


@app.callback([
    Output('result-state', 'children'),
    Output('model-state', 'children')
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
            if task_type == 'interpretable-modeling':
                data, base64_datas, train_infos = gp_predict(pop_size, generation, tournament_size, global_data,
                                                             operator_select, state)
            elif task_type == 'PS-Tree':
                data, base64_datas, train_infos = ps_tree_predict(pop_size, generation, tournament_size, global_data,
                                                                  moea_operator_select, state)
            elif task_type == 'evolutionary-forest':
                data, base64_datas, train_infos = evolutionary_forest_predict(pop_size, generation, tournament_size,
                                                                              global_data, moea_operator_select, state)
            training_infos = {
                'data': data,
                'base64_datas': base64_datas,
                'train_infos': train_infos
            }
            global_dict[session_id]['training_infos'] = training_infos
            for i in range(len(base64_datas)):
                base64_datas[i] = base64_datas[i].decode("utf-8")
            if task_type == 'PS-Tree':
                return (html.Div(
                    generate_table(data),
                ), [html.Div(style={'width': '100%'},
                             children=[
                                 html.H4(children=f'决策树', style={'textAlign': 'center', 'font-size': '2.0rem'}),
                                 html.Img(src='data:image/png;base64,{}'.format(base64_datas[0]),
                                          style={'display': 'block', 'margin-left': 'auto',
                                                 'margin-right': 'auto',
                                                 'overflow-x': 'scroll', 'overflow-y': 'scroll',
                                                 'max-height': '400px',
                                                 'max-width': '100%'})])])
            elif task_type == 'evolutionary-forest':
                return (html.Div(
                    generate_table(data),
                ), [html.Div(style={'width': '100%'},
                             children=[
                                 html.H4(children=f'演化森林',
                                         style={'textAlign': 'center', 'font-size': '2.0rem'}),
                                 html.Img(src='data:image/png;base64,{}'.format(base64_datas[0]),
                                          style={'display': 'block', 'margin-left': 'auto',
                                                 'margin-right': 'auto',
                                                 'overflow-x': 'scroll', 'overflow-y': 'scroll',
                                                 'max-height': '400px',
                                                 'max-width': '100%'})])])
            else:
                return (
                    html.Div(
                        generate_table(data),
                    ),
                    [
                        html.Div(id='model_image', style={'width': '75%'}),
                        html.Div(
                            children=[html.H6('模型信息', style={'font-size': '1.6rem'}),
                                      html.P("展示特征编号"),
                                      dcc.Dropdown(options=[{"label": i, "value": i} for i in range(len(base64_datas))],
                                                   id='model_id_selection',
                                                   value=0),
                                      html.Div(id='model_info')]
                            , style={
                                'width': '25%'
                            }
                        )]
                )
        else:
            print('Feature selection task!')
            global_data = global_dict[session_id]['global_data']
            pf_data, pf_figure = feature_selection_utils(pop_size, generation, global_data, moea_operator_select,
                                                         state)
            print('Task finished!')
            training_infos = {
                'data': pf_data,
                'base64_datas': pf_figure,
            }
            global_dict[session_id]['training_infos'] = training_infos
            return html.Div(
                generate_table(pf_data),
            ), html.Div(
                children=[html.H4(children=f'帕累托前沿图', style={'textAlign': 'center', 'font-size': '2.0rem'}),
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
            )
    else:
        return '', ''


def render_image(model_id, session_id):
    # render the image of gp
    training_infos = global_dict[session_id]['training_infos']
    data, base64_datas, train_infos = training_infos['data'], training_infos['base64_datas'], \
                                      training_infos['train_infos']
    return [html.H4(children=f'可解释模型', style={'textAlign': 'center', 'font-size': '2.0rem'}),
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
        '平均误差:',
        html.Br(),
        train_infos[model_id]['mse'],
        html.Br(),
        '表达式:',
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


@app.callback(Output('training_data_card', 'children'),
              [Input('upload_data', 'contents')],
              [State('session-id', 'children')])
def update_output(content, session_id):
    print('data loader session_id', session_id)
    if content is None:
        print(os.getcwd())
        x, y = load_boston(return_X_y=True)
        df = pd.DataFrame(np.concatenate([x, np.reshape(y, (-1, 1))], axis=1))
    else:
        content_type, content_string = content.split(',')
        df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')), header=None, delimiter='\t')
    global_dict[session_id]['global_data'] = df
    df.columns = ['变量' + str(c) if i < len(df.columns) - 1 else '目标变量' for i, c in enumerate(df.columns)]
    df.insert(0, '数据编号', np.arange(0, len(df)))
    return [html.B("训练数据"),
            html.Hr(),
            generate_table(df.loc[:4])]


@app.callback(Output("progress", "value"),
              [Input("interval", "n_intervals")],
              State("session-id", 'children'))
def advance_progress(n, session_id):
    if not session_id in global_dict or not 'state' in global_dict[session_id]:
        return 0
    state = global_dict[session_id]['state']
    return min(state[0], 100)


if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server()
    # app.run_server(host='0.0.0.0')
