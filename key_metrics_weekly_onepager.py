import datetime as dt
from datetime import datetime
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import re 
import os
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from plotly.subplots import make_subplots

#STYLE

external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100;400&display=swap'
]

def human_format(num, format='1f', sign='+'):
    if num is None:
        return None
    if type(num) == str:
        return f'{num}'

    if num >= 1e9:
        return f"{round(num / 1e9, 1):{sign}.{format}}B"
    elif num >= 1e6:
        return f"{round(num / 1e6,1):{sign}.{format}}M"
    elif num >= 1e3:
        return f"{round(num / 1e3,1):{sign}.{format}}k"
    else:
        return f"{round(num):{sign}.0f}"

layout = dict(
    template='simple_white', showlegend=False, autosize=True,
    margin=dict(l=20, r=20, b=20, t=40, pad=5),
    xaxis=dict(visible=False),
    yaxis=dict(showline = False, ticks = '', ticklabelstep=2, tickfont=dict(size=25), ticklabelposition="inside", ticklabeloverflow='allow'),
    font=dict(family='IBM Plex Sans'),
    legend=dict(orientation='v', yanchor='bottom', y=0.05, xanchor='right', x=1, font=dict(size=33)),
    uniformtext=dict(mode="hide", minsize=30)
)

config = {'displayModeBar': False}
plot2_style = {'height': '418px', 'width': '99%'}
plot3_style = {'height': '333px', 'width': '99%'}
cards_class = 'card border-5'
headers = {'text-align': 'center', 'font-weight': 'bold', 'font-size': '400%'}
titles = {'text-align': 'center', 'font-weight': 'bold', 'font-size': '300%'}
subtitles = {'text-align': 'center', 'font-weight': 'bold', 'font-size': '250%'}
anno = dict(xref="paper", yref="paper", showarrow=False, y=1.14)
anno_font = 30


_format = '1f'

table_styles = dict(
    style_as_list_view=True,
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
        'lineHeight': '66px',
        'font-family':'IBM Plex Sans',
        'textAlign': 'center'
    },
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'fontSize': 35,
        'font-family':'IBM Plex Sans'
    },
    style_cell={
        'textAlign': 'center',
        'padding': '10px',
        'fontSize': 40,
        'font-family':'IBM Plex Sans'
    },
    style_cell_conditional=[
        {
            'if': {'column_id': ''},
            'textAlign': 'left'
        }
    ]
)

def highlight(invert=False, cols=['Δm', '_Δm', 'Δw', '_Δw']):
    up = ['green']*5
    down = ['#ffd6d6']*5
    zero = ['#A1A1A1']*5
    if invert:
        up, down = down, up
    return (
        [
            {
                'if': {
                    'filter_query': '{{{}}} contains "-"'.format(col, 0),
                    'column_id': col
                },
                'backgroundColor': c
            } for col, c in zip(cols, down)
        ]
    )

#DATA
tables = dict()
tables['hr'] = pd.DataFrame({
    'KPI': ['A', 'B', 'C'],
    'Current Value': [10,15,77],
    'Plan': [12,20,100]
})
tables['sales'] = pd.DataFrame({
    '2022 Q4': ['Total', 'A', '', 'B', '', 'C', ''],
    '': ['', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Done Q4': [215_200_100, 49_700_000, 65_100_000, 50_100_000, 50_100_000, 47_300_000, 2_000_100],
    'Δ Q3': [5_100_000, 3_495_000, 6_300_000, 3_700_000, 4_340_000, 11_940_000, 1_500_000],
    'Potential': [500_000_000, 100_000_000, 80_000_000, 60_000_000, 70_000_000, 30_000_000, 90_000_000],
    # 'YtD': [683_000, 25_000, 104_000, 121_000, 342_000, '', 116_000]
})

_audience = [451, 417]
survey = pd.DataFrame({
    'axis': [1, 2, 3, 4, 5],
    'current': [0, 15, 177, 219, 75],
    'previous': [6, 18, 159, 123, 96],
    'current_percent': [0, 0.309, 0.3642, 0.4506, 0.1543],
    'previous_percent': [0.0149, 0.0447, 0.3955, 0.3059, 0.2388]
})

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
RENDER = 'UNFORMATTED_VALUE'
SPREADSHEET_ID = '1QFw1CA5sUNPyp09TcBpfdST4I4zeJHZcrSzFGHJp5hA'
creds = Credentials.from_authorized_user_file(os.environ['GOOGLE_APPLICATION_CREDS'], SCOPES)
service = build('sheets', 'v4', credentials=creds)
result = service.spreadsheets().values().get(
    spreadsheetId=SPREADSHEET_ID, 
    range='METRICS HISTORY', 
    valueRenderOption=RENDER).execute()

metrics_history = pd.DataFrame(result.get('values', [])).iloc[1:,:-1]
metrics_history.columns = metrics_history.iloc[0,:]
metrics_history.drop(1,inplace=True)
metrics_history.drop(columns=[
    'Metric Group', 
    'Metric', 
    'Comment', 
    'Sparkline 2022', 
    'Vs Prev Period',	
    'Vs Past Year',	
    'Vs Plan'], inplace=True)
metrics_history.drop(columns=metrics_history.columns[1], inplace=True)
metrics_history.set_index('Metric ID', inplace=True)
metrics_history.columns = [datetime.fromordinal(datetime(1900, 1, 1).toordinal() + x - 2).strftime('%Y-%m-%d') \
     for x in metrics_history.columns]
metrics_history['Metric ID'] = metrics_history.index
metrics_history['Last Value'] = metrics_history.iloc[:,-2]
metrics_history['Prev Value'] = metrics_history.iloc[:,-4]
metrics_history['Metric Diff'] = metrics_history['Last Value'] - metrics_history['Prev Value']
metrics_history['Value And Diff'] = metrics_history['Last Value'].apply(
    lambda x: human_format(x, sign='')
) + \
' (' + \
metrics_history['Metric Diff'].apply(
    lambda x: human_format(x, sign='+')
) + \
')'
metrics_history['Metric Name'] = metrics_history['Metric ID'].apply(
    lambda x:re.findall('m\d', x)[0].replace('m', 'Metric ')
)
metrics_history['Region'] = metrics_history['Metric ID'].apply(
    lambda x:re.findall('region\d', x)[0].replace('region', 'Region ') if len(re.findall('region\d', x)) > 0 else 'Total'
)

group_a_table = metrics_history[metrics_history['Metric ID'].str.contains('group_a')].pivot(
    index='Region', 
    values= 'Value And Diff', 
    columns='Metric Name', ).reset_index()
data_actual_on = max([x for x in metrics_history.columns if re.match('202\d',x)])






#WIDGETS
plots = dict()
# survey
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=survey['axis'],
        y=survey['current_percent'],
        marker_color=['#9A9AFE', '#5D5DFD', '#2121FD', '#0202DE', '#0202A2'],
        customdata=np.transpose([(survey['current_percent'] - survey['previous_percent']).tolist()]),
        texttemplate="%{y:,.1%} (%{customdata[0]:,.1%})",
        textposition='outside'
    )
)
_anno = anno.copy()
_anno['y'] = 1.05
_mean = np.mean([v for k in [[i]*j for i, j in zip(survey['axis'], survey['current'])] for v in k])
fig.add_annotation(_anno, x=0, align='left', font=dict(size=40, color='#0202A2'), text=f"µ {human_format(_mean, _format, sign='')}")
#

fig.add_annotation(_anno, x=0, y=0.75, align='left', font=dict(size=35, color='#0202A2'), text=f"🫂 {human_format(_audience[0], _format, sign='')}")
fig.add_annotation(_anno, x=0.1, y=0.75, align='left', font=dict(size=35, 
    color=['green' if _audience[0] >= _audience[1] else 'red'][0]), 
    text=f"({human_format(_audience[0] - _audience[1], _format)})"
)
#
_scores = [survey['current'].sum(), survey['previous'].sum()]
fig.add_annotation(_anno, x=0, y=0.5, align='left', font=dict(size=35, color='#0202A2'), text=f"⭐ {human_format(_scores[0], _format, sign='')}")
fig.add_annotation(_anno, x=0.1, y=0.5, align='left', font=dict(size=35, 
    color=['green' if _scores[0] >= _scores[1] else 'red'][0]), 
    text=f"({human_format(_scores[0] - _scores[1], _format)})"
)
#
fig.update_layout(layout, margin=dict(l=20, r=20, b=50, t=0, pad=5), uniformtext=dict(mode="hide", minsize=25))
fig.update_yaxes(visible=False, range=[0, survey['current_percent'].max()+0.15])
fig.update_xaxes(visible=True, showline = False, ticks = '', tickfont_size=33)
plots['Team survey'] = fig

#GRAPHS: GROUP A
group_a_colors = ['#B9CFD4', '#007FFF']
for i in [1,2,3]:
    x_ = metrics_history.columns[:-7]
    key_ = 'group_a_m' + str(i)
    y_ = metrics_history.iloc[:,:-7].loc[key_].T
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=x_, y=np.cumsum(y_), marker=dict(color=group_a_colors[0]),
        ), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=x_, y=y_, 
            line=dict(color=group_a_colors[1], width=3), 
            mode='lines', connectgaps=True,
            x0=0
        ), secondary_y=True
    )
    fig.add_annotation(anno, x=0, align='left', font=dict(size=40, color='#0202A2'), text=f"{human_format(y_.values[-1], _format, sign='')}")
    _w2w = y_.values[-1] - y_.values[-2]
    fig.add_annotation(anno, x=0.45, align='right', font=dict(size=anno_font, 
        color=['green' if _w2w >= 0 else 'red'][0]), 
        text=f"{human_format(_w2w, _format)} Δw")
    _y2y = y_.values[-1] - y_.values[-53]
    fig.add_annotation(anno, x=1, align='right', font=dict(size=anno_font, 
        color=['green' if _y2y >= 0 else 'red'][0]), 
        text=f"{human_format(_y2y, _format)} Δy")
    fig.update_layout(layout, yaxis2=layout['yaxis'])
    fig.update_yaxes(rangemode="tozero") 
    plots[key_] = fig

#GRAPHS: GROUP B/D/E/G
anno2 = anno.copy()
anno2['y'] = 1.19
metric_ids = [x for x in metrics_history['Metric ID'].values if re.match('group_[b|d|e|g]',x)]
for key_ in metric_ids:
    if re.match('group_[d|g]', key_):
        anno2['y'] = 1.13
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for year, c, w in zip([2021, 2022], ['grey', '#0202A2'], [3, 10]):
        x_ = [x.replace('2021','2022') for x in metrics_history.columns if re.match(str(year),x)]
        y_ = metrics_history.loc[metrics_history['Metric ID'] == key_,[x for x in metrics_history.columns if re.match(str(year),x)]].values[0]
        fig.add_trace(
            go.Scatter(
            x =  x_,
            y =  y_,
            mode = 'lines',
            line=dict(color=c, width=w), name=year
            )
        )
    x_ = [x for x in metrics_history.columns if re.match(str('202\d'),x)]
    y_ = metrics_history.loc[metrics_history['Metric ID'] == key_,[x for x in metrics_history.columns if re.match(str('202\d'),x)]].values[0]
    fig.add_annotation(anno2, x=0, align='left', font=dict(size=40, color='#0202A2'), text=f"{human_format(y_[-1], _format, sign='')}")
    _w2w = y_[-1] - y_[-2]
    fig.add_annotation(anno2, x=0.45, align='right', font=dict(size=anno_font, 
            color=['green' if _w2w >= 0 else 'red'][0]), 
            text=f"{human_format(_w2w, _format)} Δw")
    _y2y = y_[-1] - y_[-53]
    fig.add_annotation(anno2, x=1, align='right', font=dict(size=anno_font, 
        color=['green' if _y2y >= 0 else 'red'][0]), 
        text=f"{human_format(_y2y, _format)} Δy")
    fig.update_layout(layout, yaxis2=layout['yaxis'])
    plots[key_] = fig

#LAYOUT
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]+external_stylesheets)
app.layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col(
                html.H2('v. 1.0', style={'text-align': 'left', 'font-size': '250%'}), 
            width=4),
            dbc.Col(
                    html.H1('⛩️ KEY METRICS WEEKLY ⛩️', style={'text-align': 'center', 'color': '#00A55A', 'font-size': '350%', 'font-weight': 'bold'}),
            width=4),
            dbc.Col(
                html.A(
                    html.H2('RAW DATA', style={'text-align': 'right', 'font-size': '250%', 'color': '#00A55A'}),
                href='https://docs.google.com/spreadsheets/d/1tGirptXgbN1kB0n0CWsbMYiYowu5Nc7PAqKKX2WyMF4/edit?usp=sharing', target='_blank'),
            width=1),
            dbc.Col(
                html.H2(f'DATA ACTUAL ON: {data_actual_on}', style={'text-align': 'right', 'font-size': '250%'}), 
            width=3),
        ]
    , style={'height': '60px'}, className='g-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Row([
                    dbc.Col(
                        html.H2('■ cumulative', style={'text-align': 'right', 'font-size': '200%', 'color': '#B9CFD4'}), 
                    width=2),
                    dbc.Col(
                        html.H2('━ current', style={'text-align': 'left', 'font-size': '200%', 'color': '#007FFF'}), 
                    width=2),
                    dbc.Col(html.H1('🛍️ GROUP A', style=headers), width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H2('🧸 Metric 1', style=titles),
                        dcc.Graph(figure=plots['group_a_m1'], config=config, style=plot2_style)
                    ]),
                    dbc.Col([
                        html.H2('🪆 Metric 2', style=titles),
                        dcc.Graph(figure=plots['group_a_m2'], config=config, style=plot2_style)
                    ]),
                    dbc.Col([
                        html.H2('🎈 Metric 3', style=titles),
                        dcc.Graph(figure=plots['group_a_m3'], config=config, style=plot2_style)
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(
                            data=group_a_table.to_dict('records'),
                            columns=[{'id': c, 'name': c} for c in group_a_table.columns],
                            style_data_conditional=highlight(cols=['Metric ' + str(i) for i in range(1,6)]),
                            **(table_styles | {'style_data': table_styles['style_data'] | {'lineHeight': '56px'}} | {'style_cell': table_styles['style_cell'] | {'fontSize': 40}})
                        )
                    ])
                ]),
            ], style={'border': '10px solid #dfdfdf'})
        ]),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.Row([
                            dbc.Col(
                                html.H2('', style={'text-align': 'left'}), 
                            width=2),
                            dbc.Col([
                                html.H1('💸 Group C', style=headers)
                            ], width=8),
                            dbc.Col(
                                html.H2('Δ to 18.11', style={'text-align': 'right'}), 
                            width=2)
                        ]),
                        dbc.Row(
                            dbc.Col([
                                dash_table.DataTable(
                                    data=tables['sales'].apply(lambda x: x.apply(human_format) if x.name in ['Δ Q3', ' Δ'] else x.apply(human_format, sign='')).to_dict('records'),
                                    columns=[{'id': c, 'name': c} for c in tables['sales'].columns],
                                    style_data_conditional=highlight(cols=['Δ', ' Δ']),
                                    **(table_styles | {'style_data': table_styles['style_data'] | {'lineHeight': '32px'}} | {'style_cell': table_styles['style_cell'] | {'fontSize': 36}}),
                                    #fill_width=False
                                )
                            ])
                        ),
                    ], class_name=cards_class),
                ]),
            ], style={'margin-bottom': '16px'}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.Row([
                            dbc.Col(
                                html.H2('━ 2022', style={'text-align': 'right', 'font-size': '200%', 'color': '#0202A2'}), 
                            width=2),
                            dbc.Col(
                                html.H2('- 2021', style={'text-align': 'left', 'font-size': '200%', 'color': 'grey'}), 
                            width=2),                            
                            dbc.Col(
                                html.H1('💫 Group D', style=headers)
                            , width=4),
                            dbc.Col(html.H2(''), width=2),
                            dbc.Col(html.H2(''), width=2),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H2('🌝 Metric 1', style=titles),
                                dcc.Graph(figure=plots['group_d_m1'], config=config, style=plot2_style)
                            ]),
                            dbc.Col([
                                html.H2('🌚 Metric 2', style=titles),
                                dcc.Graph(figure=plots['group_d_m2'], config=config, style=plot2_style)
                            ]),
                        ], className="g-0")
                    ], class_name=cards_class),
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.Row(
                            dbc.Col(
                                html.H1('🚦 GROUP E', style=headers)
                            )
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.H2('✈️ Metric 1', style=titles),
                                dcc.Graph(figure=plots['group_e_m1'], config=config, style=plot3_style)
                            ]),
                        ], className="g-0"),
                        dbc.Row([
                            dbc.Col([
                                html.H2('🚅 Metric 2', style=titles),
                                dcc.Graph(figure=plots['group_e_m2'], config=config, style=plot3_style)
                            ]),
                        ], className="g-0"),
                        dbc.Row([
                            dbc.Col([
                                html.H2('🚗 Metric 3', style=titles),
                                dcc.Graph(figure=plots['group_e_m3'], config=config, style=plot3_style)
                            ]),
                        ], className="g-0"),
                    ], class_name=cards_class)
                ], width=4),
            ]),
        ]),
    ], style={'height': '1160px'}, className='g-3'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Row([
    
                    dbc.Col(
                        html.H2('━ 2022', style={'text-align': 'right', 'font-size': '200%', 'color': '#0202A2'}), 
                    width=1),
                    dbc.Col(
                        html.H2('- 2021', style={'text-align': 'left', 'font-size': '200%', 'color': 'grey'}), 
                    width=1),
                    dbc.Col(html.H1('🍽️ GROUP B', style=headers), width=8),
                    # dbc.Col(html.H2('starting January 22 (vs. last year)', style={'text-align': 'right'}), width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.H2('1️⃣ Subgroup 1', style=titles),
                            dbc.Row([
                                html.H2('🍔 Metric 1', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m1'], config=config, style=plot3_style)
                            ]),
                            dbc.Row([
                                html.H2('🍕 Metric 2', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m2'], config=config, style=plot3_style)
                            ]),
                            dbc.Row([
                                html.H2('🌭 Metric 3', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m3'], config=config, style=plot3_style)
                            ]),
                        ], class_name='card border-3'),
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            html.H2('2️⃣ Subgroup 2', style=titles),
                            dbc.Row([
                                html.H2('🍏 Metric 4', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m4'], config=config, style=plot3_style)
                            ]),
                            dbc.Row([
                                html.H2('🍋 Metric 5', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m5'], config=config, style=plot3_style)
                            ]),
                            dbc.Row([
                                html.H2('🍉 Metric 6', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m6'], config=config, style=plot3_style)
                            ]),
                        ], class_name='card border-3'),
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            html.H2('3️⃣ Subgroup 3', style=titles),
                            dbc.Row([
                                html.H2('🍺 Metric 7', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m7'], config=config, style=plot3_style)
                            ]),
                            dbc.Row([
                                html.H2('🍷 Metric 8', style=subtitles),
                                dcc.Graph(figure=plots['group_b_m8'], config=config, style=plot3_style)
                            ]),
                            dbc.Row([
                                dbc.Row([
                                    html.H2('🥃 Metric 9', style=subtitles)
                                ]),
                                dcc.Graph(figure=plots['group_b_m9'], config=config, style=plot3_style)
                            ]),
                        ], class_name='card border-3'),
                    ], width=4)
                ])
            ], class_name=cards_class)
        ], width=6),
        dbc.Col([
            dbc.Col([
                dbc.Card([
                    dbc.Row(
                        dbc.Col([
                            html.H1('📞 GROUP F', style=headers),
                            dash_table.DataTable(
                                data=tables['hr'].apply(lambda x: x.apply(human_format, sign='')).to_dict('records'),
                                style_data_conditional=highlight(),
                                columns=[{'id': c, 'name': c} for c in tables['hr'].columns],
                                **(table_styles | {'style_data': table_styles['style_data'] | {'lineHeight': '35px'}} | {'style_cell': table_styles['style_cell'] | {'fontSize': 35}})
                            )
                        ])
                    , className="g-3", style={'margin-bottom': '15px'}),
                    dbc.Row(
                        dbc.Col([
                            dcc.Graph(figure=plots['Team survey'], config=config, style={'height': '360px'})
                        ])
                    , className="g-3"),
                ], style={'margin-right': '8px'}, class_name=cards_class),
            ], width=8),
            dbc.Card([
                dbc.Row([
                    dbc.Col(
                        html.H2('━ 2022', style={'text-align': 'right', 'font-size': '200%', 'color': '#0202A2'}), 
                    width=1),
                    dbc.Col(
                        html.H2('- 2021', style={'text-align': 'left', 'font-size': '200%', 'color': 'grey'}), 
                    width=1),
                    dbc.Col(
                        html.H1('🎼 GROUP G', style=headers),
                    width=8),
                    dbc.Col(html.H2(''), width=1),
                    dbc.Col(html.H2(''), width=1),
                ], style={'margin-top': '5px', 'margin-bottom': '10px'}),
                dbc.Row([
                    dbc.Col([
                        html.H2('🎸 Metric 1', style=titles),
                        dcc.Graph(figure=plots['group_g_m1'], config=config, style=plot2_style)
                    ], width=4),
                    dbc.Col([
                        html.H2('🎹 Metric 2', style=titles),
                        dcc.Graph(figure=plots['group_g_m2'], config=config, style=plot2_style)
                    ], width=4),
                    dbc.Col([
                        html.H2('🎺 Metric 3', style=titles),
                        dcc.Graph(figure=plots['group_g_m3'], config=config, style=plot2_style)
                    ], width=4)
                ], className="g-0"),
            ], style={'margin-top': '20px'}, class_name=cards_class)
        ], width=6),
    ], style={'height': '1160px'}, className='g-3')
], fluid=True, style={'height': '2480px', 'width': '3508px', 'font': '15px IBM Plex Sans', 'font-weight':'bold'})

if __name__ == '__main__':
    app.run_server(debug=True)
