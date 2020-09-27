import pandas as pd
import os
from Farm_Eval import binWindResourceData
import numpy as np

data_path = 'Shell_Hackathon Dataset/'

power_curve = pd.read_csv(os.path.join(data_path, 'power_curve.csv'))
turbine_loc_test = pd.read_csv(os.path.join(data_path, 'turbine_loc_test.csv'))

wind_data_2007 = pd.read_csv(os.path.join(data_path, 'Wind Data/wind_data_2007.csv'))

wind_data = pd.DataFrame()
path_wind = os.path.join(data_path, 'Wind Data')
for filename in os.listdir(path_wind):
    if filename.endswith('.csv'):
        data = pd.read_csv(os.path.join(path_wind, filename))
        wind_data = wind_data.append(data)
        #new_table_list.append(filename.split(".")[0])

wind_data = wind_data.reset_index(drop=True)

bin_data = binWindResourceData(os.path.join(data_path, 'Wind Data/wind_data_2007.csv'))

df = pd.read_csv(os.path.join(data_path, 'Wind Data/wind_data_2007.csv'))
wind_resource = df[['drct', 'sped']].to_numpy(dtype=np.float32)

# direction 'slices' in degrees
slices_drct = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
## slices_drct   = [360, 10.0, 20.0.......340, 350]
n_slices_drct = slices_drct.shape[0]

# speed 'slices'
slices_sped = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
               18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
n_slices_sped = len(slices_sped) - 1

# placeholder for binned wind
binned_wind = np.zeros((n_slices_drct, n_slices_sped),
                       dtype=np.float32)

# 'trap' data points inside the bins.
for i in range(n_slices_drct):
    for j in range(n_slices_sped):
        # because we already have drct in the multiples of 10
        foo = wind_resource[(wind_resource[:, 0] == slices_drct[i])]

        foo = foo[(foo[:, 1] >= slices_sped[j])
                  & (foo[:, 1] < slices_sped[j + 1])]

        binned_wind[i, j] = foo.shape[0]

wind_inst_freq = binned_wind / np.sum(binned_wind)
wind_inst_freq = wind_inst_freq.ravel()


grp = df.groupby(["drct", "sped"]).size()\
        .reset_index(name="frequency")



import plotly.graph_objects as go

fig = go.Figure()

for i in slices_sped:
    fig.add_trace(go.Barpolar(
        r=[77.5, 72.5, 70.0, 45.0, 22.5, 42.5, 40.0, 62.5],
        name= i ))

binned_wind[0]
fig.add_trace(go.Barpolar(
    r=[77.5, 72.5, 70.0, 45.0, 22.5, 42.5, 40.0, 62.5],
    name='11-14 m/s',
    marker_color='rgb(106,81,163)'
))
fig.add_trace(go.Barpolar(
    r=[57.5, 50.0, 45.0, 35.0, 20.0, 22.5, 37.5, 55.0],
    name='8-11 m/s',
    marker_color='rgb(158,154,200)'
))
fig.add_trace(go.Barpolar(
    r=[40.0, 30.0, 30.0, 35.0, 7.5, 7.5, 32.5, 40.0],
    name='5-8 m/s',
    marker_color='rgb(203,201,226)'
))
fig.add_trace(go.Barpolar(
    r=[20.0, 7.5, 15.0, 22.5, 2.5, 2.5, 12.5, 22.5],
    name='< 5 m/s',
    marker_color='rgb(242,240,247)'
))

fig.update_traces(text=slices_drct)
fig.update_layout(
    title='Wind Speed Distribution',
    font_size=16,
    legend_font_size=16,
    polar_radialaxis_ticksuffix='%',
    polar_angularaxis_rotation=90,

)
fig.show()

print([str(i) for i in slices_sped])

wd=pd.DataFrame(binned_wind)
drct = []
for i in range(0, 28):
    fig.add_trace(go.Barpolar(
        r=wd.loc[i],
        dr=10,
        r0=0,
        name=str(i) + '-' + str(i + 2) + ' ' + 'm/s'
    ))
    fig.add_trace(go.Barpolar(
        r=wd.loc[i],
        name = str(i) +'-'+ str(i+2) + ' '+'m/s'
    ))

    name_drct = str(i) +'-'+ str(i+2) + ' '+'m/s'
    print(name_drct)
    name_drct = str(name_drct)
    drct = drct.append(name_drct)

fig.update_traces(text=name_drct)
fig.update_layout(
    title='Wind Speed Distribution',
    font_size=16,
    legend_font_size=16,
    polar_radialaxis_ticksuffix='%',
    polar_angularaxis_rotation=90,

)

fig.show()

radian = [i*10 for i in wd.index]
wd['direct'] = radian



