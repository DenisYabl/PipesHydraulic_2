import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
import numpy as np

#from code.DFOperations.calculate_DF import calculate_DF
#from code.Visualisation.pyvis_v0 import draw_result_graph
from DFOperations.calculate_DF import calculate_DF
from Visualisation.pyvis_v0 import draw_result_graph

start_time = time.time()


def main():
    st.set_page_config(layout="wide")
    st.title("Расчет трубопроводной сети")
    option = st.selectbox('Выберите участок для расчета',
    ('ДНС 3', 'ДНС 1', 'ДНС 2'))
    selection_dict = {"ДНС 1":  "DNS1_real_values.csv", "ДНС 2":  "DNS2_real_values.csv", "ДНС 3":  "DNS3_real_values.csv"}
    dataset = pd.read_csv("../CommonData/" + selection_dict[option])
    use_coord = st.sidebar.checkbox(label='Отрисовка сети по координатам', value=True)
    coordinate_scaling = float(st.sidebar.text_input(f"Масштаб координат", '6'))
    EffD = st.sidebar.slider("Средний эффективный диаметр сети", min_value=0.01, max_value=1.0, value = 0.85, step = 0.01)
    dataset["effectiveD"] = EffD
    with st.beta_expander("Свойства жидкости"):
        with st.form("Свойства жидкости"):
            st.write("Свойства жидкости")
            cols = st.beta_columns(8)
            baseproperties = [('sat_P_bar', 66.7, "Давление насыщения"), ('plastT_C', 84.0, "Пластовая температура"), ('gasFactor', 39.0, "Газовый фактор"),
                             ('oildensity_kg_m3', 826.0, "Плотность сепарированной нефти"), ('waterdensity_kg_m3', 1015.0, "Плотность пластовой воды"),
                             ('gasdensity_kg_m3', 1.0, "Плотность газа"), ('oilviscosity_Pa_s', 35e-3, "Вязкость нефти"),
                             ('volumeoilcoeff', 1.015, "Объемный коэффициент")]
            property_dict = {}
            for i, col in enumerate(cols):
                property = baseproperties[i]
                property_dict.update({property[0]:col.text_input(f"{property[2]}", property[1])})
            submitted = st.form_submit_button("Submit")
            if submitted:
                for key in list(property_dict.keys()):
                    dataset[key] = float(property_dict[key])

    inlets_df = dataset[dataset["startIsSource"]]
    inlets_dict = {}
    inlets_df["__pad_num"] = inlets_df["__pad_num"].astype(float)
    for index, row in inlets_df.sort_values("__pad_num").iterrows():
        with st.sidebar:
            with st.form(key = row['node_name_start']):
                st.write(row['node_name_start'])
                cols = st.beta_columns(2)
                inlets_dict.update({index:(
                cols[0].text_input(f"Давление", row["startValue"]),
                cols[1].text_input(f"Обводненность", np.round(row["VolumeWater"], decimals=1) if pd.notna(row["VolumeWater"]) else 50.0))})
                submit_button = st.form_submit_button(label='Submit')

    for key in list(inlets_dict.keys()):
        dataset.loc[key, "startValue"] = float(inlets_dict[key][0])
        dataset.loc[key, "VolumeWater"] = inlets_dict[key][1]

    dataset["VolumeWater"] = dataset["VolumeWater"].astype(float)
    outlets_df = dataset[dataset["endIsOutlet"]].copy().drop_duplicates(subset='node_id_end')
    outlets_dict = {}
    for index, row in outlets_df.sort_values("__pad_num").iterrows():
        with st.sidebar:
            with st.form(row['node_name_end']):
                st.write(row['node_name_end'])
                outlets_dict.update({index:
                st.text_input(f"Давление", row["endValue"])})
                submit_button = st.form_submit_button(label='Submit')

    for key in list(outlets_dict.keys()):
        dataset.loc[key, "endValue"] = float(outlets_dict[key])

    with st.spinner("Расчет сети"):
        mdf, G = calculate_DF(dataset, return_graph=True)
        pyvis_graph = draw_result_graph(mdf, G, use_coordinates=use_coord, coordinate_scaling=coordinate_scaling)
    try:
        mdf = mdf[[ 'node_name_start',
                   'node_name_end','L','uphillM', 'D', 'startP',
                   'endP', 'res_watercut_percent', 'res_liquid_density_kg_m3', 'X_kg_sec', 'velocity_m_sec']]

        mdf['velocity_m_sec'] = mdf['velocity_m_sec'].round(2)
        mdf[['L', 'uphillM', 'startP','endP', 'res_watercut_percent', 'res_liquid_density_kg_m3', 'X_kg_sec']] = \
            mdf[['L', 'uphillM', 'startP','endP', 'res_watercut_percent', 'res_liquid_density_kg_m3', 'X_kg_sec']].astype(float).round(1)



        mdf = mdf.rename(columns = {'L': "Длина участка", 'node_name_start': "Начало участка",
                   'node_name_end': "Конец участка",'uphillM': "Перепад высот, м.", 'startP' : "Давление в начале, атм.",
                   'endP': "Давление в конце, атм.", 'D': "Диаметр, мм.", 'res_watercut_percent':"Обводненность, %",
                                    'res_liquid_density_kg_m3':"Плотность, кг/м3", 'X_kg_sec':"Массовый расход, кг/сек", 'velocity_m_sec':"Скорость потока, м/с" })

        st.write(mdf.style.format({"Длина участка": '{:.1f}', "Перепад высот, м.": '{:.1f}',  "Давление в начале, атм.": '{:.1f}',
                                   "Давление в конце, атм.": '{:.1f}', "Обводненность, %": '{:.1f}', "Плотность, кг/м3": '{:.1f}',
                                   "Массовый расход, кг/сек": '{:.1f}', "Скорость потока, м/с" : '{:.2f}'}))
        st.success("Решение найдено")
    except:

        st.error("Решение для данной конфигурации сети не найдено")

    pyvis_graph.save_graph('temp.html')
    HtmlFile = open("temp.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=900, width=1200)

if __name__ == "__main__":
    main()