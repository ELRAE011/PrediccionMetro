import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import mplcursors
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")
# Cargar el CSV y procesar los datos
data = pd.read_csv('Lineas.csv', header=None)
linea1 = data.iloc[1:21, [0, 1]].rename(columns={0: 'Linea1', 1: 'Afluencia'})
linea2 = data.iloc[22:45, [0, 1]].rename(columns={0: 'Linea2', 1: 'Afluencia'})
linea3 = data.iloc[46:68, [0, 1]].rename(columns={0: 'Linea3', 1: 'Afluencia'})
linea4 = data.iloc[69:79, [0, 1]].rename(columns={0: 'Linea4', 1: 'Afluencia'})
linea5 = data.iloc[80:93, [0, 1]].rename(columns={0: 'Linea5', 1: 'Afluencia'})
linea6 = data.iloc[94:105, [0, 1]].rename(columns={0: 'Linea6', 1: 'Afluencia'})
linea7 = data.iloc[106:120, [0, 1]].rename(columns={0: 'Linea7', 1: 'Afluencia'})
linea8 = data.iloc[121:140, [0, 1]].rename(columns={0: 'Linea8', 1: 'Afluencia'})
linea9 = data.iloc[141:153, [0, 1]].rename(columns={0: 'Linea9', 1: 'Afluencia'})
lineaA = data.iloc[154:164, [0, 1]].rename(columns={0: 'LineaA', 1: 'Afluencia'})
lineaB = data.iloc[165:184, [0, 1]].rename(columns={0: 'LineaB', 1: 'Afluencia'})
linea12 = data.iloc[185:205, [0, 1]].rename(columns={0: 'Linea12', 1: 'Afluencia'})
afluenciaAnual = data.iloc[206:218, [0, 1]].rename(columns={0: 'AfluenciaAnual', 1: 'Afluencia'})

# Limpiar los datos
linea1['Afluencia'] = linea1['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea1['Afluencia'] = pd.to_numeric(linea1['Afluencia'], errors='coerce')

linea2['Afluencia'] = linea2['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea2['Afluencia'] = pd.to_numeric(linea2['Afluencia'], errors='coerce')

linea3['Afluencia'] = linea3['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea3['Afluencia'] = pd.to_numeric(linea3['Afluencia'], errors='coerce')

linea4['Afluencia'] = linea4['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea4['Afluencia'] = pd.to_numeric(linea4['Afluencia'], errors='coerce')

linea5['Afluencia'] = linea5['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea5['Afluencia'] = pd.to_numeric(linea5['Afluencia'], errors='coerce')

linea6['Afluencia'] = linea6['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea6['Afluencia'] = pd.to_numeric(linea6['Afluencia'], errors='coerce')

linea7['Afluencia'] = linea7['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea7['Afluencia'] = pd.to_numeric(linea7['Afluencia'], errors='coerce')

linea8['Afluencia'] = linea8['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea8['Afluencia'] = pd.to_numeric(linea8['Afluencia'], errors='coerce')

linea9['Afluencia'] = linea9['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea9['Afluencia'] = pd.to_numeric(linea9['Afluencia'], errors='coerce')

lineaA['Afluencia'] = lineaA['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
lineaA['Afluencia'] = pd.to_numeric(lineaA['Afluencia'], errors='coerce')

lineaB['Afluencia'] = lineaB['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
lineaB['Afluencia'] = pd.to_numeric(lineaB['Afluencia'], errors='coerce')

linea12['Afluencia'] = linea12['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
linea12['Afluencia'] = pd.to_numeric(linea12['Afluencia'], errors='coerce')

afluenciaAnual['Afluencia'] = afluenciaAnual['Afluencia'].replace({',': '', '"': '', ' ': ''}, regex=True)
afluenciaAnual['Afluencia'] = pd.to_numeric(afluenciaAnual['Afluencia'], errors='coerce')
# Normalizar y preparar datos para regresión
def preparar_datos(estaciones, afluencias):
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    afluencia_scaled = scaler_y.fit_transform(afluencias.reshape(-1, 1)).flatten()
    X = np.arange(len(estaciones)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(X, afluencia_scaled)
    y_pred_scaled = modelo.predict(X)
    return X, afluencia_scaled, y_pred_scaled, modelo
# Datos adicionales 
dias_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
#Datos Tiempo de espera 
TiempoDeEspera = np.array([1, 2, 3])
afluencia_espera_linea1 = np.array([3, 5, 7])
afluencia_espera_linea2 = np.array([1, 2, 3])
afluencia_espera_linea3 = np.array([1, 2, 3])
afluencia_espera_linea4 = np.array([1, 2, 3])
afluencia_espera_linea5 = np.array([3, 5, 7])
afluencia_espera_linea6 = np.array([1, 2, 3])
afluencia_espera_linea7 = np.array([1, 2, 3])
afluencia_espera_linea8 = np.array([1, 2, 3])
afluencia_espera_linea9 = np.array([3, 5, 7])
afluencia_espera_lineaA = np.array([1, 2, 3])
afluencia_espera_lineaB = np.array([1, 2, 3])
afluencia_espera_linea12 = np.array([1, 2, 3])
#Datos Afluencia por dia
afluencia_dia_linea1 = np.array([3294447, 5294447, 3294447, 4294447, 5294447, 4294447, 4294447])
afluencia_dia_linea2 = np.array([5294447, 3294447, 3294447, 5294447, 5294447, 3294447, 4294447])
afluencia_dia_linea3 = np.array([4294447, 5294447, 5294447, 3294447, 4294447, 5294447, 4294447])
afluencia_dia_linea4 = np.array([5294447, 5294447, 3594447, 3294447, 5294447, 4294447, 5294447])
afluencia_dia_linea5 = np.array([4294447, 3294447, 3294447, 5294447, 5294447, 4294447, 5294447])
afluencia_dia_linea6 = np.array([5294447, 4294447, 3294447, 3294447, 5294447, 4294447, 4294447])
afluencia_dia_linea7 = np.array([4294447, 4294447, 3294447, 4294447, 5294447, 5294447, 4294447])
afluencia_dia_linea8 = np.array([4294447, 5294447, 3594447, 5294447, 5294447, 4294447, 5294447])
afluencia_dia_linea9 = np.array([5294447, 4294447, 3294447, 3294447, 4294447, 5294447, 4294447])
afluencia_dia_lineaA = np.array([4294447, 4294447, 3294447, 4294447, 5294447, 3294447, 5294447])
afluencia_dia_lineaB = np.array([4294447, 5294447, 5294447, 3294447, 4294447, 5294447, 4294447])
afluencia_dia_linea12 = np.array([5294447, 5294447, 3594447, 4294447, 5294447, 4294447, 5294447])
#Datos Afluencia por Mes
afluencia_mes_Linea1 = np.array([59171718, 59171718, 59171718, 58658924, 58658924, 58658924, 61555017, 61555017, 61555017, 63401753, 63401753, 63401753])
afluencia_mes_Linea2 = np.array([65477847, 65477847, 65477847, 65391692, 65391692, 65391692, 68312640, 68312640, 68312640, 69967267, 69967267, 69967267])
afluencia_mes_Linea3 = np.array([54399852, 54399852, 54399852, 54124691, 54124691, 54124691, 56144037, 56144037, 56144037, 57699677, 57699677, 57699677])
afluencia_mes_Linea4 = np.array([7317543, 7317543, 7317543, 6828539, 6828539, 6828539, 7141305, 7141305, 7141305, 7725645, 7725645, 7725645])
afluencia_mes_Linea5 = np.array([21667321, 21667321, 21667321, 21242860, 21242860, 21242860, 21956977, 21956977, 21956977, 21645841, 21645841, 21645841])
afluencia_mes_Linea6 = np.array([12506909, 12506909, 12506909, 12055072, 12055072, 12055072, 12554078, 12554078, 12554078, 12829763, 12829763, 12829763])
afluencia_mes_Linea7 = np.array([26374524, 26374524, 26374524, 26630419, 26630419, 26630419, 27889395, 27889395, 27889395, 27257713, 27257713, 27257713])
afluencia_mes_Linea8 = np.array([32839859, 32839859, 32839859, 32703868, 32703868, 32703868, 33472520, 33472520, 33472520, 34604432, 34604432, 34604432])
afluencia_mes_Linea9 = np.array([28055719, 28055719, 28055719, 27639658, 27639658, 27639658, 28751288, 28751288, 28751288, 29318863, 29318863, 29318863])
afluencia_mes_LineaA = np.array([26695821, 26695821, 26695821, 26385055, 26385055, 26385055, 29129803, 29129803, 29129803, 30077385, 30077385, 30077385])
afluencia_mes_LineaB = np.array([37065229, 37065229, 37065229, 36992481, 36992481, 36992481, 38346157, 38346157, 38346157, 40142091, 40142091, 40142091])
afluencia_mes_Linea12 = np.array([32419410, 32419410, 32419410, 32673340, 32673340, 32673340, 34243765, 34243765, 34243765, 35563852, 35563852, 35563852])
# Diccionario para mapear líneas a afluencias
afluencias_por_linea = {
    "Linea 1": afluencia_mes_Linea1,
    "Linea 2": afluencia_mes_Linea2,
    "Linea 3": afluencia_mes_Linea3,
    "Linea 4": afluencia_mes_Linea4,
    "Linea 5": afluencia_mes_Linea5,
    "Linea 6": afluencia_mes_Linea6,
    "Linea 7": afluencia_mes_Linea7,
    "Linea 8": afluencia_mes_Linea8,
    "Linea 9": afluencia_mes_Linea9,
    "Linea A": afluencia_mes_LineaA,
    "Linea B": afluencia_mes_LineaB,
    "Linea 12": afluencia_mes_Linea12,
}

def graficar_regresion_multiple_3d(linea):
    eliminar_imagen()
    # Limpiar el contenido anterior del frame de las gráficas
    for widget in frame_grafica.winfo_children():
        widget.destroy()

    # Obtener los datos según la línea seleccionada
    if linea == 'Linea1':
        estaciones = linea1['Linea1'].values
        afluencias = linea1['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea1
        afluencia_dia = afluencia_dia_linea1
        afluencia_espera = afluencia_espera_linea1
    elif linea == 'Linea2':
        estaciones = linea2['Linea2'].values
        afluencias = linea2['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea2
        afluencia_dia = afluencia_dia_linea2
        afluencia_espera = afluencia_espera_linea2
    elif linea == 'Linea3':
        estaciones = linea3['Linea3'].values
        afluencias = linea3['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea3
        afluencia_dia = afluencia_dia_linea3
        afluencia_espera = afluencia_espera_linea3
    elif linea == 'Linea4':
        estaciones = linea4['Linea4'].values
        afluencias = linea4['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea4
        afluencia_dia = afluencia_dia_linea4
        afluencia_espera = afluencia_espera_linea4
    elif linea == 'Linea5':
        estaciones = linea5['Linea5'].values
        afluencias = linea5['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea5 
        afluencia_dia = afluencia_dia_linea5
        afluencia_espera = afluencia_espera_linea5    
    elif linea == 'Linea6':
        estaciones = linea6['Linea6'].values
        afluencias = linea6['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea6
        afluencia_dia = afluencia_dia_linea6
        afluencia_espera = afluencia_espera_linea6
    elif linea == 'Linea7':
        estaciones = linea7['Linea7'].values
        afluencias = linea7['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea7
        afluencia_dia = afluencia_dia_linea7
        afluencia_espera = afluencia_espera_linea7    
    elif linea == 'Linea8':
        estaciones = linea8['Linea8'].values
        afluencias = linea8['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea8
        afluencia_dia = afluencia_dia_linea8
        afluencia_espera = afluencia_espera_linea8    
    elif linea == 'Linea9':
        estaciones = linea9['Linea9'].values
        afluencias = linea9['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea9
        afluencia_dia = afluencia_dia_linea9
        afluencia_espera = afluencia_espera_linea9    
    elif linea == 'LineaA':
        estaciones = lineaA['LineaA'].values
        afluencias = lineaA['Afluencia'].values
        afluencia_mes = afluencia_mes_LineaA
        afluencia_dia = afluencia_dia_lineaA
        afluencia_espera = afluencia_espera_lineaA 
    elif linea == 'LineaB':
        estaciones = lineaB['LineaB'].values
        afluencias = lineaB['Afluencia'].values
        afluencia_mes = afluencia_mes_LineaB
        afluencia_dia = afluencia_dia_lineaB
        afluencia_espera = afluencia_espera_lineaB
    elif linea == 'Linea12':
        estaciones = linea12['Linea12'].values
        afluencias = linea12['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea12
        afluencia_dia = afluencia_dia_linea12
        afluencia_espera = afluencia_espera_linea12  
    
    # Igualar las longitudes de las listas si es necesario
    max_length = max(len(afluencia_mes), len(afluencia_dia), len(afluencia_espera), len(estaciones))
    afluencia_mes = np.pad(afluencia_mes, (0, max_length - len(afluencia_mes)), 'edge')
    afluencia_dia = np.pad(afluencia_dia, (0, max_length - len(afluencia_dia)), 'edge')
    afluencia_espera = np.pad(afluencia_espera, (0, max_length - len(afluencia_espera)), 'edge')
    estaciones = np.pad(estaciones, (0, max_length - len(estaciones)), 'edge')
    afluencias = np.pad(afluencias, (0, max_length - len(afluencias)), 'edge')

    X = np.column_stack((afluencia_mes, afluencia_dia, afluencia_espera))
    y = afluencias

    # Crear y ajustar el modelo de regresión
    modelo = LinearRegression()
    modelo.fit(X, y)
    predicciones = modelo.predict(X)
    
    # Crear un Label para mostrar el título de la gráfica
    title_label = tk.Label(frame_grafica, text=f'Grafica De Afluencias - AfluenciaMes vs AfluenciaDia vs TiempoDeEspera vsPredicción  - {linea}', font=('Helvetica', 16))
    title_label.pack()

    # Crear la figura 3D
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#f1f7dc')  # Cambiar el fondo de la figura
    ax.set_facecolor('#f1f7dc')  # Cambiar el fondo del eje
    ax.tick_params(colors='black')  # Color de los números de los ejes en negro

    # Graficar cada conjunto de datos con un color y marcador distinto
    ax.scatter(afluencia_mes, afluencia_dia, afluencias, color='blue', marker='o', label='Afluencias Reales')  # Marcador circular
    ax.scatter(afluencia_mes, afluencia_dia, afluencia_mes, color='orange', marker='s', label='Afluencia Mensual')  # Marcador cuadrado
    ax.scatter(afluencia_mes, afluencia_dia, afluencia_dia, color='purple', marker='^', label='Afluencia Diaria')  # Marcador triangular
    ax.scatter(afluencia_mes, afluencia_dia, afluencia_espera, color='green', marker='d', label='Tiempo de Espera')  # Marcador rombo
    
    # Graficar las predicciones
    ax.plot_trisurf(afluencia_mes, afluencia_dia, predicciones, color='red', alpha=0.5, label='Predicción')

    # Etiquetas y leyendas
    ax.set_xlabel('Afluencia Mensual')
    ax.set_ylabel('Afluencia Diaria')
    ax.set_zlabel('Afluencias Reales/Predicción')
    ax.legend()
    
    # Convertir la gráfica a un canvas compatible con Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack()


    # Limpiar los widgets de datos anteriores
    for widget in frame_datos.winfo_children():
        widget.destroy()


def graficar_prediccion(ax, linea, estaciones, afluencias, modelo, prediccion_dias, color_afluencia, color_prediccion, titulo):
    # Normalizar afluencias
    afluencia_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(afluencias.reshape(-1, 1)).flatten()
    print("Afluencias normalizadas:", afluencia_scaled)  # Debug: Verificar valores normalizados
    
    X = np.arange(len(estaciones)).reshape(-1, 1)

    # Verificar que no hay NaN antes de ajustar el modelo
    if np.isnan(afluencia_scaled).any():
        print("Error: afluencia_scaled contiene NaN.")
        return  # Salir si hay NaN

    modelo.fit(X, afluencia_scaled)
    
    # Preparar datos para la predicción
    X_pred = np.arange(len(estaciones), len(estaciones) + prediccion_dias).reshape(-1, 1)
    y_pred_scaled = modelo.predict(X_pred)

    # Normalizar las afluencias para revertir la normalización
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(afluencias.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Graficar la afluencia histórica y la predicción
    ax.plot(np.arange(len(estaciones)), afluencias, color=color_afluencia, label='Afluencia Estación', marker='o', markersize=8)
    ax.plot(np.arange(len(estaciones), len(estaciones) + prediccion_dias), y_pred, color=color_prediccion, linestyle='-', marker='o', markersize=8, label='Predicción')
    
    # Calcular límites dinámicos
    min_value = min(afluencias.min(), y_pred.min()) - 0.1  # Margen inferior
    max_value = max(afluencias.max(), y_pred.max()) + 0.1  # Margen superior
    ax.set_ylim(min_value, max_value)
    ax.set_xlim(-1, len(estaciones) + prediccion_dias)
    ax.set_ylabel('Predicción Con Datos Reales')
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    # Calcular y mostrar métricas de evaluación
    mse = mean_squared_error(afluencias, modelo.predict(X))
    mae = mean_absolute_error(afluencias, modelo.predict(X))
    
    # Mostrar MSE y MAE en la gráfica
    ax.text(0.5, 0.9, f'MSE: {mse:.4f}', transform=ax.transAxes, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.85, f'MAE: {mae:.4f}', transform=ax.transAxes, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.5))

    # Graficar las afluencias históricas con etiquetas
    scatter_historico = ax.scatter(np.arange(len(estaciones)), afluencias, color=color_afluencia, label='Afluencia Estación', marker='o', s=50)
    scatter_cursor_historico = mplcursors.cursor(scatter_historico, hover=True)
    @scatter_cursor_historico.connect("add")
    def on_add_historico(sel):
        estacion_index = sel.index
        # Mostrar afluencia de la estación seleccionada
        sel.annotation.set_text(f'Estación: {estaciones[estacion_index]}\nAfluencia: {int(afluencias[estacion_index])}')

    # Graficar las predicciones con etiquetas
    scatter_prediccion = ax.scatter(np.arange(len(estaciones), len(estaciones) + prediccion_dias), y_pred, color=color_prediccion, label='Predicción', marker='o', s=50)
    scatter_cursor_prediccion = mplcursors.cursor(scatter_prediccion, hover=True)
    @scatter_cursor_prediccion.connect("add")
    def on_add_prediccion(sel):
        estacion_index = sel.index
        # Mostrar predicción de la estación seleccionada
        if estacion_index < prediccion_dias:
            sel.annotation.set_text(f'Predicción: {y_pred[estacion_index]:.2f}')

    # Análisis de residuos
    residuos = afluencias - modelo.predict(X)
    # Graficar residuos
    ax_residuos = ax.twinx()  # Crear un segundo eje y
    bars = ax_residuos.bar(np.arange(len(estaciones)), residuos, color='lightgray', alpha=0.5, label='Residuos', width=0.4)
    ax_residuos.axhline(0, color='red', linestyle='--', linewidth=1)
    ax_residuos.legend(loc='upper left')
    # Añadir etiquetas a los residuos
    cursor_residuos = mplcursors.cursor(bars, hover=True)
    @cursor_residuos.connect("add")
    def on_add_residuo(sel):
        residuo_index = sel.index
        sel.annotation.set_text(f'Prediccion con Residuo: {residuos[residuo_index]:.2f}')
    

def manejar_seleccion(linea):
    eliminar_imagen()
    # Limpiar los widgets de datos anteriores
    for widget in frame_datos.winfo_children():
        widget.destroy()
    # Normalizar y preparar datos para regresión
    def preparar_datos1(estaciones, afluencias):
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        afluencia_scaled = scaler_y.fit_transform(afluencias.reshape(-1, 1)).flatten()
        X = np.arange(len(estaciones)).reshape(-1, 1)
        # Dividir los datos en entrenamiento (80%) y prueba (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, afluencia_scaled, test_size=1, shuffle=False)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred_scaled = modelo.predict(X_train)
        return modelo, X_train, y_train, X_test, y_test
    try:
        if linea == 'LineaA':
            estaciones = lineaA['LineaA'].values
            afluencias = lineaA['Afluencia'].values
        elif linea == 'LineaB':
            estaciones = lineaB['LineaB'].values
            afluencias = lineaB['Afluencia'].values
        else:
            estaciones = eval(f"{linea.lower()}['{linea}'].values")  # Asegúrate de que el nombre del DataFrame sea correcto
            afluencias = eval(f"{linea.lower()}['Afluencia'].values")  # Asegúrate de que el DataFrame tiene esta columna
    except KeyError:
        print("Error al acceder a los datos de la línea seleccionada.")
        return
    # Limpiar la gráfica anterior
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    # Crear la figura y los ejes para la nueva gráfica
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#f1f7dc')  # Cambiar el fondo de la figura
    ax.set_facecolor('#f1f7dc')  # Cambiar el fondo del eje
    ax.tick_params(colors='black') 
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.get_tk_widget().pack()
    opciones = {
        'Linea1': (20, 'pink', '#ff00e8', 'Predicción de Afluencia para Línea 1'),
        'Linea2': (23, 'blue', '#0087ff', 'Predicción de Afluencia para Línea 2'),
        'Linea3': (22, 'green', '#00ff17', 'Predicción de Afluencia para Línea 3'),
        'Linea4': (10, '#43e7e7', '#00fff7', 'Predicción de Afluencia para Línea 4'),
        'Linea5': (13, '#e1e813', '#fff300', 'Predicción de Afluencia para Línea 5'),
        'Linea6': (11, 'red', '#cc0000', 'Predicción de Afluencia para Línea 6'),
        'Linea7': (14, 'orange', '#ff7800', 'Predicción de Afluencia para Línea 7'),
        'Linea8': (19, '#19bc68', '#87ff00', 'Predicción de Afluencia para Línea 8'),
        'Linea9': (12, 'brown', '#5f2727', 'Predicción de Afluencia para Línea 9'),
        'LineaA': (10, 'purple', '#ec00ff', 'Predicción de Afluencia para Línea A'),
        'LineaB': (19, '#0a8352', '#34c43d', 'Predicción de Afluencia para Línea B'),
        'Linea12': (20, 'gold', '#d8df00', 'Predicción de Afluencia para Línea 12')
    }
    if linea in opciones:
        prediccion_dias, color_afluencia, color_prediccion, titulo = opciones[linea]
        # Aquí extraemos solo el modelo
        modelo, X_train, y_train, X_test, y_test = preparar_datos1(estaciones, afluencias)
        # Llamar a la función para graficar
        graficar_prediccion(ax, linea, estaciones, afluencias, modelo, prediccion_dias, color_afluencia, color_prediccion, titulo)
        
    canvas.draw()
    
    
def mostrar_tiempo_espera(linea):
    eliminar_imagen()
    global linea1, linea2, linea3, linea4, linea5, linea6, linea7, linea8, linea9, lineaA, lineaB, linea12
    # Limpiar la gráfica actual
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    # Inicializar la variable afluencia_mes
    afluencia_espera = None
    # Seleccionar los datos correspondientes a la línea elegida
    if linea == 'Linea1':
        afluencia_espera = afluencia_espera_linea1
    elif linea == 'Linea2':
        afluencia_espera = afluencia_espera_linea2
    elif linea == 'Linea3':
        afluencia_espera = afluencia_espera_linea3
    elif linea == 'Linea4':
        afluencia_espera = afluencia_espera_linea4
    elif linea == 'Linea5':
        afluencia_espera = afluencia_espera_linea5
    elif linea == 'Linea6':
        afluencia_espera = afluencia_espera_linea6
    elif linea == 'Linea7':
        afluencia_espera = afluencia_espera_linea7   
    elif linea == 'Linea8':
        afluencia_espera = afluencia_espera_linea8   
    elif linea == 'Linea9':
        afluencia_espera = afluencia_espera_linea9
    elif linea == 'LineaA':
        afluencia_espera = afluencia_espera_lineaA
    elif linea == 'LineaB':
        afluencia_espera = afluencia_espera_lineaB
    elif linea == 'Linea12':
        afluencia_espera = afluencia_espera_linea12
    # Verificar si afluencia_mes no es None
    if afluencia_espera is not None:
        # Gráfica 4: Afluencia por mes
        fig, ax = plt.subplots(figsize=(18, 10))
        # Establecer el color de fondo de la figura y del eje
        fig.patch.set_facecolor('#f1f7dc')  # Cambiar el fondo de la figura
        ax.set_facecolor('#f1f7dc')  # Cambiar el fondo del eje
        ax.tick_params(colors='black')  # Color de los números de los ejes en negro
        # Gráfica 3: Afluencia por día de la semana
        # Gráfica 2: Tiempo de espera
        ax.bar(TiempoDeEspera, afluencia_espera, color='green', tick_label=['10%', '30%', '60%'])
        ax.set_xlabel('Porcentaje de usuarios')
        ax.set_ylabel('Minutos')
        ax.set_title('Gráfica de Tiempo de Espera De Los Usuarios (minutos)')
        ax.grid()
        # Mostrar la gráfica en el frame de la interfaz
        canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        canvas.draw()
        # Cambiar color de fondo del canvas
        canvas.get_tk_widget().config(bg='#f1f7dc')  # Cambiar el color de fondo del canvas
        canvas.get_tk_widget().pack(fill='both', expand=True)
    else:
        print(f"No se encontraron datos para la {linea}")
    for widget in frame_datos.winfo_children():
        widget.destroy()
        

def mostrar_afluencia_dia(linea):
    eliminar_imagen()
    global linea1, linea2, linea3, linea4, linea5, linea6, linea7, linea8, linea9, lineaA, lineaB, linea12
    # Limpiar la gráfica actual
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    # Inicializar la variable afluencia_mes
    afluencia_dia = None
    # Seleccionar los datos correspondientes a la línea elegida
    if linea == 'Linea1':
        afluencia_dia = afluencia_dia_linea1
    elif linea == 'Linea2':
        afluencia_dia = afluencia_dia_linea2
    elif linea == 'Linea3':
        afluencia_dia = afluencia_dia_linea3
    elif linea == 'Linea4':
        afluencia_dia = afluencia_dia_linea4
    elif linea == 'Linea5':
        afluencia_dia = afluencia_dia_linea5
    elif linea == 'Linea6':
        afluencia_dia = afluencia_dia_linea6
    elif linea == 'Linea7':
        afluencia_dia = afluencia_dia_linea7   
    elif linea == 'Linea8':
        afluencia_dia = afluencia_dia_linea8   
    elif linea == 'Linea9':
        afluencia_dia = afluencia_dia_linea9
    elif linea == 'LineaA':
        afluencia_dia = afluencia_dia_lineaA
    elif linea == 'LineaB':
        afluencia_dia = afluencia_dia_lineaB
    elif linea == 'Linea12':
        afluencia_dia = afluencia_dia_linea12
    # Verificar si afluencia_mes no es None
    if afluencia_dia is not None:
        # Gráfica 4: Afluencia por mes
        fig, ax = plt.subplots(figsize=(18, 10))
        # Establecer el color de fondo de la figura y del eje
        fig.patch.set_facecolor('#f1f7dc')  # Cambiar el fondo de la figura
        ax.set_facecolor('#f1f7dc')  # Cambiar el fondo del eje
        ax.tick_params(colors='black')  # Color de los números de los ejes en negro
        # Gráfica 3: Afluencia por día de la semana
        bars_dia = ax.bar(dias_semana, afluencia_dia, color='purple')
        ax.set_xlabel('Día de la Semana')
        ax.set_ylabel('Afluencia en Millones')
        ax.set_title('Afluencia de Pasajeros por Día de la Semana')
        ax.grid()
        mplcursors.cursor(bars_dia, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Dia: {dias_semana[sel.index]}\nAfluencia: {int(afluencia_dia[sel.index])}'))
        # Mostrar la gráfica en el frame de la interfaz
        canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        canvas.draw()
        # Cambiar color de fondo del canvas
        canvas.get_tk_widget().config(bg='#f1f7dc')  # Cambiar el color de fondo del canvas
        canvas.get_tk_widget().pack(fill='both', expand=True)
        mostrar_datos_dias(dias_semana, afluencia_dia)
    else:
        print(f"No se encontraron datos para la {linea}")
        

def mostrar_afluencia_mes(linea):
    eliminar_imagen()
    global linea1, linea2, linea3, linea4, linea5, linea6, linea7, linea8, linea9, lineaA, lineaB, linea12
    # Limpiar la gráfica actual
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    # Inicializar la variable afluencia_mes
    afluencia_mes = None
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    # Seleccionar los datos correspondientes a la línea elegida
    if linea == 'Linea1':
        afluencia_mes = afluencia_mes_Linea1
    elif linea == 'Linea2':
        afluencia_mes = afluencia_mes_Linea2
    elif linea == 'Linea3':
        afluencia_mes = afluencia_mes_Linea3
    elif linea == 'Linea4':
        afluencia_mes = afluencia_mes_Linea4
    elif linea == 'Linea5':
        afluencia_mes = afluencia_mes_Linea5
    elif linea == 'Linea6':
        afluencia_mes = afluencia_mes_Linea6
    elif linea == 'Linea7':
        afluencia_mes = afluencia_mes_Linea7   
    elif linea == 'Linea8':
        afluencia_mes = afluencia_mes_Linea8   
    elif linea == 'Linea9':
        afluencia_mes = afluencia_mes_Linea9
    elif linea == 'LineaA':
        afluencia_mes = afluencia_mes_LineaA
    elif linea == 'LineaB':
        afluencia_mes = afluencia_mes_LineaB
    elif linea == 'Linea12':
        afluencia_mes = afluencia_mes_Linea12
    # Verificar si afluencia_mes no es None
    if afluencia_mes is not None:
        # Gráfica 4: Afluencia por mes
        # Color de los números de los ejes en negro
        fig, ax = plt.subplots(figsize=(18, 10))
        # Establecer el color de fondo de la figura y del eje
        fig.patch.set_facecolor('#f1f7dc')  # Cambiar el fondo de la figura
        ax.set_facecolor('#f1f7dc')  # Cambiar el fondo del eje
        ax.tick_params(colors='black')
        bars_mes = ax.bar(meses, afluencia_mes, color='orange')
        ax.set_xlabel('Mes')
        ax.set_ylabel('Afluencia en Millones')
        ax.set_title('Afluencia de Pasajeros por Mes')
        ax.grid()
        mplcursors.cursor(bars_mes, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Mes: {meses[sel.index]}\nAfluencia: {int(afluencia_mes[sel.index])}'))
        # Mostrar la gráfica en el frame de la interfaz
        canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        canvas.draw()
        # Cambiar color de fondo del canvas
        canvas.get_tk_widget().config(bg='#f1f7dc')  # Cambiar el color de fondo del canvas
        canvas.get_tk_widget().pack(fill='both', expand=True)
        mostrar_datos_mes(meses, afluencia_mes)
    else:
        print(f"No se encontraron datos para la {linea}")
    #for widget in frame_datos.winfo_children():
        widget.destroy()
        
# Función para generar la gráfica de regresión lineal
def mostrar_regresion_lineal(linea):
    eliminar_imagen()
    global linea1, linea2, linea3, linea4, linea5, linea6, linea7, linea8, linea9, lineaA, lineaB, linea12, anio
    # Limpiar la gráfica actual
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    # Seleccionar los datos correspondientes a la línea elegida
    if linea == 'Linea1':
        estaciones = linea1['Linea1'].values
        afluencias = linea1['Afluencia'].values
    elif linea == 'Linea2':
        estaciones = linea2['Linea2'].values
        afluencias = linea2['Afluencia'].values
    elif linea == 'Linea3':
        estaciones = linea3['Linea3'].values
        afluencias = linea3['Afluencia'].values
    elif linea == 'Linea4':
        estaciones = linea4['Linea4'].values
        afluencias = linea4['Afluencia'].values
    elif linea == 'Linea5':
        estaciones = linea5['Linea5'].values
        afluencias = linea5['Afluencia'].values
    elif linea == 'Linea6':
        estaciones = linea6['Linea6'].values
        afluencias = linea6['Afluencia'].values
    elif linea == 'Linea7':
        estaciones = linea7['Linea7'].values
        afluencias = linea7['Afluencia'].values   
    elif linea == 'Linea8':
        estaciones = linea8['Linea8'].values
        afluencias = linea8['Afluencia'].values   
    elif linea == 'Linea9':
        estaciones = linea9['Linea9'].values
        afluencias = linea9['Afluencia'].values
    elif linea == 'LineaA':
        estaciones = lineaA['LineaA'].values
        afluencias = lineaA['Afluencia'].values
    elif linea == 'LineaB':
        estaciones = lineaB['LineaB'].values
        afluencias = lineaB['Afluencia'].values
    elif linea == 'Linea12':
        estaciones = linea12['Linea12'].values
        afluencias = linea12['Afluencia'].values
    elif linea =='AfluenciaAnual':
        estaciones = afluenciaAnual['AfluenciaAnual'].values
        afluencias = afluenciaAnual['Afluencia'].values   
    # Normalizar afluencias
    afluencia_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(afluencias.reshape(-1, 1)).flatten()
    print("Afluencias normalizadas:", afluencia_scaled)  # Debug: Verificar valores normalizados
    X = np.arange(len(estaciones)).reshape(-1, 1)
    modelo = LinearRegression()
    # Verificar que no hay NaN antes de ajustar el modelo
    if np.isnan(afluencia_scaled).any():
        print("Error: afluencia_scaled contiene NaN.")
        return  # Salir si hay NaN
    modelo.fit(X, afluencia_scaled)
    y_pred_scaled = modelo.predict(X)
    y_pred_scaled = np.abs(y_pred_scaled)
    # Obtener el coeficiente de la regresión
    coeficiente = modelo.coef_[0]
    # Graficar regresión lineal
    X = np.arange(len(estaciones)).reshape(-1, 1)  # Índice de estaciones como variable independiente
    y = afluencias  # Afluencias como variable dependiente
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    # Crear la gráfica
    fig, ax = plt.subplots(figsize=(18, 10))
    # Establecer el color de fondo de la figura y del eje
    fig.patch.set_facecolor('#f1f7dc')  # Cambiar el fondo de la figura
    ax.set_facecolor('#f1f7dc')  # Cambiar el fondo del eje
    ax.tick_params(colors='black')  # Color de los números de los ejes en negro
    # Graficar los datos
    scatter = ax.scatter(X, y, color='blue', label='Afluencia Real')  # Puntos de afluencia real
    ax.plot(X, y_pred, color='red', label='Regresión Lineal')
    ax.set_title(f'Regresión Lineal para {linea}')
    ax.set_xlabel('Índice de Estaciones')
    ax.set_ylabel('Afluencia')
    ax.legend()
    ax.grid()
    coeficientes_individuales = afluencia_scaled
    # Crear cursor interactivo
    scatter_cursor = mplcursors.cursor(scatter, hover=True)
    @scatter_cursor.connect("add")
    def on_add(sel):
        estacion_nombre = estaciones[sel.index]
        coeficiente_individual = coeficientes_individuales[sel.index]  # Obtener coeficiente individual
        sel.annotation.set_text(f'Estación: {estacion_nombre}\nCoeficiente: {coeficiente_individual:.4f}')
    # Mostrar la gráfica en el frame de la interfaz
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    # Cambiar color de fondo del canvas
    canvas.get_tk_widget().config(bg='#f1f7dc')  # Cambiar el color de fondo del canvas
    canvas.get_tk_widget().pack(fill='both', expand=True)
    # Eliminar widgets antiguos
    for widget in frame_datos.winfo_children():
        widget.destroy()
    # Mostrar los datos de las estaciones
    mostrar_datos(estaciones, afluencias)
    
# Función para generar gráficas basadas en la selección
def mostrar_grafica(linea):
    
    global AfluenciaAnual, linea1, linea2, linea3, linea4, linea5, linea6, linea7, linea8, linea9, lineaA, lineaB, linea12
    # Limpiar el contenido anterior del frame de las gráficas
    for widget in frame_grafica.winfo_children():
        widget.destroy()
    eliminar_imagen()
    if linea == 'Linea1':
        estaciones = linea1['Linea1'].values
        afluencias = linea1['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea1
        afluencia_dia = afluencia_dia_linea1
        afluencia_espera = afluencia_espera_linea1
        HoraRegular = np.array([ 2000000, 3000000, 3500000, 3500000, 4500000, 2000000, 3500000])
        HoraPico = np.array([    4000000, 5500000, 4500000, 4500000, 5500000, 4000000, 5000000])
        HoraRegular1 = np.array([2000000, 3000000, 3500000, 3500000, 4500000, 2000000, 3500000])
        HoraPico1 = np.array([   4000000, 5500000, 4500000, 4500000, 5500000, 4000000, 5000000])
    elif linea == 'Linea2':
        estaciones = linea2['Linea2'].values
        afluencias = linea2['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea2
        afluencia_dia = afluencia_dia_linea2
        afluencia_espera = afluencia_espera_linea2
        HoraRegular = np.array([ 4500000, 3000000, 3500000, 4500000, 4500000, 2000000, 3500000])
        HoraPico = np.array([    5500000, 3500000, 4500000, 5500000, 5500000, 4000000, 5000000])
        HoraRegular1 = np.array([4500000, 3000000, 3500000, 3500000, 4500000, 2000000, 3500000])
        HoraPico1 = np.array([   5500000, 3500000, 4500000, 5500000, 5500000, 4000000, 5000000])
    elif linea == 'Linea3':
        estaciones = linea3['Linea3'].values
        afluencias = linea3['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea3
        afluencia_dia = afluencia_dia_linea3
        afluencia_espera = afluencia_espera_linea3
        HoraRegular = np.array([ 3000000, 4000000, 4000000, 3000000, 3500000, 4000000, 3500000])
        HoraPico = np.array([    4000000, 5500000, 5500000, 4000000, 4500000, 5500000, 4500000])
        HoraRegular1 = np.array([3000000, 4000000, 4000000, 3000000, 3500000, 4000000, 3500000])
        HoraPico1 = np.array([   4000000, 5500000, 5500000, 4000000, 4500000, 5500000, 4500000])
    elif linea == 'Linea4':
        estaciones = linea4['Linea4'].values
        afluencias = linea4['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea4
        afluencia_dia = afluencia_dia_linea4
        afluencia_espera = afluencia_espera_linea4
        HoraRegular = np.array([ 400000, 400000, 300000, 200000, 400000, 400000, 400000])
        HoraPico = np.array([    500000, 500000, 400000, 350000, 500000, 500000, 500000])
        HoraRegular1 = np.array([400000, 400000, 300000, 200000, 400000, 400000, 400000])
        HoraPico1 = np.array([   500000, 500000, 400000, 350000, 500000, 500000, 500000])
    elif linea == 'Linea5':
        estaciones = linea5['Linea5'].values
        afluencias = linea5['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea5 
        afluencia_dia = afluencia_dia_linea5
        afluencia_espera = afluencia_espera_linea5
        HoraRegular = np.array([ 3500000, 3000000, 3000000, 4000000, 4000000, 3500000, 4500000])
        HoraPico = np.array([    4500000, 4500000, 4500000, 5500000, 5500000, 4500000, 5500000])
        HoraRegular1 = np.array([3500000, 3000000, 3000000, 4000000, 4000000, 3500000, 4500000])
        HoraPico1 = np.array([   4500000, 4500000, 4500000, 5500000, 5500000, 4500000, 5500000])
    elif linea == 'Linea6':
        estaciones = linea6['Linea6'].values
        afluencias = linea6['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea6
        afluencia_dia = afluencia_dia_linea6
        afluencia_espera = afluencia_espera_linea6
        HoraRegular = np.array([ 4000000, 3500000, 2500000, 2500000, 4000000, 3500000, 3500000])
        HoraPico = np.array([    5500000, 4000000, 3000000, 3000000, 5500000, 4000000, 4000000])
        HoraRegular1 = np.array([4000000, 3500000, 2500000, 2500000, 4000000, 3500000, 3500000])
        HoraPico1 = np.array([   5500000, 4000000, 3000000, 3000000, 5500000, 4000000, 4000000])
    elif linea == 'Linea7':
        estaciones = linea7['Linea7'].values
        afluencias = linea7['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea7
        afluencia_dia = afluencia_dia_linea7
        afluencia_espera = afluencia_espera_linea7
        HoraRegular = np.array([ 3500000, 3000000, 3000000, 4000000, 4500000, 4500000, 4000000])
        HoraPico = np.array([    4500000, 4500000, 3500000, 4500000, 5500000, 5500000, 4500000])
        HoraRegular1 = np.array([3500000, 3000000, 3000000, 4000000, 4500000, 4500000, 4000000])
        HoraPico1 = np.array([   4500000, 4500000, 3500000, 4500000, 5500000, 5500000, 4500000])    
    elif linea == 'Linea8':
        estaciones = linea8['Linea8'].values
        afluencias = linea8['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea8
        afluencia_dia = afluencia_dia_linea8
        afluencia_espera = afluencia_espera_linea8
        HoraRegular = np.array([ 3500000, 4500000, 2500000, 4500000, 4500000, 3500000, 4500000])
        HoraPico = np.array([    4000000, 5500000, 3000000, 5500000, 5500000, 4000000, 5500000])
        HoraRegular1 = np.array([3500000, 4500000, 2500000, 4500000, 4500000, 3500000, 4500000])
        HoraPico1 = np.array([   4000000, 5500000, 3000000, 5500000, 5500000, 4000000, 5500000])    
    elif linea == 'Linea9':
        estaciones = linea9['Linea9'].values
        afluencias = linea9['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea9
        afluencia_dia = afluencia_dia_linea9
        afluencia_espera = afluencia_espera_linea9
        HoraRegular = np.array([ 4500000, 3500000, 2500000, 2500000, 3500000, 4500000, 3500000])
        HoraPico = np.array([    5500000, 4500000, 3500000, 3500000, 4500000, 5500000, 4500000])
        HoraRegular1 = np.array([4500000, 3500000, 2500000, 2500000, 3500000, 4500000, 3500000])
        HoraPico1 = np.array([   5500000, 4500000, 3500000, 3500000, 4500000, 5500000, 4500000])    
    elif linea == 'LineaA':
        estaciones = lineaA['LineaA'].values
        afluencias = lineaA['Afluencia'].values
        afluencia_mes = afluencia_mes_LineaA
        afluencia_dia = afluencia_dia_lineaA
        afluencia_espera = afluencia_espera_lineaA
        HoraRegular = np.array([ 3500000, 3500000, 2500000, 3500000, 4000000, 2500000, 4000000])
        HoraPico = np.array([    4500000, 4500000, 3500000, 4500000, 5500000, 3500000, 5500000])
        HoraRegular1 = np.array([3500000, 3500000, 2500000, 3500000, 4000000, 2500000, 4000000])
        HoraPico1 = np.array([   4500000, 4500000, 3500000, 4500000, 5500000, 3500000, 5500000]) 
    elif linea == 'LineaB':
        estaciones = lineaB['LineaB'].values
        afluencias = lineaB['Afluencia'].values
        afluencia_mes = afluencia_mes_LineaB
        afluencia_dia = afluencia_dia_lineaB
        afluencia_espera = afluencia_espera_lineaB
        HoraRegular = np.array([ 3500000, 4500000, 4500000, 2500000, 3500000, 4500000, 3500000])
        HoraPico = np.array([    4500000, 5500000, 5500000, 3500000, 4500000, 5500000, 4500000])
        HoraRegular1 = np.array([3500000, 4500000, 4500000, 2500000, 3500000, 4500000, 3500000])
        HoraPico1 = np.array([   4500000, 5500000, 5500000, 3500000, 4500000, 5500000, 4500000])
    elif linea == 'Linea12':
        estaciones = linea12['Linea12'].values
        afluencias = linea12['Afluencia'].values
        afluencia_mes = afluencia_mes_Linea12
        afluencia_dia = afluencia_dia_linea12
        afluencia_espera = afluencia_espera_linea12
        HoraRegular = np.array([ 4500000, 4500000, 2500000, 3500000, 4500000, 3500000, 4500000])
        HoraPico = np.array([    5500000, 5500000, 3500000, 4500000, 5500000, 4500000, 5500000])
        HoraRegular1 = np.array([4500000, 4500000, 2500000, 3500000, 4500000, 3500000, 4500000])
        HoraPico1 = np.array([   5500000, 5500000, 3500000, 4500000, 5500000, 4500000, 5500000])         
    # Normalizar afluencias
    afluencia_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(afluencias.reshape(-1, 1)).flatten()
    print("Afluencias normalizadas:", afluencia_scaled)  # Debug: Verificar valores normalizados
    X = np.arange(len(estaciones)).reshape(-1, 1)
    modelo = LinearRegression()
    # Verificar que no hay NaN antes de ajustar el modelo
    if np.isnan(afluencia_scaled).any():
        print("Error: afluencia_scaled contiene NaN.")
        return  # Salir si hay NaN
    modelo.fit(X, afluencia_scaled)
    y_pred_scaled = modelo.predict(X)
    y_pred_scaled = np.abs(y_pred_scaled)
    # Obtener el coeficiente de la regresión
    coeficiente = modelo.coef_[0]  # No es necesario hacer un array aquí
    coeficientes_individuales = afluencia_scaled
    # Crear una nueva figura con 3 filas y 2 columnas para ajustar las 5 gráficas
    fig, axs = plt.subplots(3, 2, figsize=(17, 15), facecolor='#f1f7dc')
    # Establecer el fondo para todas las subgráficas
    for ax in axs.flatten():
        ax.set_facecolor('#f1f7dc')  # Fondo negro del área de cada subgráfica
        ax.tick_params(colors='black')  # Color de los números de los ejes en blanco
    # Gráfica 1: Dispersión y línea de regresión
    scatter = axs[0, 0].scatter(X, afluencia_scaled, color='blue', label='Coeficiente')
    axs[0, 0].plot(X, y_pred_scaled, color='red', linewidth=2, label='Línea de regresión')
    axs[0, 0].set_xlabel('Índice de Estaciones')
    axs[0, 0].set_ylabel('Afluencia (normalizada)')
    axs[0, 0].set_title('Regresión Lineal: Índice de Afluencia')
    axs[0, 0].legend()
    axs[0, 0].grid()
    # Crear cursor interactivo
    scatter_cursor = mplcursors.cursor(scatter, hover=True)
    @scatter_cursor.connect("add")
    def on_add(sel):
        estacion_nombre = estaciones[sel.index]
        coeficiente_individual = coeficientes_individuales[sel.index]  # Obtener coeficiente individual
        sel.annotation.set_text(f'Estación: {estacion_nombre}\nCoeficiente: {coeficiente_individual:.4f}')
    # Gráfica 2: Tiempo de espera
    axs[0, 1].bar(TiempoDeEspera, afluencia_espera, color='green', tick_label=['10%', '30%', '60%'])
    axs[0, 1].set_xlabel('Porcentaje de usuarios')
    axs[0, 1].set_ylabel('Minutos')
    axs[0, 1].set_title('Gráfica de Tiempo de Espera De Los Usuarios (minutos)')
    axs[0, 1].grid()
    # Gráfica 3: Afluencia por día de la semana
    bars_dia = axs[1, 0].bar(dias_semana, afluencia_dia, color='purple')
    axs[1, 0].set_xlabel('Día de la Semana')
    axs[1, 0].set_ylabel('Afluencia en Millones')
    axs[1, 0].set_title('Afluencia de Pasajeros por Día de la Semana')
    axs[1, 0].grid()
    mplcursors.cursor(bars_dia, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Dia: {dias_semana[sel.index]}\nAfluencia: {int(afluencia_dia[sel.index])}'))
    # Gráfica 4: Afluencia por mes
    bars_mes = axs[1, 1].bar(meses, afluencia_mes, color='orange')
    axs[1, 1].set_xlabel('Mes')
    axs[1, 1].set_ylabel('Afluencia en Millones')
    axs[1, 1].set_title('Afluencia de Pasajeros por Mes')
    axs[1, 1].grid()
    mplcursors.cursor(bars_mes, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Mes: {meses[sel.index]}\nAfluencia: {int(afluencia_mes[sel.index])}'))
    # Gráfica 5: Afluencia por Hora Regular y Hora Pico (en axs[2, 0])
    bar_width = 0.15
    x_indices = np.arange(len(dias_semana))
    axs[2, 0].bar(x_indices - 1.5 * bar_width, HoraRegular, bar_width, label='Hora Regular 10am-5pm', color='#007acc')
    axs[2, 0].bar(x_indices - 0.5 * bar_width, HoraPico, bar_width, label='Hora Pico 5am-9am', color='#ff5733')
    axs[2, 0].bar(x_indices + 0.5 * bar_width, HoraRegular1, bar_width, label='Hora Regular 10pm-12am', color='#66cc66')
    axs[2, 0].bar(x_indices + 1.5 * bar_width, HoraPico1, bar_width, label='Hora Pico 6pm-9pm', color='#ff3333')
    axs[2, 0].set_xticks(x_indices)
    axs[2, 0].set_xticklabels(dias_semana)
    axs[2, 0].set_ylabel('Afluencia en Millones')
    axs[2, 0].set_xlabel('Día de la Semana')
    axs[2, 0].set_title('Afluencia De Pasajeros Por Hora Regular y Hora Pico')
    axs[2, 0].legend()
    axs[2, 0].grid()
    bars = axs[2, 0].containers
    for container in bars:
        mplcursors.cursor(container, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Afluencia: {int(sel.target[1])}'))
    def graficar_prediccion(ax, linea, estaciones, afluencias, modelo, prediccion_dias, color_afluencia, color_prediccion, titulo):
        # Crear X para la predicción
        X_pred = np.arange(len(estaciones), len(estaciones) + prediccion_dias).reshape(-1, 1)
        # Predicción
        y_pred_scaled = modelo.predict(X_pred)
        # Revertir la normalización
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(afluencias.reshape(-1, 1))  # Ajustar el scaler a las afluencias originales
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        # Graficar la afluencia histórica y la predicción
        ax.plot(np.arange(len(estaciones)), afluencias, color=color_afluencia, label='Afluencia Estación', marker='o', markersize=8)
        ax.plot(np.arange(len(estaciones), len(estaciones) + prediccion_dias), y_pred, color=color_prediccion, linestyle='-', marker='o', markersize=8, label='Predicción')
        ax.set_title(titulo)
        # Calcular límites dinámicos
        min_value = min(afluencias.min(), y_pred.min()) - 0.1  # Margen inferior
        max_value = max(afluencias.max(), y_pred.max()) + 0.1  # Margen superior
        ax.set_ylim(min_value, max_value)
        ax.set_xlim(-1, len(estaciones) + prediccion_dias)
        ax.set_ylabel('Prediccion Con Datos Reales')
        ax.legend()
        ax.grid()
        # Interactividad con mplcursors para la afluencia histórica
        scatter_historico = ax.scatter(np.arange(len(estaciones)), afluencias, color='blue', label='Afluencia Estación', marker='o', s=50)
        # Interactividad para el gráfico histórico (afluencia real)
        scatter_cursor_historico = mplcursors.cursor(scatter_historico, hover=True)
        @scatter_cursor_historico.connect("add")
        def on_add_historico(sel):
            # Mostrar el nombre de la estación y la afluencia real
            sel.annotation.set_text(f'Estación: {estaciones[sel.index]}\nAfluencia: {int(afluencias[sel.index])}')
        # Interactividad con mplcursors para la predicción
        scatter_prediccion = ax.scatter(np.arange(len(estaciones), len(estaciones) + prediccion_dias), y_pred, color='orange', label='Predicción', marker='o', s=50)
        # Interactividad para el gráfico de predicción (solo valores de predicción)
        scatter_cursor_prediccion = mplcursors.cursor(scatter_prediccion, hover=True)
        @scatter_cursor_prediccion.connect("add")
        def on_add_prediccion(sel):
            estacion_index = sel.index
            # Mostrar solo la predicción en las etiquetas
            if estacion_index < prediccion_dias:
                sel.annotation.set_text(f'Estación: {estaciones[sel.index]}\nPredicción: {y_pred[estacion_index]:.2f}')
    # Llamar a la función para cada línea
    if linea == 'Linea1':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 20, 'pink', '#ff00e8', 'Predicción de Afluencia para Línea 1')
    elif linea == 'Linea2':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 23, 'blue', '#0087ff', 'Predicción de Afluencia para Línea 2')
    elif linea == 'Linea3':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 22, 'green', '#00ff17', 'Predicción de Afluencia para Línea 3')
    elif linea == 'Linea4':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 10, '#43e7e7', '#00fff7', 'Predicción de Afluencia para Línea 4')
    elif linea == 'Linea5':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 13, '#e1e813', '#fff300', 'Predicción de Afluencia para Línea 5')
    elif linea == 'Linea6':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 11, 'red', '#cc0000', 'Predicción de Afluencia para Línea 6')
    elif linea == 'Linea7':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 14, 'orange', '#ff7800', 'Predicción de Afluencia para Línea 7')
    elif linea == 'Linea8':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 19, '#19bc68', '#87ff00', 'Predicción de Afluencia para Línea 8')
    elif linea == 'Linea9':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 12, 'brown', '#5f2727', 'Predicción de Afluencia para Línea 9')
    elif linea == 'LineaA':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 10, 'purple', '#ec00ff', 'Predicción de Afluencia para Línea A')
    elif linea == 'LineaB':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 19, '#0a8352', '#34c43d', 'Predicción de Afluencia para Línea B')
    elif linea == 'Linea12':
        graficar_prediccion(axs[2, 1], linea, estaciones, afluencias, modelo, 20, 'gold', '#d8df00', 'Predicción de Afluencia para Línea 12') 
    plt.tight_layout()
    # Mostrar la figura en el canvas de Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # Mostrar los datos
    mostrar_datos(estaciones, afluencias)
    
# Función para mostrar el formulario con los datos de estaciones y afluencia (usando Combobox)
def mostrar_datos(estaciones, afluencias):
    eliminar_imagen()
    # Eliminar widgets antiguos
    for widget in frame_datos.winfo_children():
        widget.destroy()
    # Crear etiquetas y entradas para cada estación y su afluencia
    for estacion, afluencia in zip(estaciones, afluencias):
        estacion_label = ctk.CTkLabel(frame_datos, text=f"Estación: {estacion}", text_color='black')
        afluencia_entry = ctk.CTkEntry(frame_datos, justify='right', fg_color='#f1f7dc', text_color='black', width=150)
        afluencia_entry.insert(0, f"{int(afluencia):,}")
        # Posiciona los widgets en el frame
        estacion_label.pack(anchor='w', padx=10, pady=0)
        afluencia_entry.pack(anchor='w', padx=10, pady=2)
    # Actualiza el área de desplazamiento del canvas después de agregar nuevos widgets
    frame_datos.update_idletasks()  # Actualiza el layout
    canvas.configure(scrollregion=canvas.bbox("all"))  # Ajusta la scrollregion del canvas
# Función para mostrar datos de afluencia en el formulario
def mostrar_datos_mes(meses, afluencias):
    eliminar_imagen()
    # Eliminar widgets antiguos
    for widget in frame_datos.winfo_children():
        widget.destroy()
    # Crear etiquetas y entradas para cada mes y su afluencia
    for mes, afluencia in zip(meses, afluencias):
        mes_label = ctk.CTkLabel(frame_datos, text=f"Mes: {mes}", text_color='black')
        afluencia_entry = ctk.CTkEntry(frame_datos, justify='right', fg_color='#f1f7dc', text_color='black', width=145)
        afluencia_entry.insert(0, f"{int(afluencia):,}")
        # Posicionar los widgets en el frame
        mes_label.pack(anchor='w', padx=10, pady=0)
        afluencia_entry.pack(anchor='w', padx=10, pady=2)
        
def mostrar_datos_dias(dias, afluencias):
    eliminar_imagen()
    # Eliminar widgets antiguos
    for widget in frame_datos.winfo_children():
        widget.destroy()
    # Crear etiquetas y entradas para cada día y su afluencia
    for dia, afluencia in zip(dias, afluencias):
        dia_label = ctk.CTkLabel(frame_datos, text=f"Día: {dia}", text_color='black')
        afluencia_entry = ctk.CTkEntry(frame_datos, justify='right', fg_color='#f1f7dc', text_color='black', width=145)
        afluencia_entry.insert(0, f"{int(afluencia):,}")
        # Posicionar los widgets en el frame
        dia_label.pack(anchor='w', padx=10, pady=0)
        afluencia_entry.pack(anchor='w', padx=10, pady=2)
        
def mostrar_datos_prediccion(estaciones_prediccion, afluencias_prediccion):
    eliminar_imagen()
    # Eliminar widgets antiguos
    for widget in frame_datos.winfo_children():
        widget.destroy()
    # Crear etiquetas y entradas para cada estación y su predicción de afluencia
    for estacion, afluencia in zip(estaciones_prediccion, afluencias_prediccion):
        # Crear el label para el nombre de la estación
        estacion_label = ctk.CTkLabel(frame_datos, text=f"Estación: {estacion}", text_color='black')
        # Crear el campo de entrada para la afluencia predicha
        afluencia_entry = ctk.CTkEntry(frame_datos, justify='right', fg_color='#f1f7dc', text_color='black', width=145)
        afluencia_entry.insert(0, f"{int(afluencia):,}")  # Insertar la afluencia predicha en formato numérico
        # Posicionar los widgets en el frame
        estacion_label.pack(anchor='w', padx=10, pady=0)
        afluencia_entry.pack(anchor='w', padx=10, pady=2)              
    # Actualiza el área de desplazamiento del canvas después de agregar nuevos widgets
    frame_datos.update_idletasks()  # Actualiza el layout
    canvas.configure(scrollregion=canvas.bbox("all"))  # Ajusta la scrollregion del canvas

# Función para borrar todo lo que haya en la pantalla y mostrar solo la imagen
def mostrar_inicio():
    # Ocultar los otros frames y widgets
    frame_canvas.pack_forget()  # Oculta el canvas
    frame_datos.pack_forget()  # Oculta el frame de los datos
    lbl_img.pack(fill=tk.BOTH, expand=True)  # Muestra solo la imagen
    frame_imagen.pack(fill=tk.BOTH, expand=True)  # Asegura que el frame de la imagen esté visible
    redimensionar_imagen_a_fijo()  # Redimensiona la imagen para ocupar toda la ventana    
    
# Crear la ventana principal de tkinter
ventana = ctk.CTk()  # Inicializa una nueva ventana con customtkinter
ventana.title("Visualización de Gráficas")  # Establece el título de la ventana
ventana.geometry("1200x800")  # Define el tamaño de la ventana (ancho x alto en píxeles)
ventana.configure(bg='#f1f7dc')

# Crear un Frame exclusivo para la imagen
frame_imagen = tk.Frame(ventana)
frame_imagen.pack(fill=tk.BOTH, expand=True)
# Mostrar la imagen inicialmente en el Frame dedicado
img_original = Image.open("2.png")

# Función para redimensionar la imagen a un tamaño fijo (1920x1080)
def redimensionar_imagen_a_fijo():
    # Redimensionar la imagen original a 1920x1080
    img_redimensionada = img_original.resize((1920, 1080), Image.LANCZOS)
    # Convertir la imagen redimensionada a PhotoImage para Tkinter
    img = ImageTk.PhotoImage(img_redimensionada)
    # Actualizar la imagen en el label
    lbl_img.config(image=img)
    lbl_img.image = img  # Guardar una referencia para evitar que se borre la imagen
# Función para eliminar la imagen y ajustar el layout
def eliminar_imagen():
    if lbl_img.winfo_ismapped():  # Verificar si la imagen está visible
        lbl_img.pack_forget()  # Ocultar el label de la imagen
        frame_imagen.pack_forget()  # Ocultar el frame de la imagen
        frame_canvas.pack(fill=tk.BOTH, expand=True)  # Expandir el frame del canvas para que ocupe todo el espacio

# Crear el label de la imagen y agregarlo al frame
lbl_img = tk.Label(frame_imagen)
lbl_img.pack(fill=tk.BOTH, expand=True)

# Llamar a la función para redimensionar la imagen a 1920x1080
redimensionar_imagen_a_fijo()    
# Crear un frame para contener el canvas de las gráficas y los datos
frame_canvas = ctk.CTkFrame(ventana, fg_color='#f1f7dc')  # Establecer color de fondo a negro
frame_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Permite que el frame se expanda para llenar el espacio
# Crear un canvas para mostrar las gráficas
canvas = tk.Canvas(frame_canvas, bg="#f1f7dc")  # Usamos tk.Canvas de tkinter con fondo negro
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Permite que el canvas se expanda y llene el frame
# Agregar una barra de desplazamiento vertical al canvas
scrollbar = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview, )  # Crea una barra de desplazamiento vertical
scrollbar.pack(side=tk.RIGHT, fill="y")  # Posiciona la barra de desplazamiento a la derecha
# Configura el canvas para que use la barra de desplazamiento
canvas.configure(yscrollcommand=scrollbar.set)  # Vincula la barra de desplazamiento al canvas
# Crear un frame para las gráficas y los datos dentro del canvas
frame_grafica = ctk.CTkFrame(canvas, fg_color='#f1f7dc')  # Establecer color de fondo a negro
canvas.create_window((0, 0), window=frame_grafica, anchor="nw")  # Crea una ventana en el canvas para el frame de gráficas
# Actualiza el área de desplazamiento del canvas cuando el frame_grafica cambia de tamaño
frame_grafica.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))  # Actualiza el scrollregion del canvas
# Función para manejar el desplazamiento del ratón (scroll) en el canvas
def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")  # Desplaza el canvas en función de la rueda del ratón
# Vincula el evento de desplazamiento del ratón a la función on_mouse_wheel
ventana.bind_all("<MouseWheel>", on_mouse_wheel)  # Asegura que todos los scrolls de la rueda del ratón funcionen
# Crear un frame para mostrar los datos de afluencia
frame_datos = ctk.CTkFrame(frame_canvas, fg_color='#f1f7dc', width=200)  # Establecer color 
canvas.create_window((1690, 0), window=frame_datos, anchor="nw")  # Mueve el frame de datos más a la derecha
# Crear el menú usando tkinter
menu_bar = tk.Menu(ventana, bg='#f1f7dc', fg='black')  # Intenta aplicar color de fondo
ventana.config(menu=menu_bar, bg='#f1f7dc')
menu_bar.add_command(label="Inicio", command=mostrar_inicio) 
# Crear un menú desplegable para opciones
menu_opciones = tk.Menu(menu_bar, tearoff=0, bg='#f1f7dc', fg='black')  # Elimina la opción de separación
menu_bar.add_cascade(label="Opciones", menu=menu_opciones)

grafica_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="Gráficas", menu=grafica_menu)
# Opciones de gráfico
opciones_grafica = ['Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_grafica:
    grafica_menu.add_command(label=opcion, command=lambda opt=opcion: mostrar_grafica(opt))
# Añadir un submenú para regresión lineal
regresion_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="Regresión Lineal", menu=regresion_menu)
opciones_regresion_lineal = ['AfluenciaAnual', 'Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_regresion_lineal:
    regresion_menu.add_cascade(label=opcion, command=lambda opt=opcion: mostrar_regresion_lineal(opt))

regresion_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="Afluencia Mensual", menu=regresion_menu)    
opciones_afluencia_mes = ['Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_afluencia_mes:
    regresion_menu.add_cascade(label=opcion, command=lambda opt=opcion: mostrar_afluencia_mes(opt))

regresion_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="Afluencia Diaria", menu=regresion_menu)    
opciones_afluencia_dia = ['Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_afluencia_dia:
    regresion_menu.add_cascade(label=opcion, command=lambda opt=opcion: mostrar_afluencia_dia(opt))

regresion_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="Tiempo De Espera", menu=regresion_menu)    
opciones_tiempo_espera = ['Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_tiempo_espera:
    regresion_menu.add_cascade(label=opcion, command=lambda opt=opcion: mostrar_tiempo_espera(opt))
    
prediccion_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="Predicción", menu=prediccion_menu)
opciones_prediccion = ['Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_prediccion:
    prediccion_menu.add_command(label=opcion, command=lambda opt=opcion: manejar_seleccion(opt))
    
regresion_menu = tk.Menu(menu_opciones, tearoff=0, bg='#f1f7dc', fg='black')
menu_opciones.add_cascade(label="regresion multiple ", menu=regresion_menu)
opciones_regresion_multiple = ['Linea1', 'Linea2', 'Linea3', 'Linea4', 'Linea5', 'Linea6', 'Linea7', 'Linea8', 'Linea9', 'LineaA', 'LineaB', 'Linea12']
for opcion in opciones_regresion_multiple:
    regresion_menu.add_command(label=opcion, command=lambda opt=opcion: graficar_regresion_multiple_3d(opt))    

ventana.mainloop()  # Inicia el bucle de eventos de la interfaz gráfica