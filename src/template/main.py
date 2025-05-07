import json
from collections import defaultdict
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Variable global para la ruta de datos (ajústala según tu estructura de directorio)
data_dir = "../data"

def load_and_process_data(json_file, window_size=7):
    """Carga y procesa los datos de un archivo JSON, sin eliminar ningún dato."""
    try:
        # Cargar datos
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Diccionarios para almacenar datos procesados
        commits_by_day = defaultdict(int)
        active_users_by_day = defaultdict(set)
        
        # Procesar los datos de cada usuario, sin filtrar ningún período
        for user_id, stats in data.items():
            for fecha_str, cnt in stats.get("daily_commits", {}).items():
                current_date_obj = dt.datetime.strptime(fecha_str, "%Y-%m-%d")
                
                # Acumular datos sin filtrar
                commits_by_day[fecha_str] += cnt
                if cnt > 0:
                    active_users_by_day[fecha_str].add(user_id)

        # Ordenar fechas
        fechas = sorted(commits_by_day.keys())
        
        # Manejar caso de datos vacíos
        if not fechas:
            return {
                'fechas': [],
                'commits_rolling_avg': [],
                'users_rolling_avg': [],
                'raw_commits': [],  # Añadimos datos crudos
                'raw_users': []
            }

        # Convertir fechas a objetos datetime
        fechas_dt = [dt.datetime.strptime(f, "%Y-%m-%d") for f in fechas]
        counts = [commits_by_day[f] for f in fechas]
        active_users = [len(active_users_by_day[f]) for f in fechas]

        # Crear serie temporal completa con fechas faltantes
        if len(fechas_dt) > 1:
            # Determinar rango de fechas
            current_date_fill = fechas_dt[0] 
            end_date_fill = fechas_dt[-1]
            
            # Generar todas las fechas en el rango
            date_range = []
            while current_date_fill <= end_date_fill:
                date_range.append(current_date_fill)
                current_date_fill += dt.timedelta(days=1)
            
            # Completar series con valores para todas las fechas
            complete_counts = []
            complete_users = []
            complete_dates = []
            
            for date_in_range in date_range:
                date_str = date_in_range.strftime("%Y-%m-%d")
                complete_dates.append(date_in_range)
                
                # Asignar valor real o cero según exista datos
                if date_str in commits_by_day:
                    complete_counts.append(commits_by_day[date_str])
                    complete_users.append(len(active_users_by_day[date_str]))
                else:
                    complete_counts.append(0)
                    complete_users.append(0)
            
            # Reemplazar series originales con series completas
            fechas_dt = complete_dates
            counts = complete_counts
            active_users = complete_users
        elif not fechas_dt:
             counts = []
             active_users = []

        # Calcular promedios móviles
        commits_rolling_avg = calculate_rolling_average(counts, window_size)
        users_rolling_avg = calculate_rolling_average(active_users, window_size)
        
        return {
            'fechas': fechas_dt,
            'commits_rolling_avg': commits_rolling_avg,
            'users_rolling_avg': users_rolling_avg,
            'raw_commits': counts,  # Datos sin promediar
            'raw_users': active_users
        }
    except Exception as e:
        print(f"Error procesando {json_file}: {e}")
        return {
            'fechas': [],
            'commits_rolling_avg': [],
            'users_rolling_avg': [],
            'raw_commits': [],
            'raw_users': []
        }

def calculate_rolling_average(data_series, window_size):
    """Calcula el promedio móvil para una serie de datos"""
    rolling_avg = []
    
    if not data_series:
        return rolling_avg
        
    for i in range(len(data_series)):
        if i < window_size - 1:
            rolling_avg.append(np.mean(data_series[:i+1]))
        else:
            rolling_avg.append(np.mean(data_series[i-window_size+1:i+1]))
    
    return rolling_avg

def calculate_weekly_data(json_file):
    """Calcula datos semanales de commits y usuarios activos."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Preparar estructuras de datos para agregación semanal
        commits_by_week = defaultdict(int)
        active_users_by_week = defaultdict(set)
        
        # Procesar datos de cada usuario, sin filtrar
        for user_id, stats in data.items():
            for fecha_str, cnt in stats.get("daily_commits", {}).items():
                date_obj = dt.datetime.strptime(fecha_str, "%Y-%m-%d")
                
                # Determinar clave de semana (ISO)
                year, week, _ = date_obj.isocalendar()
                week_key = f"{year}-W{week:02d}"
                
                # Acumular datos por semana
                commits_by_week[week_key] += cnt
                if cnt > 0:
                    active_users_by_week[week_key].add(user_id)
        
        # Ordenar semanas
        weeks = sorted(commits_by_week.keys())
        
        # Manejar caso de datos vacíos
        if not weeks:
            return {
                'fechas': [],
                'commits': [],
                'active_users': [],
                'week_labels': []
            }
        
        # Convertir etiquetas de semana a fechas (primer día de cada semana)
        week_dates = []
        for week_key in weeks:
            year_str, week_str = week_key.split('-W')
            year = int(year_str)
            week_num = int(week_str)
            # Usar formato ISO para semanas: el lunes es el primer día
            date = dt.datetime.strptime(f"{year}-{week_num}-1", "%G-W%V-%u")
            week_dates.append(date)
        
        # Extraer datos de conteos
        week_commits = [commits_by_week[w] for w in weeks]
        week_users = [len(active_users_by_week[w]) for w in weeks]
        
        return {
            'fechas': week_dates,
            'commits': week_commits,
            'active_users': week_users,
            'week_labels': weeks
        }
    except Exception as e:
        print(f"Error procesando datos semanales de {json_file}: {e}")
        return {
            'fechas': [],
            'commits': [],
            'active_users': [],
            'week_labels': []
        }

def plot_daily_activity(
    countries=["Bangladesh", "India", "Philippines"],
    highlight_start="2024-07-17",
    highlight_end="2024-07-24",
    window_size=7,
    output_dir="output",
    show_highlight=True  # Nuevo parámetro para mostrar/ocultar área sombreada
):
    """Genera gráficas de actividad diaria (commits y usuarios) a escala real para individuales."""
    # Asegurar que existe el directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Diccionarios para almacenar datos procesados y estadísticas
    all_data = {}
    max_commits_overall_comparative = 0
    max_users_overall_comparative = 0
    
    # Colores para cada país
    colors = {"Bangladesh": "blue", "India": "green", "Philippines": "purple"}
    
    # Convertir fechas de interés a objetos datetime
    hl_start_dt = dt.datetime.strptime(highlight_start, "%Y-%m-%d")
    hl_end_dt = dt.datetime.strptime(highlight_end, "%Y-%m-%d")
    
    # Cargar y procesar datos para cada país
    for country in countries:
        json_file = os.path.join(data_dir, f"{country}_data.json")
        try:
            country_data = load_and_process_data(json_file, window_size)
            all_data[country] = country_data
            
            # Actualizar máximos para escala común en gráficos comparativos
            if country_data['commits_rolling_avg']:
                current_max_commits = max(country_data['commits_rolling_avg'])
                max_commits_overall_comparative = max(max_commits_overall_comparative, current_max_commits)
            
            if country_data['users_rolling_avg']:
                current_max_users = max(country_data['users_rolling_avg'])
                max_users_overall_comparative = max(max_users_overall_comparative, current_max_users)
                
        except FileNotFoundError:
            print(f"Advertencia: Archivo {json_file} no encontrado.")
            all_data[country] = {'fechas': [], 'commits_rolling_avg': [], 'users_rolling_avg': []}
    
    # Generar gráficas individuales de commits diarios
    create_individual_plots(
        all_data, 
        colors, 
        hl_start_dt, 
        hl_end_dt, 
        'commits_rolling_avg', 
        'Commits diarios a GitHub', 
        'Número de Commits (promedio)',
        'commits_diarios',
        output_dir,
        show_highlight
    )
    
    # Generar gráfica comparativa de commits diarios
    create_comparative_plot(
        all_data, 
        colors, 
        hl_start_dt, 
        hl_end_dt, 
        'commits_rolling_avg', 
        'Comparación de commits diarios a GitHub', 
        'Número de commits (promedio)',
        max_commits_overall_comparative,
        'comparacion_commits_diarios',
        output_dir,
        show_highlight
    )

def plot_daily_raw_activity(
    countries=["Bangladesh", "India", "Philippines"],
    highlight_start="2024-07-17",
    highlight_end="2024-07-24",
    output_dir="output",
    show_highlight=True
):
    """Genera gráficas de actividad diaria usando datos sin promediar."""
    # Asegurar que existe el directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Diccionarios para almacenar datos procesados y estadísticas
    all_data = {}
    max_commits_overall = 0
    max_users_overall = 0
    
    # Colores para cada país
    colors = {"Bangladesh": "blue", "India": "green", "Philippines": "purple"}
    
    # Convertir fechas de interés a objetos datetime
    hl_start_dt = dt.datetime.strptime(highlight_start, "%Y-%m-%d")
    hl_end_dt = dt.datetime.strptime(highlight_end, "%Y-%m-%d")
    
    # Cargar y procesar datos para cada país
    for country in countries:
        json_file = os.path.join(data_dir, f"{country}_data.json")
        try:
            country_data = load_and_process_data(json_file)
            
            # Actualizar máximos para escala común en gráficos comparativos
            if country_data['raw_commits']:
                current_max = max(country_data['raw_commits'])
                max_commits_overall = max(max_commits_overall, current_max)
            
            if country_data['raw_users']:
                current_max = max(country_data['raw_users'])
                max_users_overall = max(max_users_overall, current_max)
                
            all_data[country] = country_data
        except FileNotFoundError:
            print(f"Advertencia: Archivo {json_file} no encontrado.")
            all_data[country] = {'fechas': [], 'raw_commits': [], 'raw_users': []}
    
    # Generar gráficas individuales de commits diarios (RAW)
    for country, data_dict in all_data.items():
        if not data_dict['fechas'] or not data_dict['raw_commits']:
            print(f"No hay datos de commits para graficar de {country}.")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar datos crudos como puntos y línea
        ax.plot(data_dict['fechas'], data_dict['raw_commits'], 
               marker='o', markersize=3, linestyle='-', linewidth=1, 
               color=colors.get(country, 'blue'), label='Commits diarios')
        
        # Opcionalmente destacar período de apagón
        if show_highlight:
            ax.axvspan(hl_start_dt, hl_end_dt, alpha=0.2, color='yellow', label='Período de apagón')
            # Agregar líneas verticales para marcar claramente el inicio y fin del período
            ax.axvline(x=hl_start_dt, color='red', linestyle='--', alpha=0.7, label='Inicio apagón')
            ax.axvline(x=hl_end_dt, color='red', linestyle='--', alpha=0.7, label='Fin apagón')
        
        # Configurar escala local para el eje Y
        local_max_value = max(data_dict['raw_commits']) if data_dict['raw_commits'] else 0
        ax.set_ylim(bottom=0, top=local_max_value * 1.1 if local_max_value > 0 else 1)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title(f'Commits diarios (datos crudos) - {country}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Número de Commits')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{country}_commits_raw.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Figura guardada como: {output_file}")
    
    # Generar gráfica comparativa de commits diarios (RAW)
    fig, ax = plt.subplots(figsize=(14, 7))
    has_data = False
    
    for country, data_dict in all_data.items():
        if data_dict['fechas'] and data_dict['raw_commits']:
            ax.plot(data_dict['fechas'], data_dict['raw_commits'], 
                   marker='o', markersize=2, linestyle='-', linewidth=1, 
                   color=colors.get(country, 'blue'), label=country)
            has_data = True
    
    if has_data:
        # Opcionalmente destacar período de apagón
        if show_highlight:
            ax.axvspan(hl_start_dt, hl_end_dt, alpha=0.2, color='yellow', label='Período de apagón')
            ax.axvline(x=hl_start_dt, color='red', linestyle='--', alpha=0.7, label='Inicio apagón')
            ax.axvline(x=hl_end_dt, color='red', linestyle='--', alpha=0.7, label='Fin apagón')
        
        # Configurar escala para el eje Y
        ax.set_ylim(bottom=0, top=max_commits_overall * 1.1 if max_commits_overall > 0 else 1)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title('Comparación de commits diarios (datos crudos)')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Número de commits')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, "comparacion_commits_raw.png")
        plt.savefig(output_file, dpi=300)
        print(f"Figura comparativa guardada como: {output_file}")
    else:
        print("No hay datos para la gráfica comparativa después del filtrado.")
    
    plt.close(fig)

def plot_percentage_change(
    countries=["Bangladesh", "India", "Philippines"],
    highlight_start="2024-07-17",
    highlight_end="2024-07-24",
    window_size=7,
    show_highlight=True,
    output_dir="output"
):
    """Genera gráficas de cambio porcentual en actividad diaria."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = {}
    min_pct_change = 0
    max_pct_change = 0
    
    colors = {"Bangladesh": "blue", "India": "green", "Philippines": "purple"}
    
    hl_start_dt = dt.datetime.strptime(highlight_start, "%Y-%m-%d")
    hl_end_dt = dt.datetime.strptime(highlight_end, "%Y-%m-%d")
    
    # Cargar y procesar datos para cada país
    for country in countries:
        json_file = os.path.join(data_dir, f"{country}_data.json")
        try:
            country_data = load_and_process_data(json_file, window_size)
            
            if not country_data['fechas'] or not country_data['commits_rolling_avg']:
                print(f"No hay suficientes datos para {country} después del filtrado.")
                all_data[country] = {'fechas': [], 'pct_change': []}
                continue
            
            # Calcular cambio porcentual día a día
            pct_change = []
            fechas_pct = []
            
            for i in range(1, len(country_data['commits_rolling_avg'])):
                today = country_data['commits_rolling_avg'][i]
                yesterday = country_data['commits_rolling_avg'][i-1]
                
                if yesterday != 0:  # Evitar división por cero
                    change = ((today - yesterday) / yesterday) * 100
                else:
                    if today == 0:
                        change = 0
                    else:
                        change = 100  # Si ayer fue 0 y hoy no, consideramos un 100% de aumento
                
                pct_change.append(change)
                fechas_pct.append(country_data['fechas'][i])
            
            all_data[country] = {
                'fechas': fechas_pct,
                'pct_change': pct_change
            }
            
            # Actualizar límites globales
            if pct_change:
                current_min = min(pct_change)
                current_max = max(pct_change)
                min_pct_change = min(min_pct_change, current_min)
                max_pct_change = max(max_pct_change, current_max)
            
        except FileNotFoundError:
            print(f"Advertencia: Archivo {json_file} no encontrado.")
            all_data[country] = {'fechas': [], 'pct_change': []}
    
    # Generar gráficas individuales de cambio porcentual
    for country, data_dict in all_data.items():
        if not data_dict['fechas'] or not data_dict['pct_change']:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar cambio porcentual
        ax.plot(
            data_dict['fechas'], 
            data_dict['pct_change'], 
            linestyle='-', 
            linewidth=2.0, 
            color=colors.get(country, 'blue'), 
            label='Cambio porcentual diario'
        )
        
        # Agregar línea horizontal en 0% para referencia
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Opcionalmente destacar período de apagón
        if show_highlight:
            ax.axvspan(hl_start_dt, hl_end_dt, alpha=0.2, color='yellow', label='Período de apagón')
            ax.axvline(x=hl_start_dt, color='red', linestyle='--', alpha=0.7, label='Inicio apagón')
            ax.axvline(x=hl_end_dt, color='red', linestyle='--', alpha=0.7, label='Fin apagón')
        
        # Configurar límites del eje Y con margen
        buffer = 10  # Porcentaje adicional de margen
        y_min = min(data_dict['pct_change']) - buffer
        y_max = max(data_dict['pct_change']) + buffer
        ax.set_ylim(bottom=y_min, top=y_max)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title(f'Cambio porcentual diario en commits - {country}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Cambio porcentual (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{country}_cambio_porcentual.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Figura guardada como: {output_file}")
    
    # Generar gráfica comparativa de cambio porcentual
    fig, ax = plt.subplots(figsize=(14, 7))
    has_data = False
    
    for country, data_dict in all_data.items():
        if data_dict['fechas'] and data_dict['pct_change']:
            ax.plot(
                data_dict['fechas'], 
                data_dict['pct_change'], 
                linestyle='-', 
                linewidth=2.0, 
                color=colors.get(country, 'blue'), 
                label=country
            )
            has_data = True
    
    if has_data:
        # Agregar línea horizontal en 0% para referencia
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Opcionalmente destacar período de apagón
        if show_highlight:
            ax.axvspan(hl_start_dt, hl_end_dt, alpha=0.2, color='yellow', label='Período de apagón')
            ax.axvline(x=hl_start_dt, color='red', linestyle='--', alpha=0.7, label='Inicio apagón')
            ax.axvline(x=hl_end_dt, color='red', linestyle='--', alpha=0.7, label='Fin apagón')
        
        # Configurar límites del eje Y con margen
        buffer = 10  # Porcentaje adicional de margen
        ax.set_ylim(bottom=min_pct_change-buffer, top=max_pct_change+buffer)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title('Comparación de cambio porcentual diario en commits')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Cambio porcentual (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, "comparacion_cambio_porcentual.png")
        plt.savefig(output_file, dpi=300)
        print(f"Figura comparativa guardada como: {output_file}")
    else:
        print("No hay datos para la gráfica comparativa de cambio porcentual.")
    
    plt.close(fig)

def create_individual_plots(all_data, colors, hl_start_dt, hl_end_dt, data_key, title_prefix, ylabel, filename_suffix, output_dir, show_highlight=True):
    """Función auxiliar para crear gráficas individuales por país"""
    for country, data_dict in all_data.items():
        if not data_dict['fechas'] or not data_dict[data_key]:
            print(f"No hay datos de {data_key} para graficar de {country} después del filtrado.")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar serie temporal
        ax.plot(
            data_dict['fechas'], 
            data_dict[data_key], 
            linestyle='-', 
            linewidth=2.5, 
            color=colors.get(country, 'blue'), 
            label=f'Promedio móvil (7 días)'
        )
        
        # Opcionalmente destacar período de apagón
        if show_highlight:
            ax.axvspan(hl_start_dt, hl_end_dt, alpha=0.2, color='yellow', label='Período de apagón')
            # Agregar líneas verticales para marcar claramente el inicio y fin del período
            ax.axvline(x=hl_start_dt, color='red', linestyle='--', alpha=0.7, label='Inicio apagón')
            ax.axvline(x=hl_end_dt, color='red', linestyle='--', alpha=0.7, label='Fin apagón')
        
        # Configurar escala local para el eje Y
        local_max_value = max(data_dict[data_key]) if data_dict[data_key] else 0
        ax.set_ylim(bottom=0, top=local_max_value * 1.1 if local_max_value > 0 else 1)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title(f'{title_prefix} - {country}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{country}_{filename_suffix}.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Figura guardada como: {output_file}")

def create_comparative_plot(all_data, colors, hl_start_dt, hl_end_dt, data_key, title, ylabel, max_value, filename, output_dir, show_highlight=True):
    """Función auxiliar para crear gráficas comparativas"""
    fig, ax = plt.subplots(figsize=(14, 7))
    has_data = False
    
    # Graficar series de todos los países
    for country, data_dict in all_data.items():
        if data_dict['fechas'] and data_dict[data_key]:
            ax.plot(
                data_dict['fechas'], 
                data_dict[data_key], 
                linestyle='-', 
                linewidth=2.5, 
                color=colors.get(country, 'blue'), 
                label=country
            )
            has_data = True
    
    if has_data:
        # Opcionalmente destacar período de apagón
        if show_highlight:
            ax.axvspan(hl_start_dt, hl_end_dt, alpha=0.2, color='yellow', label='Período de apagón')
            ax.axvline(x=hl_start_dt, color='red', linestyle='--', alpha=0.7, label='Inicio apagón')
            ax.axvline(x=hl_end_dt, color='red', linestyle='--', alpha=0.7, label='Fin apagón')
        
        # Configurar escala para el eje Y
        ax.set_ylim(bottom=0, top=max_value * 1.1 if max_value > 0 else 1)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title(title)
        ax.set_xlabel('Fecha')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Figura comparativa guardada como: {output_file}")
    else:
        print(f"No hay datos para la gráfica comparativa {filename} después del filtrado.")
    
    plt.close(fig)

def plot_weekly_activity(
    countries=["Bangladesh", "India", "Philippines"],
    highlight_week_start="2024-W29",
    highlight_week_end="2024-W30",
    output_dir="output",
    show_highlight=True
):
    """Genera gráficas de actividad semanal (commits y usuarios) a escala real para individuales."""
    # Asegurar que existe el directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Diccionarios para almacenar datos procesados y estadísticas
    all_data = {}
    max_commits_overall_comparative = 0
    max_users_overall_comparative = 0
    
    # Colores para cada país
    colors = {"Bangladesh": "blue", "India": "green", "Philippines": "purple"}
    
    # Función de formato para etiquetas de semanas
    def format_week_date(x, pos=None):
        try:
            date_val = mdates.num2date(x)
            year, week, _ = date_val.isocalendar()
            return f"{year}-W{week:02d}"
        except ValueError: 
            return ""
    
    # Cargar y procesar datos para cada país
    for country in countries:
        json_file = os.path.join(data_dir, f"{country}_data.json")
        try:
            country_data = calculate_weekly_data(json_file)
            all_data[country] = country_data
            
            # Actualizar máximos para escala común en gráficos comparativos
            if country_data['commits']:
                current_max_commits = max(country_data['commits'])
                max_commits_overall_comparative = max(max_commits_overall_comparative, current_max_commits)
            
            if country_data['active_users']:
                current_max_users = max(country_data['active_users'])
                max_users_overall_comparative = max(max_users_overall_comparative, current_max_users)
                
        except FileNotFoundError:
            print(f"Advertencia: Archivo {json_file} no encontrado.")
            all_data[country] = {'fechas': [], 'commits': [], 'active_users': [], 'week_labels': []}
    
    # Generar gráficas individuales de commits semanales
    create_weekly_individual_plots(
        all_data, 
        colors, 
        highlight_week_start, 
        highlight_week_end, 
        'commits', 
        'Commits semanales a GitHub', 
        'Número de Commits',
        'commits_semanales',
        format_week_date,
        output_dir,
        show_highlight
    )
    
    # Generar gráficas individuales de usuarios activos semanales
    create_weekly_individual_plots(
        all_data, 
        colors, 
        highlight_week_start, 
        highlight_week_end, 
        'active_users', 
        'Usuarios activos semanales en GitHub', 
        'Número de usuarios activos',
        'usuarios_semanales',
        format_week_date,
        output_dir,
        show_highlight
    )
    
    # Encontrar país de referencia para el período de span
    ref_country_data_for_span = None
    for country, data_dict in all_data.items():
        if data_dict['week_labels'] and data_dict['fechas']:
            ref_country_data_for_span = data_dict
            break
    
    # Generar gráfica comparativa de commits semanales
    create_weekly_comparative_plot(
        all_data, 
        colors, 
        highlight_week_start, 
        highlight_week_end, 
        'commits', 
        'Comparación de commits semanales a GitHub', 
        'Número de commits',
        max_commits_overall_comparative,
        'comparacion_commits_semanales',
        format_week_date,
        ref_country_data_for_span,
        output_dir,
        show_highlight
    )
    
    # Generar gráfica comparativa de usuarios activos semanales
    create_weekly_comparative_plot(
        all_data, 
        colors, 
        highlight_week_start, 
        highlight_week_end, 
        'active_users', 
        'Comparación de usuarios activos semanales en GitHub', 
        'Número de usuarios activos',
        max_users_overall_comparative,
        'comparacion_usuarios_semanales',
        format_week_date,
        ref_country_data_for_span,
        output_dir,
        show_highlight
    )

def create_weekly_individual_plots(all_data, colors, highlight_week_start, highlight_week_end, 
                                  data_key, title_prefix, ylabel, filename_suffix, date_formatter, 
                                  output_dir, show_highlight=True):
    """Función auxiliar para crear gráficas individuales semanales por país"""
    for country, data_dict in all_data.items():
        if not data_dict['fechas'] or not data_dict[data_key]:
            print(f"No hay datos de {data_key} semanales para graficar de {country} después del filtrado.")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar serie temporal
        ax.plot(
            data_dict['fechas'], 
            data_dict[data_key], 
            linestyle='-', 
            linewidth=2.5, 
            color=colors.get(country, 'blue'), 
            label=f'{title_prefix.split(" ")[0]} semanales'
        )
        
        # Opcionalmente destacar período de apagón
        if show_highlight:
            highlight_indices = []
            if data_dict['week_labels']:
                for i, week_label in enumerate(data_dict['week_labels']):
                    if highlight_week_start <= week_label <= highlight_week_end:
                        highlight_indices.append(i)
            
            if highlight_indices and data_dict['fechas'] and highlight_indices[-1] < len(data_dict['fechas']):
                min_idx = min(highlight_indices)
                max_idx = max(highlight_indices)
                if min_idx < len(data_dict['fechas']):
                    start_date_span = data_dict['fechas'][min_idx]
                    end_date_span = data_dict['fechas'][max_idx] + dt.timedelta(days=7)
                    ax.axvspan(start_date_span, end_date_span, alpha=0.2, color='yellow', label='Período de apagón')
        
        # Configurar escala local para el eje Y
        local_max_value = max(data_dict[data_key]) if data_dict[data_key] else 0
        ax.set_ylim(bottom=0, top=local_max_value * 1.1 if local_max_value > 0 else 1)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(plt.FuncFormatter(date_formatter))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title(f'{title_prefix} - {country}')
        ax.set_xlabel('Semana')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{country}_{filename_suffix}.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Figura guardada como: {output_file}")

def create_weekly_comparative_plot(all_data, colors, highlight_week_start, highlight_week_end, 
                                  data_key, title, ylabel, max_value, filename, date_formatter, 
                                  ref_country_data, output_dir, show_highlight=True):
    """Función auxiliar para crear gráficas comparativas semanales"""
    fig, ax = plt.subplots(figsize=(14, 7))
    has_data = False
    
    # Graficar series de todos los países
    for country, data_dict in all_data.items():
        if data_dict['fechas'] and data_dict[data_key]:
            ax.plot(
                data_dict['fechas'], 
                data_dict[data_key], 
                linestyle='-', 
                linewidth=2.5, 
                color=colors.get(country, 'blue'), 
                label=country
            )
            has_data = True
    
    if has_data:
        # Opcionalmente destacar período de apagón
        if show_highlight and ref_country_data:
            highlight_indices_comp = []
            for i, week_label in enumerate(ref_country_data['week_labels']):
                if highlight_week_start <= week_label <= highlight_week_end:
                    highlight_indices_comp.append(i)
            
            if highlight_indices_comp and ref_country_data['fechas'] and highlight_indices_comp[-1] < len(ref_country_data['fechas']):
                min_idx = min(highlight_indices_comp)
                max_idx = max(highlight_indices_comp)
                if min_idx < len(ref_country_data['fechas']):
                    start_date_span = ref_country_data['fechas'][min_idx]
                    end_date_span = ref_country_data['fechas'][max_idx] + dt.timedelta(days=7)
                    ax.axvspan(start_date_span, end_date_span, alpha=0.2, color='yellow', label='Período de apagón')
        
        # Configurar escala para el eje Y
        ax.set_ylim(bottom=0, top=max_value * 1.1 if max_value > 0 else 1)
        
        # Configurar ejes y etiquetas
        ax.xaxis.set_major_formatter(plt.FuncFormatter(date_formatter))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        plt.xticks(rotation=45, ha='right')
        
        # Títulos y leyenda
        ax.set_title(title)
        ax.set_xlabel('Semana')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajustar diseño y guardar
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Figura comparativa guardada como: {output_file}")
    else:
        print(f"No hay datos para la gráfica comparativa {filename} después del filtrado.")
    
    plt.close(fig)

# Contenido que se ejecuta si este archivo se ejecuta directamente
if __name__ == "__main__":
    # Configuración
    countries_list = ["Bangladesh", "India", "Philippines"]
    
    # Período de "apagón" a destacar en las gráficas
    highlight_start_date = "2024-07-17"
    highlight_end_date = "2024-07-24"
    highlight_week_start_label = "2024-W29"  # Semana que incluye el 17 de julio
    highlight_week_end_label = "2024-W30"    # Semana que incluye el 24 de julio
    
    output_plots_dir = "output_modified" # Cambiado para no sobrescribir originales si se corre en el mismo lugar
    
    # Crear directorio de salida
    os.makedirs(output_plots_dir, exist_ok=True)
    
    # Generar diferentes visualizaciones
    print("Generando gráficas de actividad diaria (datos crudos)...")
    plot_daily_raw_activity(
        countries=countries_list,
        highlight_start=highlight_start_date,
        highlight_end=highlight_end_date,
        output_dir=output_plots_dir
    )
    
    print("\nGenerando gráficas de actividad diaria con promedios móviles...")
    plot_daily_activity(
        countries=countries_list,
        highlight_start=highlight_start_date,
        highlight_end=highlight_end_date,
        window_size=7,
        output_dir=output_plots_dir
    )
    
    # También generar gráficas sin área sombreada
    sin_sombra_dir = os.path.join(output_plots_dir, "sin_sombreado")
    os.makedirs(sin_sombra_dir, exist_ok=True)
    
    print("\nGenerando gráficas de actividad diaria sin área sombreada...")
    plot_daily_activity(
        countries=countries_list,
        highlight_start=highlight_start_date,
        highlight_end=highlight_end_date,
        window_size=7,
        output_dir=sin_sombra_dir,
        show_highlight=False
    )
    
    print("\nGenerando gráficas de cambio porcentual...")
    plot_percentage_change(
        countries=countries_list,
        highlight_start=highlight_start_date,
        highlight_end=highlight_end_date,
        window_size=7,
        output_dir=output_plots_dir
    )
    
    print("\nGenerando gráficas de actividad semanal...")
    plot_weekly_activity(
        countries=countries_list,
        highlight_week_start=highlight_week_start_label,
        highlight_week_end=highlight_week_end_label,
        output_dir=output_plots_dir
    )
    
    print("\nTodas las gráficas han sido generadas correctamente.")