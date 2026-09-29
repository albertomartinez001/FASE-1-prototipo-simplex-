import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Simplex Web - Estilo PHPSimplex",
    layout="wide"
)

st.title(" Algoritmo Simplex Tabular (Prototipo 1.0)")
st.markdown("""
Esta herramienta resuelve modelos de **Programación Lineal de Maximización** con restricciones de tipo **$\le$** 
desplegando paso a paso cada matriz de iteración, inspirada en la plataforma **PHPSimplex**.
""")

st.divider()

# -----------------------------------------------------------------------------
# PASO 1: DIMENSIONES DEL PROBLEMA
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Modelo")
num_vars = st.sidebar.number_input(
    "Número de variables de decisión (n):",
    min_value=2, max_value=10, value=2, step=1
)
num_constraints = st.sidebar.number_input(
    "Número de restricciones (m):",
    min_value=1, max_value=10, value=3, step=1
)

# Nombres de las columnas
col_vars = [f"X{j+1}" for j in range(num_vars)]
col_slacks = [f"S{i+1}" for i in range(num_constraints)]
all_cols = col_vars + col_slacks + ["RHS"]

# -----------------------------------------------------------------------------
# PASO 2: CAPTURA DE DATOS DEL MODELO
# -----------------------------------------------------------------------------
st.subheader("1. Definición del Modelo Matemático")

st.markdown("#### **Función Objetivo (Maximizar Z)**")
c_coeffs = []
c_cols = st.columns(num_vars)
for j in range(num_vars):
    with c_cols[j]:
        val = st.number_input(f"Coeficiente {col_vars[j]}", value=0.0, step=1.0, key=f"c_{j}")
        c_coeffs.append(val)

st.markdown("#### **Restricciones ($\le$)**")
A_matrix = []
b_vector = []

for i in range(num_constraints):
    st.write(f"**Restricción {i+1}:**")
    row_cols = st.columns(num_vars + 1)
    a_row = []
    for j in range(num_vars):
        with row_cols[j]:
            val = st.number_input(f"{col_vars[j]} (R{i+1})", value=0.0, step=1.0, key=f"a_{i}_{j}")
            a_row.append(val)
    with row_cols[num_vars]:
        b_val = st.number_input(f"Lado Der. (b_{i+1})", value=0.0, step=1.0, key=f"b_{i}")
        b_vector.append(b_val)
    A_matrix.append(a_row)

st.divider()

# -----------------------------------------------------------------------------
# PASO 3: EJECUCIÓN Y RESOLUCIÓN DEL ALGORITMO SIMPLEX
# -----------------------------------------------------------------------------
if st.button("🚀 Resolver con Método Simplex", type="primary", use_container_width=True):
    
    # Validar que los lados derechos sean no negativos
    if any(b < 0 for b in b_vector):
        st.error("❌ El Prototipo 1.0 solo admite términos independientes no negativos ($b_i \ge 0$).")
        st.stop()
        
    st.subheader("2. Proceso Iterativo Paso a Paso")
    
    # Construcción de la Tabla Inicial de NumPy
    c = np.array(c_coeffs, dtype=float)
    A = np.array(A_matrix, dtype=float)
    b = np.array(b_vector, dtype=float)
    
    # Matriz del Simplex de dimensiones (m+1) x (n+m+1)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1), dtype=float)
    
    # Fila 0 (Z): -c_1, -c_2, ..., -c_n, 0, ..., 0
    tableau[0, :num_vars] = -c
    
    # Filas de restricciones: A | I | b
    tableau[1:, :num_vars] = A
    tableau[1:, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[1:, -1] = b
    
    # Rastreo de la Base
    base_vars = col_slacks.copy()
    
    iteration = 0
    max_iterations = 20
    is_optimal = False
    is_unbounded = False
    
    while iteration < max_iterations:
        # Formatear la tabla actual para mostrarla en Pandas/Streamlit
        df_tableau = pd.DataFrame(
            tableau, 
            columns=all_cols, 
            index=["Z"] + base_vars
        )
        
        # Evaluar Optimalidad (revisar coeficientes negativos en la Fila Z)
        z_row_coefs = tableau[0, :-1]
        min_z_val = np.min(z_row_coefs)
        
        if min_z_val >= -1e-9:
            is_optimal = True
            st.success(f"✨ **¡Solución Óptima Encontrada en la Iteración {iteration}!**")
            st.dataframe(df_tableau.style.format("{:.2f}"), use_container_width=True)
            break
            
        # Criterio de Entrada (Columna Pivote)
        col_pivote_idx = np.argmin(z_row_coefs)
        var_entrada = all_cols[col_pivote_idx]
        
        # Criterio de Salida (Prueba del Cociente Mínimo)
        ratios = []
        for i in range(1, num_constraints + 1):
            elem = tableau[i, col_pivote_idx]
            rhs = tableau[i, -1]
            if elem > 1e-9:
                ratios.append(rhs / elem)
            else:
                ratios.append(np.inf)
                
        min_ratio = np.min(ratios)
        
        if min_ratio == np.inf:
            is_unbounded = True
            st.error("⚠️ **El problema tiene una Solución No Acotada (no hay variable de salida válida).**")
            st.dataframe(df_tableau.style.format("{:.2f}"), use_container_width=True)
            break
            
        fila_pivote_idx = np.argmin(ratios) + 1
        var_salida = base_vars[fila_pivote_idx - 1]
        elem_pivote = tableau[fila_pivote_idx, col_pivote_idx]
        
        # Renderizar la iteración actual estilo PHPSimplex
        with st.expander(f"📌 **Iteración {iteration}**", expanded=(iteration == 0)):
            st.dataframe(df_tableau.style.format("{:.2f}"), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Variable de Entrada:** {var_entrada} (Columna {col_pivote_idx + 1})")
            c2.warning(f"**Variable de Salida:** {var_salida} (Fila {fila_pivote_idx})")
            c3.success(f"**Elemento Pivote:** {elem_pivote:.2f}")
        
        # Actualizar la variable básica
        base_vars[fila_pivote_idx - 1] = var_entrada
        
        # Operaciones de Gauss-Jordan
        tableau[fila_pivote_idx, :] /= elem_pivote
        for k in range(num_constraints + 1):
            if k != fila_pivote_idx:
                factor = tableau[k, col_pivote_idx]
                tableau[k, :] -= factor * tableau[fila_pivote_idx, :]
                
        # Limpieza de precisión numérica
        tableau[np.abs(tableau) < 1e-9] = 0.0
        
        iteration += 1

    # -------------------------------------------------------------------------
    # PASO 4: INTERPRETACIÓN DE RESULTADOS
    # -------------------------------------------------------------------------
    if is_optimal:
        st.divider()
        st.subheader("3. Interpretación de Resultados Finales")
        
        opt_z = tableau[0, -1]
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="**Valor Óptimo (Z*)**", value=f"{opt_z:.2f}")
            
            st.markdown("#### **Variables de Decisión**")
            for j in range(num_vars):
                var_name = col_vars[j]
                if var_name in base_vars:
                    row_idx = base_vars.index(var_name) + 1
                    val = tableau[row_idx, -1]
                else:
                    val = 0.0
                st.write(f"• **{var_name}:** {val:.2f}")
                
        with res_col2:
            st.markdown("#### **Variables de Holgura**")
            for i in range(num_constraints):
                slack_name = col_slacks[i]
                if slack_name in base_vars:
                    row_idx = base_vars.index(slack_name) + 1
                    val = tableau[row_idx, -1]
                else:
                    val = 0.0
                st.write(f"• **{slack_name}:** {val:.2f}")