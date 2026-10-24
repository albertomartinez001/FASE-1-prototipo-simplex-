import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Investigación de Operaciones", page_icon="📈", layout="wide")


def simplex_page():
    st.title("Método Simplex Tabular")
    st.markdown(
        "Resuelve modelos de **maximización** con restricciones `≤` y términos independientes no negativos. "
        "Consulta cada tabla, pivote y cambio de base."
    )
    st.sidebar.header("Configuración del modelo")
    n = st.sidebar.number_input("Variables de decisión", 2, 10, 2, key="simplex_n")
    m = st.sidebar.number_input("Restricciones", 1, 10, 3, key="simplex_m")
    variables = [f"X{j + 1}" for j in range(n)]
    holguras = [f"S{i + 1}" for i in range(m)]
    columnas = variables + holguras + ["RHS"]

    st.subheader("1. Definición del modelo")
    st.markdown("#### Función objetivo (maximizar Z)")
    coef_obj = []
    for j, col in enumerate(st.columns(n)):
        with col:
            coef_obj.append(st.number_input(f"Coeficiente {variables[j]}", value=0.0, step=1.0, key=f"simplex_c_{j}"))
    st.markdown("#### Restricciones (≤)")
    matriz, rhs = [], []
    for i in range(m):
        st.write(f"**Restricción {i + 1}**")
        cols = st.columns(n + 1)
        fila = []
        for j in range(n):
            with cols[j]:
                fila.append(st.number_input(f"{variables[j]} (R{i + 1})", value=0.0, step=1.0, key=f"simplex_a_{i}_{j}"))
        with cols[n]:
            rhs.append(st.number_input(f"Lado derecho b{i + 1}", value=0.0, step=1.0, key=f"simplex_b_{i}"))
        matriz.append(fila)

    with st.expander("Ejemplo para probar (dos variables)"):
        st.markdown("**Maximizar** `Z = 3X1 + 5X2`\n\nSujeto a `X1 ≤ 4`, `2X2 ≤ 12`, `3X1 + 2X2 ≤ 18`, `X1, X2 ≥ 0`. El óptimo es `X1 = 2, X2 = 6, Z = 36`. Ingresa estos coeficientes arriba para validarlo.")

    if not st.button("Resolver con Simplex", type="primary", use_container_width=True, key="simplex_solve"):
        return
    if any(v < 0 for v in rhs):
        st.error("El prototipo Simplex solo admite lados derechos no negativos (bᵢ ≥ 0).")
        return

    a = np.asarray(matriz, dtype=float)
    b = np.asarray(rhs, dtype=float)
    c = np.asarray(coef_obj, dtype=float)
    tableau = np.zeros((m + 1, n + m + 1), dtype=float)
    tableau[0, :n] = -c
    tableau[1:, :n] = a
    tableau[1:, n:n + m] = np.eye(m)
    tableau[1:, -1] = b
    base = holguras.copy()
    epsilon = 1e-9
    limite = 100
    terminado = False
    st.subheader("2. Proceso iterativo")

    for iteracion in range(limite + 1):
        df = pd.DataFrame(tableau.copy(), columns=columnas, index=["Z"] + base.copy())
        coef_z = tableau[0, :-1]
        if np.min(coef_z) >= -epsilon:
            st.success(f"Solución óptima encontrada en la iteración {iteracion}.")
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
            terminado = True
            break
        if iteracion == limite:
            st.error(f"Se alcanzó el límite de {limite} iteraciones; revisa si hay degeneración o ciclos.")
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
            break
        col_pivote = int(np.argmin(coef_z))
        razones = [tableau[i, -1] / tableau[i, col_pivote] if tableau[i, col_pivote] > epsilon else np.inf for i in range(1, m + 1)]
        if not np.isfinite(razones).any():
            st.error("El problema es no acotado: no hay variable de salida válida.")
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
            break
        fila_pivote = int(np.argmin(razones)) + 1
        pivote = tableau[fila_pivote, col_pivote]
        entra, sale = columnas[col_pivote], base[fila_pivote - 1]
        with st.expander(f"Iteración {iteracion}: entra {entra}, sale {sale}", expanded=(iteracion == 0)):
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
            st.write(f"**Criterio:** entra {entra} (coeficiente Z más negativo); sale {sale} (cociente mínimo {razones[fila_pivote - 1]:.2f}).")
            st.write(f"**Pivote:** {pivote:.2f}. Se divide la fila pivote por este valor y se elimina {entra} de las otras filas con operaciones de Gauss-Jordan.")
        base[fila_pivote - 1] = entra
        tableau[fila_pivote, :] /= pivote
        for k in range(m + 1):
            if k != fila_pivote:
                tableau[k, :] -= tableau[k, col_pivote] * tableau[fila_pivote, :]
        tableau[np.abs(tableau) < epsilon] = 0.0

    if terminado:
        st.subheader("3. Interpretación de resultados")
        r1, r2 = st.columns(2)
        r1.metric("Valor óptimo Z*", f"{tableau[0, -1]:.2f}")
        with r1:
            st.markdown("#### Variables de decisión")
            for var in variables:
                valor = tableau[base.index(var) + 1, -1] if var in base else 0.0
                st.write(f"**{var}:** {valor:.2f}")
        with r2:
            st.markdown("#### Holguras")
            for var in holguras:
                valor = tableau[base.index(var) + 1, -1] if var in base else 0.0
                st.write(f"**{var}:** {valor:.2f}")


def graphical_page():
    import matplotlib.pyplot as plt
    from scipy.optimize import linprog

    st.title("Método Gráfico")
    st.markdown("Resuelve modelos de dos variables con objetivo de **maximización o minimización**, restricciones `≤` o `≥`, y `X, Y ≥ 0`.")
    st.sidebar.header("Función objetivo")
    sentido = st.sidebar.radio("Optimizar", ("Maximizar", "Minimizar"), key="graf_sense")
    oc1, oc2 = st.sidebar.columns(2)
    with oc1:
        c1 = st.number_input("Coeficiente de X", value=1.0, step=1.0, key="graf_c1")
    with oc2:
        c2 = st.number_input("Coeficiente de Y", value=1.0, step=1.0, key="graf_c2")
    st.sidebar.header("Restricciones")
    m = st.sidebar.number_input("Cantidad de restricciones", 1, 8, 2, key="graf_m")
    restricciones = []
    for i in range(m):
        st.sidebar.subheader(f"Restricción {i + 1}")
        signo = st.sidebar.selectbox("Tipo", ("≤", "≥"), key=f"graf_sign_{i}")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            a1 = st.number_input("X", value=1.0, key=f"graf_a1_{i}")
        with col2:
            a2 = st.number_input("Y", value=1.0, key=f"graf_a2_{i}")
        with col3:
            b = st.number_input("Límite", value=10.0, key=f"graf_b_{i}")
        restricciones.append((a1, a2, b, signo))

    if not st.button("Resolver problema gráfico", type="primary", use_container_width=True, key="graf_solve"):
        return
    if any(abs(a1) < 1e-12 and abs(a2) < 1e-12 and ((signo == "≤" and b < 0) or (signo == "≥" and b > 0)) for a1, a2, b, signo in restricciones):
        st.error("El modelo contiene una restricción imposible con ambos coeficientes iguales a cero.")
        return
    aub, bub = [], []
    for a1, a2, b, signo in restricciones:
        mult = 1 if signo == "≤" else -1
        aub.append([mult * a1, mult * a2])
        bub.append(mult * b)
    c_obj = [-c1, -c2] if sentido == "Maximizar" else [c1, c2]
    res = linprog(c_obj, A_ub=aub, b_ub=bub, bounds=[(0, None), (0, None)], method="highs")
    if res.status == 2:
        st.error("El modelo es infactible: las restricciones se contradicen.")
        return
    if res.status == 3:
        st.error("El modelo es no acotado: la función objetivo puede mejorar sin límite.")
        return
    if not res.success:
        st.error(f"No se pudo resolver el modelo: {res.message}")
        return

    x, y = res.x
    z = -res.fun if sentido == "Maximizar" else res.fun
    st.success("Solución óptima encontrada.")
    r1, r2, r3 = st.columns(3)
    r1.metric("Valor de X", f"{x:.2f}")
    r2.metric("Valor de Y", f"{y:.2f}")
    r3.metric("Valor óptimo Z", f"{z:.2f}")
    st.subheader("Región factible")

    puntos = [(0.0, 0.0)]
    for a1, a2, b, _ in restricciones:
        if abs(a1) > 1e-12:
            puntos.append((b / a1, 0.0))
        if abs(a2) > 1e-12:
            puntos.append((0.0, b / a2))
    for i, (a1, a2, b, _) in enumerate(restricciones):
        for a1b, a2b, bb, _ in restricciones[i + 1:]:
            det = a1 * a2b - a1b * a2
            if abs(det) > 1e-12:
                point = ((b * a2b - bb * a2) / det, (a1 * bb - a1b * b) / det)
                puntos.append(point)

    def es_factible(px, py):
        return px >= -1e-7 and py >= -1e-7 and all((a1 * px + a2 * py <= b + 1e-7) if signo == "≤" else (a1 * px + a2 * py >= b - 1e-7) for a1, a2, b, signo in restricciones)

    vertices = [(px, py) for px, py in puntos if np.isfinite(px) and np.isfinite(py) and es_factible(px, py)]
    xs = [p[0] for p in vertices] + [x]
    ys = [p[1] for p in vertices] + [y]
    # Mostrar también los interceptos aunque estén fuera de la región factible.
    # Así se ven completas las rectas en el primer cuadrante, con 50 % de margen.
    for a1, a2, b, _ in restricciones:
        if abs(a1) > 1e-12 and np.isfinite(b / a1) and b / a1 >= 0:
            xs.append(b / a1)
        if abs(a2) > 1e-12 and np.isfinite(b / a2) and b / a2 >= 0:
            ys.append(b / a2)
    x_max = max(10.0, max(xs, default=0.0) * 1.5)
    y_max = max(10.0, max(ys, default=0.0) * 1.5)
    xv = np.linspace(0, x_max, 300)
    xx, yy = np.meshgrid(xv, np.linspace(0, y_max, 300))
    region = (xx >= 0) & (yy >= 0)
    for a1, a2, b, signo in restricciones:
        region &= (a1 * xx + a2 * yy <= b + 1e-7) if signo == "≤" else (a1 * xx + a2 * yy >= b - 1e-7)

    fig, ax = plt.subplots(figsize=(9, 6))
    colores = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2", "#db2777", "#4b5563"]
    for i, (a1, a2, b, signo) in enumerate(restricciones):
        color = colores[i % len(colores)]
        if abs(a2) > 1e-12:
            ax.plot(xv, (b - a1 * xv) / a2, color=color, label=f"R{i + 1}: {a1:g}X + {a2:g}Y {signo} {b:g}")
        elif abs(a1) > 1e-12:
            ax.axvline(b / a1, color=color, label=f"R{i + 1}: {a1:g}X {signo} {b:g}")
        else:
            ax.plot([], [], color=color, label=f"R{i + 1}: 0 {signo} {b:g}")
    ax.contourf(xx, yy, region.astype(float), levels=[0.5, 1.5], colors=["#a5f3fc"], alpha=0.55)
    ax.plot(x, y, "o", color="#dc2626", markersize=9, label=f"Óptimo ({x:.2f}, {y:.2f})")
    ax.set(xlim=(0, x_max), ylim=(0, y_max), xlabel="X", ylabel="Y", title="Región factible y solución óptima")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Vista amplia: incluye los cruces positivos de las rectas con los ejes y un margen adicional del 50 %. El área celeste representa la región factible dentro del rango visible. Los ejes representan las restricciones de no negatividad.")


st.sidebar.title("Investigación de Operaciones")
seccion = st.sidebar.radio("Selecciona un método", ("Simplex tabular", "Método gráfico"))
if seccion == "Simplex tabular":
    simplex_page()
else:
    graphical_page()

