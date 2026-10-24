# Investigación de Operaciones — Método Simplex

El archivo `app.py` en la raíz conserva el primer avance del proyecto.

## Versión entregable

La carpeta `entregable/` incorpora la aplicación unificada de Simplex tabular y método gráfico. No contiene la versión experimental.

Desde una terminal en `entregable/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Una vez instalado el entorno, también se puede ejecutar `iniciar_app.bat` en Windows. Streamlit muestra la dirección local en la terminal.

### Alcance

- Simplex: maximización, restricciones de tipo menor o igual, lados derechos no negativos y variables no negativas; tablas, pivotes y holguras.
- Método gráfico: dos variables no negativas, maximización o minimización y restricciones de tipo menor o igual/mayor o igual.
- Las gráficas incluyen los interceptos positivos con los ejes y un 50 % de margen adicional.

Ejemplo comprobado: maximizar `Z = 3x + 2y`, sujeto a `2x + y <= 18`, `2x + 3y <= 42`, `3x + y <= 24`, `x,y >= 0`. Resultado: `x = 3`, `y = 12`, `Z = 33`.

No se incluyen entornos virtuales, ejecutables de túneles, credenciales ni configuraciones personales.
