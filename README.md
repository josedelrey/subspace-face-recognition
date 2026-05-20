# subspace-face-recognition

Implementacion de Eigenfaces y Fisherfaces para reconocimiento facial en ORL.

## Ejecucion rapida

```bash
pip install -e .
```

```bash
python scripts/run_fisherfaces.py
```

## Runner de experimentos

El runner lee todos los ficheros `.yaml` y `.yml` de `configs/`, los ejecuta en
orden alfabetico y anade una fila por valor de `d'` al CSV de resultados.

```bash
python scripts/run_experiments.py --config-dir configs --results-csv results/experiments.csv
```

Tambien se puede ejecutar un unico plan YAML con defaults compartidos y una
lista de experimentos:

```bash
python scripts/run_experiments.py --config-file configs/experiment_plan.yaml --overwrite
```

La estructura del plan es:

```yaml
defaults:
  data:
    dataset: orl
    data_dir: data
  outputs:
    results_csv: results/experiment_plan.csv

experiments:
  -
    experiment:
      name: eigenfaces_baseline
    model:
      name: eigenfaces
  -
    experiment:
      name: fisherfaces_baseline
    model:
      name: fisherfaces
```

Usa `--overwrite` para regenerar el CSV desde cero.

Cada configuracion permite parametrizar dataset, particion de imagenes,
preprocesado, modelo, regularizacion numerica, dimensiones evaluadas, proyeccion,
clasificador y salidas.

Opciones utiles para los experimentos:

```yaml
projection:
  drop_first_components: 0
  normalize: false

classifier:
  name: nearest_neighbor
  params:
    k: 1
    distance: euclidean  # euclidean | cosine
```
