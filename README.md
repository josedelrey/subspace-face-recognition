# subspace-face-recognition

Implementacion de representacion directa, Eigenfaces y Fisherfaces para
reconocimiento facial en ORL.

## Ejecucion rapida

```bash
pip install -e .
```

## Runner de experimentos

El runner recibe un plan YAML con `defaults` compartidos y una lista de
experimentos. Los ejecuta secuencialmente y anade una fila por valor de `d'` al
CSV de resultados.

```bash
python scripts/run_experiments.py configs/experiment_plan.yaml --overwrite
```

Si no se indica ruta, usa `configs/experiment_plan.yaml`:

```bash
python scripts/run_experiments.py --overwrite
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
preprocesado, modelo, regularizacion numerica, dimensiones evaluadas,
proyeccion, clasificador y salidas. Los modelos disponibles son `direct`,
`eigenfaces` y `fisherfaces`.

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

model:
  name: direct  # direct | eigenfaces | fisherfaces
  params:
    center: false
```
