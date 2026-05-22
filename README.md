# Subspace Face Recognition

Classical face recognition study on the ORL face database using direct pixel
matching, Eigenfaces, and Fisherfaces. The project implements the complete
experimental pipeline in Python/NumPy: PGM loading, illumination
preprocessing, subspace projection, nearest-neighbor classification, metric
evaluation, result summarization, and error-curve generation.

The goal is to compare how much discriminative performance can be retained
after projecting 92 x 112 grayscale face images into low-dimensional subspaces.

## Highlights

- Implements three representations: full-image direct matching, PCA
  Eigenfaces, and PCA+LDA Fisherfaces.
- Evaluates global normalization, histogram equalization, local normalization,
  clipped local histogram equalization, cosine distance, Euclidean distance,
  feature L2 normalization, different `k` values, and dropped leading
  components.
- Uses a reproducible YAML experiment plan with one CSV row per tested
  projection dimensionality.
- Reaches **91.5% accuracy** on the held-out ORL split with Eigenfaces using
  only **55 coefficients** instead of the original **10,304 pixels**.

## Dataset

The experiments use the ORL face database: 40 subjects, 10 grayscale PGM images
per subject, and 92 x 112 pixels per image. The default protocol trains on
images 1-5 for each subject and tests on images 6-10, producing:

| Split | Images per subject | Total images |
| --- | ---: | ---: |
| Train | 5 | 200 |
| Test | 5 | 200 |

The dataset is expected under `data/` as `data/s1`, `data/s2`, ..., `data/s40`.
The `data/` directory is intentionally ignored by git.

Please cite the original ORL/AT&T face database when using these images:
F. Samaria and A. Harter, "Parameterisation of a stochastic model for human
face identification", 2nd IEEE Workshop on Applications of Computer Vision,
1994.

## Methods

**Direct representation** uses the flattened image vector directly and serves
as the high-dimensional nearest-neighbor baseline.

**Eigenfaces** performs PCA in the sample covariance dual space, normalizes the
resulting eigenvectors, and classifies projected samples with nearest
neighbors. This gives a compact unsupervised subspace that preserves dominant
variance in the training faces.

**Fisherfaces** first applies PCA to avoid singular scatter matrices, then
learns an LDA subspace with regularized within-class scatter. With 40 subjects,
the Fisherface subspace is bounded by at most 39 discriminant directions.

All reported accuracies use nearest-neighbor classification on the ORL test
split. For subspace models, the runner evaluates every valid projection size
`d'` unless a configuration overrides the component grid.

## Results

The full experiment sweep contains **6,663 evaluated configurations** across
**59 named experiments**. The best result per experiment is stored in
`results/summary.csv`; the full per-dimensionality table is stored in
`results/full_results.csv`.

Top configurations from the current run:

| Rank | Representation | Preprocessing | Classifier | Best `d'` | Accuracy | Test errors |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | Eigenfaces | none | cosine 1-NN, L2 features | 55 | **91.5%** | 17 / 200 |
| 2 | Fisherfaces | none | cosine 1-NN, L2 features | 34 | **91.0%** | 18 / 200 |
| 3 | Eigenfaces | none | Euclidean 1-NN | 78 | **90.5%** | 19 / 200 |
| 4 | Direct pixels | none | Euclidean 1-NN | 10,304 | **90.0%** | 20 / 200 |
| 5 | Eigenfaces | clipped global histogram equalization | Euclidean 1-NN | 86 | **90.0%** | 20 / 200 |

Best result by model family:

| Model | Best experiment | Best `d'` | Accuracy |
| --- | --- | ---: | ---: |
| Direct pixels | `direct_raw_euclidean_k1` | 10,304 | 90.0% |
| Eigenfaces | `eigenfaces_raw_cosine_l2_k1` | 55 | 91.5% |
| Fisherfaces | `fisherfaces_raw_cosine_l2_k1` | 34 | 91.0% |

Key observations:

- Eigenfaces delivered the strongest overall result while reducing the feature
  dimension by roughly 187x relative to raw pixels.
- Fisherfaces was nearly tied with Eigenfaces and reached 91.0% accuracy using
  only 34 dimensions, consistent with the compactness expected from LDA.
- Cosine distance with L2-normalized projected features improved the strongest
  Eigenfaces and Fisherfaces runs.
- Raw grayscale inputs outperformed most local illumination preprocessing in
  this fixed ORL split. Local normalization and small-window local histogram
  equalization were often harmful, suggesting that these transformations can
  distort identity-bearing texture when the original acquisition conditions
  are already controlled.
- Increasing `k` from 1 to 3 or 5 did not improve the best Eigenfaces baseline;
  for this split, nearest-neighbor identity matching is strongest with `k=1`.

These results should be interpreted as a controlled classical-computer-vision
benchmark rather than a production face recognition system. ORL is small,
frontal, and relatively constrained; a modern deployment would require larger
datasets, cross-validation or subject-disjoint protocols appropriate to the
task, fairness and privacy analysis, and robustness testing.

## Reproducing the Experiments

Install the project into the `cv` conda environment:

```bash
conda run -n cv python -m pip install -e .
```

Run the full YAML experiment plan:

```bash
conda run -n cv python scripts/run_experiments.py configs/experiment_plan.yaml --overwrite --quiet
```

Summarize the best projection dimensionality for each named experiment:

```bash
conda run -n cv python scripts/summarize_results.py results/full_results.csv --output-csv results/summary.csv
```

Generated artifacts are intentionally excluded from version control:

- `results/full_results.csv`: one row per experiment and tested `d'`.
- `results/summary.csv`: best row per experiment.
- `outputs/figures/full_plan/*.png`: error curves for the experiment plan.

## Experiment Configuration

Experiments are defined in `configs/experiment_plan.yaml`. A minimal entry
looks like this:

```yaml
-
  experiment:
    name: eigenfaces_raw_euclidean_k1
  model:
    name: eigenfaces
    params:
      eps: 1.0e-10
  classifier:
    name: nearest_neighbor
    params:
      k: 1
      distance: euclidean
```

The runner supports configurable dataset splits, preprocessing, model
parameters, projection options, classifier options, and output paths.

Available preprocessing modes:

- `global_normalization`
- `global_histogram_equalization`
- `local_normalization`
- `local_histogram_equalization`

Available model families:

- `direct`
- `eigenfaces`
- `fisherfaces`

## Repository Structure

```text
configs/                         YAML experiment plans
scripts/run_experiments.py        Experiment runner
scripts/summarize_results.py      Result summarizer
scripts/visualize_preprocessing.py
src/subspace_face_recognition/    Library code
  data.py                         ORL PGM loader
  preprocessing.py                Normalization and histogram equalization
  models/direct.py                Direct pixel representation
  models/eigenfaces.py            PCA Eigenfaces
  models/fisherfaces.py           PCA + LDA Fisherfaces
  evaluation.py                   Projection and classification loop
  knn.py                          Nearest-neighbor classifier
results/                          Generated CSV results, ignored by git
outputs/                          Generated figures, ignored by git
```

## License

This project is released under the MIT License. The ORL dataset has its own
citation and usage expectations; keep the dataset separate from the repository
unless redistribution is explicitly permitted.
