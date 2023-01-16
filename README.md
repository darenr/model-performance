# model-performance

Example showing how to use datapane/evidently to generate a model report in html

## Development

```bash
conda config --add channels conda-forge
conda update -n base -c defaults conda
conda env create --file environment.yaml # or use mamba for better performance
conda activate metrics
```

## Run

```bash
python make_report.py
open report.html
```