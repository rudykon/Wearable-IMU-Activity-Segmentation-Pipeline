# Installation

The project targets **Python 3.12 or newer** and packages its source under
`src/imu_activity_pipeline/`. Use an editable install so root entry points,
experiment scripts, notebooks, and direct Python imports all resolve the same
package.

## Requirements

- Python ≥ 3.12
- Conda, or Python `venv` + pip
- Git
- A CUDA-capable environment for normal model training
- JDK 17 + Android SDK only when building the Android app

!!! note

    The public smoke test is intentionally CPU-safe and uses tiny temporary
    files. Full training needs the documented data; inference downloads missing
    public weights as described in [Data & model assets](../reference/assets.md).

## Conda

~~~bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline

conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
~~~

The environment pins the numerical and machine-learning stack used by the
repository, including PyTorch 2.5.1.

## pip

=== "Linux / macOS"

    ~~~bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ~~~

=== "Windows PowerShell"

    ~~~powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ~~~

For a CUDA-specific PyTorch build, choose the wheel appropriate for the host
driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/),
then install the remaining project dependencies. The pip requirements use
PyTorch 2.8.0 for Hugging Face ZeroGPU compatibility; the reproducibility Conda
environment retains the original PyTorch 2.5.1 research stack.

## Verify

~~~bash
python -c "import imu_activity_pipeline; print(imu_activity_pipeline.__version__)"
python tests/smoke_test.py
~~~

Expected package version:

~~~text
0.1.0
~~~

The smoke test checks:

1. package imports and canonical paths;
2. a tiny temporary tab-separated signal stream;
3. annotation parsing; and
4. prediction workbook writing.

## Preview

Documentation dependencies are isolated from the research environment:

~~~bash
python -m pip install -r requirements-docs.txt
mkdocs serve
~~~

Open `http://127.0.0.1:8000/`. A strict production build is:

~~~bash
mkdocs build --strict
~~~

## Troubleshooting

??? question "The package cannot be imported"

    Run `python -m pip install -e .` from the repository root and confirm that
    the active interpreter is the one shown by `python -m pip --version`.

??? question "PyTorch cannot see the GPU"

    Verify the driver and installed PyTorch wheel independently. The project
    requirements pin the framework version, but CUDA wheel selection is
    host-specific.

??? question "The smoke test passes but inference cannot start"

    That usually means the code installation is healthy but required local
    signal files are missing, or the public model download failed. Continue with the
    [quick start](quickstart.md) and [asset map](../reference/assets.md).
