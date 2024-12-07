# Client_Segmentation_Dashboard guide installation

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- Optional [Mamba](https://mamba.readthedocs.io/en/latest/)

## Installing Miniconda
### Windows

These three commands quickly and quietly download the latest 64-bit Windows installer, rename it to a shorter file name, silently install, and then delete the installer:

``` bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" .\miniconda.exe /S
del miniconda.exe
```
After installing, open the Anaconda Prompt (miniconda3) program to use Miniconda3.

### Linux

These four commands download the latest 64-bit version of the Linux installer, rename it to a shorter file name, silently install, and then delete the installer:

``` bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
After installing, add the following line to your .bashrc to initialize conda automatically:

``` bash
export PATH=~/miniconda3/bin:$PATH
```
Then run:
``` bash
source ~/.bashrc
```

Now you can use Miniconda3 on your Linux system.

## Create environment

```bash
conda env create -f environment.yml
conda activate customer_segmentation
```

or 

```bash
mamba env create -f environment.yml
mamba activate customer_segmentation
```
or 

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

The packages necessary to run the project are now installed inside the conda environment.

**Note: The following sections assume you are located in your environment.**

## Set up project's module

To move beyond notebook prototyping, all reusable code should go into the `customer_segmentation/` folder package. To use that package inside your project, install the project's module in editable mode, so you can edit files in the `customer_segmentation` folder and use the modules inside your notebooks :

### Step 1: Install the module in editable mode

To install the project module in an editable mode (which means you can edit the code and see the changes reflected immediately without needing to reinstall), follow these steps:

1. Open the **Anaconda Prompt** (or terminal if using Linux/macOS) and navigate to the root folder of your project (where the `setup.py` file is located).
2. Run the following command:

```bash
   pip install --editable .
```
This will install the project module in "editable" mode so that any changes you make to the code in the customer_segmentation/ folder will be reflected immediately without needing to reinstall.

### Step 2: Use the module in your Jupyter Notebooks

To use the module inside your notebooks, add `%autoreload` at the top of your notebook :

```python
%load_ext autoreload
%autoreload 2
```
This will automatically reload the modules in your project whenever changes are made, ensuring you are working with the most recent version of your code.

Example of using the module:

```python
from customer_segmentation.utils.paths import data_dir
data_dir()
```
## Working with Paths in the Project

In this project, paths are handled dynamically to easily access important directories like `data`, `models`, or `notebooks`. Instead of hardcoding paths, the project uses a utility function to generate paths relative to the project’s root directory.

### Predefined Paths

Several key paths are predefined in the `utils/paths.py` file, such as:

- `data_dir`: Root directory for all data.
- `data_raw_dir`: Directory for raw data.
- `data_processed_dir`: Directory for processed data.
- `models_dir`: Directory for saved models.
- `notebooks_dir`: Directory for notebooks.

These paths are available for direct use throughout your code.

### Example Usage

To use a path in your script or notebook, you can import and call the respective function:

```python
from customer_segmentation.utils.paths import data_raw_dir

# Access the raw data directory
raw_data_path = data_raw_dir()
print(raw_data_path)
```
If your project is located at /home/user/project, this would print the following path:

```python
/home/user/project/data/raw
```

## Update Environment Files After Installing New Dependencies

```bash
conda install <package_name>
conda env export --no-builds | grep -v "^prefix: " > environment.yml
```

or 

```bash
mamba install <package_name>
mamba env export --no-builds | grep -v "^prefix: " > environment.yml
```
or 

```bash
pip install <package_name>
pip list --format=freeze > requirements.txt
```