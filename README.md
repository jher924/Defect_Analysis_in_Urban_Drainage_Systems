# Sewer Defects Analysis

This repository contains the code developed for the **Distribution and Properties of Defects in Urban Drainage Systems: An analysis of Auckland's Sewer Network** paper by Juana Herrán, María A. González, Jakobus E. van Zyl, and Theunis F. P. Henning.

This repository uses sewer pipe and inspection data to perform a descriptive analysis of defects and their properties. The code allows users to:

- Describe a sewer network and compare it with the subset of pipes inspected via CCTV, assessing representativeness.
- Analyze and visualize properties of defects observed during inspections, including:
  - Average number of defects per km and per pipe
  - Defect type
  - Defect size
  - Longitudinal distance
  - Extent
  - Circumferential position

The repository contains the code used to generate the tables and figures presented in the referenced paper. Although the paper presents results for the Auckland, New Zealand sewer network, the repository can be used with data from any sewer system.

---
## Code Structure

The codebase consists of six Python files and one Jupyter Notebook, where the full analysis workflow is executed. The core logic is implemented in the Python scripts, while the Jupyter Notebook is used to orchestrate their execution and visualize results. Below is a general description of each file. Detailed documentation of individual functions can be found within the corresponding source files.

#### Run_defect_description.ipynb
This Jupyter Notebook serves as the main entry point of the project. It orchestrates the execution of all scripts and displays the tables and figures generated during the analysis. The notebook does not contain core processing logic; instead, it focuses on configuration, execution flow, and result visualization. It also manages the selection of materials (e.g., AC, CONC, VC, PVC, PE) and factors (e.g., age, length, slope, etc.) to be included in the analysis.
#### config.py
Defines the global variables that are accessed and used by the other files.
#### load_excel.py
Contains functions for loading and importing input data from an Excel file.
#### data_preparation.py
Prepares the data for analysis. It performs the required data merges and includes functions to select and validate the chosen materials and factors. In addition, it defines color maps for materials and defect types.
#### dataset_description.py
Provides the functions for a detailed description of the dataset, summarizing key properties and characteristics of the sewer network and inspections. It also analyzes the representativeness of the pipes inspected by CCTV.
#### defect_correlation.py
Contains the function used to calculate pearson correlations between defect types.
#### defect_general_description.py
Includes functions that generate a general description of the network, such as the number of defects per kilometer and per pipe, as well as the distribution of defect types by material.
#### defect_properties_description.py
Provides functions to analyze the properties of observed defects, including size, longitudinal distance, extent, and clock reference position.

---
## Input data

To run the code with your own data, prepare an input file containing four sheets:

1. **PIPES** – Description of the pipes in the network.
2. **CCTV** – Data related to pipe inspections.
3. **DEFECTS** – Details of observed defects.
4. **HYDRAULIC PROPERTIES (optional)** - Information on the hydraulic characteristics of the pipes, such as flow rate and velocity. This sheet should be included only if hydraulic properties are required as part of the network description; otherwise, it can be omitted.

Below is a description of the required columns for each sheet.

<table style="text-align:center;">
  <tr>
    <th>Sheet</th>
    <th>Required Column</th>
    <th>Description of the Column</th>
  </tr>

  <!-- PIPES -->
  <tr>
    <td rowspan="3" style="vertical-align:middle;">PIPES</td>
    <td>Pipe_ID</td>
    <td>Unique identifier for each pipe in the network.</td>
  </tr>
  <tr>
    <td>Material</td>
    <td>Pipe material.</td>
  </tr>
  <tr>
    <td>Factors (multiple columns) </td>
    <td>Pipe characteristics to be included in the network description. Examples include installation year, diameter, length, depth, among others. Each attribute should be provided in a separate column.</td>
  </tr>
  

  <!-- Separator -->
  <tr><td colspan="3"><hr></td></tr>

  <!-- CCTV -->
  <tr>
    <td rowspan="3" style="vertical-align:middle;">CCTV</td>
    <td>Pipe_ID</td>
    <td>Unique identifier linking each CCTV inspection to the corresponding pipe.</td>
  </tr>
  <tr>
    <td>Inspection_direction</td>
    <td>Inspection direction, usually indicating whether the survey was carried out upstream or downstream.</td>
  </tr>
  <tr>
    <td>Survey_length</td>
    <td>Length of the pipe surveyed during the CCTV inspection (m).</td>
  </tr>

  <!-- Separator -->
  <tr><td colspan="3"><hr></td></tr>

  <!-- DEFECTS -->
  <tr>
    <td rowspan="9" style="vertical-align:middle;">DEFECTS</td>
    <td>Defect_ID</td>
    <td>Unique identifier for each defect.</td>
  </tr>
  <tr>
    <td>Pipe_ID</td>
    <td>Unique identifier for each pipe in the network.</td>
  </tr>
  <tr>
    <td>Defect_code</td>
    <td>Type of defect.</td>
  </tr>
  <tr>
    <td>Quantification</td>
    <td>Size of the defect. Must be classified as S, M, or L. If the size is in a different format, it is recommended to adjust it to these categories in order to run the size plot.</td>
  </tr>
  <tr>
    <td>Circumferential_start</td>
    <td>Circumferential position where the defect begins. This is given by a clock reference (integer 1–12). If 0, the defect has no circumferential extent.</td>
  </tr>
  <tr>
    <td>Circumferential_end</td>
    <td>Circumferential position where the defect ends. This is given by a clock reference (integer 1–12). If 0, the defect has no circumferential extent.</td>
  </tr>
  <tr>
    <td>Longitudinal_distance_normalized</td>
    <td>Normalized longitudinal distance of the defect (defect position along the pipe divided by the pipe length). It represents the starting point on extend defects</td>
  </tr>
  <tr>
    <td>Defect_length</td>
    <td>Normalized defect length (distance from the start of the defect to its end, relative to pipe length).</td>
  </tr>
<!-- Separator -->
  <tr><td colspan="3"><hr></td></tr>

  <!-- HYDRAULIC PROPERTIES -->
  <tr>
    <td rowspan="2" style="vertical-align:middle;">HYDRAULIC PROPERTIES</td>
    <td>Pipe_ID</td>
    <td>Unique identifier linking each property value to the corresponding pipe.</td>
  </tr>
  <tr>
    <td>Hydraulic properties (multiple columns)</td>
    <td>These are the hydraulic properties associated with each pipe. Examples include velocity, flow rate, and pipe capacity. Each property should be provided in a separate column.</td>
  </tr>
<table>

#### Data Assumptions and Notes
- Pipe identifiers must be consistent across the PIPES, CCTV, and DEFECTS sheets.
- Defect sizes (Quantification) are expected to be categorized as S, M, or L.
- Clock reference positions must be integers between 1 and 12.
- Normalized distances and lengths must be provided in the range [0, 1].

---
## Data validation
The following repository provides a tool for validating the data used in this analysis. It generates a report containing errors and warnings for pipes, CCTV inspections, defects, and hydraulic properties, helping identify data that may be unreliable and require further review.

https://github.com/jher924/Defect_data_validation.git

The use of the data validation repository is not mandatory to run this repository. However, it is a useful tool for validating the input data prior to performing the analysis.

---
## Installation and Setup
Before using the code, ensure you have the following packages installed:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical plotting
- `openpyxl`: Excel file support

You can install all required packages using pip:

```python
pip install numpy pandas matplotlib seaborn openpyxl
```

---
## How to run the code
To execute the analysis, open the notebook `Run_defect_description.ipynb.`

In the _Load Data_ section of the notebook, update the following line with the correct path to the input Excel file and adjust the sheet names if needed. The input file must contain separate sheets for pipes, inspections, and defects.

If hydraulic model data are available, include an additional sheet for hydraulic properties and list it in sheet_names. If no hydraulic data are available, this sheet can be omitted and its name removed from the list.

```python
df_information = load_multiple_sheets(
    r"..\2.Data_validation\Validation_rules\Validated_data.xlsx",
    sheet_names=["PIPES", "CCTV", "DEFECTS", "HYDRAULIC_PROPERTIES"]
)
```

---
## Outputs
The code generates:

- A dataset description, including summary tables and a comparison between all pipes in the network and those inspected via CCTV.

- An analysis of the average number of defects per kilometer and per pipe for the analyzed pipe materials.

- The distribution of defect types by material.

- Plots illustrating the distribution of defect properties by material.

---
## Citation

If you use this repository in your research, please cite the corresponding paper:

Herrán, J., González, M. A., van Zyl, J. E., & Henning, T. F. P. (2026). 
Distribution and Properties of Defects in Urban Drainage Systems: An analysis of Auckland's Sewer Network. _Journal of Water Resources Planning and Management_.

---
## License

This project is distributed under the MIT License.
See the `LICENSE` file for the full text.

---
## Contact

For questions, feedback, or collaboration inquiries related to the paper or this repository, please contact the corresponding author:

**Juana Herrán**  
Email: _jher924@aucklanduni.ac.nz_  
Affiliation: University of Auckland










