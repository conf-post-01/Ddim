## Simulation codes for detecting change symptom with Descriptive Dimension
This repository has simulation codes for detecting symptom with Descriptive Dimension (Ddim).

### How to use
1. Create file of normalization term of NML.
    - Create conf file:
    ```
    ./calcNML/conf/calcGaussianNML.conf
    ```
    - Run codes for calculating normalization term as below:
    ```sh
    python ./calcNML/calc_Gaussian_NormalizationNML.py
    ```

1. Simulate for detecting change symptom.
    - Move file of normalization term of NML created above to this directory:
    ```
    ./ddim_change_symptom/data/normfile
    ```
    - Create conf file:
    ```
    # parameter sets for single change
    ./ddim_change_symptom/conf/ddim_paramset_single.json

    # parameter sets for multi change
    ./ddim_change_symptom/conf/ddim_paramset_multi.json
    ```

    - Run the simulation codes on Jupyter Notebook.
    ```
    ./ddim_change_symptom/run_ddim_simulation.ipynb
    ```

### Acknowledgements
The real market data is provided by a data provider.

