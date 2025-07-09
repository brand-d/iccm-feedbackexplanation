ICCM Feedback Explanation
=========================

Companion repository for the 2021 article "Model-Based Explanation of Feedback Effects in Syllogistic Reasoning" published in the proceedings of the 19th International Conference on Cognitive Modeling (ICCM)

### Overview

- `analysis`: Contains scripts for replicating the analysis.
- `analysis/benchmarks/`: Contains the CCOBRA benchmarks and the resulting parameters.
- `analysis/benchmarks/parameters`: Contains parameters from the benchmark runs.
- `analysis/benchmarks/parameters/brand2021`: Contains JSON files with the parameter results for the Brand2021 dataset.
- `analysis/benchmarks/parameters/brand2021/brand_control.json`: Parameters for the control group.
- `analysis/benchmarks/parameters/brand2021/brand_feedback.json`: Parameters for the feedback group.
- `analysis/benchmarks/parameters/dames2020`: Contains JSON files with the parameter results for the Dames2020 dataset.
- `analysis/benchmarks/parameters/dames2020/dames_control.json`: Parameters for the control group.
- `analysis/benchmarks/parameters/dames2020/dames_feedback.json`: Parameters for the feedback group.
- `analysis/benchmarks/brand_control.json`: CCOBRA benchmark for the control group of the Brand2021 dataset.
- `analysis/benchmarks/brand_feedback.json`: CCOBRA benchmark for the feedback group of the Brand2021 dataset.
- `analysis/benchmarks/dames_control.json`: CCOBRA benchmark for the control group of the Dames2020 dataset.
- `analysis/benchmarks/dames_feedback.json`: CCOBRA benchmark for the feedback group of the Dames2020 dataset.
- `analysis/plot_param_distribution.py`: Plots the parameter distributions from the paper.
- `analysis/statistics.py`: Calculates descriptive statistics and tests the hypotheses.
- `data`: Contains the datasets in CCOBRA format.
- `data/Brand2021`: Contains the Brand2021 dataset.
- `data/Brand2021/Brand2021_FeedbackEffect.csv`: Brand2021 dataset for the feedback effect in syllogistic reasoning.
- `data/Brand2021/Brand2021_FeedbackEffect_Control.csv`: Brand2021 dataset filtered to only include the control group.
- `data/Brand2021/Brand2021_FeedbackEffect_Feedback.csv`: Brand2021 dataset filtered to only include the feedback group.
- `data/Dames2020`: Contains the Dames2020 dataset.
- `data/Dames2020/data_1s.csv`: Data containing results for the 1-second feedback condition.
- `data/Dames2020/data_10s.csv`: Data containing results for the 10-second feedback condition.
- `data/Dames2020/data_control.csv`: Data containing results for the control group.
- `data/experiment`: Contains the webexperiment and the source code.
- `data/experiment/page`: Web experiment.
- `data/experiment/source`: Source code for the web experiment.
- `models`: Contains the models used for the analysis.
- `models/mreasoner`: Contains the [ccobra implementation](https://github.com/nriesterer/pymreasoner) of mReasoner.
- `models/phm`: Contains the [ccobra implementation](https://github.com/nriesterer/phm) of the Probability Heuristics Model (PHM).
- `models/transset`: Contains the [ccobra implementation](https://github.com/Shadownox/iccm-transset-indiv) of the TransSet.

### Dependencies

- Python 3
    - [CCOBRA](https://github.com/CognitiveComputationLab/ccobra)
    - [pandas](https://pandas.pydata.org)
    - [numpy](https://numpy.org)
    - [seaborn](https://seaborn.pydata.org)
	- [scipy](https://www.scipy.org)
	- [statsmodels](https://www.statsmodels.org)

### Run the benchmarks and extract parameter information

After installing CCOBRA, run the following command to execute the benchmark (e.g., for the control group of the Brand2021 dataset):

```
cd /path/to/repository/analysis/benchmarks/
$> ccobra brand_control.json
```

An HTML-file will be created in the same folder. When opening the file, the predictive performance of the models is shown. At the bottom of the page, the parameter fits can be downloaded (in the category `Model Logs`, click on `Save Full Log`).

### Run the scripts

To generate the plots, execute the script with the following command:

```
cd /path/to/repository/analysis/
$> python plot_param_distribution.py
```

The plots will be generated as PDF-files in the same folder.
In the same manner, you can run the statistics script:

```
cd /path/to/repository/analysis/
$> python statistics.py
```

### Run the web experiment

The web experiment needs a webserver environment with PHP to run. On a local system, you need [XAMPP](https://www.apachefriends.org) or a similar solution installed. When using XAMPP, copy the folder `data/experiment/page` to the `htdocs` folder of your XAMPP installation. The experiment can then be tested by navigating to the following URL in your browser:

```
http://localhost/page/
```

The experiment will save the data locally in the `storage` subfolder.

### Open the source code of the web experiment

The webexperiment was created using [Construct3](https://www.construct.net), which is a browser-based authoring tool to generate HTML5/Javascript applications and games.

The webexperiment can be opened using the editor at https://editor.construct.net/ by choosing the `Open local project folder` option and then loading the `source` folder.

### References

Brand, D., Riesterer, N., & Ragni, M. (2021). Model-Based Explanation of Feedback Effects in Syllogistic Reasoning. In Proceedings of the 19th International Conference on Cognitive Modeling.

Dames, H., Schiebel, C., & Ragni, M. (2020). The role of feedback and post-error adaptations in reasoning. In S. Denison, M. Mack, Y. Xu, & B. C. Armstrong (Eds.), Proceedings of the 42nd annual conference of the cognitive science society (pp. 3275–3281). Cognitive Science Society.