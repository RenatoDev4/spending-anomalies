spending-anomalies
==============================

This project aims to develop a system to identify anomalies in the spending data of Brazilian deputies through the application of Machine Learning techniques, with a special focus on neural networks (AutoEncoder)

The dataset used in this study contains detailed information on expenses incurred by deputies, including amounts, expense categories and dates. However, this data can often contain inconsistent or discrepant information, such as posting errors, standard deviations or even fraud.

*This project was created for student purposes only.



<!DOCTYPE html>
<html>
<body>
    <h1>Project Organization</h1>
    <p>This GitHub repository is structured to provide a clear and organized approach to managing a data science / machine learning project. Below is a description of the directory tree:</p>
    <ul>
        <li><strong>data</strong>: This directory contains different subdirectories to manage data at various stages of processing:</li>
        <ul>
            <li><strong>external</strong>: Data from third-party sources is stored here.</li>
            <li><strong>interim</strong>: Intermediate data that has been transformed during the data preparation phase is kept here.</li>
            <li><strong>processed</strong>: The final, canonical data sets that are ready for modeling and analysis are stored in this directory. For example, the file <em>cota-parlamentar.csv</em> is one of the processed datasets.</li>
            <li><strong>raw</strong>: The original, immutable data dump is placed here.</li>
        </ul>
        <li><strong>notebooks</strong>: This directory is meant for storing Jupyter notebooks related to the project. The naming convention used for the notebooks is a combination of a number for ordering, the creator's initials, and a short description. For instance, a notebook named <em>ML.ipynb</em> would be an example.</li>
        <li><strong>src</strong>: This directory contains the source code for the project, organized into different subdirectories:</li>
        <ul>
            <li><strong>data</strong>: Scripts to download or generate data are placed in this directory. For example, <em>make_dataset.py</em> is a script that might help download or generate data.</li>
            <li>To create charts based on the dataset information, use the command: '''python src/data'''</li>
            <li><strong>models</strong>: This directory contains scripts for training models and using trained models</li>
            <li>To train the neural network and create graphs to present the anomalous data use the command: '''python src/models'''</li>
        </ul>
        <li><strong>Other files</strong>:</li>
        <ul>
            <li><strong>LICENSE</strong>: The license under which the project is shared with the community.</li>
            <li><strong>README.md</strong>: The top-level README file that provides essential information for developers using this project.</li>
            <li><strong>requirements.txt</strong>: A file that lists all the required Python packages and their versions for reproducing the analysis environment.</li>
        </ul>
    </ul>
    <p>This organized structure allows for easy navigation. It separates data, source code, making it easier to manage and understand the project's components.</p>
</body>
</html>