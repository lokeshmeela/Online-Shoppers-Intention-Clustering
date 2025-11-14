Online Shoppers Intention Clustering
Sai Lokesh Meela

------------------------------------------------------------
README - Instructions to Run the Python Program
------------------------------------------------------------

1. Requirements
---------------
Make sure you have Python 3 installed. You also need the following Python packages:

- pandas
- numpy
- scikit-learn

You can install them using pip:

    pip install pandas numpy scikit-learn


2. Files Needed
---------------
Ensure the following files are in the same directory:

- Untitled-1.py                  → The main Python program
- online_shoppers_intention.csv → The dataset file
- README.txt                     → This instructions file

Note: The script reads directly from 'online_shoppers_intention.csv', so do not rename it.


3. How to Run the Program
--------------------------
To execute the clustering analysis, run the following command in the terminal or command prompt:

    python Untitled-1.py

The script will:
- Load and preprocess the dataset
- Apply K-means and Agglomerative clustering (k=4)
- Calculate Rand Index for both models
- Print cluster sizes and means
- Save the output summary in a file called:

    clustering_report.txt


4. Output
---------
After running, the script will generate:

- **clustering_report.txt**: Contains:
  - Rand Index for both models
  - Cluster size distribution
  - Which model performed better


5. Notes
--------
- The 'Revenue' column is used only for evaluation (not during clustering).
- 'Month' and 'VisitorType' are mean-encoded based on Revenue.
- Features are standardized using `StandardScaler`.
