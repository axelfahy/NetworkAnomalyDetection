# NetworkAnomalyDetection

Anomaly Detection in Network Traffic using different clustering algorithm.

Data must be located in the *data* folder. Due to the size of the dataset, it has been ignored.

The full dataset is available at <https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data>

Program can be run using Intellij or with sbt and spark-submit:

```$ sbt package```

```$ spark-submit --class "NetworkAnomalyDetection" --driver-memory 6g target/scala-2.11/networkanomalydetection_2.11-0.1.jar```

This project has been developped using DataFrames from Spark MLlib.


## Data

The datasets used for the training phase contain 4,898,431 packets, and its size is of 743 MB. Here is a preview of the data:

```0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.```

The data come from a competition held in 1999 by KDD Cup organization (i.e. an organisation bringing together professional data miners). The competition was about creating a model capable to detect anomalies in network traffic. The packets available were gathered from an US military network.

## Preprocessing

- **Numerical features**: at first, only numerical features were used as features.

- **Categorical features**: then, categorical features were encoded using the One-hot encoder.

- **Normalization**: finally, normalization of vector features using standard deviation

## Algorithms

- **K-means**: the center of a cluster is named a centroid. At the first iteration, K centroids are chosen randomly. Then, at each iteration, data points are affected to their closest centroid and the centroid becomes the mean of the points from this cluster.

- **Bisecting K-means**: same approach as K-means. However, at first there is only one cluster that contains all data points. Then, at each iteration, cluster are divided using K-means.

- **Gaussian Mixture (GMM)**: the goal is to maximise the log-likelihood. The algorithm will iterate until a threshold is reached or a maximum of iterations. The algorithm will converge, but not necessarly in a global minimum.

## Evaluation

To evaluate the models with the same technique, an evaluation function was created. The evaluation is the Euclidean squared distance of each centroids.

