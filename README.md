# NetworkAnomalyDetection

Anomaly Detection in Network Traffic using different clustering algorithm.

Data must be located in the *data* folder. Due to the size of the dataset, it has been ignored.

The full dataset is available at <https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data>

Program can be run using Intellij or with sbt and spark-submit: 

```$ sbt package```

```$ spark-submit --class "NetworkAnomalyDetection" --driver-memory 6g target/scala-2.11/networkanomalydetection_2.11-0.1.jar```

