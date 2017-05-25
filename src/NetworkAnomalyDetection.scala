/**
 * Anomaly Detection in Network Traffic with different clustering algorithm.
 *
 * The implementation is done using the Dataframe-based API of SparkMLlib.
 *
 * Algorithms:
 *
 *  - K-means
 *  - Gaussian Mixture Model (GMM)
 *  - Latent Dirichlet allocation (LDA)
 *
 * Categorical features are transformed into numerical features using one-hot encoder.
 * Afterwards, all features are normalized.
 *
 * These different implementation are compared.
 *
 * Metrics used:
 *
 *  - K-means: Within Set Sum of Squared Errors (WSSSE)
 *  - GMM: Mean, covariance and weights
 *
 * Basic implementation is based on the chapter 5 (Anomaly Detection in Network Traffic with K-means clustering)
 * of the book Advanced Analytics with Spark.
 * However, this implementation is using the Dataframe-based API instead of the RDD-based API.
 *
 * @author Axel Fahy
 * @author Rudolf Höhn
 * @author Brian Nydegger
 * @author Assaf Mahmoud
 *
 * @date 18.05.2017
 *
 */

import java.io.{File, PrintWriter}
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.spark.ml.{Pipeline}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vector}


object NetworkAnomalyDetection {

  val DataPath = "data/kddcup.data.corrected"

  // Schema of data from csv file
  // Used when loading the data to have a correct structure
  val DataSchema = StructType(Array(
    StructField("duration", IntegerType, true),
    StructField("protocol_type", StringType, true),
    StructField("service", StringType, true),
    StructField("flag", StringType, true),
    StructField("src_bytes", IntegerType, true),
    StructField("dst_bytes", IntegerType, true),
    StructField("land", IntegerType, true),
    StructField("wrong_fragment", IntegerType, true),
    StructField("urgent", IntegerType, true),
    StructField("hot", IntegerType, true),
    StructField("num_failed_logins", IntegerType, true),
    StructField("logged_in", IntegerType, true),
    StructField("num_compromised", IntegerType, true),
    StructField("root_shell", IntegerType, true),
    StructField("su_attempted", IntegerType, true),
    StructField("num_root", IntegerType, true),
    StructField("num_file_creations", IntegerType, true),
    StructField("num_shells", IntegerType, true),
    StructField("num_access_files", IntegerType, true),
    StructField("num_outbound_cmds", IntegerType, true),
    StructField("is_host_login", IntegerType, true),
    StructField("is_guest_login", IntegerType, true),
    StructField("count", IntegerType, true),
    StructField("srv_count", IntegerType, true),
    StructField("serror_rate", DoubleType, true),
    StructField("srv_serror_rate", DoubleType, true),
    StructField("rerror_rate", DoubleType, true),
    StructField("srv_rerror_rate", DoubleType, true),
    StructField("same_srv_rate", DoubleType, true),
    StructField("diff_srv_rate", DoubleType, true),
    StructField("srv_diff_host_rate", DoubleType, true),
    StructField("dst_host_count", IntegerType, true),
    StructField("dst_host_srv_count", IntegerType, true),
    StructField("dst_host_same_srv_rate", DoubleType, true),
    StructField("dst_host_diff_srv_rate", DoubleType, true),
    StructField("dst_host_same_src_port_rate", DoubleType, true),
    StructField("dst_host_srv_diff_host_rate", DoubleType, true),
    StructField("dst_host_serror_rate", DoubleType, true),
    StructField("dst_host_srv_serror_rate", DoubleType, true),
    StructField("dst_host_rerror_rate", DoubleType, true),
    StructField("dst_host_srv_rerror_rate", DoubleType, true),
    StructField("label", StringType, true)))

  def main(args: Array[String]): Unit = {
    // Creation of configuration and session
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("NetworkAnomalyDetection")
      .set("spark.driver.memory", "6g")

    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("NetworkAnomalyDetection")
      .getOrCreate()

    val rawDataDF = spark.read.format("com.databricks.spark.csv")
      .option("header", "false")
      .option("inferSchema", "true")
      .schema(DataSchema)
      .load(DataPath)

    val runClustering = new RunClustering(spark, rawDataDF)

    // K-means
    //(20 to 100 by 20).map(k => (k, runClustering.kmeansSimple(k)))
    //(20 to 100 by 20).map(k => (k, runClustering.kmeansOneHotEncoder(k)))
    //(20 to 100 by 20).map(k => (k, runClustering.kmeansOneHotEncoderWithNormalization(k)))
    //runClustering.gaussianMixtureOneHotEncoderWithNormalization(20)
    runClustering.kmeansOneHotEncoderWithNormalization(20)

    //// Bisecting K-means
    //(20 to 100 by 20).map(k => (k, runClustering.bisectingKmeansOneHotEncoderWithNormalization(k)))

    // Gaussian Mixture
    //(20 to 100 by 20).map(k => (k, runClustering.gaussianMixtureOneHotEncoderWithNormalization(k)))
    //runClustering.gaussianMixtureOneHotEncoderWithNormalization(20)
  }

  class RunClustering(private val spark: SparkSession, var data: DataFrame) {

    import spark.implicits._

    // Select only numerical features
    val CategoricalColumns = Seq("protocol_type", "service", "flag")

    def distance(centroid: Vector, data: Vector) =
      // Tranforming vector to array of double since operations
      // on vector are not implemented
      math.sqrt(centroid.toArray.zip(data.toArray)
        .map(p => p._1 - p._2).map(d => d * d).sum)

    def distanceAllCluster(centroid: Vector, dataCentroid: Array[DenseVector]) = {
      dataCentroid.map(d => distance(centroid, d))
    }

    def clusteringScore(centroids: Array[Vector], data: DataFrame, k: Int) = {
      // For each k, select data belonging to the centroid
      // and calculating the distance.
      val score = (0 until k).map{ k =>
        val dataCentroid = data.filter($"prediction" === k)
          .select("features")
          .collect()
          .map {
            // Get the feature vectors in dense format
            case Row(v: Vector) => v.toDense
          }
        val s = distanceAllCluster(centroids(k), dataCentroid)
        if (s.length > 0)
          s.sum / s.length
        else
          s.sum
      }
      if (score.nonEmpty)
        score.sum / score.length
      else
        score.sum
    }

    def write2file(score: Double, startTime: Long, technique: String) = {
      // Writes result to file
      val format = new SimpleDateFormat("yyyyMMddHHmm")
      val pw = new PrintWriter(new File("results" + format.format(Calendar.getInstance().getTime) +
        "_" + technique.replaceAll(" ", "_") + ".txt"))
      try {
        println(technique)
        pw.write(s"$technique\n")
        println(s"Score=$score")
        pw.write(s"Score=$score\n")
        val duration = (System.nanoTime - startTime) / 1e9d
        println(s"Duration=$duration")
        pw.write(s"Duration=$duration\n")
      } finally {
        pw.close()
      }
    }

    def kmeansSimple(k: Int): Unit = {
      println(s"Running kmeansSimple ($k)")
      val startTime = System.nanoTime()
      // Remove the label column
      val dataDF = this.data.drop("label")
      dataDF.cache()
      val numericalColumns = dataDF.columns.diff(CategoricalColumns)

      // Creation of vector with features
      val assembler = new VectorAssembler()
        .setInputCols(numericalColumns)
        .setOutputCol("features")

      val kmeans = new KMeans()
        .setK(k)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(1L)

      val pipeline = new Pipeline()
        .setStages(Array(assembler, kmeans))

      val dataDFSample = dataDF.sample(false, 0.01, 42)
      dataDFSample.cache()
      val pipelineModel = pipeline.fit(dataDFSample)
      dataDF.unpersist()
      //val pipelineModel = pipeline.fit(dataDF)
      //dataDF.unpersist()

      val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

      // Prediction
      val cluster = pipelineModel.transform(dataDFSample)

      // Get the centroids
      val centroids = kmeansModel.clusterCenters

      // Calculate the score
      val score = this.clusteringScore(centroids, cluster, k)

      this.write2file(score, startTime, "K-means (" + k + ") simple")
    }

    def kmeansOneHotEncoder(k: Int): Unit = {
      println(s"Running kmeansOneHotEncoder ($k)")
      val startTime = System.nanoTime()
      // Remove the label column
      val dataDF = this.data.drop("label")
      dataDF.cache()

      // Indexing categorical columns
      val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new StringIndexer()
          .setInputCol(c)
          .setOutputCol(s"${c}_index")
      ).toArray

      // Encoding previously indexed columns
      val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new OneHotEncoder()
          .setInputCol(s"${c}_index")
          .setOutputCol(s"${c}_vec")
      ).toArray

      // Creation of list of columns for vector assembler (with only numerical columns)
      val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

      // Creation of vector with features
      val assembler = new VectorAssembler()
        .setInputCols(assemblerColumns)
        .setOutputCol("features")

      val kmeans = new KMeans()
        .setK(k)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(1L)

      val pipeline = new Pipeline()
        .setStages(indexer ++ encoder ++ Array(assembler, kmeans))

      val pipelineModel = pipeline.fit(dataDF)
      dataDF.unpersist()

      val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

      // Prediction
      val cluster = pipelineModel.transform(dataDF)

      // Get the centroids
      val centroids = kmeansModel.clusterCenters

      // Calculate the score
      val score = this.clusteringScore(centroids, cluster, k)

      this.write2file(score, startTime, "K-means (" + k + ") with one-hot encoder")
    }

    def kmeansOneHotEncoderWithNormalization(k: Int): Unit = {
      println(s"Running kmeansOneHotEncoderWithNormalization ($k)")
      val startTime = System.nanoTime()
      // Remove the label column
      val dataDF = this.data.drop("label")
      dataDF.cache()

      // Indexing categorical columns
      val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new StringIndexer()
          .setInputCol(c)
          .setOutputCol(s"${c}_index")
      ).toArray

      // Encoding previously indexed columns
      val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new OneHotEncoder()
          .setInputCol(s"${c}_index")
          .setOutputCol(s"${c}_vec")
      ).toArray

      // Creation of list of columns for vector assembler (with only numerical columns)
      val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

      // Creation of vector with features
      val assembler = new VectorAssembler()
        .setInputCols(assemblerColumns)
        .setOutputCol("featuresVector")

      // Normalization using standard deviation
      val scaler = new StandardScaler()
        .setInputCol("featuresVector")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(false)

      val kmeans = new KMeans()
        .setK(k)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(1L)

      val pipeline = new Pipeline()
        .setStages(indexer ++ encoder ++ Array(assembler, scaler, kmeans))

      val dataDFSample = dataDF.sample(false, 0.01, 42)
      dataDFSample.cache()
      val pipelineModel = pipeline.fit(dataDFSample)
      dataDF.unpersist()
      //val pipelineModel = pipeline.fit(dataDF)
      //dataDF.unpersist()

      // Prediction
      val cluster = pipelineModel.transform(dataDF)

      val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

      // Get the centroids
      val centroids = kmeansModel.clusterCenters

      // Calculate the score
      val score = this.clusteringScore(centroids, cluster, k)

      this.write2file(score, startTime, "K-means (" + k + ") with one-hot encoder with normalization")
    }

    def bisectingKmeansOneHotEncoderWithNormalization(k: Int): Unit = {
      println(s"Running bisectingKmeansOneHotEncoderWithNormalization ($k)")
      val startTime = System.nanoTime()
      // Remove the label column
      val dataDF = this.data.drop("label")
      dataDF.cache()

      // Indexing categorical columns
      val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new StringIndexer()
          .setInputCol(c)
          .setOutputCol(s"${c}_index")
      ).toArray

      // Encoding previously indexed columns
      val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new OneHotEncoder()
          .setInputCol(s"${c}_index")
          .setOutputCol(s"${c}_vec")
      ).toArray

      // Creation of list of columns for vector assembler (with only numerical columns)
      val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

      // Creation of vector with features
      val assembler = new VectorAssembler()
        .setInputCols(assemblerColumns)
        .setOutputCol("featuresVector")

      // Normalization using standard deviation
      val scaler = new StandardScaler()
        .setInputCol("featuresVector")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(false)

      val kmeans = new BisectingKMeans()
        .setK(k)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(1L)

      val pipeline = new Pipeline()
        .setStages(indexer ++ encoder ++ Array(assembler, scaler, kmeans))

      val pipelineModel = pipeline.fit(dataDF)
      dataDF.unpersist()

      // Prediction
      val cluster = pipelineModel.transform(dataDF)

      val kmeansModel = pipelineModel.stages.last.asInstanceOf[BisectingKMeansModel]

      // Get the centroids
      val centroids = kmeansModel.clusterCenters

      // Calculate the score
      val score = this.clusteringScore(centroids, cluster, k)

      this.write2file(score, startTime, "Bisecting K-means (" + k + ") with one-hot encoder with normalization")
    }

    def gaussianMixtureOneHotEncoderWithNormalization(k: Int): Unit = {
      println(s"Running gaussianMixtureOneHotEncoderWithNormalization ($k)")
      val startTime = System.nanoTime()
      // Remove the label column
      val dataDF = this.data.drop("label")
      dataDF.cache()

      // Indexing categorical columns
      val indexer: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new StringIndexer()
          .setInputCol(c)
          .setOutputCol(s"${c}_index")
      ).toArray

      // Encoding previously indexed columns
      val encoder: Array[org.apache.spark.ml.PipelineStage] = CategoricalColumns.map(
        c => new OneHotEncoder()
          .setInputCol(s"${c}_index")
          .setOutputCol(s"${c}_vec")
      ).toArray

      // Creation of list of columns for vector assembler (with only numerical columns)
      val assemblerColumns = (Set(dataDF.columns: _*) -- CategoricalColumns ++ CategoricalColumns.map(c => s"${c}_vec")).toArray

      // Creation of vector with features
      val assembler = new VectorAssembler()
        .setInputCols(assemblerColumns)
        .setOutputCol("featuresVector")

      // Normalization using standard deviation
      val scaler = new StandardScaler()
        .setInputCol("featuresVector")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(false)

      val gaussianMixture = new GaussianMixture()
        .setK(k)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(1L)

      val pipeline = new Pipeline()
        .setStages(indexer ++ encoder ++ Array(assembler, scaler, gaussianMixture))

      val dataDFSample = dataDF.sample(false, 0.01, 42)
      dataDFSample.cache()
      val pipelineModel = pipeline.fit(dataDFSample)
      dataDF.unpersist()

      val gmm = pipelineModel.stages.last.asInstanceOf[GaussianMixtureModel]

      // Prediction
      val cluster = pipelineModel.transform(dataDFSample)

      // Get the centroids
      val centroids = (0 until k).map(i => gmm.gaussians(i).mean).toArray

      // Calculate the score
      val score = this.clusteringScore(centroids, cluster, k)

      this.write2file(score, startTime, "GaussianMixture (" + k + ") with one-hot encoder with normalization")
    }
  }
}
