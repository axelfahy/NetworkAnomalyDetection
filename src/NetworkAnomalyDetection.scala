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
 *
 *  Basic implementation is based on the chapter 5 (Anomaly Detection in Network Traffic with K-means clustering) of the book Advanced Analytics with Spark. However, this implementation is using the Dataframe-based API instead of the RDD-based API.
 *
 * @author Axel Fahy
 * @author Rudolf Höhn
 * @author Brian Nydegger
 * @author Assaf Mahmoud
 *
 * @date 15.05.2017
 *
 */

import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}

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

  val K = 10

  def main(args: Array[String]): Unit = {
    // Creation of configuration and session
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("NetworkAnomalyDetection")
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

    // Select only numerical features
    val categoricalColumns = Seq("protocol_type", "service", "flag")
    // Remove the label column
    val dataDF = rawDataDF.drop("label")
    val numericalColumns = dataDF.columns.diff(categoricalColumns)
    numericalColumns.foreach(println)

    // Indexing categorical columns
    val indexer: Array[org.apache.spark.ml.PipelineStage] = categoricalColumns.map(
      c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(s"${c}_index")
    ).toArray

    // Encoding previously indexed columns
    val encoder: Array[org.apache.spark.ml.PipelineStage] = categoricalColumns.map(
        c => new OneHotEncoder()
         .setInputCol(s"${c}_index")
         .setOutputCol(s"${c}_vec")
    ).toArray

    // Creation of list of columns for vector assembler (with only numerical columns)
    val assemblerColumns = (Set(dataDF.columns: _*) -- categoricalColumns ++ categoricalColumns.map(c => s"${c}_vec")).toArray

    // Creation of vector with features
    val assembler = new VectorAssembler()
      .setInputCols(assemblerColumns)
      .setOutputCol("featuresVector")

    // Normalization using standard deviation
    val scaler = new StandardScaler()
      .setInputCol("featuresVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans()
      .setK(K)
      .setFeaturesCol("scaledFeatureVector")
      .setPredictionCol("prediction")
      .setSeed(1L)

    val pipeline = new Pipeline()
      .setStages(indexer ++ encoder ++ Array(assembler, scaler, kmeans))

    val pipelineModel = pipeline.fit(dataDF)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val WSSSE = kmeansModel.computeCost(pipelineModel.transform(dataDF))
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    kmeansModel.clusterCenters.foreach(println)
  }
}
