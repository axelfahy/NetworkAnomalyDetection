
lazy val root = (project in file(".")).
  settings(
    name := "NetworkAnomalyDetection",
    organization := "hes-so",
    scalaVersion := "2.11.7",
    version := "0.1",
    mainClass in Compile := Some("NetworkAnomalyDetection")
  )

val sparkVersion = "2.1.0"

libraryDependencies ++= Seq(
  // Spark and Mllib
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "com.databricks" % "spark-csv_2.11" % "1.5.0"
)

scalaSource in Compile <<= baseDirectory(_ / "src")

