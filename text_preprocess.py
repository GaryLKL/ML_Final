# https://towardsdatascience.com/natural-language-processing-with-pyspark-and-spark-nlp-b5b29f8faba
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, regexp_replace, concat_ws
from pyspark.sql.types import *
from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.corpus import wordnet
from nltk.corpus import stopwords
eng_stopwords = stopwords.words('english')
#from nltk.tokenize import punkt
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import sparknlp
from sparknlp.annotator import (Tokenizer, Normalizer,
                                LemmatizerModel, StopWordsCleaner)
from sparknlp.common import *
from sparknlp.base import Finisher, DocumentAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as f

# pyspark --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0
# spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0

def remove_html_tags(text):
    ''' remove html tags from text '''
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_extra_whitespace(text):
    ''' remove the extra whitespace '''
    text = text.strip()
    return " ".join(text.split())

def remove_accented_chars(text):
    ''' e.g. café -> cafe '''
    text = unidecode(text)
    return text

def text_preprocessing(df):
	# remove url and convert to lower case
	df = df.withColumn("clean_review", (lower(regexp_replace("review", r"https?:\/\/.*[\r\n]*", ""))))
	# remove html tags
	#remove_html_udf = udf(lambda row: remove_html_tags(row), StringType())
	#df = df.withColumn("clean_review", remove_html_udf(col("clean_review")))
	# remove extra whitespaces
	remove_space_udf = udf(lambda row: remove_extra_whitespace(row), StringType())
	df = df.withColumn("clean_review", remove_space_udf(col("clean_review")))
	# remove accented characters
	remove_accented_udf = udf(lambda row: unidecode(row), StringType())
	remove_accented_df = df.select(remove_accented_chars("clean_review"))
	df = df.withColumn("clean_review", remove_accented_df.clean_review)
	# remove punctuation
	df = df.withColumn("clean_review", regexp_replace("clean_review", "[^a-zA-Z\\s]", ""))
	return df

def sparknlp_transform(df):
	documentAssembler = DocumentAssembler() \
	     .setInputCol('review') \
	     .setOutputCol('document')
	tokenizer = Tokenizer() \
	     .setInputCols(['document']) \
	     .setOutputCol('token')
	normalizer = Normalizer() \
	     .setInputCols(['token']) \
	     .setOutputCol('normalized') \
	     .setLowercase(True)
	lemmatizer = LemmatizerModel.pretrained() \
	     .setInputCols(['normalized']) \
	     .setOutputCol('lemma')
	stopwords_cleaner = StopWordsCleaner() \
	     .setInputCols(['lemma']) \
	     .setOutputCol('clean_token') \
	     .setCaseSensitive(False) \
	     .setStopWords(eng_stopwords)
	# finisher converts tokens to human-readable output
	finisher = Finisher() \
	     .setInputCols(['clean_token']) \
	     .setCleanAnnotations(True)
	pipeline = Pipeline() \
	     .setStages([
	           documentAssembler,
	           tokenizer,
	           normalizer,
	           lemmatizer,
	           stopwords_cleaner,
	           finisher
	     ])
	data = pipeline.fit(df).transform(df)
	return data

def text_pipeline(df):
	data = text_preprocessing(df)
	final_df = sparknlp_transform(data)
	final = final_df.withColumn('cleaned_review', concat_ws(' ', 'finished_clean_token')) \
			.select("ex_id", "user_id", "prod_id", "rating", "label", "date", "cleaned_review")
	return final

def input_schema():
    data_schema = StructType([
        StructField("ex_id", IntegerType()),
        StructField("user_id", IntegerType()),
        StructField("prod_id", IntegerType()),
        StructField("rating", FloatType()),
        StructField("label", IntegerType()),
        StructField("date", StringType()),
        StructField("review", StringType()),
    ])
    return data_schema

def output_schema():
    data_schema = StructType([
        StructField("ex_id", IntegerType()),
        StructField("user_id", IntegerType()),
        StructField("prod_id", IntegerType()),
        StructField("rating", FloatType()),
        StructField("label", IntegerType()),
        StructField("date", StringType()),
        StructField("cleaned_review", StringType()),
    ])
    return data_schema

def settings(memory):
    ### setting ###
    conf = pyspark.SparkConf() \
        .setAll([('spark.app.name', 'downsampling code'),
                 ('spark.master', 'local'),
                 ('spark.executor.memory', memory),
                 ('spark.driver.memory', memory)])
    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()
    return spark

def write_json(df, folder_name):
	outputschema = output_schema()
	df \
   .coalesce(1) \
   .write \
   .format("json") \
   .mode ("overwrite")	\
   .option("header", "true") \
   .option("schema", outputschema) \
   .save("fake-reviews/"+folder_name)

if __name__ == "__main__":
	# spark = SparkSession.builder \
	# .appName("Spark NLP")\
	# .master("local[4]")\
	# .config("spark.driver.memory","16G")\
	# .config("spark.driver.maxResultSize", "2G") \
	# .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0")\
	# .config("spark.kryoserializer.buffer.max", "1000M")\
	# .getOrCreate()

	# spark = SparkSession.builder \
	#  .master('local[*]') \
	#  .appName('Spark NLP') \
	#  .config("spark.driver.memory","10G")\
	#  .config('spark.jars.packages', 
	#          'com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.0') \
	#  .getOrCreate()


	#df = spark.read.csv("hdfs:///user/kll482/fake-reviews/data.csv", header=True)
	spark = settings("10g")

	inputschema = input_schema()
	train = spark.read.json("hdfs:///user/kll482/fake-reviews/train.json",
							multiLine=True,
							schema=inputschema)
	dev = spark.read.json("hdfs:///user/kll482/fake-reviews/dev.json",
							multiLine=True,
							schema=inputschema)

	test = spark.read.json("hdfs:///user/kll482/fake-reviews/test.json",
							multiLine=True,
							schema=inputschema)

	train_df = train.where(col("review") != '')
	dev_df = dev.where(col("review") != '')
	test_df = test.where(col("review") != '')

	train_final = text_pipeline(train_df)
	dev_final = text_pipeline(dev_df)
	test_final = text_pipeline(test_df)

	write_json(train_final, "cleaned_train")
	write_json(dev_final, "cleaned_dev")
	write_json(test_final, "cleaned_test")
	
