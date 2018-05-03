import pyspark
import numbers
import os
import shutil
import numpy as np
import string
import warnings

from functools import reduce

from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import when
from pyspark.sql.functions import monotonically_increasing_id
from math import sqrt
from operator import add
from pyspark.mllib.clustering import KMeans, KMeansModel

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import NGram
from pyspark.ml.feature import MinHashLSH

from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.ml.feature import CountVectorizer

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Normalizer

class cleaner():
    def __init__(self,data,save_dir,sparkcontext=None):
        '''
        Constructor

        Args:
            data: string, path of the data file
            save_dir: string, path for saving different versions of data

        '''
        if sparkcontext != None:
            self.sc = sparkcontext
        else:
            self.sc = pyspark.SparkContext()
        self.ss = SQLContext(self.sc)
        try:
            self.data=self.ss.read.format('csv').options(header='true',inferschema='true').load(data)
        except Exception as e:
            self.sc.stop()
            print(e)
            raise ValueError("Can't read file, sparkcontext stopped.")
        self.out='Spark-lean default output'
        if not os.path.isdir(save_dir):
            raise ValueError('Directory does not exist! Create it first!')
        self.save_dir=save_dir

    def initialize(self):
        self.data.printSchema()
        return
    def stop_context(self):
        '''
        Stop the current spark context.
        Call it after done with the class to avoid confliction of multiple spark contexts.

        '''
        self.sc.stop()
        return

    def save_data_version(self,name):

        '''
        Save current data to a checkpoint version. In case you want a backup copy.

        Args:
            name:string,name of this data version.

        '''
        self.data.write.parquet(os.path.join(self.save_dir,name))
        return
    def load_data_version(self, name):

        '''
        Load a saved checkpoint version.

        Args:
            name:string,name of this data version.

        '''
        if not os.path.isdir(os.path.join(self.save_dir,name)):
            raise ValueError('Directory does not exist!')
        self.data=self.ss.read.parquet(os.path.join(self.save_dir,name))
        return
    def delete_data_version(self,name):
        '''
        Delete a saved checkpoint version.

        Args:
            name:string,name of this data version.

        '''
        if not os.path.isdir(os.path.join(self.save_dir,name)):
            raise ValueError('Directory does not exist!')
        shutil.rmtree(os.path.join(self.save_dir,name))
        return
    def detect_missing_value(self,columns='DEFAULT',keywords=["N/A","None"]):
        '''
        Provide a missing value diagnosis and further operations.

        Args:
            keywords: list of strings, possible keywords that represent null values.
            columns: list of strings, range of columns to inspect. Default: all columns.

        '''
        output=[]
        targets={}
        null_targets=[]
        if columns=='DEFAULT':
            columns=self.data.columns

        for col in columns:
            n=self.data.where(self.data[col].isNull()).count()
            if n >0:
                message='column '+col+' has '+str(n)+' null values!'
                output.append(message)
                null_targets.append(col)
        for naword in keywords:
            for col in self.data.columns:
                n=self.data.filter(self.data[col]==naword).count()
                if  n> 0:
                    message='Column '+col+' has '+str(n)+' rows that include keyword: '+naword
                    output.append(message)
                    if col in targets:
                        targets[col].append(naword)
                    else:
                        targets[col]=[naword]
        self.out=(null_targets,targets)
        for i in output:
            print(i)
        for i in range(5):
            print('\n')
        print('-'*16)

        print("Please select an approach:")
        print("1. Delete all suspicious rows")
        print("2. Replacing suspicious rows with 0")
        print("3. Replacing suspicious rows with input")
        print("4. Only delete rows with null values")
        print("5. Replacing null values with input")
        print("6. Replacing null values with 0")
        print("7. Do nothing")

        try:
            mode=int(input('Input(number):'))
        except ValueError:
            print ("Not a number")
        # if mode==10:
        #     for col,naword in targets.items():
        #         c.replace(col,naword)
        if mode==1:
            for col in null_targets:
                self.data=self.data.where(self.data[col].isNotNull())
            for k,v in targets.items():
                self.remove_by_column_and_value(k,v)
            print("All deleted!")
        elif mode==2:
            self.data=self.data.na.fill('0')
            for k,v in targets.items():
                self.remove_by_column_and_value(k,v)
        elif mode==3:
            target=str(input('Input(to replace null):'))
            self.data=self.data.na.fill(target)
            for k,v in targets.items():
                self.replace_by_value(k, v, target)
        elif mode==4:
            for col in null_targets:
                self.data=self.data.where(self.data[col].isNotNull())

        elif mode==5:
            target=str(input('Input(to replace null):'))
            self.data=self.data.na.fill(target)

        elif mode==6:
            self.data=self.data.na.fill('0')
        return


    def reset_index(self, drop = False):
        '''
        Reset index of the input

        Args:

            drop: whether drop the original index column, default False

        '''
        if drop:
            data = self.data.drop('index')
        else:
            df = self.data.drop('index')
            data = df.select("*").withColumn("index", monotonically_increasing_id())

        self.data = data
        return

    def replace_by_value(self, col_name, org_value, new_value):
        '''
        Replace value of a instance by its column name and original value

        Args:

            col_name: column name of the target instance (string)
            org_value: original value of the target instance
            new_value: replaced value
        '''
        if isinstance(col_name, str):
            if col_name in self.data.columns:
                org_value_type = type(org_value)
                if isinstance(self.data.first()[col_name], org_value_type):
                    new_value_type = type(new_value)
                    if org_value_type == new_value_type:
                        data = self.data.replace(org_value, new_value, col_name)
                    else:
                        raise TypeError("{} is {} object, not the same type as {}: {}".format(new_value, new_value_type,org_value,org_value_type))
                else:
                    raise TypeError("{} is {} object, not the same type as {}: {}".format(org_value, type(org_value), col_name, type(self.data.first()[col_name])))
            else:
                raise NameError('{} is not a valid column name'.format(col_name))
        else:
            raise TypeError('column name must be a string object. your input type is {}'.format(type(col_name)))

        self.data = data
        return

    def replace_by_index(self, col_name, index, new_value):
        '''
        Replace value of a instance by its column name and index

        Args:

            col_name: column name of the target instance (string)
            index: index of the target instance(int)
            new_value: replaced value

        '''
        if isinstance(col_name, str):
            if "index" in self.data.columns:
                if isinstance(index, int):
                    new_value_type = type(new_value)
                    if isinstance(self.data.first()[col_name], new_value_type):
                        data = self.data.withColumn(col_name,when(self.data["index"] == index, new_value).otherwise(self.data[col_name]))
                    else:
                        raise TypeError("{} is {} object, not the same type as {}: {}".format(new_value, type(new_value), col_name, type(self.data.first()[col_name])))
                else:
                    raise TypeError("index must be a int object. your input type is {}".format(type(index)))
            else:
                raise ValueError('The input DataFrame object does not contain a index column')
        else:
            raise TypeError('column name must be a string object. your input type is {}'.format(type(col_name)))

        self.data = data
        return
    def remove_by_index(self,index):
        """
        Remove rows by looking at index

        index: given index to remove
        """
        df = self.data
        if 'index' in df.columns:
            pass
        else:
            return "DataFrame dosen't have index"

        df.createOrReplaceTempView("df")
        num_before_remove = df.count()
        if isinstance(index,tuple):
            pass
        elif isinstance(index,list):
            if len(index) == 0:
                return "Empty list"
            elif len(index) == 1:
                index = index[0]
                df = self.ss.sql("select * from df where index != '{}'".format(index))
                num_after_remove = df.count()
                self.data = df
                print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
                return
            else:
                index = tuple(index)
        elif isinstance(index, int) or isinstance(index,str):
            df = self.ss.sql("select * from df where index != '{}'".format(index))
            num_after_remove = df.count()
            self.data = df
            print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
            return
        else:
            return "Wrong index type, only list, tuple, str, or int is accepted, you input type is {}".format(type(index))

        df = self.ss.sql("select * from df where index not in {}".format(index))
        num_after_remove = df.count()
        self.data = df
        print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
        return



    def remove_by_column_and_value(self,column,values):
        '''
        Remove rows by looking at column index and value

        column: str
        values: number/string
        '''
        df = self.data
        if isinstance(column,str):
            pass
        else:
            return 'column must be str type'
        if column in df.columns:
            pass
        else:
            return 'feature not in scheme'
        if isinstance(values, int) or isinstance(values,str):
            pass
        else:
            return 'The value mush be str or number, your input is'+str(type(values))
        df.createOrReplaceTempView("df")
        num_before_remove = df.count()
        the_number_of_same_value = self.ss.sql("select * from df where  {} = '{}'".format(column,values)).count()
        if the_number_of_same_value > 0:
            df = self.ss.sql("select * from df where  {} != '{}'".format(column,values))
        num_after_remove = df.count()
        self.data = df
        print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
        return
    def dropDuplicateColumn(self):
        """
        Detect and drop duplicated features
        """
        df = self.data
        i = 0
        df.createOrReplaceTempView("df")
        total_row = df.count()
        while i < len(df.columns)-1:
            df_col_len = len(df.columns)
            left_col = df.columns[i]
            print("Checking column {}".format(left_col))
            left_table = self.ss.sql("select {} from df".format(left_col))
            left_table.createOrReplaceTempView("left_table")
            j = i + 1
            while j < len(df.columns):
                right_col = df.columns[j]
                right_table = self.ss.sql("select {} from df".format(right_col))
                right_table.createOrReplaceTempView("right_table")
                #firt, find if the first num_try of row is same
                n_try = 10
                left_temp_n_try = left_table.take(n_try)
                right_temp_n_try = right_table.take(n_try)
                indicator = False
                for l_row,r_row in zip(left_temp_n_try,right_temp_n_try):
                    if (l_row[left_col]) == (r_row[right_col]):
                        indicator = True
                        pass
                    else:
                        indicator = False
                        #print(left_col,right_col,'not equal')
                        break
                #left_table and right_table have same first 10 value
                if indicator:
                    the_number_of_same_line = self.ss.sql("select * from left_table inner join right_table on {}={}".format(left_col,right_col)).count()
                    #print(the_number_of_same_line,total_row)
                    if the_number_of_same_line == total_row:
                        option = input("Column {} and column {} are same.\n \
                        Press 1 to drop {}, press 2 to drop {}, press 3 or other to keep both".format(left_col,right_col,left_col,right_col))
                        if option =='1':
                            print('The column',left_col,'will be dropped')
                            currenct_list_columns = df.columns
                            currenct_list_columns.remove(left_col)
                            str_currenct_list_columns = ','.join(currenct_list_columns)
                            df = self.ss.sql("select {} from df".format(str_currenct_list_columns))
                            df.createOrReplaceTempView("df")
                            i = i - 1
                            print('Success')
                            break
                        elif option == '2':
                            print('The column',right_col,'will be dropped')
                            currenct_list_columns = df.columns
                            currenct_list_columns.remove(right_col)
                            str_currenct_list_columns = ','.join(currenct_list_columns)
                            df = self.ss.sql("select {} from df".format(str_currenct_list_columns))
                            df.createOrReplaceTempView("df")
                            j = j - 1
                            print('Success')
                        else:
                            print("Not column will be dropped")
                    else:
                        #print('10 same, not same')
                        pass
                else:
                    #print('10 not same, not same')
                    pass
                j = j + 1
            i = i + 1
        self.data =df
        return
    def dropColumnWithAllTheSameValues(self):
        """
        Drop columns with only one unique element
        """
        df = self.data
        df.createOrReplaceTempView("df")
        for col in df.columns:
            print("Checking column {}".format(col))
            first_set = set()
            num_of_row = df.count()
            num_first_try = 10
            temp_df = self.ss.sql("select {} from df limit 10".format(col)).take(num_first_try)
            for i in range(num_first_try):
                first_set.add(temp_df[i][col])
            if len(first_set) == 1:
                target_value = first_set.pop()
                num_of_row_with_same_value = self.ss.sql("select * from df where {} = '{}'".format(col,target_value)).count()
                if num_of_row_with_same_value == df.count():
                    option = input("Column {} has the same value for all cells, do you want to drop it?\n\
                    Press 1 to drop it, press 2 or other to keep it: ".format(col))
                    if option == '1':
                        print('The column',col,'is dropped')
                        currenct_list_columns = df.columns
                        currenct_list_columns.remove(col)
                        str_currenct_list_columns = ','.join(currenct_list_columns)
                        df = self.ss.sql("select {} from df".format(str_currenct_list_columns))
                        df.createOrReplaceTempView("df")
                    else:
                        pass
            else:
                pass
                #print('not same')
        self.data =df
        return

    def get_similar_word(self, column, text, n_words=10, n_hash=5,verbose=True):
        """
        Get similar strings in a column by MinHash

        target_col: target column to search
        text: input string
        n_words: number of similar strings
        n_hash: number of hash functions for MinHash
        verbose:True if you want to see interactive output

        Output:
            DataFrame of Nearest Neighbours
        """
        rdd = self.data.rdd
        rdd=rdd.filter(lambda row:row[column] != None)
        rdd=rdd.filter(lambda row:row[column] != "" )
        rdd=rdd.filter(lambda row:len(row[column]) >1)
        cdf=self.ss.createDataFrame(rdd.map(lambda row:(row[column] if row[column] !=None else " ",list(row[column].lower()) if row[column] !=None else [" "])))

        ngram = NGram(n=2, inputCol="_2", outputCol="ngrams")
        if verbose:
            print("Counting Ngram...")
        ngramDataFrame = ngram.transform(cdf)
        if verbose:
            print("Vectorizing...")
        # fit a CountVectorizerModel from the corpus.
        cv = CountVectorizer(inputCol="ngrams", outputCol="features", vocabSize=3000, minDF=0)

        cv_model = cv.fit(ngramDataFrame)

        result = cv_model.transform(ngramDataFrame)

        mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=n_hash)
        if verbose:
            print("Min Hashing...")
        model = mh.fit(result)

        input_text=text
        input_df = [{'text':input_text,'characters': list(input_text)}]
        input_df=self.ss.createDataFrame(input_df)

        ngram = NGram(n=2, inputCol="characters", outputCol="ngrams")
        input_df= ngram.transform(input_df)

        key= cv_model.transform(input_df).first()['features']

        if (key.toArray().sum()<1):
            print("No Match! Please try another input..")
            return

        if verbose:
            print("Finding nearest neighbors...")

        NNs=model.approxNearestNeighbors(result, key, n_words)
        NNs.show()

        self.out=NNs
        #self.out=NNs.select('_1').distinct()
        return

    def clean_strings_removing(self, target_col,rename_to='',punc=True,number=True,others='',verbose=True):
        """
        Clean string by removing certain characters.

        target_col: target column to clean
        rename_to: assign it if you want to rename the column
        number: True if you do not want to keep digits
        punc: True if you do not want to keep punctuations
        others: A string contains other characters you want to remove
        verbose:True if you want to see interactive output
        """
        new_name='nn'
        bad_charac=punc*'!."#$%&\()*+,-/:;<=>?@[\\]^_`{|}~'+number*'1234567890'+others
        if verbose:
            print("Remove any character contained in the set below:")
            print(bad_charac)
        translator = str.maketrans('', '',bad_charac)
        rdd = self.data.rdd
        rdd=rdd.filter(lambda row:row[target_col] != None)
        cleaned=self.ss.createDataFrame(rdd.map(lambda row:(row[target_col],row[target_col].translate(translator),)),
                                   [target_col,new_name])
        if verbose:
            print("original column v.s. cleaned")
            cleaned.show(10)
        self.data=self.data.join(cleaned, self.data[target_col] == cleaned[target_col])

        self.data=self.data.drop(target_col)
        self.data=self.data.drop(target_col)

        if rename_to != '':
            names=[]
            for c in self.data.columns:
                if c==new_name:
                    names.append(rename_to)
                else:
                    names.append(c)
        else:
            names=[]
            for c in self.data.columns:
                if c==new_name:
                    names.append(target_col)
                else:
                    names.append(c)
        self.data=self.data.toDF(*names)
        if verbose:
            print("Final dataframe: (orders could be different)")
            self.data.show(10)
        return

    def clean_strings_keeping(self, target_col,rename_to='',char=True,number=True,punc=True,space=True,verbose=True):
        """
        Clean string by keeping certain characters.

        target_col: target column to clean
        rename_to: assign it if you want to rename the column
        char: True if you want to keep english characters
        number: True if you want to keep digits
        punc: True if you want to keep punctuations
        space: True if you want to keep space
        verbose:True if you want to see interactive output
        """
        new_name='cleaned'

        SafeSet=char*string.ascii_letters+string.punctuation*punc+string.digits*number+space*' '

        def isOK(c,safeset=SafeSet):
            return (c in safeset)

        rdd = self.data.rdd
        rdd=rdd.filter(lambda row:row[target_col] != None)
        cleaned=self.ss.createDataFrame(rdd.map(lambda row:(row[target_col],''.join(e for e in row[target_col] if isOK(e)),)),
                                   [target_col,new_name])
        if verbose:
            print("original column v.s. cleaned")
            cleaned.show(10)
        self.data=self.data.join(cleaned, self.data[target_col] == cleaned[target_col])
        self.data=self.data.drop(target_col)
        self.data=self.data.drop(target_col)
        if rename_to != '':
            names=[]
            for c in self.data.columns:
                if c==new_name:
                    names.append(rename_to)
                else:
                    names.append(c)
        else:
            names=[]
            for c in self.data.columns:
                if c==new_name:
                    names.append(target_col)
                else:
                    names.append(c)
        self.data=self.data.toDF(*names)
        if verbose:
            print("Final dataframe: (orders could be different)")
            self.data.show(10)
            print("Similar names are stored in self.out!")
        return




    def to_double(self, col_name):
        '''
        cast a column to float type

        Args:

            col_name: name of the target column (string)
        '''

        self.data =self.data.withColumn(col_name, self.data[col_name].cast(DoubleType()))
        return

    def featurize(self, col_name, option, skip=False):
        '''
        featurize a specific numerical feature

        Args:

            col_name: name of the target feature (string)
            option: featurization option (int), 1 for Standarization, 2 for L2-Normalization, 3 for Min-Max transformation
            skip: whether skip a non-numerical feature, default False
        '''
        n_null=self.data.where(self.data[col_name].isNull()).count()
        if n_null>0:
            print("Drop {} null values in {}!".format(n_null, col_name))
            self.data=self.data.dropna()

        def ith_(v, i):
            '''
            helper function
            '''
            try:
                return float(v[i])
            except ValueError:
                return None
        if not isinstance(col_name, str):
            raise TypeError('column name must be a string object. your input type is {}'.format(type(col_name)))
        if isinstance(option, int):
            options = [1,2,3]
            if option not in options:
                raise ValueError("option should be 1(Standarization), 2(L2-Normalization) or 3(Min-Max), your input is: {}".format(option))
        else:
            raise TypeError('option must be an int object. your input type is {}'.format(type(option)))
        df=self.data
        temp = df.select(col_name)
        types = [f.dataType for f in temp.schema.fields]
        type_list = ["IntegerType","LongType","DoubleType"]
        if str(types[0]) not in type_list:
            if not skip:
                raise TypeError('The column you try to featurize is {}, which is not a valid data type for this function. You may mant to use function to_double() to cast the column first '.format(types[0]))
            else:
                warnings.warn("you are skipping a non-numerical feature!")

        if option == 1:
            df_stats = df.select(F.mean(F.col(col_name)).alias('mean'),
                                 F.stddev(F.col(col_name)).alias('std')).collect()
            mean = df_stats[0]['mean']
            std = df_stats[0]['std']
            data = df.withColumn(col_name, (df[col_name] - mean)/std)
            data_stats = data.select(F.mean(F.col(col_name)).alias('mean'),
                                 F.stddev(F.col(col_name)).alias('std')).collect()
            new_mean = data_stats[0]['mean']
            new_std = data_stats[0]['std']
            print("Standarization on {} is successful!".format(col_name))
            print("new mean: {}, new std: {}".format(new_mean, new_std))

        elif option == 2:
            assembler = VectorAssembler(inputCols=[col_name], outputCol="feature")
            assembled = assembler.transform(df)
            normalizer = Normalizer(inputCol="feature", outputCol="l2normFeature")
            l2NormData = normalizer.transform(assembled).drop("feature").drop(col_name)
            data = l2NormData.withColumnRenamed("l2normFeature", col_name)
            print("L2-Normalization on {} is successful!".format(col_name))
            ith = F.udf(ith_, DoubleType())
            data = data.withColumn(col_name, ith(col_name, F.lit(0)))

        elif option == 3:
            col_max = df.agg({col_name: "max"}).collect()[0][0]
            col_min = df.agg({col_name: "min"}).collect()[0][0]
            data = df.withColumn(col_name, (df[col_name] - col_min)/(col_max-col_min))
            new_max = data.agg({col_name: "max"}).collect()[0][0]
            new_min = data.agg({col_name: "min"}).collect()[0][0]
            print("Min-Max Transformation on {} is successful!".format(col_name))
            print("new lower bound: {}, new upper bound: {}".format(new_min, new_max))

        self.data = data
        return
    def featurize_all(self, option):
        '''
        featurize the whole DataFrame object

        Args:

            option: featurization option (int), 1 for Standarization, 2 for L2-Normalization, 3 for Min-Max transformation

        '''
        df=self.data
        for i in df.columns:

            temp = df.select(i)
            types = [f.dataType for f in temp.schema.fields]
            type_list = ["IntegerType","LongType","DoubleType"]
            if str(types[0]) not in type_list:
                message="The column ({}) you try to featurize is {}, which is not a valid data type for this function. We skip it for you at this point".format(str(i), str(types[0]))
                warnings.warn(message)
                continue
            self.featurize(i, option, False)

        return





    def outlier_detect(self, col_name, error = 0.1, threshold=1.5):
        '''
        detect outliers of a numerical feature based on IQR rule

        Args:

            col_name: name of the target feature (string)
            error: the relative target precision to achieve, default is 0.1
            threshold: Times of IQR, default is 1.5
        Output(in self.out):
            DataFrame without possible outliers
        '''
        if not isinstance(col_name, str):
            raise TypeError('column name must be a string object. your input type is {}'.format(type(col_name)))
        try:
            self.data = self.data.withColumn(col_name, self.data[col_name].cast(DoubleType()))
        except:
            pass

        initial_count = self.data.count()
        quantiles = self.data.approxQuantile(col_name,[0.25, 0.75], error)
        IQR = quantiles[1] - quantiles[0]
        low_ = quantiles[0] - threshold*IQR
        high_ = quantiles[1] + threshold*IQR

        new_df = self.data.where((self.data[col_name] < low_ ) | (self.data[col_name] > high_ ))
        new_count = new_df.count()

        print("Outlier detection on {} finished!".format(col_name))
        print("There are {} instances in {}, we detect {} possible outliers, shown below:".format(initial_count, col_name, new_count))
        print("-"*50)
        print("Potential Outliers:")
        new_df.select(col_name).show()

        self.out=self.data.where((self.data[col_name] >= low_ ) & (self.data[col_name] <= high_ ))
        print("-"*50)
        print("CLeaned DataFrame without potential outliers:")
        self.out.show()
        return

    def distinguish_numerical_formats(self):
        '''
        provide our estimation of data type (whether numerical) of each feature
        that may be different from the default input type


        '''
        print("The original data types of each feature are:")
        print(self.data.dtypes)

        for i in range(len(self.data.columns)):
            col_name = self.data.dtypes[i][0]
            sample = self.data.select(col_name).limit(10)
            last = sample.select(col_name).orderBy(sample[col_name].desc()).limit(1)
            try:
                int(sample.first()[0])
                int(last.first()[0])
                print("We think {} is a numerical type".format(col_name))
            except:
                print("We think {} is not a numerical type".format(col_name))

        return

    def anomaly_detection_by_KMeans(self,columns,k=3,threshold=4,normalize=False):
        '''
        Detect anomaly combination of features through K-Means

        columns: list of columns to be detected
        k: the number of clusters
        threshold:  if (distance - mean_dsitance )> threshold*std, then point is considered to be anomaly
        normalize: whether normalize the data before fitting into the cluster

        Output(in self.out):
            output[0]: Index of outliers (list of int)
            output[1]: DataFrame of outliers
        '''
        def error(point):
            center = clusters.centers[clusters.predict(point)]
            return sqrt(sum([x**2 for x in (point-center)]))
        def addclustercols(x):
            point = np.array(x[1:])
            center = clusters.centers[0]
            mindist = sqrt(sum([y**2 for y in (point-center)]))
            cl = 0
            for i in range(1,len(clusters.centers)):
                center = clusters.centers[i]
                distance = sqrt(sum([y**2 for y in (point-center)]))
                if distance < mindist:
                    cl = i
                    mindist = distance
            clcenter = clusters.centers[cl]
            #return [x[0]]+list(clcenter) + [distance]
            result = list(clcenter) + [distance]
            return [x[0],cl]+[float(x) for x in result]
        def featurize(df, col_name):
            df_stats = df.select(F.mean(F.col(col_name)).alias('mean'),
                                   F.stddev(F.col(col_name)).alias('std')).collect()
            mean = df_stats[0]['mean']
            std = df_stats[0]['std']
            data = df.withColumn(col_name, (df[col_name] - mean)/std)
            data_stats = data.select(F.mean(F.col(col_name)).alias('mean'),
                                     F.stddev(F.col(col_name)).alias('std')).collect()
            new_mean = data_stats[0]['mean']
            new_std = data_stats[0]['std']
            return data
        def featurize_all(df, columns):
            for i in columns:
                df = featurize(df, i)
            data = df
            return data
        data = self.data
        if 'index' not in data.columns:
            print('Please create index first')
            return
        new_cols_len = len(columns)
        number_type = ["BinaryType" "DecimalType", "DoubleType", "FloatType", "IntegerType","LongType", "ShortType"]
        all_number_type = True
        new_columns_name = ['index','cluster_number']
        for column in columns:
            if column in data.columns:
                all_number_type = (str(data.schema[column].dataType) in number_type) and (all_number_type)
                if not all_number_type:
                    print('The type of'+column+" is "+str(data.schema["_c6"].dataType))
                    print("Only numerical type is accepted ")
                    return
                else:
                    new_columns_name.append(column+"_cluster")
            else:
                print(column,"doesn't exist")
                return
        new_columns_name.append('distance_to_cluster')
        origin_data = data.cache()
        data = data.select(['index']+columns)
        data = data.dropna()
        if normalize:
            data = featurize_all(data,columns)
        target_numpy = data.select(columns).rdd.map(lambda x : np.array(x))
        clusters = KMeans.train(target_numpy, k, maxIterations=20)
        result_data = data.rdd.map(lambda x: addclustercols(x)).toDF(new_columns_name)
        full_data = origin_data.join(result_data,'index',how='inner')
        stat = full_data.groupBy('cluster_number').agg(F.mean('distance_to_cluster').alias('distance_mean'),F.stddev('distance_to_cluster').alias('distance_std'))
        anomaly_data = full_data.join(stat,'cluster_number','inner').rdd.filter(lambda x : x['distance_to_cluster']>(x['distance_mean']+threshold*x['distance_std']))
        try:
            anomaly_data = anomaly_data.toDF()
            anomaly_indices = anomaly_data.select('index')
        except:
            print("None anomaly data based on your setting")
            return
        else:
            self.out= ([int(i['index']) for i in anomaly_indices.collect()],anomaly_data)
            return

    def train_test_split(weights,seed=None):
        """
        Test Train split

        weights: weights for splits, will be normalized if they don’t sum to 1
        """
        if isinstance(weights,list):
            self.out=self.data.randomSplit(weights, seed=seed)
            return
        else:
            print('weights should be list')
            return
