import tensorflow as tf
import pandas as pd
import numpy as np

#读取训练数据
df=pd.read_csv(r"C:\Users\Administrator\Desktop\shujujingsai\salor\public.train.csv",header=None)
datatrain=np.array(df)
#提取特征值，形成输入数据
#dataxs=datatrain[1:,[0,1,2,3,4,8,9,10,11,12,13,17,18,19]]#9000x8
dataxs=datatrain[1:,:20]
dataxshlen=len(dataxs)#训练输入数据的行数
dataxsllen=len(dataxs[0])#训练输入数据的列数
for i in range(dataxshlen):
    for j in range(dataxsllen):
         dataxs[i][j]=float(dataxs[i][j])
#形成输出数据
datays=datatrain[1:,[20]]#9000x1
datayshlen=dataxshlen#训练输出数据的行数
dataysllen=len(datays[0])#训练输出数据的列数
for i in range(datayshlen):
    for j in range(dataysllen):
         datays[i][j]=float(datays[i][j])
print(datays[0][0])
#归一化,datays取值范围为（0.05，0.95）
#datays_nor=0.05+(datays-np.min(datays))/(np.max(datays)-np.min(datays))*(0.95-0.05)

print(dataxs,datays,dataxshlen,dataxsllen,datayshlen,dataysllen)

#读取测试数据
df1=pd.read_csv(r"C:\Users\Administrator\Desktop\shujujingsai\salor\public.test.csv",header=None)
datatest=np.array(df1)
#提取特征值，形成输入数据
#dataxs_test=datatest[1:,[0,1,2,3,4,8,9,10,11,12,13,17,18,19]]#9000x8
dataxs_test=datatest[1:,:20]
dataxs_testhlen=len(dataxs_test)#训练输入数据的行数
dataxs_testllen=len(dataxs_test[0])#训练输入数据的列数
for i in range(dataxs_testhlen):
    for j in range(dataxs_testllen):
         dataxs_test[i][j]=float(dataxs_test[i][j])
#测试比对数据
df2=pd.read_csv(r"C:\Users\Administrator\Desktop\shujujingsai\salor\submit_example.csv",header=None)
datatest1=np.array(df2)
datays_test=datatest1[0:,1:2]#9000x1
datays_testhlen=dataxs_testhlen#训练输出数据的行数
datays_testllen=len(datays_test[0])#训练输出数据的列数
for i in range(datays_testhlen):
    for j in range(datays_testllen):
         datays_test[i][j]=float(datays_test[i][j])
print(datays_test[0][0])
#归一化,datays取值范围为（0.05，0.95）
#datays_nor_test=0.05+(datays_test-np.min(datays_test))/(np.max(datays_test)-np.min(datays_test))*(0.95-0.05)

print(dataxs_test,datays_test,dataxs_testhlen,dataxs_testllen,datays_testhlen,datays_testllen)



def add_layer(inputs,in_size,out_size,activation_function=None): 
        Weights=tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
        Biases=tf.Variable(tf.constant(0.1,shape=[out_size]))
        xW_plus_b=tf.matmul(inputs,Weights)+Biases
        if activation_function is None:
            outputs=xW_plus_b
        else:
            outputs=activation_function(xW_plus_b)
        return outputs

#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,20])#
ys=tf.placeholder(tf.float32,[None,1])
keep_prop=tf.placeholder(tf.float32)


#layer1
datays_nor_layer1=add_layer(xs,20,10,activation_function=tf.sigmoid)

#layer2
datays_nor_layer2=add_layer(datays_nor_layer1,10,20,activation_function=tf.nn.relu)

#perdiction
prediction=add_layer(datays_nor_layer2,20,1,activation_function=None)


#the error between prediction and real data
loss=tf.sqrt(tf.reduce_mean(tf.square(prediction-ys)))
train_step=tf.train.GradientDescentOptimizer(0.9).minimize(loss)
#train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)


#初始化
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#模型精确度
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prop:0.5})
    #y_pre_tru=(y_pre-0.05)*(np.max(datays)-np.min(datays))/(0.95-0.05)
    accuracy_prediction=1/(1+tf.sqrt(tf.reduce_mean(tf.square(y_pre-ys))))
    #accuracy_prediction_tru=1/(1+tf.sqrt(tf.reduce_mean(tf.square(y_pre_tru-datays_test))))
    result=sess.run(accuracy_prediction,feed_dict={xs:v_xs,ys:v_ys,keep_prop:0.5})
    print(y_pre)
    return result


         
for k in range(2000):
    #print(batch_xs,batch_ys,len(batch_xs))
    sess.run(train_step,feed_dict={xs:dataxs,ys:datays,keep_prop:0.5})
    #sess.run(train_step,feed_dict={xs:dataxs_test,ys:datays_nor_test})
    if k%500==0:
        #print(k,sess.run(loss,feed_dict={xs:dataxs,ys:datays_nor}))
        print(compute_accuracy(dataxs_test,datays_test))
        
























         
