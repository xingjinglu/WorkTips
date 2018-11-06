# 分布式编程： 数据并行   
TF的“集群”由参与一个Tensorflow graph分布式执行的“任务”集合构成。每一个任务与一个TF  的“服务器”相关联，该服务器（server）包含一个“master"和一个“worker”，前者用于创建  sessions，或者用于执行图中的操作符。一个集群可以划分成一个或者多个“job”（工作），每  个工作包含一个或者多个任务。   
- 创建tf.train.ClusterSpec用来描述该集群中的所有任务，任意任务来说具有相同的集群。 
- 创建tf.trainServer,集群会作为参数传递给其构造函数，并且会用工作名字和任务编号来  标识该任务。  


## 关键术语解释  
- client  
  一个客户端是这样一个程序，它会创建一个TF 图并且会构造一个tensorflow::session  来与一个集群交互。   一个单独的客户端进程可以与多个TF服务器交互，一个单独的服务器也  可以为多个客户端提供服务。  
     - 客户端相当于是工作中的任务，可以创建子线程用于执行在同一个服务器的不同gpu or cpu 上？
- cluster（集群）：包含一个或者多个工作，每个工作被划分成一个或者多个任务列表。   
- Job（工作）：由一个任务列表组成，相当于是完成相同的任务的一个分组。  
- Master service（主服务/控制服务）：  
- Task（任务）
      - 
       

## 1. 并行实现路径

用户在开发TF程序时需要考虑下面几个问题  
1. 单GPU或者CPU   
- I/O：如何读入数据，建议采用tf.dataset
- 构建计算图      
- 数据的管理：Tensor，变量，已经namescope, Variablescope, 数据重用  
- 数据或者计算到设备的映射： 包括对GPU设备的配置 
- 会话的建立和管理：有些高层API，可以自动处理checkpoint，Summary等操作  

2. 多GPU模式  
- I/O：如何为每个GPU上的计算图提供数据？ tf.Record or tf.Dataset  
- 并行模式：采用allreduce还是PS模式？如果是后者，采用同步还是异步？    
- 构建计算图 
     * 为每个GPU建立任务？
- 数据的管理  
     * 注意重用变量的声明？（Variablescope  and reuse)      
- 数据和计算到设备的映射   
     * 保存I/O数据的变量   
     * Learningrate，loss  
     * 梯度操作  
- 任务间的数据计算：同步模式下，需要合并计算不同GPU得到的结果，例如平均值   
     * 哪些tensor/变量或者计算定义在设备的上下文？  
     * 哪些可以不用指定tf.device?   
     * 变量的重用  
- 会话的建立和管理  
- 在数据集上的迭代并行   

3. 多机多GPU  
- 建立ClusterSpec和Server：  
- 如果是PS模式：job分为ps和worker两类  
     * ps 任务和worker 任务需要完成不同的任务  
     * worker分为is_chief和 !is_chief，两者负责完成的任务不同   
     * 每个node上的任务数量，建议放置一个task，然后为每个GPU clone任务  
     * 如果是异步并行：采用节点内同步、节点间异步 
- I/O  
- 构建计算图  
- 数据的管理  
- 数据和计算到设备的映射  
     * Tensor和Operator到设备的映射： 采用工具slim？
- 节点内/间数据的merge操作   
- 会话的建立和管理： 采用较新的接口   
- Tensor和Operator到设备的映射： 采用工具slim？
- PS节点的数量：？    
     * 变量在PS上的分布（多ps节点情况）    
     * 
- 数据集上的迭代并行  


### 1.1 创建cluster和Server  

#### 1.1.1 ClusterSpec  
在定义集群时，利用词典定义需要用到的job name和对应的计算节点，实际是将工作名字映射到网络   地址列表，并会将该词典传递给tf.train.ClusterSpec  的构造函数。    

```
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})  

tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
```
问题：  
-  该定义方法导致TF程序不具有平台可移植性  
     * kubernetes无法解决该问题
     * Kubeflow看起来可以（需后续验证）    
- 如何指定PS运行在CPU上
     * 目前没有看到相关的配置方法？
     * 当机器资源比较充足时，PS设置到GPU上，可能性能更好？   



#### 1.1.2 配置server  
Sever属于Clusterspec，每个Clusterspec一般只有一个Server。  

- 指定在当前机器上创建服务器。   
```
tf.train.Server.create_local_server()  
```


- 在ClusterSpec上创建Server

```python
 # Get hosts for both ps and worker.
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, 
                              job_name = FLAGS.job_name,     
                              task_index = FLAGS.task_index) 

```


### 1.2 并行模式   
Tensorflow支持Paramerter+Worker模式和只有worker的模式。
- 无PS模式    
后者一般是通过all-reduce方法来实现参数的通信，例如百度PaddlePaddle支持
的all-reduce+MPI、英伟达的NCCL1和NCCL2（后者支持节点内和节点间）和Uber的Horovod方法，一般推荐用第三种方法。
- PS模式   
PS模式根据worker间更新训练参数的方式分为同步和异步方式。
    * 同步方式  
前者是指所有的worker每处理完一个batch，都需要等待其他    
所有worker完成当前batch的计算，然后将参数传递到PS，由PS计算出新的参数值后，再传递给各个worker，进行下一个batch   
的数据处理。   
     * 异步方式  
异步方式下，每个worker完成自己的计算后，就可以将新的参数传递到PS服务器，PS根据该计算结果重新更新参数后，将参数值   
传递给worker，然后该worker就会继续进行计算。     
总的来说，同步方式更容易收敛，但是编程较复杂，扩展性较差；异步方式编程简单，扩展性好。在收敛性，实际上两者差距不大，   
只是从理论上分析，同步方式更容易收敛。  

#### 1.2.1 计算图的复制

##### In-graph replication    
客户端会建立包含一个参数集合（）的独立tf.Graph，并且会建立模型计算密集部分的多个拷贝，每一个拷贝会被绑定到/job:worker的不同任务上。  
相当于是创建一个大图（包含多个拷贝），然后将这个大图的每个子图分配到不同的任务上执  行。

[5] 给出了in-graph的例子。   


##### Between-graph replication  
每个/job:worker中的任务都有独自的客户端创建，与worker的任务在同一个进程中（client和  任务是不同的线程？）。每个客户端会创建包含参数（绑定到/job:ps，利用  tf.train.replica_device_setter来将参数确定地映射到相同的任务上）的相似计算图、  模型中计算密集部分的独立拷贝，并且其绑定到/job:worker的局部任务上。

- 解释[1]   
     - 该方法每个client只会创建一个任务，每个任务都有自己的参数拷贝；   
     - 与1.4.1的区别是什么？1.4.1是一个大图，1.4.2是多个独立的图。   
     - between-graph实现同步训练比较复杂，但是扩展性比较好；
     - in-graph实现异步训练比较复杂，适合在较小的集群上；   




#### 1.2.1 PS：同步    

#### 1.2.2 PS：异步   

```python

```


### 1.3  I/O  

tf.data.Dataset
Feeding
QueueRunner
Preloaded data   

tf.data.TFRecordDataset(filename)


### 1.4 构建计算图  

- tf.train.Saver()
   保存和restore变量。   


- tf.train.MonitoredTrainingSession   
 会配置保存summaries，checkpoint相关信息的参数；   
 会创建Scaffold对象，该对象会具有tf.train.Saver成员；   
返回值是MonitoredSession对象。   

```pyhton
from tensorflow.python.training.monitored_session import MonitoredTrainingSession  
with tf.train.MonitoredTrainingSession() as mts_sess:

``` 
-
    * 不用显式调用saver.save()函数  


- mts_sess不能用作
    * 不能设置为默认session.
    * 不能传递给 saver.save.
    * it cannot be sent to tf.train.start_queue_runners. 

tf.train.MonitoredSession  


tf.train.SessionRunHook  

tf.train.Coordinator
tf.train.QueueRunner

tf.train.start_queue_runners



### 1.5 数据的管理  


### 1.6 数据和计算到设备的映射  
     * Tensor和Operator到设备的映射： 采用工具slim？
### 1.7 节点内/间数据的merge操作   
### 1.8 会话的建立和管理： 采用较新的接口   
### 1.9 Tensor和Operator到设备的映射： 采用工具slim？
### 1.10 PS节点的数量：？    
     * 变量在PS上的分布（多ps节点情况）    
     * 
### 1.11 数据集上的迭代并行  



### 1.2 定义模型到device的映射  


- 手工配置这些参数，非常复杂；  
- 用户可以试用Kubernetes来自动化管理；  


### 1.4 训练过程的并行  






#### 1.4.5 模型复制

- tf.train.replica_device_setter  

```python  
tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', 
                                merge_devices=True, cluster=None, ps_ops=None, ps_strategy=None)

tf.contrib.training.GreedyLoadBalancingStrategy  
```    


返回值是device funciton，作为With tf.device()的参数来用。默认情况下，只有tf.Variable才会被放到PS上，如果定义了多个PS，  
会按照round-robin模式在多个ps之间放置tf.Variables。tf.Variables会被放在PS上，其他数据会被放在worker上。  


### 1.5 配置GPU
#### 1.5.1 支持的设备  
```
"/cpu:0": The cpu of your machine
"/device:GPU:0": 第一个GPU
"/device:GPU:1": 第二个GPU
```
如果用户不指定设备，那么如果存在/device:GPU:0，那么TF会将操作符默认分配到GPU:0上。 所以，当用户手工指定设备时，一定要完整、   
全面的设计，未指定部分操作会被分配到GPU:0上  执行。    

### 1.5.2 设备上下文  
```
 with tf.device("/cpu:0"):
    a = tf.constant()
    b = tf.constant()
```

### 1.5.3 tf.Session的配置参数  

```
sess = tf.Session(
config = tf.ConfigProto(allow_soft_placement = true,
log_device_placement = FLAGS.log_device_placement,
                       ))
```


- 配置信息log
```
 sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

```

- 配置GPU存储分配  

```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 逐步按需分配内存
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #  限制分配的上限  
session = tf.Session(config=config, ...)
```

- 指定单个设备（多卡机器上）
如果用户指定，则系统默认使用id最小的设备。  


```
with tf.device('/device:GPU:2'):  # 指定第3个GPU

sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True,   #  有TF自动指定设备
      log_device_placement=True))
```

- 使用多个设备

```
for d in ['/device:GPU:d2', '/device:GPU:3']:
  with tf.device(d):  e
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
```

- tf.ConfigProto  
属于Proto定义，具体内容参考[3].  


- Runoptions[3]

- tf.GraphOptions 
有一个域用来指定  
```
optimizer_options = tf.OptimizerOptions()
```



- tf.OptimizerOptions  

```
def _session_config(self):
        """Creates the session config with t2t default parameters."""
        graph_options = tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                                opt_level=tf.OptimizerOptions.L1, 
                                do_function_inlining=False))

        if self._single_cpu_thread:
            config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                graph_options=graph_options,
                log_device_placement=False)
        else:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=0.95)
            config = tf.ConfigProto(
                allow_soft_placement=True,
                graph_options=graph_options,
                gpu_options=gpu_options,
                log_device_placement=False)
        return config 
```

## 1.5 同步训练
tf.train.SyncReplicasOptimizer  
tf.train.replica_device_setter  
- tf.moving_average_variables  
Returns all variables that maintain their moving averages.   

ExponentialMovingAverage 

tf.clip_by_global_norm[6]  

```python
opt = GradientDescentOptimizer(learning_rate=0.1)  
grads_and_vars = opt.compute_gradients(loss, <list of variables>)   
compute_gradients()
opt.apply_gradients(capped_grads_and_vars)     
```

## 1.6 异步训练

```python
tf.train.ExponentialMovingAverage
```

[7]给出了多机多卡训练的case，比较好的实践方案是每个node一个worker，然后在node内，每个GPU clone一份图，  
局部同步，全局异步，不需要每个GPU都与PS通信。   
[9] 给出了一个写分布式TF程序的例子，对细节讲解的比较清楚，但是没有给出源代码例子。  

## 2. 关键问题


- 1. 外部有tf.train.replica_device_setter函数修饰，内部的标量为什么还需要指定设备？ 

```python 
with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d"% FLAGS.task_index,
            cluster = cluster)):
```

- 2. 在异步分布式数据并行中，GPU之间的Compute Graph是否可以重用变量？  

- 3. 在设置变量重用时，一定要先定义VariableScope，否则，可能会影响到其他区域[10]    

# reference  
[1] https://clindatsci.com/blog/2017/5/31/distributed-tensorflow     
[2] https://www.tensorflow.org/deploy/distributed     
[3] https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/protobuf/config.proto    
[4] https://hackmd.io/s/HJxsUvOpg                                     
[5] https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py    
[6] https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm        
[7] https://github.com/tensorflow/models/issues/54             
[8] https://github.com/weixsong/LearnTF/blob/master/TensorFlow-Examples/examples/6_DistributedTF/mnist_dist.py       
[9] https://medium.com/clusterone/how-to-write-distributed-tensorflow-code-with-an-example-on-tensorport-70bf3306adcb       
[10] https://github.com/tensorflow/tensorflow/issues/6220  