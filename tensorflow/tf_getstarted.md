Tensorflow user guide

# 1.  Basic knowledges

## 1.1 数据
 
### 1.1.1 Tensor   
在TF程序中，所有的数据都是通过张量的形式来表示，可以理解是多维数组，但在实现  上又不是多维数组，只是对TF中运算结果的运用，共有三个属性（name, shape,type）。  
所以， 程序中并不存在用户直接声明或者定义的张量，而是对程序中数据(计算过程)的描述。  下面例子中, a, b, 和 result都是张量。  
```python
a = tf.constant([1.0, 2.0], name = "a") 
b = tf.constant([2.0, 3.0], name = "b")
result = a + b
print (a)
print (b)
print (result) 
```
执行上述代码，其输出是   
```python
Tensor("a:0", shape=(2,), dtype=float32)
Tensor("b:0", shape=(2,), dtype=float32)
Tensor("add:0", shape=(2,), dtype=float32)
```

### 1.1.2 Constant  

### 1.1.3 Variable  
用来表示可以被程序共享和永久保存的状态,共享是指能够在单个sess.run上下文之  外，仍然能够被访问。  
- Placeholder  

- tf.get_variable   

## 1.2 Scope   


- Namescope  

- Variablescope


## Contruct graph

### feed_dict

### tf.data

## train

## valid

## test

## input


## File I/O

## Command Line process

## Checkpoint 


