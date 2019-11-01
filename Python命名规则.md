# Python 变量命名规则——PyCharm默认命名规则
### 1. 普通变量/普通函数/模块名/程序包  
字母全小写，分割单词用`_` ，例如：batch_size  
### 2.类名  
每个单词首字母大写，例如：HattedDataset  
### 3. 常量/全局变量   
字母全大写，分割单词用`_` ，例如：COLOR_WRITE  
### 4.下划线开头  
字母小写，分割单词用`_`  
* 实例变量：单下划线开头   
* 私有变量: 双下划线开头    
* 专有变量：双下划线开头，双下划线结尾   
* 私有函数：双下划线开头
# 其它命名
### 1.文件夹名  
小驼峰式命名：第一个单词首字母小写，从第二个单词开始首字母大写，例如： myStn。  
适当结合下划线  
### 2.Python文件名  
采用普通变量命名方法，字母全小写，分割单词用`_` ，例如：hatted_dataset.py  
### 3.避免关键词冲突  
结尾加下划线   
### 4.常用缩写
* function  --->  fn  
* text      --->  txt  
* object    --->  obj  
* count     --->  cnt  
* number    --->  num  
# 下划线开头注意事项：  
* 单下划线开头的成员变量叫保护变量，只有包含它的类或子类对象能访问，外部不能访问，也不能用from module import 导入。    
* 双下划线开头的时私有成员，只有类对象自己能访问，子类都不能访问。  
* 双下划线开头，双下划线结尾的时大多是系统定义的名字。  
# 命名规则如图：
注：普通变量可采用下划线分割的命名方式，也可采用驼峰式命名  
![naming_ruls](https://github.com/lix3972/Python-Learning/blob/master/picture/varNamingRules/PythonVarNamingRules.png)  


参考：  https://www.cnblogs.com/liuyanhang/p/11099897.html;    https://www.cnblogs.com/zhangyafei/p/10429426.html   
