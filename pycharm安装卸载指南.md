# 1、安装  
使用工具箱进行安装  
##### https://www.jetbrains.com/toolbox/app/ 
下载jetbrains-Toolbox App.  
(1)建议解压到/opt文件夹下(官方建议，不放在/opt下似乎也能运行)，用命令：   
##### sudo tar -xzf jetbrains-toolbox-1.14.5179.tar.gz -C /opt  
(2)到/opt文件夹下，进入jetbrains-toolbox-1.14.5179文件夹，双击“jetbrains-toolbox” 运行该工具箱,或用命令：
##### sudo ./jetbrains-toolbox (在/opt/jetbrains-toolbox-1.14.5179/文件夹中)  
(3)在工具箱的选项中，选择需要安装的程序 pycharm 进行安装.  
# 2、 卸载(以2018.1.4为例)
### 删除文件也可以直接使用 “磁盘使用情况分析器”，找到相关文件删除。
(1)删除解压缩目录
可以找到类似 pycharm-community-2018.1.4 文件夹直接删除，位置根据自己安装(解压)位置寻找。参考命令：
##### sudo rm -r /opt/pycharm-community-2018.1.4/

(2)删除用于保存配置信息的隐藏目录(显示全部内容，包括隐藏目录：ls -a)

##### rm -r ~/.PyCharmCE2018.1

(3)删除快捷方式(jetbrains-pycharm*.desktop)  
可能在下列位置找到该文件：～/.local/share/applications/ ; 或/usr/share/applications/;  
##### ls 查看文件夹下的文件以及文件夹
##### sudo rm /usr/share/applications/jetbrains-pycharm-ce.desktop; 或 sudo rm jet* -i 逐条删除jet开头文件


