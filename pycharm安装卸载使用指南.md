# 1、安装  
使用工具箱进行安装  
#### https://www.jetbrains.com/toolbox/app/ 
下载jetbrains-Toolbox App.  
(1)建议解压到/opt文件夹下(官方建议，不放在/opt下似乎也能运行)，用命令：   
#### sudo tar -xzf jetbrains-toolbox-1.14.5179.tar.gz -C /opt  # 版本不同文件名要修改
(2)到/opt文件夹下，进入jetbrains-toolbox-1.14.5179文件夹，双击“jetbrains-toolbox” 运行该工具箱,或用命令：
#### sudo ./jetbrains-toolbox (在/opt/jetbrains-toolbox-1.14.5179/文件夹中)  
(3)在工具箱的选项中，选择需要安装的程序 pycharm 进行安装.  
# 2、 卸载
### 删除文件也可以直接使用 “磁盘使用情况分析器”，找到相关文件删除。
(1)删除解压缩目录
可以找到类似 pycharm-community-2018.1.4 文件夹直接删除，位置根据自己安装(解压)位置寻找。参考命令：
#### sudo rm -r /opt/pycharm-community-2018.1.4/

(2)删除用于保存配置信息的隐藏目录(显示全部内容，包括隐藏目录：ls -a)

#### rm -r ~/.PyCharmCE2018.1

(3)删除快捷方式(jetbrains-pycharm*.desktop)  
可能在下列位置找到该文件：～/.local/share/applications/ ; 或/usr/share/applications/;  
#### ls (查看文件夹下的文件以及文件夹)
#### sudo rm /usr/share/applications/jetbrains-pycharm-ce.desktop; 或 sudo rm jet* -i 逐条删除jet开头文件

# 3、使用
## (1)实用技巧1
1.设置断点
在行号右边点鼠标左键添加断点，再点取消断点。在断点处右键有菜单，待研究。
2.选择整行
点击行号选中整行。
3.按列选择
选择某一行的若干列，按住鼠标左键不松，按Alt键，鼠标上下移动可以选择不同行的若干列。(有的不行，不知是安装的问题或是设置问题)

## (2)实用技巧2
0 快速查找文件
开发项目时，文件数量越来越庞大，有时要在不同的文件之间来回切换，如果还是从左侧工程目录中按层级去查找的话，效率非常低效，通常，我们要用的都是最近查看过或编辑的文件，用快捷 Ctrl + E 可打开最近访问过的文件或者用 Ctrl+Shift+E打开最近编辑过的我文件。

从Tab页逐个地扫描也不快，如果你有强迫症不想显示Tab页的话可以在 Settings 中将 Tabs 设置为 None，直接使用快捷键来打开最近文件来提高效率。

1. 万能搜索

如果要评选Pycharm中最实用的快捷键，非 Double Shift 莫属，连续按两下 Shitf 键可以搜索文件名、类名、方法名，还可以搜索目录名，搜索目录的技巧是在在关键字前面加斜杠/。

如果你要全局项目范围内搜索文件里面的关键字，那么就需要使用 Ctrl + Shfit + F，或者 Ctrl + Shfit + R全局替换。
2. 历史粘贴版

如果你是Mac用户，一定熟悉 Alfred， Alfred是一款历史粘贴板神器，它缓存了过去一段时间的复制的内容，在 P月charm 中可通过 Ctrl + Shift + V 可访问历史粘贴板。

3. 分割窗口

在大屏显示器上写代码倍儿爽，很多时候我们在两个文件中来回的切换，这时把屏幕切割成两半就无需来回切换了，效率大大提高。Pycharm的默认配置没有设置分割的快捷键，你可以在Settings中的Keymap自定义快捷键。

不仅支持纵向分隔，还可以横向分隔

4. 智能提示

智能提示是 IDE 的标配功能，Pycharm 默认有自动提示功能，但是还不够智能，比如要使用还没有引入到模块则没法自动提示了，使用 Alt + Enter 智能提示你选择合适的操作。

5. 任意位置换行

无论你的光标处在何位置，你都可以通过快捷键 Shfit + Enter 另起一行，这样无需把光标移到末尾去操作。
————————————————  
版权声明：本文为CSDN博主「liu志军」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/lantian_123/article/details/78245514
