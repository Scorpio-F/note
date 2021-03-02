# ADB常用命令

ADB，即 Android Debug Bridge，它是 Android 开发/测试人员不可替代的强大工具，也是 Android 设备玩家的好玩具。安卓调试桥 (Android Debug Bridge, adb)，是一种可以用来操作手机设备或模拟器的命令行工具。它存在于 sdk/platform-tools 目录下。虽然现在 Android Studio 已经将大部分 ADB 命令以图形化的形式实现了，但是了解一下还是有必要的。



## 查看当前连接设备

```shell
adb devices

```

**如果发现多个设备：**

```shell
adb -s 设备号 其他指令

```







## 获取手机分辨率

```shell
adb shell wm size

```



## 查看所有包名

```shell
adb shell pm list packages

```



## 查看当前启动中的应用信息(包名和启动名)

```shell
adb shell dumpsys window | findstr mCurrentFocus
# adb shell am monitor
# 打开app再使用命令即可

```



