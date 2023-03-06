# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


num1 = 20
num2 = 30
print('num1=%d, num=%d' % (num1, num2))
print('num1={}, num2={}'.format(num1, num2))
num = 3.141526
print('%0.2f' %num)      # 保留两位小数
print('%10.1f' %num)     # 占10个空格, 右对齐
print('%-10.2f' %num)    # 占10个空格, 左对齐
str1 = '小沐'
str2 = '小冷'
print('{0}{1}{1}{0}'.format(str1, str2))   # 从0开始对映变量值
 # 0：3.14    |  *：填充多出的空格（便于观察）    | 10：3.14占的位置大小
print('--{0:*<10}--{0:*^10}--{0:*>10}--{0:*=7}'.format(3.14))

num = 3.141526
print(F'保留两位小数：{num:.2f}')
name = "小沐"
age = 20
print(f'我叫{name}，'
      f'今年{age}岁了。')    #  多行用法


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
