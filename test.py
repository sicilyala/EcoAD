import torch
from torch.backends import cudnn

# import xlrd

print(torch.__version__)
print(torch.cuda.is_available())

# 验证cuDNN安装
print(cudnn.is_available())  # 返回True说明已经安装了cuDNN


class Father:
    year = 1999

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.month = 23
        print("father month: %d" % self.month)
        Father.print_age()
        self.print_age()

    def get_name(self):
        return self.name

    def get_age(self):
        print(self.age+self.year)

    @classmethod
    def print_age(cls):
        print("father's class method % d" % Father.year)
        # print("father's class method % d" % cls.age)


class Son(Father):
    def __int__(self, name, age):

        self.month = 34
        print("son month: %d" % self.month)

        super().__init__(name, age)

    def print_age(self):
        print("Son is printing age. %d" % self.month)


if __name__ == '__main__':
    son = Son('son', 9)
    print(son.month)
    # son.print_age()
    # father = Father('father', 90)
    # father.get_age()
    # father.print_age()
    # Father.print_age()



