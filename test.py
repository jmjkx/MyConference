given1 = input() 
given1 = given1.replace('已知字符串：{', '')
given1 = given1.replace('}', '')
given1 = given1.split(',')
given2 = input().replace('给定字符串：','')

for string in given1:
    if set(given2) == set(string):
        print(string) 
