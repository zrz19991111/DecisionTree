def input_string():
    string = input()
    yes = 0
    yes1 = 0
    yes2 = 0
    yes3 = 0
    yes4 = 0
    yes5 = 0
    if len(string.strip()) == 0:
        print('false')
    if len(string.strip()) > 0:
        if len(string) % 2 == 1:
            print('false')
        if len(string) % 2 == 0:
            for n in range(int(len(string))):
                for i in range(len(string) - 2):
                    left_right1 = string[i] == '(' and string[i + 1] == ')'
                    left_right2 = string[i] == '[' and string[i + 1] == ']'
                    left_right3 = string[i] == '{' and string[i + 1] == '}'
                    left_right4 = string[i] == '<' and string[i + 1] == '>'
                    left_right5 = string[i] == '/' and string[i + 1] == "\\"
                    if left_right1 :
                        yes1 = 1
                        string = string[:i] + string[i + 2:]  # 字符串拼接
                        break
                    if left_right2:
                        yes2 =  1
                        string = string[:i] + string[i + 2:]  # 字符串拼接
                        break
                    if left_right3 :
                        yes3 = 1
                        string = string[:i] + string[i + 2:]  # 字符串拼接
                        break
                    if left_right4:
                        yes4 =  1
                        string = string[:i] + string[i + 2:]  # 字符串拼接
                        break
                    if left_right5:
                        yes5 =  1
                        string = string[:i] + string[i + 2:]  # 字符串拼接
                        break
            if len(string) > 2:
                print('false')
            if len(string) == 2:
                d = ['()', '[]', '{}', '<>', '/\\']
                    if string=='()' :
                        yes1 = 1
                    if string=='[]':
                        yes2 = 1
                    if string=='{}':
                        yes3 = 1
                    if string=='<>':
                        yes4 = 1
                    if string=='/\\':
                        yes5 = 1
                yes = yes1 + yes2 + yes3 + yes4 + yes5
                print("true " + str(yes))
                    if string not in d:
                        print('false')

input_string()




