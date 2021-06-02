def input_string():
    string = input()
    yes = 0
    # 判断是否为奇数
    if len(string.strip()) > 0:
        if len(string) % 2 == 1:
            print( 'false')

        # 循环遍历字符串，存在()[]{}的剔除
        if len(string) % 2 == 0:
            for n in range(int(len(string))):
                for i in range(len(string) - 2):
                    left_right1 = string[i] == '(' and string[i + 1] == ')'
                    left_right2 = string[i] == '[' and string[i + 1] == ']'
                    left_right3 = string[i] == '{' and string[i + 1] == '}'
                    left_right4 = string[i] == '<' and string[i + 1] == '>'
                    left_right5 = string[i] == '/' and string[i + 1] == "\\"
                    if left_right1 or left_right2 or left_right3 or left_right4 or left_right5:
                        string = string[:i] + string[i + 2:]  # 字符串拼接
                        break
            if len(string) > 2:
                print('false')
            if len(string) == 2:
                d = ['()', '[]', '{}','<>','/\\']
                if string not in d:
                    print('false')
                else:
                    yes = yes+1
                    print("true " + str(yes))


input_string()