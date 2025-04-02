# {key0:value0, key1:value1}
my_dict = {'a':'阿', 'tai':'太','ni':'你'}
# 遍历 value
for k in my_dict:
    print(my_dict[k])
# 遍历 key
for k in my_dict:
    print(k)
# 遍历字典
for k, v in my_dict.items():
    print(f"{k}, {v}")

