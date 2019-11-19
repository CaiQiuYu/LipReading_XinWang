
result = {}
for i in range(1, 14):
    file_name = 'final_submit' + str(i) + '.csv'
    with open(file_name, 'r', encoding='utf-8') as fw:
        for line in fw:
            word = line.strip().split(',')
            if word[0] in result.keys():
                if word[1] in result[word[0]].keys():
                    result[word[0]][word[1]] += 1
                else:
                    result[word[0]][word[1]] = 1

            else:
                result[word[0]] = {word[1]:1}
print(result)
with open('zero_merge_submit.csv', 'w', encoding='utf-8') as fw:
    for k, v in result.items():
        re = sorted(v.items(), key=lambda x: x[1], reverse=True)
        fw.write(k + ',')
        fw.write(re[0][0])
        fw.write('\n')
