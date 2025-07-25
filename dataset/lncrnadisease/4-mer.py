Letter = ['A', 'T', 'C', 'G']
k_mer = []
for i in range(4):
    for j in range(4):
        for m in range(4):
            for n in range(4):
                newLetter = Letter[i] + Letter[j] + Letter[m] + Letter[n]
                k_mer.append(newLetter)
print(k_mer)

list_rna = []
list_lab = []
list_pro = []
file = open(r"seq.csv", 'r', encoding="utf8")
output_file = open(r'4_mer.csv', 'a', encoding='utf8')

max_len = 240
min_len = 400
count = 0

list_temp_rna = []

line_temp = file.readline()
while line_temp != '':
    rna = line_temp.strip('\n')
    str_temp = ''
    # len_count = len_count + len(rna) / 4
    # if int(len(rna) / 4) > max_len:
    #     max_len = int(len(rna) / 4)
    # if int(len(rna) / 4) < min_len:
    #     min_len = int(len(rna) / 4)
    for i in range(int(len(rna) / 4)):
        if i > max_len:
            break
        str_temp += (str(k_mer.index(rna[i * 4: (i + 1) * 4])) + ',')
    list_temp_rna.append(str_temp)
    line_temp = file.readline()
file.close()

for each_seq in list_temp_rna:
    count_len = each_seq.count(',')
    if count_len < max_len:
        for i in range(max_len - count_len + 1):
            each_seq += '256,'
            count = count + 1
    output_file.write(each_seq[:-1] + '\n')
output_file.close()
print(count)


