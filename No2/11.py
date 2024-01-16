with open('popular-names.txt', 'r') as file:
    content = file.read()

modified_content = content.replace('\t', ' ')

print(modified_content)

'''
sed 's/\t/ /g' popular-names.txt

's/\t/ /g' : \t(tab)をスペースに置換する。
gはグローバルマッチング指定。
(対象文字列全体ではなく、対象文字列内のすべてのマッチを対象とする)

tr '\t' ' ' < popular-names.txt

'\t' ' ' : \t(tab)をスペースに置換する。

expand -t 1 popular-names.txt

-t 1 : \t(tab)をスペース1つ分に置換する。
'''