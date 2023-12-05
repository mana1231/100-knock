def generate_ngram(sequence, n):
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = sequence[i:i + n]
        ngrams.append(''.join(ngram))
    return set(ngrams)

s1 = "paraparaparadise"
s2 = "paragraph"

X = generate_ngram(s1, 2)
Y = generate_ngram(s2, 2)

# 和集合
union_set = X.union(Y)
print("和集合:", union_set)

# 積集合
intersection_set = X.intersection(Y)
print("積集合:", intersection_set)

# 差集合
difference_set = X.difference(Y)
print("差集合 (X - Y):", difference_set)

# 'se'がXおよびYに含まれるかどうか
is_se_in_X = 'se' in X
is_se_in_Y = 'se' in Y

print("'se'がXに含まれるか:", is_se_in_X)
print("'se'がYに含まれるか:", is_se_in_Y)
