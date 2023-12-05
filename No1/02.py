police = "パトカー"
taxis = "タクシー"

result = "".join(s1 + s2 for s1, s2 in zip(police, taxis))
print(result)