import collections
import copy

test_dict = {'10D_a': {"(-6.10955, 106.88212)": [
    ["a", -6.109496475659279, 106.8821254367165],
    ["b", -6.109496274231438, 106.8821393207384]],
    "(-6.13308, 106.89311)": [
        ["c", -6.133101112368332, 106.8931311335058]]},
    "2A_A1": {"(-6.18323, 106.90912)": [
        ["a", -6.183359376409075, 106.9092034982567],
        ["e", -6.183360241023678, 106.9091902866807]],
        "(-6.1308, 106.81311)": [
            ["a", -6.133101112368332, 106.8931311335058],
            ["f", -6.133101112368332, 106.8931311335058]]}}

test = [p[0] for r in test_dict.values() for c in r.values() for p in c]
test_dupl = [item for item, count in collections.Counter(test).items() if count > 1]
test_uniq = list(set(test))
test_copy = copy.deepcopy(test_dict)

copy = []
for r, p in test_copy.items():
    for c, i in p.items():
        ids = []
        ids_d = []
        for d in i:
            print(d)
            if d[0] not in copy:
                if d[0] in test_copy:
                    copy.append(d[0])
                    ids_d.append(d[0])
                    ids.append(d)

    p[c] = ids
print(test_copy)
print([p[0] for r in test_copy.values() for c in r.values() for p in c])
