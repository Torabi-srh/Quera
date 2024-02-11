r = int(input())
number = ""
for i in range(r):
    rr = input()
    if len(number) < 1:
        number = rr
        continue
    number = list(reversed(number))
    rr = list(reversed(rr))
    more = 0
    max_length = max(len(number), len(rr))
    number += ['0'] * (max_length - len(number))
    rr += ['0'] * (max_length - len(rr))
    for j in range(max_length):
        n1 = int(rr[j]) if j < len(rr) else 0
        n = int(number[j]) if j < len(number) else 0
        n1 += n + more
        if n1 > 9:
            more = 1
            n1 -= 10
        else:
            more = 0
        number[j] = str(n1)

    if more > 0:
        number.append('1')
    number = ''.join(reversed(number))

print(number)