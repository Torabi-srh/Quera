def beautifulNumber(n):
    return all(digit not in ["0", "1"] for digit in str(n))


def makeDigits(n):
    product = 1
    for digit in str(n):
        product *= int(digit)
    return product


def countRange(l, r):
    count = 0
    for n in range(2, 10**6 + 1):
        if beautifulNumber(n):
            prod = makeDigits(n)
            if l <= prod <= r:
                count += 1
    return count

i = int(input())
for j in range(i):
    l, r = map (int, input().split())
    count = countRange(l, r)
    print(count % (10**9 + 7))