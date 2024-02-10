def primeFactors(n):
    prime = []
    divisor = 2
    while n >= divisor * divisor:
        while n % divisor == 0:
            prime.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        prime.append(n)
    print(f"for {n} found {prime}")
    return sum(set(prime))


def sumNumbers(n):
    return sum(int(number) for number in str(n))


def D(x):
    return x + sumNumbers(x) + primeFactors(x)


n = int(input())
z = []
for i in range(n):
    z.append(int(input()))

for i in z:
    found_father = False
    for m in range(1, i):
        # print(f"D({D(m)}) = {c}")
        if i == D(m):
            found_father = True
            break
    if found_father:
        print("YES")
    else:
        print("NO")
