gcd = lambda a,b : a if not b else gcd(b, a%b)
a, b = map(int, input().split())
gcdr = gcd(a,b)
lcm = int(a*b / gcdr)
print(f"{gcdr} {lcm}")