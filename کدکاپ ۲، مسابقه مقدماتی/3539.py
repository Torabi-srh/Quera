number = input()
while len(number) > 1:
    t = 0
    for i in number:
        t += int(i)
    number = str(t)

print(number)
