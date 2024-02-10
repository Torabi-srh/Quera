fing = int(input())
hand = int(input())
firs = int(input())
seco = int(input())

fins = (fing * hand)
numb = (firs + seco)
resu = numb % fins
if numb == 0:
    print(0)
elif resu == 0:
    print(fins)
else:
    print(resu)