stones = {
    ("space", "blue"),
    ("mind", "yellow"),
    ("reality", "red"),
    ("power", "purple"),
    ("time", "green"),
    ("soul", "orange"),
}


stone_name = input()
print(next((color for name, color in stones if name == stone_name), None))