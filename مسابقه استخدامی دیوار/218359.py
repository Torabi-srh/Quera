# دیوار من
# مسابقه استخدامی دیوار

users = []
advertises = []
userAdverties = []
favorites = []
tags = []


def register(username):
    if username in users:
        return "invalid username"
    else:
        users.append(username)
        return "registered successfully"


def addAdvertise(username, title, tagList=None):
    if username not in users:
        return "invalid username"
    if title in [ad[1] for ad in advertises]:
        return "invalid title"
    user_id = users.index(username)
    ad_id = len(advertises)
    advertises.append((ad_id, title))
    userAdverties.append((user_id, ad_id))
    if tagList:
        for tag in tagList:
            tags.append((ad_id, tag))
    return "posted successfully"


def remAdvertise(username, title):
    global tags, favorites
    if username not in users:
        return "invalid username"
    ad_ids = [ad[0] for ad in advertises if ad[1] == title]
    if not ad_ids:
        return "invalid title"
    ad_id = ad_ids[0]
    user_id = users.index(username)
    if (user_id, ad_id) not in userAdverties:
        return "access denied"
    advertises.remove((ad_id, title))
    userAdverties.remove((user_id, ad_id))
    tags = [(x, y) for x, y in tags if x != ad_id]
    favorites = [(x, y) for x, y in favorites if y != ad_id]
    return "removed successfully"


def listMyAdvertises(username, tag=None):
    if username not in users:
        return "invalid username"
    uid = users.index(username)
    user_ads = [ad_id for uid2, ad_id in userAdverties if uid == uid2]
    filtered_ads = []
    for ad_id in user_ads:
        ad_title = next((title for id_, title in advertises if id_ == ad_id), None)
        if tag:
            if any(t for aid, t in tags if aid == ad_id and t in tag):
                filtered_ads.append(ad_title)
        else:
            filtered_ads.append(ad_title)
    return " ".join(filtered_ads)


def addFavorite(username, title):
    if username not in users:
        return "invalid username"
    ad_ids = [ad[0] for ad in advertises if ad[1] == title]
    if not ad_ids:
        return "invalid title"
    user_id = users.index(username)
    ad_id = ad_ids[0]
    if (user_id, ad_id) in favorites:
        return "already favorite"
    favorites.append((user_id, ad_id))
    return "added successfully"


def remFavorite(username, title):
    if username not in users:
        return "invalid username"
    ad_ids = [ad[0] for ad in advertises if ad[1] == title]
    if not ad_ids:
        return "invalid title"
    user_id = users.index(username)
    ad_id = ad_ids[0]
    if (user_id, ad_id) not in favorites:
        return "already not favorite"
    favorites.remove((user_id, ad_id))
    return "removed successfully"


def listFavoriteAdvertises(username, tag=None):
    if username not in users:
        return "invalid username"
    uid = users.index(username)
    user_ads = [ad_id for uid2, ad_id in favorites if uid == uid2]
    filtered_fav = []
    for ad_id in user_ads:
        ad_title = next((title for id_, title in advertises if id_ == ad_id), None)
        if tag:
            if any(t for aid, t in tags if aid == ad_id and t in tag):
                filtered_fav.append(ad_title)
        else:
            filtered_fav.append(ad_title)
    return " ".join(filtered_fav)


row = int(input())
commands = []

for i in range(row):
    command = input().split()
    commands.append(command)

for c in commands:
    if c[0] == "register":
        print(register(c[1]))
    elif c[0] == "add_advertise":
        if len(c) >= 4:
            print(addAdvertise(c[1], c[2], c[3:]))
        else:
            print(addAdvertise(c[1], c[2], None))
    elif c[0] == "list_my_advertises":
        if len(c) >= 3:
            print(listMyAdvertises(c[1], c[2:]))
        else:
            print(listMyAdvertises(c[1], None))
    elif c[0] == "list_favorite_advertises":
        if len(c) >= 3:
            print(listFavoriteAdvertises(c[1], c[2:]))
        else:
            print(listFavoriteAdvertises(c[1], None))
    elif c[0] == "rem_advertise":
        print(remAdvertise(c[1], c[2]))
    elif c[0] == "add_favorite":
        print(addFavorite(c[1], c[2]))
    elif c[0] == "rem_favorite":
        print(remFavorite(c[1], c[2]))
