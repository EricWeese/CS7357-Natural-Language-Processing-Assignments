str = "I came in in the middle of this film so I had no idea about \
any credits or even its title till I looked it up here, where I see\
 that it has received a mixed reception by your commentators. I'm on\
 the positive side regarding this film but one thing really caught \
my attention as I watched: the beautiful and sensitive score written\
 in a Coplandesque Americana style. My surprise was great when I \
discovered the score to have been written by none other than John Williams himself."

numWords = len(str.split())
print(f'Number of tokens: {numWords}')

types = set(str.split())
numTypes = len(types)
print(f'Number of types: {numTypes}')
print(f'All types: {types}')