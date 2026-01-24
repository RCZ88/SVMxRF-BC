import string
import random
import time

letters = list(string.ascii_letters)
wordTarget = 'SHUT UR ASS UP NIGGER'
currentWord = ''
letterFound = ''
for i in range(len(wordTarget)):
    curLetter = ''
    letterIndex = 0
    if wordTarget[i] == ' ':
        letterFound+= ' '
        continue
    else:
        while curLetter != wordTarget[i]:
            word = ''
            for j in range(i, len(wordTarget)):
                if wordTarget[j] == ' ':
                    word += ' '
                else:
                    if j == i:
                        word += letters[letterIndex]
                        curLetter = letters[letterIndex]
                    else:
                        word += random.choice(letters)
            letterIndex += 1
            print(letterFound, end='')
            print(word)
            time.sleep(0.001)
        letterFound += wordTarget[i]
