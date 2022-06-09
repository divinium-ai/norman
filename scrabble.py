
SCORES = {
    'a':1, 'b':3, 'c':3, 'd':2, 'e':1, 'f':4, 'g':2, 'h':4,
    'i':1, 'j':8, 'k':5, 'l':1, 'm':3, 'n':1, 'o':1, 'p':3, 
    'q':10, 'r':1, 's':1, 't':1, 'u':1, 'v':4, 'w':4, 'x':8, 'y':4, 'z':10
}

def score_word(word:str) -> int:
    # https://gist.github.com/amundo/894107/369bdbef4a5268283478cef6ffe5b05dbb195e75
    return sum([SCORES[char] for char in word.lower().strip()])

if __name__ == "__main__":
    print(score_word('viagra'))