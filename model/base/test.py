import string
letters = string.ascii_letters + "+"
def cur_sentence_by_word(sentence):
    result = []
    temp = ""
    for word in sentence:
        if word in letters:
            temp += word
        else:
            if temp != "":
                result.append(temp.lower())
                temp = ""
            if word != ' ':result.append(word.strip())
    if temp != "":result.append(temp)
    return result

if __name__ == '__main__':
    # sentence = "python 和 C++ 哪个用起来比较舒服haha"
    sentence = 'the sky is blue'
    print(cur_sentence_by_word(sentence))