from cs50 import *

user_input = get_string("Text: ")

def count_letters():
    letters = 0

    for letter in user_input:
        if letter.isalpha():
            letters += 1

    return letters

def count_words():
    words = 1

    for character in user_input:
        if character == ' ':
            words += 1
    return words


def count_sentences():
    sentences = 0

    for character in user_input:
        if character == '.' or character == '!' or character == '?':
            sentences += 1
    return sentences

total_letters = count_letters()
total_words = count_words()
total_sentences = count_sentences()

letters_average = (total_letters/total_words) * 100
sentences_average = (total_sentences/total_words) * 100



grade_level = round(0.0588 * letters_average - 0.296 * sentences_average - 15.8)

if grade_level < 1:
    print("Before Grade 1")
elif grade_level >= 16:
    print("Grade 16+")
else:
    print(f"Grade {grade_level}")

