import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def find_acc_for_dir(directory):
    class_words_list = []
    for file in os.listdir(directory):
        words = file.split('.')
        actual_class, num_words, _, _ = words
        num_words = int(num_words)
        acc = os.system(f"textcat.py gen.model spam.model 0.7 ../data/gen_spam/dev/{actual_class}/{file}")
        # print(actual_class, num_words, acc)
        class_vs_words = {'Actual Class': actual_class, 'Words': num_words, 'Accuracy': acc}
        class_words_list.append(class_vs_words)

    return pd.DataFrame(class_words_list)


def main():
    gen = sys.argv[1]
    spam = sys.argv[2]

    df_gen = find_acc_for_dir(gen)
    df_spam = find_acc_for_dir(spam)

    plt.plot(df_gen['Words'], df_gen['Accuracy'], label='gen')
    plt.plot(df_spam['Words'], df_spam['Accuracy'], label='spam')
    plt.xlabel('Number of words (Sentence length)')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.grid('--')
    plt.show()


if __name__ == "__main__":
    main()
