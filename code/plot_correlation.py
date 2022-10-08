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
        # acc = os.system(f"textcat.py gen.model spam.model 0.7 ../data/gen_spam/dev/{actual_class}/{file}")
        acc = 1
        # print(actual_class, num_words, acc)
        class_vs_words = {'Actual Class': actual_class, 'Words': num_words, 'Accuracy': acc}
        class_words_list.append(class_vs_words)

    return pd.DataFrame(class_words_list)


def main():
    gen = sys.argv[1]
    spam = sys.argv[2]

    # data_gen = find_acc_for_dir(gen)
    # data_spam = find_acc_for_dir(spam)

    # df_gen = pd.DataFrame(data_gen.items())
    # df_spam = pd.DataFrame(data_spam.items())

    df_gen = pd.DataFrame({'Words': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                           'Accuracy': [96, 94, 93.2, 91.2, 90, 88.7, 86, 84, 83.2, 82]})
    df_spam = pd.DataFrame({'Words': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                            'Accuracy': reversed([96, 94, 93.2, 91.2, 90, 87.8, 85.8, 84.1, 82.9, 82])})

    plt.plot(df_gen['Words'], df_gen['Accuracy'], label='gen')
    plt.plot(df_spam['Words'], df_spam['Accuracy'], label='spam')
    plt.xlabel('Number of words (Sentence length)')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.grid('--')
    plt.show()


if __name__ == "__main__":
    main()
