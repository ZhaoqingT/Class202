from textgenrnn import textgenrnn
from os import listdir
from os.path import isfile, join
import csv
import os

# textgen = textgenrnn()
# res = textgen.generate(return_as_list=True)

# print("result:", res)

# contents = []

# with open('tweets.csv', encoding='utf-8', errors='replace') as csvfile:
#     spamreader = csv.reader(csvfile)
#     for row in spamreader:
#         contents.append(row)

# res = []
# for content in contents:
#     if '�' not in content[0]:
#         # print(content)
#         res.append(content)
# res = [content for content in contents if '�' not in content]

# print(len(contents))
# print(res)

# print(len(content))
# test = content[52542]
# string = test[0]
# for char in string:
#     if char == '�':
#         print(1)
# print(content[52542])

def train():
    textgen = textgenrnn()
    textgen.train_from_file('myfile.txt', num_epochs=1)
    textgen.save(weights_path="test.hdf5")

def generateText():
    textgen = textgenrnn()
    res = textgen.generate(return_as_list=True)

    print("result:", res)

def generateTweets():
    textgen = textgenrnn('test.hdf5')
    res = textgen.generate(return_as_list=True)

    print("result:", res)

if __name__ == '__main__':
    # contents = []

    # with open('tweets.csv', encoding='utf-8', errors='replace') as csvfile:
    #     spamreader = csv.reader(csvfile)
    #     for row in spamreader:
    #         contents.append(row)

    # res = []
    # for content in contents:
    #     if '�' not in content[0]:
    #         res.append(content)

    # print(res[3])
    
    # file1 = open("myfile.txt","w") 
    # for line in res:
    #     file1.write(line[0])
    #     file1.write('\n')
    # file1.close()
    # L = ["This is Delhi \n","This is Paris \n","This is London \n"]  
    
    # # \n is placed to indicate EOL (End of Line) 
    # file1.write("Hello \n") 
    # file1.writelines(L) 
    # file1.close()

    # textgen = textgenrnn()
    # textgen.train_from_file('myfile.txt', num_epochs=1)
    # textgen.save(weights_path="test.hdf5")
    # res = textgen.generate(return_as_list=True)

    # generateText()
    for i in range(10):
        generateTweets()