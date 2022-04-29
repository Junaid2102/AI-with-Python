import numpy as np
import fnmatch
import os
file_count = 0
files_dict = {}
unique_word_set = set()
path = (r"C:\Users\PC\PycharmProjects\simple_IRS")
ext = ('.txt')
'''for files in os.listdir(path):
   if files.endswith(ext):
        files_dict = files'''
        #print(files_dict)
index = 0
for files in os.listdir(path):
    if files.endswith(ext):
        file_count += 1
        entry = files_dict.get(files, index)
        files_dict[files] = entry
        index += 1

print("Total Number of files are: ",file_count)
print("Dictionary of files", files_dict)

for files in os.listdir(path):
    if files.endswith(ext):
        new_path = f"{path}\{files}"
        with open(new_path, 'r') as f:
            text = f.read()
            text = text.lower()
            words = text.split()
            words = [word.strip('.,!;()[]') for word in words]
            words = [word.replace("'s", '') for word in words]
            for word in words:
                if word not in unique_word_set:
                    unique_word_set.add(word)

print("Unique words in files: ", unique_word_set)

#unique_word_dict = {}
#unique_word_dict = dict.fromkeys(unique_word_set)
unique_word_dict = dict(zip(unique_word_set, range(0, len(unique_word_set)+1)))

#print(type(unique_word_dict))
print("Dictionary of Unique Words: ", unique_word_dict)
col = len(unique_word_set)
term_doc_matrix = np.zeros((file_count, col))

print(term_doc_matrix)

for file, n_file in files_dict.items():
    new_file = open(new_path, "r")
    for word in new_file.read().split():
        term_doc_matrix[n_file][unique_word_dict[word.lower()]] = 1
    new_file.close()
print("Term Document Matrix after filling: \n", term_doc_matrix)

column_vector = np.zeros((len(unique_word_set), 1))
print("Column Vector Initially: \n", column_vector)

query = input("Write something for searching: ")
usr = np.zeros((len(files_dict), 1))
for var in query.split():
    for new_var , index in files_dict.items():
        new_file = open(new_var, 'r')
        for words in new_file.read().split():
            if (words.lower() == words):
                usr[index][0] += 1

max = np.argmax(usr)
print(usr)
print('\n Maximum in resultant is: ', np.max(usr))
print('\n Index of maximum rsultant is: ', max)

for new_var, index in files_dict.items():
    if(index == max):
        name = new_var.split('/')
        print("File named " + name[-1] + " has maximum value in resultant vector.")
        print("\n Contents of file: ")
        file = open(new_var, 'r')
        for line in file:
            print(line)
        break