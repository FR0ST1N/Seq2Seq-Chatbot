#Read from input file
input_file = open('test1.txt', 'rw')
output_file = open('output.txt', 'w')

#Write the symbols to outpur files
symbols = ['_PAD', '_GO', '_EOS', '_UNK']
for symbol in symbols:
    output_file.write(str(symbol) + '\n')

#Read the contents from the file and split words
file_contents = input_file.read()
word_list = file_contents.split()

#write the unique words to the output file
unique_words = set(word_list)
for word in unique_words:
    output_file.write(str(word) + '\n')
    #print(str(word)+'\n')
input_file.close()
output_file.close()
print('Completed!')
