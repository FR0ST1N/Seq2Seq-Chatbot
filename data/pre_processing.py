
#Read from the input file
input_file = open('triggertest.txt', 'rw')

lines = []
for line in input_file:
    lines.append(line.lower().replace('\n',''))
print (lines)

#Separate questions and answers
count = 1
questions = []
answers = []
for temp_line in lines:
    if count % 2 != 0:
        print('Question: ' + temp_line)
        questions.append(temp_line)
    else:
        print('Answer: ' + temp_line)
        answers.append(temp_line)
    count += 1


#Make the training encoder and decoder files
train_enc = open('train.enc','w')
train_dec = open('train.dec','w')

for temp_question in questions:
    train_enc.write(temp_question + '\n')
for temp_answer in answers:
    train_dec.write(temp_answer + '\n')

train_enc.close()
train_dec.close()
