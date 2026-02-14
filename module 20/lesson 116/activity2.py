#activity2/acp
score = 0

def q1():
    a1 = input("Question 1 - What do we call the first layer of this neural network?\nEnter your answer : ").lower()
    return 1 if a1 == 'input layer' else 0

def q2():
    a2 = int(input("Question 2 - How many number of neurons can be there in the first layer of this neural network?\nEnter your answer : "))
    return 1 if a2 == 8 else 0

def q3():
    a3 = int(input("Question 3 - How many hidden layers are present in this neural network?\nEnter your answer : "))
    return 1 if a3 == 3 else 0

def q4():
    a4 = input("Question 4 - There can be any number of neurons in the hidden layer of this neural network. True or False?\nEnter your answer : ").lower()
    return 1 if a4 == 'true' else 0

def q5():
    a5 = input("Question 5 - What do we call the last layer of this neural network?\nEnter your answer : ").lower()
    return 1 if a5 == 'output layer' else 0

def q6():
    a6 = int(input("Question 6 - How many number of neurons can be there in the last layer of this neural network?\nEnter your answer : "))
    return 1 if a6 == 1 else 0

score += q1()
score += q2()
score += q3()
score += q4()
score += q5()
score += q6()

print("Your score is :", score)
