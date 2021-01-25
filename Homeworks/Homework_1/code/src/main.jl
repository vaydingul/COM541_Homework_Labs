include("./NBClassifier.jl")
using .NBClassifier  #importing necessary module

# Constant variables for file directories and some parameters
TRAIN_SET_FOLDER_POS = "./../data/aclImdb/train/pos"
TRAIN_SET_FOLDER_NEG = "./../data/aclImdb/train/neg"
TEST_SET_FOLDER_POS =  "./../data/aclImdb/test/pos"
TEST_SET_FOLDER_NEG =  "./../data/aclImdb/test/neg"
POPPING_CRITERIA = 0

# Get all files in specified directory 
trainPosFiles = readdir(TRAIN_SET_FOLDER_POS, join=true) 
trainNegFiles = readdir(TRAIN_SET_FOLDER_NEG, join=true) 
testPosFiles = readdir(TEST_SET_FOLDER_POS, join=true) 
testNegFiles = readdir(TEST_SET_FOLDER_NEG, join=true) 


X_train = Array{String, 1}()
y_train = Array{Int8, 1}()
X_test = Array{String, 1}()
y_test = Array{Int8, 1}()


#In this "parallel" loop, all the data we need is extracted and pushed to the necessary variables.
#Input sentences are stored as String array, classes are stored as Int8 array.
for (folder, kind, class) in zip([trainPosFiles, trainNegFiles, testPosFiles, testNegFiles], [1, 1, 0, 0], [1, 0, 1, 0])

    for file in folder

        fileHandle = open(file, "r")
        review = read(fileHandle, String)
        
        kind == 1 ? push!(X_train, review) :  push!(X_test, review)
        kind == 1 ? push!(y_train, class) :  push!(y_test, class)

        close(fileHandle)
        
    end

end 


model = NB_NLP(X_train, y_train, 0.0) #Model construction
train(model) #Model training


y_pred = predict(model, X_test) #Prediction of all test data
accuracy = NBClassifier.calculateAccuracy(y_pred, y_test)#Calculate accuracy

println("The calculated accuracy is: ",accuracy)





