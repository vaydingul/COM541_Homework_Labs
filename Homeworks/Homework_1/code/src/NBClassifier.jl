module NBClassifier

export NB_NLP, predict, train

mutable struct NB_NLP
#= 
    It is the main struct that stores all the related data for Naive Bayes Classifier ( specifically for NLP )

    X = Input data in the form of String array, where each string represents a whole review for our case.
    
    Y = Output data in the binary form. 1 represents positive review, 0 represents negative review for our case

    popCriteria = It is the criteria for popping out some words from vocabulary by looking at its frequency.

    vocabPosProb = It is the dictionary that stores unique words and their counts for positive class. Then, this variable
        starts to represent probabilities for a given word.

    vocabNegProb = It is the dictionary that stores unique words and their counts for negative class. Then, this variable
        starts to represent probabilities for a given word.

    classProb = This variable stores class probabilities

    Usage:

            nb_nlp = NB_NLP(X, Y, popCriteria)
            train(nb_nlp)
            predict(nb_nlp, array_of_sentences_or_sentence) =#
    X::Array{String,1}
    Y::Array{Int8,1}
    popCriteria::Float32
    vocabPosProb::Any
    vocabNegProb::Any
    classProb::Any

    # This constructor only take X, Y, and popCriteria as input. All other struct members can be thought as private member.
    function NB_NLP(X, Y, popCriteria; vocabPosProb=Dict{String,Float64}(), vocabNegProb=Dict{String,Float64}(), classProb=[0.0, 0.0])
        return new(X, Y, popCriteria, vocabPosProb, vocabNegProb, classProb)
    end
end                                



function tokenize(s::String)
# This function clears all the punctuation and unnecessary space in given sentence. 
# Also, it splits the sentence into word chunks
    s = replace(s, r"[^a-zA-Z\s:-]" => " ")
    s = replace(s, r"\s+" => " ")
    s = split(lowercase(s), " ")
    return s

end

function calculateWordProbabilityPerClass(model::NB_NLP)
# This function calculates probability for each word based on they are classified as either positive or negative

    # This parallel for loop is responsible for unique_word - count dictionary for each class (1 - 0)
    @time for (review, class) in zip(model.X, model.Y)

        review = tokenize(review)

        for word in review
            if class == 1

                haskey(model.vocabPosProb, word) ? model.vocabPosProb[word] += 1 : push!(model.vocabPosProb, word => 1)  

            else        

                haskey(model.vocabNegProb, word) ? model.vocabNegProb[word] += 1 : push!(model.vocabNegProb, word => 1)

            end            

        end

    end

    # Application of popping criteria
    for (key, value) in model.vocabPosProb
        value < model.popCriteria ? delete!(model.vocabPosProb, key) : nothing
    end

    # Application of popping criteria
    for (key, value) in model.vocabNegProb
        value < model.popCriteria ? delete!(model.vocabNegProb, key) : nothing
    end 

    # Extraction of some information from data
    vocabPosValueSum = sum(values(model.vocabPosProb))
    vocabNegValueSum = sum(values(model.vocabNegProb))
    vocabPosLength = length(model.vocabPosProb)
    vocabNegLength = length(model.vocabNegProb)

    # Calculation of unique word probability for each class based on maximum-likelihood and add-one procedure
    for (key, value) in model.vocabPosProb
        model.vocabPosProb[key] += 1.0
        model.vocabPosProb[key] /= vocabPosValueSum + vocabPosLength
    end

    # Calculation of unique word probability for each class based on maximum-likelihood and add-one procedure
    for (key, value) in model.vocabNegProb
        model.vocabNegProb[key] += 1.0
        model.vocabNegProb[key] /= vocabNegValueSum + vocabNegLength
    end

    # Construction of probability for a unknown word case
    model.vocabPosProb["<?>"] = 1 / (vocabPosValueSum + vocabPosLength)
    model.vocabNegProb["<?>"] = 1 / (vocabNegValueSum + vocabNegLength)


end

function calculateClassProbability(model::NB_NLP)

    # Calculation of class probabilities
    for class in model.Y

        model.classProb[class + 1] += 1

    end

    model.classProb ./= length(model.Y)

end


function train(model::NB_NLP)
# Main training routine
    calculateWordProbabilityPerClass(model)
    calculateClassProbability(model)
    
end



function predict(model::NB_NLP, ss::Array{String,1})
# Predictions are made based on Bayes Theorem. For each prediction, the joint
# probability of the given sentence should be calculated.

    predictedClasses = Array{Int8,1}()
    @time for s in ss
        predictedProbs = [0.0, 0.0]

        s = tokenize(s)

        # In probability calculations, 'log's are used to eliminate underflow error!
        for word in s

            haskey(model.vocabNegProb, word) ? predictedProbs[1] += log(10, model.vocabNegProb[word]) : predictedProbs[1] += log(10, model.vocabNegProb["<?>"])
            haskey(model.vocabPosProb, word) ? predictedProbs[2] += log(10, model.vocabPosProb[word]) : predictedProbs[2] += log(10, model.vocabPosProb["<?>"])

        end
    
        for k = 1:2
            predictedProbs[k] += log(10, model.classProb[k])
        end
        
        # The class having the higher probability is the winning class!
        predictedProbs[2] > predictedProbs[1] ? predictedClass = 1.0 : predictedClass = 0.0
        push!(predictedClasses, predictedClass)
    
    end
    return predictedClasses
end


function calculateAccuracy(y_pred::Array{Int8,1}, y_test::Array{Int8,1})
# This function calculates the accuracy based on both the prediction of the model
# and the real class.
    correct = 0;
    counter = 0;

    for (y_p, y_t) in zip(y_pred, y_test)
        
        y_p == y_t ? correct += 1 : nothing
        counter += 1
    
    end
    
    
    accuracy = (correct / counter) * 100
    return accuracy
end


end