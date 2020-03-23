dataset<-read.csv("../exec1.csv", header = TRUE)

signalsDataset<-dataset[,1:(ncol(dataset)-1)]
initialsWeights<-runif((ncol(signalsDataset)), -1.0, 1.0)
biasWeights<-runif(1, -1.0, 1.0)
eta <- (0.1)

dataset[,ncol(dataset)]<-dataset[,ncol(dataset)]-1

linear<-function(input){
    if(input<=(-0.5)){
        output<-0
    }else if(input>=(0.5)){
        output<-1
    }else{
        output<-input+0.5
    }
    return (output)
}

linear2<-function(input){
    if(input<=(0)){
        output<-0
    }else if(input>=(1)){
        output<-1
    }else{
        output<-input
    }
    return (output)
}


Adaline<-function(signals,weights, bWeight){
  SumWeight<-sum(signals*weights)+bWeight
  return (SumWeight)
}

Train<-TRUE

print("Initial weights:")
print(initialsWeights)
print(biasWeights)

print("Start train")
while (Train) {
    classification<-c()
    AbsoluteError<-0
    AccumulatedError<-0
    for(i in 1:nrow(signalsDataset)){
        output<-Adaline(signalsDataset[i,], initialsWeights, biasWeights) # Resposta do Perceptron
        output<-linear2(output)# Resposta da função da ativação e o ajuste (+1) do resultado da função de ativação
#         print(response)
        response<-0
        if(output<=0.5){
            response<-0
        }else{
            response<-1
        }
        classification<-c(classification, response)
        if(response!=dataset[i,ncol(dataset)]){
            Err<-(output-dataset[i,ncol(dataset)])
            initialsWeights<-initialsWeights-(eta*Err*signalsDataset[i,])
            biasWeights<-biasWeights-(eta*Err)
        }
        AbsoluteError<-AbsoluteError+abs(response-dataset[i,ncol(dataset)])
        AccumulatedError<-AccumulatedError+abs((output-dataset[i,ncol(dataset)]))
    }
    if(AbsoluteError == 0){
        Train<-FALSE
    }
    print(paste("Erro absoluto de classificação:", AbsoluteError))
#     print(paste("Função de custo acumulado:", AccumulatedError))
}
print("Final weights:")
print(initialsWeights)
print(biasWeights)


# print(classification)