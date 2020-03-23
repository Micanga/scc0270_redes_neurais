dataset<-read.csv("../exec2.csv", header = TRUE)

signalsDataset<-dataset[,1:(ncol(dataset)-1)]
initialsWeights<-runif((ncol(signalsDataset)), -1.0, 1.0)
biasWeights<-runif(1, -1.0, 1.0)
eta <- (0.1)

threshold<-function(input){
  output<-ifelse(input<0.5, 0, 1)
  return (output)
}

sigmoide<-function(input){
    output <- 1/(1+exp(-0.5*input))
    return (output)
}

perceptron<-function(signals,weights, bWeight){
  SumWeight<-sum(signals*weights)+bWeight
  return (SumWeight)
}

Train<-TRUE

print("Initial weights:")
print(initialsWeights)
print(biasWeights)

print("Start train")
while (Train) {
  
  AccumulatedAbsoluteError<-0
  for(i in 1:nrow(signalsDataset)){
    output<-perceptron(signalsDataset[i,], initialsWeights, biasWeights) # Resposta do Perceptron
    
    ##################################
    ### Bloco para uso da função thrshold
#     response<-(threshold(output)+1) # Resposta da função da ativação e o ajuste (+1) do resultado da função de ativação
    ##################################
      
      
    ##################################
    ### Bloco para uso da função sigmoide
    response<-sigmoide(output)
    if(response<=0.5){
           response<-1
       }else{
           response<-2
    }
    ##################################
    
    #print(paste("Classificação:", response))
    
    if(response != dataset[i,ncol(dataset)]){
      initialsWeights<-initialsWeights-(eta*signalsDataset[i,]*(response-dataset[i,ncol(dataset)]))
      biasWeights<-biasWeights-(eta*(response-dataset[i,ncol(dataset)]))
    }
    
    AccumulatedAbsoluteError<-AccumulatedAbsoluteError+abs(response-dataset[i,ncol(dataset)])
  }
  if(AccumulatedAbsoluteError == 0){
    Train<-FALSE
  }
  print(AccumulatedAbsoluteError)
}
print("Final weights:")
print(initialsWeights)
print(biasWeights)
