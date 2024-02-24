"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Implementation of MOFS-BACO (Binary Ant Colony Optimization 
    with Self-learning for Multi Objective Feature Selection)
"""

from bas import Ant, BinaryAntSystem
from evaluation import evaluation
from evaluation import features
from ops import OPS
import pandas as pd
import numpy as np
import time


def MOFS_BACO(numFeatures, x, y, iterations=60, P=5, lambda_=0.01, m=20, ro=0.65,k=1,baco=False):
    """
        Function name : MOFS-BACO
        Arguments : 
            -- numFeatures : The total number of features in the dataset
            -- x : input features
            -- y : input types/classes
            -- iterations : Number of iterations (default = 200)
            -- P : time period for ops (default = 5)
            -- lambda_ : A variable to control the effect of #features in fitness function
            -- m : Number of ants
            -- ro : evaporation factor
            -- k : k in knn
    """
    
    # parameters printing
    print("List of all parameters : ")
    print(" {}".format("".join(["_" for i in range(40)])))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("Number of features",numFeatures))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("Number of iterations",iterations))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("Number of Ants",m))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("Evaporation factor",ro))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("Time period for OPS",P))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("Lambda",lambda_))
    print("| {:23} | {:12} |".format("",""))
    print("| {:23} | {:12} |".format("k for KNN",k))
    print("|{}|\n".format("".join(["_" for i in range(40)])))

    # intialization
    graph = BinaryAntSystem(numFeatures,m=m,ro=ro)
    Supd = [
        Ant(numFeatures),
        Ant(numFeatures),
        Ant(numFeatures)
    ]   # ib, rb and gb respectively

    count = 0
    prev = Supd[2]

    # loop start
    print("Loop starts ...")
    for i in range(iterations):
        # generate ants (new generation)
        graph.generateNewAnts()
        
        # construct solutions
        graph.traverse()

        # OPS periodic condition
        if (i+1) % P == 0:
        # if False: #change here for ops
            print("Iteration {} : In OPS".format(i+1))
            graph.population = OPS(graph.population, x, y, lambda_,k,baco=baco)        # add ops here
        
        else:
            # solution evaluation
            for j in range(graph.m):
                x_temp = features(x,graph.population[j].solution)
                # fitness function is f = accuracy / (1 + lambda * #features)
                graph.population[j].fitness, graph.population[j].accuracy = evaluation(x_temp,y,lambda_,k,baco=baco)

        best = Ant(numFeatures)
        for j in range(graph.m):
            # fitness >= best
            if graph.population[j].fitness >= best.fitness:
                best = graph.population[j]         
        
        Supd[0] = best #Iteration best
        if best.fitness > Supd[1].fitness and graph.re_init == 0: #If re-initialization not happened then compare ib with rb.
            Supd[1] = best
        elif graph.re_init == 1: #If re-initialization has happened then rb = ib and set re_init flag to zero again.
            Supd[1] = best
            graph.re_init = 0
        if best.fitness >= Supd[2].fitness:
            Supd[2] = best
        
        if prev.fitness == Supd[2].fitness:
            count += 1
        else:
            prev = Supd[2]
            count = 1
        
        if count == 10:
            break

        print("Iteration {} : Accuracy - {} and #features - {}; fitness - {}".format(i+1,round(Supd[2].accuracy*100,2),Supd[2].numFeaturesSelected,Supd[2].fitness))
        # calculate convergence factor
        graph.convergenceFactor()   # graph.cf

        # update pheromone values
        graph.updatePheromone(Supd)

    return Supd[2]  # global best solution




if __name__ == "__main__":
    # read dataset
    datasets = [
        "wine.csv",
        # "ionosphere.csv",
        # "movement_libras.csv",
        # "SCADI.csv",
        # "parkinsons.csv",
        # "sonar.csv",
        # "vehicle.csv",
        # "iris.csv",
        # "yeast.csv",
        # "glass.csv"
    ]
    
    for data in datasets:   
        # filename = "datasets/" + input("Enter the dataset name : ")
        print("\nDataset : {}\n".format(data))
        filename = "datasets/" + data
        dataset = pd.read_csv(filename)
        y = dataset.iloc[:,0].to_numpy()
        x = dataset.iloc[:,1:].to_numpy()

        # all the variable which require tuning
        n_iterations = 5  # number of iterations
        P = 5               # time period for ops
        lambda_ = 10 ** (-1*(len(str(len(x[0])))))      # tuning required
        m = 50              # number of ants
        ro = 0.049           # evaporation factor
        k = 1               # k in knn
        baco = True         # change if not comparing with baco and abaco

        # ros = np.arange(start=0.05,stop=1,step=0.05)
        # for ro in ros: 
        accuracy = []
        feature = []
        runtime = []
        sols = []
        total_time = 0
        num_runs = 1
        print()
        # run the algorithm
        for num in range(num_runs):
            # print("\n----------------- {} {} -----------------".format(data[:data.index('.')],num+1))
            t1 = time.time()
            best = MOFS_BACO(len(x[0]),x,y,iterations=n_iterations,P=P,lambda_=lambda_,m=m,ro=ro,k=k,baco=baco)
            fit, acc = best.fitness, best.accuracy
            t2 = time.time()
            total_time += (t2-t1)
            print("\n\nNumber of features selected = {} out of {} features for {} dataset".format(best.numFeaturesSelected,len(x[0]),data[:data.index('.')]))
            print("Accuracy = {}%\n\n".format(round(acc*100,2)))
            accuracy.append(round(acc*100,2))
            feature.append(best.numFeaturesSelected)
            sols.append(best.solution)
            runtime.append(t2-t1)
        total_time /= 10
        # print(f"{data} average time taken per run : {round(total_time,2)} seconds")
        results = pd.DataFrame([i+1 for i in range(num_runs)],columns=["S.No."])
        results['Accuracy'] = accuracy
        results['Features'] = feature
        results['Solution'] = sols
        print(results)
        print()
        print(results.mean(axis=0,numeric_only=True))
        # results['Run time'] = runtime
        # results.to_excel(f'results/new/without ops/{filename[9:]}.xlsx',index=False)