using DataFrames, CSV, StatsBase

# An Organism is an instance from the Data.
# Has features, and also a fitness, which is determined by the fitness function.
# Which is a measure of how far off this organism is from the target
mutable struct Organism
    features::Vector
    fitness::Real
end

# Species, are the classes to be classified.
# There is a population consisting entirely of all the same class
# And we also maintain an evolutionhistory (with respect to the target)
# And a number of turns, which is how long it took to evolve to target
mutable struct Species
    name::String
    population::Vector{Organism}
    evolutionhistory::Vector
    turns::Real
    converged::Bool
end

# The Habitat, consists of different species to be evolved towards
# Evolution is somewhat governed by an evolution rate
mutable struct Habitat
    species::Vector{Species}
    speciescopy::Vector{Species}
    evolutionrate::Real
end

# Function to prepare Data and return as a Habitat
# df: A data frame, with only "training" instances.
# classifyby: The labels, what you wish to classify this data by.
# evolutionrate: The constant in front of randn (normal distribution sampling)
function createhabitat(df::DataFrame,classifyby::String,evolutionrate::Real=1)

    # Get the unique classes from DataFrame
    classes = unique(df[Symbol(classifyby)])
    numclasses = length(classes)

    # Create the species Vector for initialization of Habitat
    species = Vector{Species}(undef,numclasses)

    # Begin to iterate each class, to population species
    for i = 1:numclasses
        # Filter out current class from DataFrame, dropping classification feature
        dfspecies = filter(x -> x[Symbol(classifyby)] == classes[i],df)
        dfspecies = select(dfspecies,Not([Symbol(classifyby)]))

        # Convert to a Matrix
        matspecies = convert(Matrix,dfspecies)

        # Iterate through rows of matrix, and create Organisms (default fitness of Inf)
        m,n = size(matspecies)
        population = Vector{Organism}(undef,m)
        for j = 1:m
            population[j] = Organism(matspecies[j,:],Inf)
        end

        # Create species (Default values for evolutionhistory and similarity)
        species[i] = Species(classes[i],population,[],Inf,false)
    end

    # Create and return Habitat
    habitat = Habitat(species,deepcopy(species),evolutionrate)
    return habitat
end

# Function to split data in "train" and test samples
# df: The data to split between training and test
# samplesize: The size of the test data
function splitdata(df::DataFrame,samplesize::Int)
    # Get indices to split on
    m,n = size(df)
    trainindices = collect(1:m)
    testindices = StatsBase.sample(trainindices,samplesize,replace=false)

    # Get the difference
    setdiff!(trainindices,testindices)

    # Get train and test
    train = df[trainindices,:]
    test = df[testindices,:]

    return train,test
end

function evolve(species::Species,target::Vector,evolutionrate::Real)
    # Clone population to make children
    numfeatures = length(target)
    children = deepcopy(species.population)
    l = length(children)

    # For every child
    for child = 1:l
        # Chance to mutate each feature of every child
        for feature = 1:numfeatures
            if rand() < 1/numfeatures
                children[child].features[feature] += evolutionrate*randn()
            end
        end
        # Calculate the fitness of this new child
        children[child].fitness = sum((children[child].features .- target).^2)/numfeatures
    end

    # Combine child with current population
    append!(species.population,children)

    # Sort by fitness
    sort!(species.population,by=(x -> x.fitness))

    # Grab the first half
    species.population = species.population[1:l]
end

# Check for convergence
function hasconverged(species::Species,tol::Real)
    val = sqrt(sum(map(x -> x.fitness,species.population)))
    return val < tol
end

# The main evolution function
# df: DataFrame with all instances and features/labels
# classifyby: The class to predict and classify by
# samplesize: Size of test data
# evolutionrate: Constant in front of randn
# maxiter: Stopping criteria
# tol: Tolerance for achieving convergence
function EA(df::DataFrame,classifyby::String;
            samplesize::Int=div(size(df)[1],5),
            evolutionrate::Real=1,maxiter::Int=10000,
            tol::Real=1e-2)

    # Get train and test data
    train,test = splitdata(df,samplesize)

    # Create the habitat with the training data
    habitat = createhabitat(train,classifyby,evolutionrate)

    # Split the test data into features and labels
    testlabels = test[Symbol(classifyby)]
    testfeatures = convert(Matrix,select(test,Not([Symbol(classifyby)])))

    # Get important info
    numspecies = length(habitat.species)
    numfeatures = length(testfeatures[1,:])

    # Predictions
    predictions = fill("",length(testlabels))

    # For every instace in the test data
    for row = 1:size(testfeatures)[1]

        # Create target vector
        target = testfeatures[row,:]

        # Update initial fitness of entire population in speciescopy
        for ids = 1:numspecies
            for ido = 1:length(habitat.speciescopy[ids].population)
                organism = habitat.speciescopy[ids].population[ido]
                organism.fitness = sum((organism.features .- target).^2)/numfeatures
            end
        end

        # While every species has not converged, or under max iter
        iter = 0
        while iter < maxiter && !reduce((x,y) -> x && y.converged, habitat.speciescopy,init=true)

            # For every species in the habitat
            for ids = 1:numspecies

                # If this species has not converged
                if !habitat.speciescopy[ids].converged

                    # Evolve this species
                    evolve(habitat.speciescopy[ids],target,habitat.evolutionrate)

                    # If the species has now convered, mark it has converged and mark timestamp
                    if hasconverged(habitat.speciescopy[ids],tol)
                        habitat.speciescopy[ids].converged = true
                        habitat.speciescopy[ids].turns = iter
                    end
                end
            end

            # Increment iter, and repeat
            iter += 1
        end

        # Grab the probability that for each species
        turns = map(x -> x.turns,habitat.speciescopy)
        turns = minimum(turns)./turns

        # Print the results
        println("Correct labeling of ",testlabels[row]," with predictions..")
        for k = 1:numspecies
            println(habitat.speciescopy[k].name,": ",turns[k])
            predictions[row] = turns[k] == 1 ? habitat.speciescopy[k].name : predictions[row]
        end
        println()

        # Deepcopy back from species to restart
        habitat.speciescopy = deepcopy(habitat.species)
    end

    # Returns predictions and labels
    return testlabels,predictions
end
