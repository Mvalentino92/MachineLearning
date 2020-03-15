using DataFrames, CSV, StatsBase, MLPreprocessing

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
    subpopulation::Vector{Organism}
    turns::Real
    converged::Bool
end

# The Habitat, consists of different species to be evolved towards
# Evolution is somewhat governed by an evolution rate
# preditions: The vector of aggregated predictions of each run
# subsize: The percent of the populations you wish to use for sub populations
# submax: Max size of sub population regardless of desired percentage
mutable struct Habitat
    species::Vector{Species}
    subsize::Real
    submax::Int
    evolutionrate::Real
end

# Function to prepare Data and return as a Habitat
# df: A data frame, with only "training" instances.
# classifyby: The labels, what you wish to classify this data by.
# evolutionrate: The constant in front of randn (normal distribution sampling)
function createhabitat(df::DataFrame,classifyby::String,evolutionrate::Real,
		      subsize::Real,submax::Int)

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
	species[i] = Species(string(classes[i]),population,[],Inf,false)
    end

    # Create and return Habitat
    habitat = Habitat(species,subsize,submax,evolutionrate)
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
    children = deepcopy(species.subpopulation)
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
    append!(species.subpopulation,children)

    # Sort by fitness
    sort!(species.subpopulation,by=(x -> x.fitness))

    # Grab the first half
    species.subpopulation = species.subpopulation[1:l]
end

# Check for convergence
function hasconverged(species::Species,tol::Real)
    val = sum(map(x -> x.fitness,species.subpopulation))/length(species.subpopulation)
    return val < tol
end

# Preps the species for next round
function prepspecies(habitat::Habitat,target::Vector,numspecies::Int,numfeatures::Int)

	# For every species in the habitat
        for ids = 1:numspecies

	    # Reset attributes
	    habitat.species[ids].turns = 0
	    habitat.species[ids].converged = false

	    # Create subpopulation
	    subdesired = trunc(Int64,habitat.subsize*length(habitat.species[ids].population))
	    subsize = subdesired < habitat.submax ? subdesired : habitat.submax
	    subindices = StatsBase.sample(1:length(habitat.species[ids].population),subsize,replace=true)
	    habitat.species[ids].subpopulation = habitat.species[ids].population[subindices]

	    # Calculate fitness for subpopulation
            for ido = 1:length(habitat.species[ids].subpopulation)
                organism = habitat.species[ids].subpopulation[ido]
                organism.fitness = sum((organism.features .- target).^2)/numfeatures
            end
        end
end

# Gets naive predictions
function getnaive(species::Vector,targets::Matrix)
    scores = zeros(length(species))
    m,n = size(targets)
    predictions = Vector{String}(undef,m)
    for i = 1:m
        target = targets[i,:]
        for j = 1:length(species)
            scores[j] = sum(map(x -> sum((x.features .- target).^2)/n,species[j].population))/length(species[j].population)
        end
        predictions[i] = species[findmin(scores)[2]].name
    end
    return predictions
end

# The main evolution function
# df: DataFrame with all instances and features/labels
# classifyby: The class to predict and classify by
# samplesize: Size of test data
# evolutionrate: Constant in front of randn
# maxiter: Stopping criteria
# tol: Tolerance for achieving convergence
function EA(df::DataFrame,classifyby::String;
            samplesize::Real=0.2,
            evolutionrate::Real=0.02,maxiter::Int=10000,
            tol::Real=1e-1/2,subsize::Real=0.65,submax::Int=100,
	        ensemblesize::Int=5)

    # Get train and test data
    samplesize = trunc(Int64,size(df)[1]*samplesize)
    train,test = splitdata(df,samplesize)

    # Create the habitat with the training data
    habitat = createhabitat(train,classifyby,evolutionrate,subsize,submax)

    # Split the test data into features and labels
    testlabels = test[Symbol(classifyby)]
    testfeatures = convert(Matrix,select(test,Not([Symbol(classifyby)])))

    # Get important info
    numspecies = length(habitat.species)
    numfeatures = length(testfeatures[1,:])

    # Predictions
    predictions = Vector{String}(undef,length(testlabels))

    # Get naive predictions!
    naive = getnaive(habitat.species,testfeatures)

    # For every instace in the test data
    for row = 1:size(testfeatures)[1]

        # Create target vector
        target = testfeatures[row,:]

	# Create ensembleturns
	ensemblescore = zeros(numspecies)

	# Ensemble number of iterations
	for en in 1:ensemblesize

		# Prep the species of the habitat
		prepspecies(habitat,target,numspecies,numfeatures)

		# While every species has not converged, or under max iter
		iter = 0
		while iter < maxiter && !reduce((x,y) -> x && y.converged, habitat.species,init=true)

		    # For every species in the habitat
		    for ids = 1:numspecies

			# If this species has not converged
			if !habitat.species[ids].converged

			    # Evolve this species
			    evolve(habitat.species[ids],target,habitat.evolutionrate)

			    # If the species has now convered, mark it has converged and mark timestamp
			    if hasconverged(habitat.species[ids],tol)
				habitat.species[ids].converged = true
				habitat.species[ids].turns = iter
			    end
			end
		    end

		    # Increment iter, and repeat
		    iter += 1
		end

		# Grab the probability that for each species and add it to ensemblescore
		turns = map(x -> x.turns,habitat.species)
		turns = minimum(turns)./turns
		ensemblescore += turns
	end

	# The highest ensemble score, will be the prediction
	mxdex = findmax(ensemblescore)[2]
	predictions[row] = habitat.species[mxdex].name
    end

    # Returns predictions and labels
    return testlabels,predictions,naive
end
