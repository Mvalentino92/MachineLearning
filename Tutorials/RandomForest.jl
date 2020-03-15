using DataFrames
using CSV
using ScikitLearn

# Import the features using CSV
features = DataFrame(CSV.File("/home/michaelvalentino92/temps.csv"))

# One hot encode
@sk_import preprocessing: LabelBinarizer
mapper = DataFrameMapper([(:week,LabelBinarizer())])

# Retrieve the transform in a matrix
onehot = fit_transform!(mapper,features)

# Grab the actual column as a vector
labels = Vector(features[:actual])

# Convert features a matrix after taking out week and actual
features = convert(Matrix,select(features,Not([:week,:actual])))

# Concatenate the onehot into this Matrix
features = hcat(features,onehot)

# Split test data
@sk_import model_selection: train_test_split
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.25)

# Import random forest
@sk_import ensemble: RandomForestRegressor

# Construct random forest
rf = RandomForestRegressor(n_estimators=100,n_jobs=-1)

ScikitLearn.fit!(rf,train_features,train_labels)

predictions = predict(rf,test_features)

errors = abs.((predictions - test_labels)./test_labels)

average_error = sum(errors)/length(errors)*100
println(average_error)
