load('Product_Product.mat')
train_data.train_features = features';
train_data.train_labels = labels;
load('Product_Realworld.mat')
test_data.test_features = features';
test_data.test_labels = labels;
clear features
clear labels