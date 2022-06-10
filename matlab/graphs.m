value_names = ["Fixed Acidity","Volatile Acidity","Citric Acid","Residual Sugar","Chlorides","Free Sulfur Dioxide","Total Sulfur Dioxide","Density","pH","Sulphates","Alcohol","Quality"];

red_wine = readmatrix("../data/winedata_red.csv");
[red_wine_c, red_wine_r] = size(red_wine);

%red_wine_mean = mean(red_wine);

%histogram(mean(red_wine))
red_wine_mean = mean(red_wine);
red_wine_var = var(red_wine) ./ (red_wine_mean).^2;

for i = 1:12
    subplot(4,3,i)
    plot(1:red_wine_c, red_wine(:,i), 'o')
    title(append(value_names(i)," (Mean = ", string(red_wine_mean(i)), ", Variance = ", string(red_wine_var(i)), ")"))
end

%plot(1:red_wine_c, red_wine(:,1), 'o')