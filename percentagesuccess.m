function percentage = percentagesuccess(conf)

% add diagonal elements of conf to get total success
correct = sum(diag(conf));

%add all elements of conf, subtract correct to get incorrect
total = sum(conf(:));

percentage = correct / total * 100;

