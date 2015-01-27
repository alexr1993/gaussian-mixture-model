%N is size(X,2)
%mu is mean

% MU 

co= x*x'/N - mu*mu' % cov
[U,L] = eig(C) % evd

L = diag(L) % evalues
[L,i] = sort(L, "decsend"); % sort evalues
U=u(:,i); % sort evectors the same way 
M=2; 

energy = cumsum(L) / sum(L);
energy = energy < 0.99;

U = U(:, energy);
L = L(energy);

%

%project
y = U'*(x - repmat(mu,1,size(x,2)));