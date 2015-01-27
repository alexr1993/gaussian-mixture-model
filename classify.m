function class = classify(x, GMM, priors)

prob = zeros(4,1);

for i = 1:4
   mu = GMM.mu(i,:)';
   sigma = GMM.Sigma(:,:,i);
   m_distance = m_dist(x, mu, sigma);
   prob(i) = exp(-0.5 * m_distance) / sqrt( det(sigma) * (2*pi) ^ (GMM.NDimensions));
   prob(i) = priors(i) * prob(i);
end


[~, class] = max(prob);

return;