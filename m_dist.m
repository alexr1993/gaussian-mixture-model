function m_distance = m_dist( x, y, covariance )

m_distance = (x - y)' * inv(covariance) * (x - y);

return;