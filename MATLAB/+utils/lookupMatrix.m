function M = lookupMatrix(t, t_arr, store)
    idx = interp1(t_arr, 1:length(t_arr), t, 'linear', 'extrap');
    idx = max(1, min(size(store,3), round(idx)));
    M   = store(:,:,idx);
end