def merton_model(me, sigma_e, debt, rf, time_horizon, max_iterations=200, tolerance=1e-4):
    """
    Implement the Merton (1974) model to compute default probability with improved convergence.
    """
    # Ensure positive values
    me = me*1000 # convert to millions
    me = max(me, 1e-6)
    sigma_e = max(sigma_e, 1e-6)
    debt = max(debt, 1e-6)

    # Improved initial guess
    asset_value = me + debt
    asset_volatility = sigma_e * (me / asset_value)

    for i in range(max_iterations):
        d1 = (np.log(asset_value / debt) + (rf + 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
        d2 = d1 - (asset_volatility * np.sqrt(time_horizon))
        
        new_asset_value = (me + debt * np.exp(-rf * time_horizon) * norm.cdf(d2))/norm.cdf(d1)
        new_asset_volatility = sigma_e * me / (new_asset_value * norm.cdf(d1))
        
        # Prevent extreme values
        # new_asset_value = np.clip(new_asset_value, me, me + debt * 10)
        # new_asset_volatility = np.clip(new_asset_volatility, sigma_e * 0.1, sigma_e * 10)
        
        # Check for convergence using relative tolerance
        if (abs(new_asset_value - asset_value) < tolerance * asset_value and 
            abs(new_asset_volatility - asset_volatility) < tolerance * asset_volatility):
            break
        
        asset_value = new_asset_value
        asset_volatility = new_asset_volatility
    
    # Compute final values even if max iterations reached
    d2 = (np.log(asset_value / debt) + (rf - 0.5 * asset_volatility**2) * time_horizon) / (asset_volatility * np.sqrt(time_horizon))
    distance_to_default = d2
    default_probability = norm.cdf(-distance_to_default)

    # Return results along with convergence status
    converged = i < max_iterations - 1
    return default_probability, distance_to_default, converged

def compute_equity_volatility(group, window=12):
    group['sigma_e'] = group['RET'].rolling(window=window).std() * np.sqrt(12)
    # group['sigma_e'] = group['RET'].std() * np.sqrt(12)
    return group

def apply_merton(row):
    if pd.isna(row['sigma_e']):
        return pd.Series({'default_probability': np.nan, 'distance_to_default': np.nan, 'converged': False})
    default_probability, distance_to_default, converged = merton_model(
        row['me'],
        row['sigma_e'],
        row['ltq'],
        row['rf'],
        # row['atq'],
        time_horizon=40.56
    )
    return pd.Series({'default_probability': default_probability, 'distance_to_default': distance_to_default, 'converged': converged})

def compute_default_probabilities(df, progress_interval=1000):
    df_copy = df.copy()
    # df_copy = df_copy.reset_index()
    df_copy = df_copy.sort_values(['GVKEY', 'year_month'])

    # Compute equity volatility for each firm
    print("Computing equity volatility...")
    grouped = df_copy.groupby('GVKEY')
    df_copy = grouped.apply(compute_equity_volatility)

    print("Applying Merton model...")
    total_rows = len(df_copy)
    results = []
    non_converged = 0
    
    for count, (_, row) in enumerate(df_copy.iterrows(), 1):
        result = apply_merton(row)
        results.append(result)
        
        if not result['converged']:
            non_converged += 1
        
        if count % progress_interval == 0:
            conv_rate = (count - non_converged) / count * 100
            print(f"Processed {count} / {total_rows} observations ({count / total_rows * 100:.2f}%)")
            print(f"Convergence rate: {conv_rate:.2f}%")
    
    df_copy[['default_probability', 'distance_to_default', 'converged']] = pd.DataFrame(results, index=df_copy.index)

    print("Computation complete!")
    print(f"Overall convergence rate: {(total_rows - non_converged) / total_rows * 100:.2f}%")
    return df_copy