import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ExtractDataForTraining import compute_data_gait_period_training
from scipy.optimize import minimize
import joblib
import warnings

# Load dataset
dataset_file = "processed_data.h5"
x, y_ = compute_data_gait_period_training(dataset_file=dataset_file)

X = np.array(x)
y = np.array(y_)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Define the kernel search space with adjusted hyperparameter bounds
k1 = C(1.0, (1e-4, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e2))
k2 = C(1.0, (1e-4, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e2), nu=1.5)
k3 = C(1.0, (1e-4, 1e2)) * RationalQuadratic(length_scale=1.0, alpha=0.1)
k4 = C(1.0, (1e-4, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=3.0, periodicity_bounds=(1e-2, 1e2))
k5 = C(1.0, (1e-4, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e2))

# Dictionary to hold the kernels with adjusted hyperparameter bounds
param_grid = [
    {'kernel': [C(1.0, (1e-4, 1e2)) * RBF(length_scale_bounds=(1e-6, 1e2))]},
    {'kernel': [C(1.0, (1e-4, 1e2)) * Matern(length_scale_bounds=(1e-6, 1e2), nu=nu_value)]
     for nu_value in [0.5, 1.5, 2.5]},
    {'kernel': [C(1.0, (1e-4, 1e2)) * RationalQuadratic(length_scale_bounds=(1e-6, 1e2), alpha_bounds=(1e-4, 1e2))]},
    {'kernel': [C(1.0, (1e-4, 1e2)) * ExpSineSquared(length_scale_bounds=(1e-6, 1e2), periodicity_bounds=(1e-2, 1e2))]}
]


# Custom optimizer function with increased max iterations
# Custom optimizer function with increased max iterations
def custom_optimizer(obj_func, initial_theta, bounds):
    def objective_function(theta):
        value = obj_func(theta)
        # Ensure the function returns a scalar
        if isinstance(value, (list, np.ndarray)):
            value = value[0]
        elif np.isnan(value[0]) or np.isinf(value[0]):
            return np.inf
        return value[0]
    result = minimize(objective_function, initial_theta, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
    return result.x, result.fun


# Set up the cross-validation search with increased alpha
gp = GaussianProcessRegressor(n_restarts_optimizer=20, optimizer=custom_optimizer, alpha=1e-5)
grid_search = GridSearchCV(estimator=gp, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Train and evaluate
grid_search.fit(X_train_scaled, y_train)
best_gp = grid_search.best_estimator_

# Make predictions with the best model
y_pred, sigma = best_gp.predict(X_test_scaled, return_std=True)

# Get results and sort by MSE (note that we need to take the negative since GridSearchCV minimizes negative MSE)
results = grid_search.cv_results_
sorted_indices = np.argsort(results['mean_test_score'])

print("Models from best to worst:")
for i in sorted_indices:
    kernel = results['param_kernel'][i]
    mean_test_mse = -results['mean_test_score'][i]
    rmse = np.sqrt(mean_test_mse)
    print(f"Kernel: {kernel}, Mean Test RMSE: {rmse:.4f}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'\nBest Model Mean Squared Error in Test: {mse}')
print(f'Best Model RMSE in Test: {rmse}')
print(f'Best Kernel: {best_gp.kernel_}')

# Save the scaler and best model
joblib.dump(scaler, 'scaler_fait_period.pkl')
joblib.dump(best_gp, 'best_gp_model_gait_period.pkl')

# Generate points for plotting
X_plot_scaled = np.linspace(np.min(X_scaled, axis=0), np.max(X_scaled, axis=0), 1000).reshape(-1, X.shape[1])

# Ensure the plotting data is also scaled
y_pred_plot, sigma_plot = best_gp.predict(X_plot_scaled, return_std=True)

# Unstandardize the input data for plotting
X_plot = scaler.inverse_transform(X_plot_scaled)
X_train_unscaled = scaler.inverse_transform(X_train_scaled)
X_test_unscaled = scaler.inverse_transform(X_test_scaled)

# Plotting Age
plt.figure(1, figsize=(10, 6))

# Training data
plt.scatter(X_train_unscaled[:, 0], y_train, color='blue', label='Training data')

# Test data
plt.scatter(X_test_unscaled[:, 0], y_test, color='green', label='Test data')

# Mean prediction
plt.plot(X_plot[:, 0], y_pred_plot, color='red', label='Predicted mean')

# Confidence intervals
plt.fill_between(X_plot[:, 0],
                 y_pred_plot - 1.96 * sigma_plot,
                 y_pred_plot + 1.96 * sigma_plot,
                 alpha=0.2, color='red', label='95% confidence interval')

plt.title('Gaussian Process Regression')
plt.xlabel('Input feature (Age)')
plt.ylabel('Target (Gait Period in s)')
plt.savefig('gait_period_pred_model_age_rep3.png', dpi=300)

# Plotting Walking speed
plt.figure(2, figsize=(12, 8))

# Training data
plt.scatter(X_train_unscaled[:, 4], y_train, color='blue', label='Training data')

# Test data
plt.scatter(X_test_unscaled[:, 4], y_test, color='green', label='Test data')

# Mean prediction
plt.plot(X_plot[:, 4], y_pred_plot, color='red', label='Predicted mean')

# Confidence intervals
plt.fill_between(X_plot[:, 4],
                 y_pred_plot - 1.96 * sigma_plot,
                 y_pred_plot + 1.96 * sigma_plot,
                 alpha=0.2, color='red', label='95% confidence interval')

plt.title('Gaussian Process Regression')
plt.xlabel('Input feature (Walking speed)')
plt.ylabel('Target (Gait Period in s)')
plt.savefig('gait_period_pred_model_walking_speed_rep3.png', dpi=300)
plt.legend()
plt.show()
