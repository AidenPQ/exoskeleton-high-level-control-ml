import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.kernel_approximation import Nystroem
from ExtractDataForTraining import compute_data_intent_recognition
from scipy.optimize import minimize
import joblib
import warnings

# Load dataset
dataset_file = "processed_data.h5"
x, y_ = compute_data_intent_recognition(dataset_file=dataset_file, gait_percentage_number=5)

X = np.array(x)
y = np.array(y_)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Define the kernel search space with adjusted hyperparameter bounds
# k1 = C(1.0, (1e-4, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e2))
# k2 = C(1.0, (1e-4, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e2), nu=1.5)
# k3 = C(1.0, (1e-4, 1e2)) * RationalQuadratic(length_scale=1.0, alpha=0.1)
# k4 = C(1.0, (1e-4, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=3.0, periodicity_bounds=(1e-2, 1e2))
# k5 = C(1.0, (1e-4, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e2))

# Dictionary to hold the kernels with adjusted hyperparameter bounds
param_grid = [
    {'kernel': [C(1.0, (1e-4, 1e4)) * RBF(length_scale_bounds=(1e-6, 1e2))]},
    {'kernel': [C(1.0, (1e-4, 1e4)) * Matern(length_scale_bounds=(1e-7, 1e3), nu=nu_value)]
        for nu_value in [0.5, 1.5, 2.5]},
    {'kernel': [C(1.0, (1e-4, 1e4)) * RationalQuadratic(length_scale_bounds=(1e-6, 1e2), alpha_bounds=(1e-4, 1e2))]},
    {'kernel': [C(1.0, (1e-4, 1e4)) * ExpSineSquared(length_scale_bounds=(1e-6, 1e2), periodicity_bounds=(1e-3, 1e2))]},
    {'kernel': [C(1.0, (1e-4, 1e4)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-4, 1e2))]}
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
    result = minimize(objective_function, initial_theta, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
    return result.x, result.fun


# Use Nystroem approximation for large datasets
nystroem = Nystroem(n_components=200)

# Set up the cross-validation search with Nystroem approximation
gp = GaussianProcessRegressor(n_restarts_optimizer=5, optimizer=custom_optimizer, alpha=1e-4)
grid_search = GridSearchCV(estimator=gp, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Transform the training data using Nystroem approximation
X_train_transformed = nystroem.fit_transform(X_train_scaled)
X_test_transformed = nystroem.transform(X_test_scaled)

# Train and evaluate
grid_search.fit(X_train_scaled, y_train)
best_gp = grid_search.best_estimator_

# Make predictions with the best model
y_pred, sigma = best_gp.predict(X_test_scaled, return_std=True)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Best Kernel: {best_gp.kernel_}')

# Save the scaler and best model
joblib.dump(scaler, 'scaler_intent_recognition.pkl')
joblib.dump(nystroem, 'nystroem_transformer.pkl')
joblib.dump(best_gp, 'best_gp_model_intent_recognition.pkl')

# Plotting the results
plt.figure(figsize=(10, 6))

# Assuming X_test_scaled has shape (n_samples, n_features)
# The number of samples
num_samples = np.arange(1, len(y_test) + 1)

# Training data
plt.scatter(num_samples, y_test, color='blue', label='Actual Gait Percentages')

# Predictions
plt.plot(num_samples, y_pred, color='red', label='Predicted Gait Percentages')

# Confidence intervals
plt.fill_between(num_samples, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='red', label='95% confidence interval')

plt.title('Gaussian Process Regression: Gait Percentages Prediction')
plt.xlabel('Sample Number')
plt.ylabel('Gait Percentages')
plt.savefig('intent_recognition_pred_model_age_rep2.png', dpi=300)
plt.legend()
plt.show()
