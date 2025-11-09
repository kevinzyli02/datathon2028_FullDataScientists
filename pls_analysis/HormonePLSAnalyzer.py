from sklearn.preprocessing import StandardScaler

class HormonePLSAnalyzer:
    def __init__(self):
        self.pls_model = None
        self.scaler_X = None
        self.scaler_Y = None
        self.feature_names = None
        self.target_names = None

    def prepare_pls_data(self, data):
        """Prepare data for PLS analysis"""

        # Define predictor variables (X)
        predictor_categories = {
            'physiological': [
                'rmssd', 'low_frequency', 'high_frequency', 'value',  # HRV and resting HR
                'minutesasleep', 'efficiency', 'minutestofallasleep',  # Sleep
                'nightly_temperature', 'temperature_diff_from_baseline',  # Temperature
                'sedentary', 'lightly', 'moderately', 'very',  # Activity
                'steps',  # Steps
            ],
            'metabolic': [
                'glucose_value',  # Glucose
                'demographic_vo2_max', 'filtered_demographic_vo2_max'  # VO2 max
            ],
            'symptoms': [
                'appetite', 'exerciselevel', 'headaches', 'cramps',
                'sorebreasts', 'fatigue', 'sleepissue', 'moodswing',
                'stress', 'foodcravings', 'indigestion', 'bloating'
            ],
            'stress_scores': [
                'stress_score', 'sleep_points', 'responsiveness_points', 'exertion_points'
            ]
        }

        # Flatten predictor list
        all_predictors = []
        for category in predictor_categories.values():
            all_predictors.extend(category)

        # Target variables (Y) - hormones
        targets = ['lh', 'estrogen', 'pdg']

        # Filter to available columns
        available_predictors = [col for col in all_predictors if col in data.columns]
        available_targets = [col for col in targets if col in data.columns]

        print(f"Available predictors: {len(available_predictors)}")
        print(f"Available targets: {len(available_targets)}")

        # Remove rows where target variables are missing
        data_clean = data.dropna(subset=available_targets)

        X = data_clean[available_predictors]
        Y = data_clean[available_targets]

        self.feature_names = available_predictors
        self.target_names = available_targets

        return X, Y, data_clean

    def perform_pls_analysis(self, X, Y, n_components=5):
        """Perform PLS regression analysis"""

        # Standardize the data
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)

        # Perform cross-validation to find optimal components
        print("Performing cross-validation...")
        mse_scores = []
        components_range = range(1, min(n_components + 1, X.shape[1]))

        for n_comp in components_range:
            pls = PLSRegression(n_components=n_comp)
            score = -np.mean(cross_val_score(pls, X_scaled, Y_scaled,
                                             cv=5, scoring='neg_mean_squared_error'))
            mse_scores.append(score)

        # Find optimal number of components
        optimal_components = np.argmin(mse_scores) + 1
        print(f"Optimal number of components: {optimal_components}")

        # Fit final PLS model with optimal components
        self.pls_model = PLSRegression(n_components=optimal_components)
        self.pls_model.fit(X_scaled, Y_scaled)

        # Calculate performance metrics
        Y_pred_scaled = self.pls_model.predict(X_scaled)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)

        # Calculate R² for each hormone
        r2_scores = {}
        for i, hormone in enumerate(self.target_names):
            r2 = r2_score(Y.iloc[:, i], Y_pred[:, i])
            r2_scores[hormone] = r2
            print(f"R² for {hormone}: {r2:.3f}")

        return optimal_components, mse_scores, r2_scores

    def analyze_variable_importance(self):
        """Analyze variable importance in PLS model"""
        if self.pls_model is None:
            raise ValueError("PLS model not fitted yet.")

        # Calculate VIP scores
        T = self.pls_model.x_scores_  # X scores
        W = self.pls_model.x_weights_  # X weights
        Q = self.pls_model.y_loadings_  # Y loadings

        # VIP formula: VIP = sqrt(p * (SS of weights * R²Y) / total R²Y)
        p = W.shape[0]  # number of features
        SS_weights = np.sum(W ** 2, axis=0)  # sum of squares of weights
        R2Y = np.sum(Q ** 2, axis=0)  # R² for each Y variable
        total_R2Y = np.sum(R2Y)  # total R²Y

        VIP_scores = np.sqrt(p * np.sum(SS_weights * R2Y / total_R2Y * (W ** 2 / SS_weights), axis=1))

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'VIP_score': VIP_scores
        }).sort_values('VIP_score', ascending=False)

        return importance_df

    def plot_results(self, importance_df, mse_scores, Y, Y_pred):
        """Create visualization plots for PLS results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: VIP scores
        top_features = importance_df.head(15)
        axes[0, 0].barh(range(len(top_features)), top_features['VIP_score'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('VIP Score')
        axes[0, 0].set_title('Top 15 Most Important Features (VIP Scores)')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cross-validation MSE
        axes[0, 1].plot(range(1, len(mse_scores) + 1), mse_scores, 'o-')
        axes[0, 1].set_xlabel('Number of PLS Components')
        axes[0, 1].set_ylabel('Cross-Validation MSE')
        axes[0, 1].set_title('PLS Component Selection')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Predicted vs Actual for each hormone
        for i, hormone in enumerate(self.target_names):
            axes[1, 0].scatter(Y.iloc[:, i], Y_pred[:, i], alpha=0.6, label=hormone)

        axes[1, 0].plot([Y.min().min(), Y.max().max()], [Y.min().min(), Y.max().max()], 'r--')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Predicted vs Actual Hormone Levels')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Loadings for first two components
        if hasattr(self.pls_model, 'x_loadings_'):
            loadings = self.pls_model.x_loadings_
            axes[1, 1].scatter(loadings[:, 0], loadings[:, 1], alpha=0.7)
            for i, feature in enumerate(self.feature_names):
                axes[1, 1].annotate(feature, (loadings[i, 0], loadings[i, 1]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, alpha=0.7)
            axes[1, 1].axhline(0, color='grey', linestyle='--')
            axes[1, 1].axvline(0, color='grey', linestyle='--')
            axes[1, 1].set_xlabel('Component 1 Loadings')
            axes[1, 1].set_ylabel('Component 2 Loadings')
            axes[1, 1].set_title('PLS Loadings Plot')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()