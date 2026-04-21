# Diffusion Models for Time-Series Forecasting

## Conceptual Core
Diffusion models (or score-based generative models) learn a data distribution by reversing a noising process. In financial time-series, we apply a **Conditional Diffusion** approach.

### Forward Process (Noising)
We add Gaussian noise to the target returns $y$:
$$ y_t = \sqrt{\bar{\alpha}_t} y_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$
where $\epsilon \sim N(0, I)$.

### Reverse Process (Denoising)
The model $\epsilon_\theta$ is trained to predict the added noise $\epsilon$ given the noisy state $y_t$, the timestep $t$, and the historical context $h$:
$$ \min_\theta \| \epsilon - \epsilon_\theta(y_t, t, h) \|^2 $$

## Application to EMH
By predicting the "noise" in future returns, the model effectively learns the conditional probability density $p(y_{t+1:t+H} | X_{t-L:t})$. If the model can consistently predict even a small portion of the target variance, it suggests that the historical context $X$ is not independent of future returns—violating the random walk property of weak-form EMH.

## Advantages for Finance
- **Probabilistic Forecasts**: Provides a distribution of possible return paths.
- **Robustness to Noise**: Specifically designed to handle stochastic signals.
- **XAI Compatibility**: Differentiable through time, allowing for gradient-based attribution (Integrated Gradients).
