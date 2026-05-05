# Deep Reinforcement Learning for Proportional Fairness Power Allocation

## 1. Project Overview
The objective of this framework is to dynamically allocate a finite continuous transmit power budget ($P_{max}$) across multiple competing users in a wireless interference network. While classical optimization algorithms like Iterative Water-Filling (IWF) and Weighted Minimum Mean Square Error (WMMSE) provide theoretical mathematical bounds, they possess high asymptotic computational complexities (e.g., $\mathcal{O}(N^3)$ for WMMSE) that render them physically unscalable for sub-millisecond Transmission Time Intervals (TTIs)[cite: 1].

To overcome this execution latency bottleneck, we deploy a Deep Reinforcement Learning (DRL) agent. The DRL framework eliminates the necessity for a highly iterative mathematical solver during online execution[cite: 1]. By shifting the processing complexity to the offline training phase, the neural network enables near-instantaneous inference[cite: 1]. The agent interacts directly with the network environment, observing complex states, executing power allocation actions, and adjusting its policy based on delayed numerical reward signals to strictly enforce Proportional Fairness[cite: 1].

---

## 2. Markov Decision Process (MDP) Formulation
The dynamic resource and power allocation problem is formalized as a Markov Decision Process defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{R})$[cite: 1].

### 2.1 State Space ($\mathcal{S}$)
The state space provides the agent with both the instantaneous physical environment and the historical context required to make equitable decisions. At each time step $t$, the state vector comprises:
*   **Instantaneous Channel State Information (CSI):** The magnitudes of the $N \times N$ channel matrix $H$, allowing the agent to observe both direct channel gains and inter-user interference channels[cite: 1].
*   **Historical Average Throughput:** The Exponential Moving Average (EMA) of previous data rates for each user[cite: 1]. This historical data is an absolute requirement for computing dynamic Proportional Fairness weights across sequential time intervals[cite: 1].

### 2.2 Action Space ($\mathcal{A}$)
The action vector determines the continuous transmit power levels selected for the $N$ clients[cite: 1].
*   **Format:** A continuous vector of length $N$, with outputs bounded between $[0, 1]$.
*   **Scaling:** The raw network output is multiplied by the total power budget $P_{max}$ to map the actions to feasible physical wattages. 

### 2.3 Reward Function ($\mathcal{R}$)
To adhere strictly to Proportional Fairness, the reward must mathematically correlate with the Proportional Fairness logarithmic utility[cite: 1].
*   **Utility Metric:** The agent calculates the sum of the natural logarithms of the instantaneous data rates: $\sum \log(R_k + \epsilon)$. Because the mathematical gradient of the logarithmic function approaches infinity as the data rate approaches zero, this formulation heavily penalizes total data deprivation, explicitly preventing the starvation of cell-edge users[cite: 1].
*   **Action Penalization:** If the neural network proposes a continuous power vector that exceeds the overarching physical sum-power constraint $P_{max}$, the reward is heavily penalized and the action is deterministicly projected back to the feasible hardware limit[cite: 1]. This forces the agent to explore only valid operational regions[cite: 1].

---

## 3. Environment Dynamics (Dynamic CSI)
To train the agent to be proactive rather than purely reactive, the wireless environment cannot rely on temporally independent random channels. It utilizes a **First-Order Gauss-Markov Process** to model continuous, correlated mobility (Jakes' fading model).

*   **Update Rule:** $H_t = \rho H_{t-1} + \sqrt{1-\rho^2} E_t$
*   **Correlation ($\rho$):** Calculated using the zeroth-order Bessel function based on the physical speed of the user equipment and the carrier frequency.

---

## 4. Implementation Architecture

The framework is implemented using Python, the `Gymnasium` API for environment standardization, and `Stable-Baselines3` (SB3) for the reinforcement learning algorithm.

### 4.1 Algorithm Selection: Proximal Policy Optimization (PPO)
For this continuous allocation task, we utilize Proximal Policy Optimization (PPO). PPO is an on-policy gradient method that utilizes a specialized, clipped surrogate objective function[cite: 1]. This clipping mechanism prevents destructively large policy updates during backpropagation, thereby ensuring monotonic improvement and high training stability in complex multi-user deployments[cite: 1].

### 4.2 Code Structure Outline
1.  **`WirelessAllocationEnv(gym.Env)`**: 
    *   `__init__()`: Defines `action_space` (Box) and `observation_space` (Box).
    *   `step(action)`: Scales action to $P_{max}$, calculates SINR, calculates Shannon capacity, updates historical EMA throughput, computes the logarithmic reward, applies constraint penalties, and steps the Gauss-Markov CSI generator.
    *   `reset()`: Reinitializes the channel matrix and historical trackers.
2.  **Training Script**: 
    *   Wraps the environment in SB3's `check_env`.
    *   Instantiates `PPO("MlpPolicy", env)`.
    *   Executes `model.learn(total_timesteps=100000)`.
3.  **Inference Script**:
    *   Loads the trained PPO model.
    *   Executes a forward-pass `model.predict(obs, deterministic=True)` inside a simulated TTI loop.
    *   Logs outputs to compare execution latency against mathematical baselines.

---

## 5. Evaluation & Benchmarking
The trained PPO agent will be explicitly benchmarked against classical algorithms to validate its performance:

1.  **Iterative Water-Filling (IWF):** Evaluated strictly on orthogonal parallel channels to establish a baseline for pure sum-rate optimization[cite: 1].
2.  **Fractional Programming (FP):** Provides the mathematical upper bound for pure sum-rate maximization across multi-user interference channels, demonstrating the extreme limit of capacity at the expense of user equity[cite: 1].
3.  **Max-Min Fairness:** Demonstrates the opposing extreme, establishing the baseline for absolute network equality and the subsequent cost to total system capacity[cite: 1].
4.  **Proportional Fair WMMSE (PF-WMMSE):** Serves as the primary comparative benchmark, providing the theoretical mathematical optimum for balancing fairness and throughput[cite: 1]. 

The primary success metric is demonstrating that the PPO agent achieves a Jain's Fairness Index and aggregate throughput equivalent to PF-WMMSE, while reducing the execution latency from the $\mathcal{O}(N^3)$ matrix inversions of classical methods to an $\mathcal{O}(1)$ sub-millisecond neural network forward pass[cite: 1].