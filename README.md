# CartPole Q-Learning 🧠🤖

CartPole Q-Learning is a reinforcement learning project that trains an agent to balance a pole on a moving cart using the **Q-learning algorithm**.  
The agent interacts with the CartPole environment, learns from rewards, and gradually improves its policy through trial and error.  
To handle the environment’s **continuous state space**, observations are discretized into bins, enabling learning with a tabular Q-table.

---

## 🚀 Features

- Implements **Q-learning** from scratch in Python  
- Solves the classic **CartPole control problem**  
- Handles **continuous state spaces** via discretization  
- Uses an **epsilon-greedy policy** for exploration vs. exploitation  
- Learns an optimal policy over multiple training episodes  
- Easily configurable hyperparameters (learning rate, discount factor, epsilon)

---

## 🧩 Tech Stack

**Language:** Python  
**Libraries:** Gymnasium / OpenAI Gym, NumPy  
**Concepts:** Reinforcement Learning, Q-Learning, Markov Decision Processes

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository

git clone [https://github.com/<your-username>/cartpole-qlearning.git](https://github.com/ArjunBharadwaj123/cart-pole)

cd cart-pole


### 2. Run the training script

python driverCode.py


---

## 🧠 How It Works

1. The CartPole environment provides four continuous observations:
   - Cart position  
   - Cart velocity  
   - Pole angle  
   - Pole angular velocity  

2. These observations are **discretized into bins** to form a finite state space.

3. A **Q-table** is initialized and updated using the Q-learning update rule:

Q(s, a) = Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]


4. The agent selects actions using an **epsilon-greedy policy**, balancing exploration and exploitation.

5. Over many episodes, the agent learns a policy that maximizes cumulative reward by keeping the pole balanced longer.

---

## 📈 Results

After sufficient training, the agent is able to consistently balance the pole for extended periods, demonstrating convergence toward an effective control policy.

---

## 📚 Future Improvements

- Replace state discretization with function approximation  
- Implement **Deep Q-Networks (DQN)**  
- Add reward and performance visualizations  
- Perform systematic hyperparameter tuning

---

## ✍️ Author

**Arjun Bharadwaj**  
Computer Science, University of Maryland
