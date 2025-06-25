\chapter{Background\markboth{Background}{}}

The overall dynamics of a two-wheeled humanoid robot are highly complex due to the nonlinear, high-dimensional, and unstable nature of the system. To manage this complexity, we decompose the control problem into two stages:

\begin{enumerate}
    \item \textbf{Stage One:} We focus on stabilizing the robot on flat ground by controlling only the two wheels. This is achieved using a combination of PID control and reinforcement learning (RL), where PID ensures basic stability and the RL component improves robustness and adaptability.
    
    \item \textbf{Stage Two:} To extend stability to uneven terrains, we introduce an articulated control strategy that maps the robot's roll angle to adaptive adjustments in the hip and knee joints. This mapping is designed to maintain the base of support under varying terrain slopes. A reinforcement learning policy is then trained to complement this mapping, enhancing overall stability under challenging surface conditions.
\end{enumerate}


\section{Classical Control Approaches: PID Control}
the classical PID (proportional-integral-derivative) control is generally defined as :

\[
u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}
\]

where:
\begin{itemize}
    \item \( e(t) = S(t) - S_{target}(t) \) is the error (output minus reference),
    \item \( u(t) \) is the control input at time t,
    \item \( K_p, K_i, K_d \) are the proportional, integral, and derivative gains. which are required to be carefully tuned in running
\end{itemize}


\section*{Convergence Analysis of the Proportional Control Term}

Although the overall robotic system is nonlinear, we apply PID control specifically to the wheel dynamics, which can be locally approximated as an inverted pendulum. Around the upright equilibrium point, this system can be interpreted as a locally linear system within a bounded error range. Based on this approximation, we consider a locally discrete-time linear system controlled by a proportional term:

\[
S_{k+1} = A S_k + B U_k
\]
\[
U_K = K_p (S_k - S_{\text{target}})
\]

Rewriting the above equations:

\[
S_{k+1} = (A - B K_p) S_k + B K_p S_{\text{target}}
\]

\subsubsection*{Define the Error}

Let the tracking error be:

\[
e_k := S_k - S_{\text{target}} \Rightarrow S_k = e_k + S_{\text{target}}
\]

Substituting into the update equation:

\begin{align*}
S_{k+1} &= (A - B K_p)(e_k + S_{\text{target}}) + B K_p S_{\text{target}} \\
&= (A - B K_p)e_k + (A - B K_p)S_{\text{target}} + B K_p S_{\text{target}} \\
&= (A - B K_p)e_k + A S_{\text{target}}
\end{align*}

Now compute the next error term:

\[
e_{k+1} = S_{k+1} - S_{\text{target}} = (A - B K_p)e_k + A S_{\text{target}} - S_{\text{target}} = (A - B K_p)e_k + (A - I) S_{\text{target}}
\]

\subsubsection*{Linear Recurrence}

This gives the inhomogeneous recurrence:

\[
e_{k+1} = M e_k + c
\]
where
\[
M = A - B K_p, \quad c = (A - I) S_{\text{target}}
\]

\subsubsection*{General Solution}

The general solution of the recurrence is:

\[
e_k = M^k e_0 + \sum_{i=0}^{k-1} M^i c
\]
We define spectral radius: 
\[
\rho(A) = \max \left\{ |\lambda| : \lambda \text{ is an eigenvalue of } A \right\}
\]
If \( \rho(M) < 1 \) , then:

\[
\lim_{k \to \infty} e_k = (I - M)^{-1} c
\]

Substitute \( M \) and \( c \):

\[
\lim_{k \to \infty} e_k = (I - A + B K_p)^{-1}(A - I) S_{\text{target}}
\]

Therefore, the steady-state value of \( S_k \) is:

\[
\lim_{k \to \infty} S_k = \lim_{k \to \infty} (e_k + S_{\text{target}}) = \left[ (I - A + B K_p)^{-1}(A - I) + I \right] S_{\text{target}}
\]

\subsubsection*{Conclusion}

\begin{itemize}
    \item The system converges if \( \rho(A - B K_p) < 1 \).
    \item The limit of \( S_k \) depends on system dynamics and proportional gain.
    \item Proper tuning of \( K_p \) is necessary to make the limit match \( S_{\text{target}} \).
\end{itemize}


\section{Deep Reinforcement Learning in Robotics}
\subsection*{Deep Reinforcement Learning Overview:}
Deep Reinforcement Learning (Deep RL) is a class of machine learning algorithms where an agent (represented by a deep neural network) learns to make decisions by interacting with an environment in order to maximize a certain objective, typically referred to as the reward. This is achieved through a trial-and-error process, where the agent receives feedback (rewards) based on the actions it takes, and gradually improves its decision-making ability over time.

\subsection*{Deep RL Concepts}
\subsubsection*{States:}
A representation of the current situation of the robot, such as sensory data. This may include images from cameras, position data from the robot’s joints, or information about obstacles in the environment.

\subsubsection*{Actions:}
Refers to the decisions that can be made by the agent in the environment, such as joint torques or wheel velocity signals.

\subsubsection*{Rewards:}
A scalar value received from the environment after the agent takes an action. Typically, positive values are used as encouragement and negative values as punishment. The objective of the agent is to maximize the cumulative rewards over time.

\subsubsection*{Policy:}
A function or strategy used by the agent to decide which action to take given the current state. In our case, this would be a deep neural network that takes the current state as input and outputs a distribution over possible actions.

\subsubsection*{Return:}
The accumulated sum of discounted rewards along a trajectory. It is used to evaluate the quality of a policy.

\subsubsection*{Markov Decision Process (MDP):}
A framework for modeling decision-making where outcomes are partly random and partly under the control of a decision-maker. 

\subsubsection*{Probability Density Function (PDF)}

For a continuous random variable \( X \), the probability density function (PDF) \( f_X(x) \) is a function that describes the relative likelihood of \( X \) taking a particular value.

It satisfies the following properties:

\begin{itemize}
    \item \( f_X(x) \geq 0 \quad \forall x \in \mathbb{R} \)
    \item \( \displaystyle \int_{-\infty}^{\infty} f_X(x) \, dx = 1 \)
\end{itemize}

The probability that \( X \) lies within an interval \([a, b]\) is given by:

\[
P(a \leq X \leq b) = \int_a^b f_X(x) \, dx
\]

Note that for continuous distributions, the probability at a single point is zero:

\[
P(X = x) = 0
\]
\cite{ross2014probability}


\subsubsection*{Contraction Mapping:}
A function \(f\) on a metric space \((X, d)\) is called a contraction mapping if there exists a constant \(0 \leq c < 1\) such that for all \(x, y \in X\), the following holds:
\[
d(f(x), f(y)) \leq c \cdot d(x, y)
\]


\subsubsection*{Banach Fixed Point Theorem (Contraction Mapping Theorem)}

Let $(X, d)$ be a non-empty complete metric space, and let $T: X \to X$ be a contraction mapping. That is, there exists a constant $0 \leq c < 1$ such that for all $x, y \in X$,
\[
d(T(x), T(y)) \leq c \cdot d(x, y).
\]
Then:
\begin{itemize}
    \item There exists a unique fixed point $x^* \in X$ such that $T(x^*) = x^*$.
    \item For any $x_0 \in X$, the sequence defined by $x_{n+1} = T(x_n)$ converges to $x^*$.
\end{itemize}

This theorem is fundamental in the analysis of iterative algorithms in reinforcement learning\cite{rudin1976principles}.



\section{Bellman Equation}

Bellman equation states that the value of a state is the expected return when starting in that state and following the policy \(\pi\)~\cite{ghasemi2024introductionreinforcementlearning}.

\subsection*{Existence of iterative solution}

We begin with the Bellman expectation equation for the value function \(V^\pi(s)\) under policy \(\pi\):

\[
V^\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s' \mid s) V^\pi(s')
\]

where:
\begin{itemize}
    \item \(r_\pi(s)\) is the expected reward when in state \(s\) and following policy \(\pi\),
    \item \(P_\pi(s' \mid s)\) is the probability of transitioning from state \(s\) to state \(s'\) under policy \(\pi\),
    \item \(\gamma\) is the discount factor.
\end{itemize}

This equation defines the value function \(V^\pi(s)\) as the expected return starting from state \(s\). The goal is to solve this equation for \(V^\pi(s)\).

\subsubsection*{Define the Iterative Solution}

An iterative solution involves updating an initial guess for the value function \(V_0\) using the Bellman expectation operator \(T^\pi\):

\[
V_{k+1} = T^\pi V_k
\]

where we define the Bellman operator \(T^\pi\) as:

\[
(T^\pi V)(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s' \mid s) V(s')
\]

The process is repeated until convergence, and the fixed point of this iteration will give the value function \(V^\pi\).

\subsubsection*{the Bellman Operator is a Contraction}

We now show that the Bellman operator \(T^\pi\) is a contraction with respect to the supremum norm. Let \(V\) and \(W\) be two value functions. We want to show that:

\[
\|T^\pi V - T^\pi W\|_\infty \leq \gamma \|V - W\|_\infty
\]

where the supremum norm \(\|V - W\|_\infty\) is defined as:

\[
\|V - W\|_\infty = \max_{s \in \mathcal{S}} |V(s) - W(s)|
\]

Consider the difference between \(T^\pi V(s)\) and \(T^\pi W(s)\):

\[
|T^\pi V(s) - T^\pi W(s)| = \left| \gamma \sum_{s'} P_\pi(s' \mid s) (V(s') - W(s')) \right|
\]

Using the triangle inequality, we have:

\[
\leq \gamma \sum_{s'} P_\pi(s' \mid s) |V(s') - W(s')|
\]

Since \( |V(s') - W(s')| \leq \|V - W\|_\infty \) for all \(s' \in \mathcal{S}\), we can further bound the sum:

\[
\leq \gamma \sum_{s'} P_\pi(s' \mid s) \|V - W\|_\infty
\]

Since the sum of the transition probabilities \( P_\pi(s' \mid s) \) over all \( s' \) is equal to 1, we have:

\[
\leq \gamma \|V - W\|_\infty
\]

Thus, we obtain:

\[
\|T^\pi V - T^\pi W\|_\infty \leq \gamma \|V - W\|_\infty
\]

\subsubsection*{Apply the Banach Fixed Point Theorem}

Since \( \gamma < 1 \), this shows that \(T^\pi\) is a contraction mapping with respect to the supremum norm.

By the Banach Fixed Point Theorem, since \(T^\pi\) is a contraction, the sequence of value functions \(V_k\) generated by the iteration \(V_{k+1} = T^\pi V_k\) will converge to a unique fixed point \(V^\pi\) as \(k \to \infty\). This fixed point is the solution to the Bellman expectation equation:

\[
V^\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s' \mid s) V^\pi(s')
\]

\subsubsection*{Conclusion}

Thus, the iterative approach to solving the Bellman expectation equation converges to the optimal value function \(V^\pi(s)\) under the given policy \(\pi\). The convergence is guaranteed because the Bellman expectation operator \(T^\pi\) is a contraction mapping in the supremum norm.

This proves that the iterative method for solving the Bellman equation will always converge to the unique fixed point, which corresponds to the value function under policy \(\pi\).

The convergence analysis and uniqueness of optimal policy stated~\cite{mathefoundation}

\section{Temporal Difference (TD) Error}

Temporal-Difference (TD) learning is a model-free, incremental approach commonly used for estimating state-value functions. Given a policy \(\pi\), the objective is to estimate the state-value function \(V^\pi(s)\) for all \(s \in \mathcal{S}\). The TD(0) algorithm, also known as Temporal Difference with zero-step lookahead, is defined as:

\[
V_{t+1}(s) =
\begin{cases}
V_t(s) + \alpha_{t} \left[ r_{t+1} + \gamma V_t(s_{t+1}) - V_t(s) \right], & \text{if } s = s_t \\
V_t(s), & \text{if } s \neq s_t
\end{cases}
\]

\noindent \textbf{Note:} The same state \(s \in \mathcal{S}\) can appear multiple times at different time steps during an episode, such as \(s_0 = s_4\), depending on the environment dynamics and the policy in use. Temporal-Difference learning updates the value of the state encountered at time \(t\), irrespective of whether it has been visited previously or will be visited again later in the episode.

Where:
\begin{itemize}
    \item \(\alpha_t\) is the learning rate at time step \(t\),
    \item \(\gamma\) is the discount factor, controlling the importance of future rewards,
    \item \(r_{t+1}\) is the reward received after transitioning from state \(s_t\) to state \(s_{t+1}\),
    \item \(V_t(s)\) is the current estimate of the state-value function \(V^\pi(s)\).
\end{itemize}

worth to note that, the TD learning algorithm can be viewed as a special stochastic approximation method for solving bellman equation with the property of convergence, the detailed prove can be found at ~\cite{mathefoundation}

In case of future use, we extract the following term from TD (0) algorithm 
\subsubsection*{TD Error}

The Temporal Difference (TD) error, denoted as \( \delta_t \), measures the discrepancy between the predicted value of the current state and the value obtained from a one-step lookahead. It is defined as:

\[
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\]

where:
\begin{itemize}
    \item \( r_{t+1} \) is the reward received after transitioning from state \( s_t \) to state \( s_{t+1} \),
    \item \( \gamma \) is the discount factor,
    \item \( V(s_{t+1}) \) is the estimated value of the next state,
    \item \( V(s_t) \) is the estimated value of the current state.
\end{itemize}

This temporal-difference (TD) error captures the discrepancy between the current value estimate and the expected return, incorporating both the immediate reward and the estimated value of the next state. In Proximal Policy Optimization (PPO), it serves as the foundation for estimating the advantage function, which guides policy updates.

\section{Policy Gradient Methods}
The goal in policy gradient methods is to maximize the expected return by directly optimizing the policy parameters \(\theta\). This is commonly formalized through the objective function~\cite{sutton2000policy}:

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r_t \right]
\]

Using the policy gradient theorem~\cite{sutton2000policy}, the gradient of the objective can be expressed as:

\[
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \, Q^\pi(s, a) \right]
\]

This formulation requires access to the action-value function \(Q^\pi(s, a)\), which may be difficult to estimate directly. 

To reduce variance and improve learning stability, we introduce the state-value function \(V^\pi(s)\) and define the \textbf{advantage function}:

\[
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
\]

Substituting \(Q^\pi(s, a) = A^\pi(s, a) + V^\pi(s)\) into the policy gradient expression yields:

\[
\nabla_\theta J(\theta)
= \mathbb{E}_{s, a} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \left( A^\pi(s, a) + V^\pi(s) \right) \right]
\]

If we unpack the expectation and observe that \( V^\pi(s) \) is independent of the action \( a \), and also recall the identity

\[
\mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \right] = 0,
\]

Therefore, the term involving \(V^\pi(s)\) vanishes:

\[
\mathbb{E}_{s, a} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot V^\pi(s) \right] = \mathbb{E}_s \left[ V^\pi(s) \cdot \underbrace{\mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \right]}_{= 0} \right] = 0
\]

Hence, we obtain the \textbf{advantage-based policy gradient}:

\[
\nabla_\theta J(\theta) = \mathbb{E}_{s, a} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot A^\pi(s, a) \right]
\]

This expression shows that the advantage function can substitute for the action-value function in the policy gradient update without changing the expectation, thus offering a more stable and interpretable learning signal.

\subsubsection*{TD-based Advantage Estimation}

In practice, the advantage function is often approximated using the \textbf{Temporal-Difference (TD) error}:

\[
\hat{A}_t = \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\]

This TD error serves as a one-step estimate of \(A^\pi(s_t, a_t)\), and can be plugged into the gradient update:

\[
\nabla_\theta J(\theta) \approx \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \hat{A}_t \right]
\]

This connection bridges policy gradient methods and value-based TD learning, and is the basis for many actor-critic algorithms.

\section{Proximal Policy Optimization (PPO) }

In Proximal Policy Optimization (PPO), the policy network and the value network can be optimized either separately using distinct loss functions—when implemented as separate architectures—or jointly when they share a common backbone. In our work, we adopt the former approach, employing separate architectures for the policy and value networks.

\subsubsection*{Policy Network Optimization (Actor)}

Before introducing the clipped objective used in PPO, it is important to address the inefficiency associated with sampling from the current policy. Since the policy changes after every update, collecting new data for each iteration is expensive and sample-inefficient. To overcome this, PPO reuses samples from the previous policy by applying \textit{importance sampling}~\cite{rubinstein1981simulation}.

The policy gradient can then be estimated using:

\[
\nabla_\theta J(\theta) \approx \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \, \nabla_\theta \log \pi_\theta(a \mid s) \, \hat{A}_t \right]
\]

Note that the gradient of the log-probability can be expressed as:

\[
\nabla_\theta \log \pi_\theta(a \mid s) = \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}
\]

This identity is fundamental to the derivation of the policy gradient using the likelihood ratio trick.

The \textbf{surrogate objective} used for policy optimization in PPO is then defined as:

\[
L^{\text{PG}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ r_\theta(s,a) \, \hat{A}_t \right]
\]

where \( \hat{A}_t \) is the estimated advantage function, and the probability ratio is given by:

\[
r_\theta(s,a) = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}
\]


To prevent overly large policy updates that may degrade performance, PPO introduces a clipped version of the objective:

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_{(s,a)} \left[ \min \left( r_\theta(s,a) \, \hat{A}_t, \, \text{clip}(r_\theta(s,a), 1 - \epsilon, 1 + \epsilon) \, \hat{A}_t \right) \right]
\]

This clipped surrogate objective constrains the size of the policy update, encouraging more stable training while still allowing for effective policy improvement.


\subsubsection*{Value Network Optimization (Critic)}

The value network is optimized by minimizing the squared error loss between the predicted value and a bootstrapped return target:

\[
L^{\text{value}}(\phi) = \mathbb{E}_t \left[ \left( V_\phi(s_t) - \hat{V}_t \right)^2 \right]
\]

where
\begin{itemize}
    \item \( V_\phi(s_t) \) is the predicted state value from the value network,
    \item \( \hat{V}_t \) is the target value, e.g., could be computed by using Advantage Estimation.
\end{itemize}
