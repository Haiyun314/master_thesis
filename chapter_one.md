\chapter{Introduction\markboth{Introduction}{}}

Balancing control for two-wheeled robots has long been a fundamental and challenging problem in robotics. These systems are inherently unstable and underactuated, requiring precise control strategies to maintain upright posture while potentially performing locomotion or interacting with the environment \cite{Two-WheeledBalancingVehiclesonUncertainTerrains}. Over the years, a range of approaches has been proposed to address this challenge, including classical model-based methods such as Proportional-Integral-Derivative (PID) control, Linear Quadratic Regulator (LQR)\cite{lqrtwo-wheeled}, and Model Predictive Control (MPC)\cite{mpc-twowheeled}, as well as modern model-free approaches like Deep Reinforcement Learning (DRL).\cite{piddqntwo-wheel}

\section{Motivation}

Classical model-based control methods offer analytical guarantees of stability and are often computationally efficient. However, they depend heavily on accurate models of the robot’s dynamics, which may not fully capture the complexity of real-world environments. As a result, these methods tend to be brittle under conditions involving dynamic terrain, modeling errors, or external perturbations.\cite{difficultyonuneventerrain}

On the other hand, DRL methods have demonstrated impressive robustness and adaptability by learning directly from interaction with the environment. However, they suffer from their own limitations: training is often data-intensive and unstable, especially in environments where the reward is sparse or delayed.\cite{limitations, sparserewards} Furthermore, end-to-end DRL policies are typically  lack interpretability, posing challenges for debugging and safety assurance.\cite{interpretable}

This creates a natural motivation to explore hybrid control architectures that combine the strengths of classical control and learning-based approaches.

\section{Problem Statement}

The central challenge addressed in this thesis is the development of a control strategy that enables a two-wheeled humanoid robot to maintain balance in dynamic and uncertain environments, such as uneven terrain.\cite{robot_urdf_2023}
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{images/PibSim.jpg}
    \caption{URDF model of the two-wheeled humanoid robot used in this study.}
    \label{fig:robot_model}
\end{figure}

Traditional controllers—such as PID, LQR, and MPC—have long been used in balancing and locomotion tasks due to their simplicity and ability to guarantee stability under well-modeled dynamics \cite{piddqntwo-wheel, lqrtwo-wheeled, mpc-twowheeled}. These methods work reliably when system parameters are well-known and disturbances are minimal. However, their performance degrades significantly under unmodeled dynamics or external perturbations, limiting their robustness in real-world scenarios.

In contrast, learning-based approaches—particularly Deep Reinforcement Learning (DRL)—have demonstrated strong adaptability in complex, uncertain environments. DRL has been successfully applied to tasks like bipedal locomotion and robotic manipulation, showing the ability to learn from interaction without explicit modeling \cite{2017-TOG-deepLoco}. Yet, these methods often suffer from high sample complexity, instability during training, and a lack of theoretical guarantees for safety or convergence\cite{limitations}.

While classical controllers such as PID have long been used for stabilizing two-wheeled and bipedal robots due to their simplicity and reliability\cite{piddqntwo-wheel}, they often struggle in environments with unmodeled dynamics or external perturbations. Recent work has explored combining such controllers-PID with deep reinforcement learning (DRL) to enhance adaptability and robustness of suspension control system in cars\cite{pidandrl}. Inspired by this trend, this thesis proposes a hybrid control framework that integrates classical PID control for low-level balance stabilization with a Proximal Policy Optimization (PPO)-based DRL agent for high-level decision-making. This approach aims to leverage the complementary strengths of both paradigms: the predictability and structure of PID, and the data-driven adaptability of DRL. Similar architectures have been explored in locomotion and manipulation tasks \cite{pidandrl}, but few studies have focused on applying this hybrid strategy to two-wheeled self-balancing robots in dynamic environments.

The proposed approach is evaluated in simulation using a two-wheeled robot navigating randomized uneven terrain (see Figure 1). Two quantitative metrics are used to assess performance and robustness: height fluctuation, measured using a GPS sensor mounted on the robot’s head, which indicates the degree of stability or falling; and standing success rate, defined as the proportion of episodes in which the robot, initialized upright on randomly generated uneven terrain, successfully maintains balance for at least 10 seconds after being trained to stand upright within the first 6 seconds. These metrics allow for a clear comparison between the proposed hybrid method and a baseline standalone PID controller.

\section{Research Objectives}

The key objectives of this thesis are as follows. First, a convergence analysis of the proportional control term in the PID controller is conducted to better understand its impact on system stability. Second, the bellman equation study aims to establish the existence of a unique and optimal value function to ensure consistent decision-making. Third, a Proximal Policy Optimization (PPO) policy is introduced to improve sampling efficiency and training performance. Fourth, a hybrid control strategy is developed by integrating PID control for real-time feedback stability with reinforcement learning for long-term adaptability. Fifth, the approach is designed to enhance robustness and generalization, enabling the robot to maintain balance across a variety of terrain conditions. Finally, the proposed framework is validated through extensive simulation-based evaluations on diverse and challenging terrains, demonstrating its practical feasibility and effectiveness.


\section{Contributions}

This thesis makes several key contributions across theory, design, and empirical validation. First, it presents a convergence analysis of the proportional control component in PID control, offering insights into its stability behavior. Second, the thesis studies the existence of a unique and optimal value function using Bellman-based analysis, forming the theoretical foundation for integrating learning-based control. Third, a hybrid control architecture is developed that combines PID control for low-level feedback stability with a Proximal Policy Optimization (PPO) reinforcement learning agent for high-level adaptability. Fourth, a guided learning mechanism is introduced, wherein the PID controller provides reference signals to the PPO agent, significantly enhancing sample efficiency and training convergence. Finally, the hybrid control strategy is evaluated in simulation on a two-wheeled robot navigating randomized uneven terrains, demonstrating improved robustness and performance compared to standalone PID controllers.


\section{Thesis Outline}

The outline of this thesis is organized as follows:

\textbf{Chapter 1} reviews relevant literature on classical model-based control techniques, such as PID and LQR, as well as recent developments in deep reinforcement learning and hybrid control strategies. It identifies gaps and limitations in prior work, motivating the need for the proposed approach.

\textbf{Chapter 2} presents the theoretical background and analysis. It includes a convergence study of the proportional term in the PID controller and an investigation into the existence and uniqueness of the optimal value function, both of which provide the foundation for integrating learning-based control methods.

\textbf{Chapter 3} describes the proposed hybrid control architecture and its implementation. It details the structure and function of the PID controller, the design of the Proximal Policy Optimization (PPO) agent, a tailored action-mapping mechanism, and the overall integration strategy employed during training phases

\textbf{Chapter 4} presents the experimental setup and results. It includes simulation scenarios, evaluation metrics, performance comparisons between control strategies.

\textbf{Chapter 5} discusses the limitations of the proposed method and outlines potential directions for future work. These include incorporating high-level directional commands, addressing the sim-to-real transfer gap, and exploring the use of neural networks to learn and model the robot’s dynamics.

\textbf{Chapter 6} concludes the thesis by summarizing the main contributions and findings, reinforcing the value of the hybrid approach in balancing performance, stability, and adaptability in dynamic environments.
