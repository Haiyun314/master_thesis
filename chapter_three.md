\chapter{Methodology\markboth{Methodology}{}}
\section{Hybrid Control Approaches}

Training deep reinforcement learning (RL) agents often suffers from sparse rewards, low sample efficiency, and convergence challenges. To address these issues, we propose a hybrid control approach that combines traditional control methods with reinforcement learning to guide the agent toward better strategies during training. \cite{hynes2020optimising}

Specifically, we define the hybrid policy as:

\[
\pi_{\text{hybrid}} = (1 - \beta) \cdot \pi_{\text{PID}} + \beta \cdot \pi_{\text{RL}}
\]

Here, the hyperparameter \( \beta \in [0, 1] \) balances the contribution of the PID controller and the RL policy. A smaller \( \beta \) favors the PID component, while a larger \( \beta \) increases the influence of the learned RL policy.

\subsubsection*{Training Procedure}

The hybrid approach is implemented in two stages:

\begin{enumerate}
    \item \textbf{Initialization on a flat plane:} The RL policy is trained in a simple, flat environment. During this phase, \( \beta \) is initialized at a small value (e.g., 0.05) and gradually increased to 0.2, allowing the RL agent to slowly take more control as it learns.
    
    \item \textbf{Transfer to uneven terrain:} Once initial training stabilizes, the environment is switched to a randomly generated uneven terrain. The same scheduling strategy is applied, starting with \( \beta = 0.05 \) and increasing to 0.2. This gradual transition ensures better generalization and stability in more complex environments.
\end{enumerate}

This hybrid approach leverages the stability of classical control (PID) and the adaptability of reinforcement learning, particularly during early training phases where exploration is critical.

\subsubsection*{A Tailored Action Mapping}

One of the primary challenges in stabilizing a robot on uneven terrain is the shifting of its base of support. A practical approach to mitigate this issue is to adjust the wheel height so that the base of support remains unchanged. Notably, if the hip joint rotates forward by an angle \(\theta\), and the knee joint rotates backward by \(2\theta\), the wheel stays aligned with the body, effectively maintaining a stable base. This insight allows us to reduce the control problem to a mapping from the roll angle to the hip angle. Additionally, to avoid excessively large control signals, we apply a clipping function. Thus, we define this mapping as follows:


\begin{align*}
f(x) &= \mathrm{clip}\left( \arccos \left( \mathrm{clip}\left(1 - \frac{x}{0.7}, -1, 1 \right) \right), -1, 1 \right), \\
\text{where} \quad \tilde{\theta}_{\mathrm{roll}} &= \left| \theta_{\mathrm{roll}} + 1.569959869416047 \right|, \\
x &= \frac{1.4}{\frac{0.25}{1.422} + \frac{1.4}{1.422 \cdot \tan(\tilde{\theta}_{\mathrm{roll}})}}, \\[6pt]
\theta_{\mathrm{hip}} &= f(x), \\
\theta_{\mathrm{knee}} &= -2 \cdot \theta_{\mathrm{hip}}, \\[6pt]
\pi_{\mathrm{tailor}} &=
\begin{bmatrix}
\theta_{\mathrm{hip}} \\
\theta_{\mathrm{knee}}
\end{bmatrix}
\end{align*}


\subsubsection*{Hybrid Control Strategy for Uneven Terrain}
Finally, by integrating the tailored action-mapping strategy with the hybrid control framework, we define the overall control strategy as:
\[
\pi = \beta \cdot 
\begin{bmatrix}
\pi_{\text{PID}} \\
\pi_{\text{tailor}}
\end{bmatrix}
+ (1 - \beta) \cdot \pi_{\text{RL}},
\]
where $\beta \in [0, 1]$ is a blending factor that balances the contributions of the classical and learned control components.


\section{Implementation}

In our implementation, we control a total of six primary joints: two wheels, two knee joints, and two hip joints. The wheel joints are governed by a PID controller, denoted as \(\pi_{\text{PID}}\), which receives the robot's pitch angle as input and outputs symmetric target velocities for the left and right wheels. The PID gains were manually tuned on flat terrain to ensure baseline stability.

The hip and knee joints are controlled using a tailored mapping strategy, denoted as \(\pi_{\text{tailor}}\). This controller maps the robot's roll angle to joint angles that maintain the base of support on uneven terrain, providing reactive stabilization.

For the learning-based components, we adopt an actor-critic architecture. The actor network consists of four fully connected layers with sizes \([64, 128, 128, 64]\), and the critic network uses a \([64, 128, 64]\) structure.

The system state is represented by an 18-dimensional vector:
\[
\mathbf{s} := 
\begin{bmatrix}
\theta_{\text{roll}}, \theta_{\text{pitch}}, \theta_{\text{yaw}}, \dot{v_{\text{roll}}}, \dot{v_{\text{pitch}}}, \dot{v_{\text{yaw}}}, h_{\text{head}}, v_{\text{left}}, v_{\text{right}}, \\
\theta_{\text{lk}}, \theta_{\text{rk}}, \theta_{\text{lh}}, \theta_{\text{rh}}, \theta_{\text{lsh}}, \theta_{\text{rsh}}, \theta_{\text{tilt}}, u_{\text{left}}, u_{\text{right}}
\end{bmatrix}
\]
where each component corresponds to inertial measurements, joint angles, and command inputs.

The action space is defined as an 11-dimensional vector:
\[
\pi_{\text{RL}} := [v_{\text{left}}, v_{\text{right}}, \theta_{\text{lk}}, \theta_{\text{rk}}, \theta_{\text{lh}}, \theta_{\text{rh}}, \theta_{\text{lsh}}, \theta_{\text{rsh}}, \theta_{\text{tilt}}, u_{\text{left}}, u_{\text{right}}]
\]
While some of the action outputs—such as side hip movements and command signals—are not actively utilized in the current stabilization task, they are retained in the action space to allow for future architectural flexibility and potential task extensions.

Accordingly, the reinforcement learning policy optimized in this work focuses on the following subset of actions:

\[
\pi_{\text{RL}} := 
\begin{bmatrix}
v_{\text{left}},\ v_{\text{right}},\ \theta_{\text{lk}},\ \theta_{\text{rk}},\ \theta_{\text{lh}},\ \theta_{\text{rh}}
\end{bmatrix}
\]

where \( v_{\text{left}} \) and \( v_{\text{right}} \) are the target velocities for the left and right wheels respectively, and \( \theta_{\text{lk}}, \theta_{\text{rk}}, \theta_{\text{lh}}, \theta_{\text{rh}} \) represent the desired angles for the left and right knee and hip joints.

\subsubsection*{Reward Function}

The reward \( r \) is defined as

\begin{align*}
r = \; & \min\left(\left|\frac{0.01}{\theta_{\text{pitch}}}\right|, 1\right) \\
& + \min\left(\left|\frac{0.01}{\theta_{\text{roll}} + \theta_{\text{ref}}}\right|, 1\right) \\
& + \min\left(\left|\frac{0.001}{h - h_{\text{ref}}}\right|, 2\right) \\
& - 0.1 \cdot \left(|\theta_{\text{lk}}| + |\theta_{\text{rk}}| + |\theta_{\text{lh}}| + |\theta_{\text{rh}}|\right)
\end{align*}



where

\begin{align*}
\theta{\text{pitch}} &:= \text{IMU pitch angular  (imu\_values[1])}, \\
\theta{\text{roll}}  &:= \text{IMU roll angular  (imu\_values[0])}, \\
\theta_{ref}         &:= 1.56996, \text{ use it to centralize} \\
h                     &:= \text{robot head height (position[2])}, \\
h_{\text{ref}}        &:= 1.5, \text{Standing upright head height}\\
\theta_{\text{lk}}, \theta_{\text{rk}}, \theta_{\text{lh}}, \theta_{\text{rh}} 
                      &:= \text{left/right knee and hip joint angles}.
\end{align*}

\text{Special rewards are given when reaching long durations, e.g., 10 seconds.} \\
\text{An additional reward of } +10 \text{ is granted upon success; otherwise, a penalty of } -100 \text{ is applied.}
