# A Hybrid Approach for Balancing a Two-Wheeled Humanoid Robot

## Abstract
Balancing a two-wheeled humanoid robot on uneven terrain presents significant
challenges due to its nonlinear, underactuated dynamics and environmental unpredictability. This project proposes a hybrid control framework that integrates classical Proportional-Integral-Derivative (PID) control, a specifically designed action-
mapping strategy, and reinforcement learning (RL) using Proximal Policy Optimization (PPO). The PID controller and action-mapping provide a structured baseline
for initial stabilization, while the RL component refines the control policy, using
Kullback–Leibler (KL) divergence as a performance metric to guide exploration.
This “guided learning” approach significantly enhances training efficiency and policy robustness compared to purely RL-based methods, which often suffer from sparse rewards and unstable convergence. Simulation results demonstrate that the proposed
method achieves reliable and consistent balancing performance across a variety of
uneven terrains, outperforming standard PPO in both training speed and stability. These findings highlight the effectiveness of combining domain knowledge with
learning-based techniques for advanced robotic control tasks.

# Report Outline

## 1. Introduction
- ### 1.1 Motivation
- ### 1.2 Problem Statement
- ### 1.3 Research Objectives
- ### 1.4 Contributions

## 2. Background
- ### 2.1 Classical Control Approaches: PID Control
- ### 2.2 Deep Reinforcement Learning in Robotics
- ### 2.3 Bellman Equation
- ### 2.4 Temporal Difference (TD) Error
- ### 2.5 Policy Gradient Methods
- ### 2.6 Proximal Policy Optimization (PPO)

## 3. Methodology
- ### 3.1 Hybrid Control Approaches
- ### 3.2 Implementation
- ### Continue...

## 4. Results

📹 **[Simulation Results Available on LinkedIn →](https://www.linkedin.com/feed/update/urn:li:activity:7311644777316306944/)**

## 5. Discussion
- ### 5.1 Limitations
- ### 5.2 Future Work

## 6. Conclusions
- ### 6.1 Conclusion