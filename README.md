# A Hybrid Approach for Balancing a Two-Wheeled Humanoid Robot

## Abstract
Balancing a two-wheeled humanoid robot on uneven terrain presents significant challenges due to its nonlinear, underactuated dynamics and environmental unpredictability. This project proposes a hybrid control framework that integrates classical \ac{PID} control, a specifically designed action-mapping strategy, and \ac{RL} using \ac{PPO}. The \ac{PID} controller and action-mapping provide a structured baseline for initial stabilization, while the \ac{RL} component refines the control policy. This “guided learning” approach significantly enhances training efficiency and policy robustness compared to purely \ac{PID} or \ac{RL}-based methods. Simulation results demonstrate that the proposed method achieves reliable and consistent balancing performance across a variety of uneven terrains, outperforming the standard \ac{PID} controller in terms of stability and balancing success rate. These findings highlight the effectiveness of combining domain knowledge with learning-based techniques for advanced robotic control tasks.
## Results

### A Miraculous Recovery  
Demonstration of the robot regaining balance after a near fall.  
![Miraculous Recovery](./simulation_result/A%20Miraculous%20Recovery_lowquality.gif)

---

### Baseline Stability  
Baseline performance on uneven terrain using the hybrid controller.  
![Baseline Stability](./simulation_result/Baseline%20Stability%20on%20Uneven%20Terrainlow%20quality.gif)

---

### Destabilization  
Failure scenario: loss of traction on uneven terrain leads to destabilization.  
![Destabilization](./simulation_result/Destabilization%20Due%20to%20Loss%20of%20Traction%20on%20Uneven%20Terrain%20low%20quality.gif)

---

### Flat Ground Comparison  
Comparison between pure PID (left) and DRL-based control (right) on flat terrain.  
![Flat Ground](./simulation_result/pid%20and%20drl%20on%20flat%20ground.gif)

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