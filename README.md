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

# Outline

## 1. Introduction
- 1.1 Motivation
- 1.2 Problem Statement
- 1.3 Research Objectives
- 1.4 Thesis Outline

## 2. Background
- 2.1 Problem Setting
- 2.2 Classical Control Approaches: PID Control
- 2.3 Deep Reinforcement Learning
  - 2.3.1 Deep Reinforcement Learning Overview
  - 2.3.2 Deep Reinforcement Learning Concepts
- 2.4 Bellman Equation
  - 2.4.1 Existence of Iterative Solution
- 2.5 Temporal Difference (TD) Learning
- 2.6 Policy Gradient Methods
- 2.7 Proximal Policy Optimization (PPO)
  - 2.7.1 Policy Network Optimization
  - 2.7.2 Value Network Optimization
  - 2.7.3 Proximal Policy Optimization

## 3. Methodology
- 3.1 Hybrid Control Approaches
  - 3.1.1 PID-Based Stabilization via Wheel Control
  - 3.1.2 A Tailored Action Mapping
  - 3.1.3 DRL Control
  - 3.1.4 Hybrid Control Strategy for Uneven Terrain
- 3.2 Implementation
  - 3.2.1 Reward Function
  - 3.2.2 Training Procedure

## 4. Results
- 4.1 Impact of Blending Factor on Hybrid Control Stability on Uneven Terrain
- 4.2 Impact of Terrain Slope on Control Performance
- 4.3 Effect of Deterministic vs. Stochastic Policy on Test Performance
  - 4.3.1 Tilting Oscillation
  - 4.3.2 Torque Consumption
  - 4.3.3 2D Locomotion Behavior

## 5. Discussion
- 5.1 Limitations
- 5.2 Future Work
  - 5.2.1 Incorporating Directional Control Commands
  - 5.2.2 Integration of Environmental Sensing
  - 5.2.3 Preliminary Exploration: Learning Local Linear Dynamics Models

## 6. Conclusions
- 6.1 Conclusion
- 6.2 Future Work

## Appendix: Robot Configuration
- A. Robot Platform Specifications
  - A.1 Mechanical Structure
  - A.2 Dimensions
  - A.3 Drive System
  - A.4 Mass Distribution
  - A.5 Sensors
