\chapter{Results\markboth{Results}{}}

To demonstrate the stability of the hybrid control approach, we conducted 56 trials for each method—hybrid and PID—on uneven terrain. Each trial lasted 10 seconds, during which the robot was evaluated on its ability to maintain upright posture. The terrain was randomly regenerated for each trial, with a maximum slope of 17.73° per period.

At the beginning of each trial, the robot was placed at a fixed height of 0.25 meters above the terrain, with slight randomness added to its initial state to prevent it from sinking into the surface due to terrain variation.

Below, we present 5 representative samples selected from the trials to illustrate a range of outcomes observed under both control schemes. In addition, \textbf{Table 1} summarizing height fluctuations across all 112 trials. The hybrid controller consistently demonstrates lower falling rate and longer duration even for falling cases compared to the traditional PID approach.

\vspace{1em}
\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{images/height_position_comparison3.png}
    \caption{Height trajectory comparison}
    \label{fig:enter-label}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{images/height_position_comparison12.png}
    \caption{Height trajectory comparison}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{images/height_position_comparison34.png}
    \caption{Height trajectory comparison}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{images/height_position_comparison38.png}
    \caption{Height trajectory comparison}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{images/height_position_comparison42.png}
    \caption{Height trajectory comparison}
\end{figure}


\begin{table}
\centering
\begin{threeparttable}
\caption{Performance Comparison: Hybrid vs PID Controller}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Hybrid Controller} & \textbf{PID Controller} \\
\midrule
Falling Rate (\%) & 41.0714\% & 58.9285\% \\
Average Fall Duration (\%) & 39.9835\% & 34.7855\% \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Falling Rate}: Percentage of trials in which the robot fell; smaller values indicate better performance.
\item \textit{Average Fall Duration}: Average duration before falling expressed as a percentage of the total normal simulation time; larger values indicate better performance.
\end{tablenotes}
\end{threeparttable}
\end{table}


These results highlight the hybrid controller's improved resilience and performance under challenging terrain conditions, supporting its potential for more robust robotic locomotion.
