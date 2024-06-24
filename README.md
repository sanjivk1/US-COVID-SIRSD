# Python package to fit and forecast the COVID-19 infection dynamics in US Counties.
Based on the paper "Hidden ParametersHidden Parameters Impacting Resurgence of SARS-CoV-2 Pandemic."


The equations used are:

\begin{figure}[h!]
\begin{center}
\resizebox{0.6\textwidth}{!}{
\begin{tikzpicture}[->,>=stealth',shorten >=2pt, line width=2pt, 
                                  node distance=2cm, style ={minimum size=20mm}]
\tikzstyle{every node}=[font=\Large]
\node [rectangle, draw, font=\Huge] (S) {$S$};
\node [rectangle, draw, font=\Huge] (L) [below=of S] {$L$};
\node [rectangle, draw, font=\Huge] (I) [right=of S] {$I$};
\node [rectangle, draw, font=\Huge] (I_H) [right=of I] {$I_H$};
\node [rectangle, draw, font=\Huge] (I_N) [below=of I_H] {$I_N$};
\node [rectangle, draw, font=\Huge] (D) [right=of I_H] {$D$};
\node [rectangle, draw, font=\Huge] (R) [right=of I_N] {$R$};

\draw[->] (L) -- node [right=-0.4] {$\kappa_1$} (S);
\draw[->] (S) -- node [above=-0.5] {
$\beta SI/N$
} (I);
\draw[->] (I) -- node [above=-0.5] {$ \gamma I$
} (I_H);
\draw[->] (I) -- node [left=0.2] {$\gamma_2 I$
} (I_N);
\draw[->] (I_H) --node [above=-0.5] {$\alpha_2 I_H$
} (D);
\draw[->] (I_H) --node [right] {$\alpha_1 I_H$
} (R);
\draw[->] (I_N) --node [below=-0.5] {$\alpha_3 I_N$
} (R);
\end{tikzpicture}
}
\end{center}
    \caption{The SIR-SD-L Compartment Model incorporating lockdown removal}
    \label{fig:model}
\end{figure}

## Fitting:
Uses python package scipy.optimize with randomized initial parameters and their ranges with multiple threads for
multiple starting points.

The fitting is in two phases: 
The first phase starts with the date at which the county has 5 confirmed cases and ends on 15th May, 2020.
The second phase starts at the time of reopening of the county after lifting restrictions. This reopen date varied for each county.
In the second phase a linear release of population was fitted.

## Forecast: 
The forecast uses the best parameters acquired by the fitting.

## How to use
The main code to be used is : MultiPhase_JHU50.py

The key subroutine used is the minimize from scipy.optimize

The following calls in the main code can be used:

1. fit_all_init() will fit the intiial phase only. 

2. fit_all_combined() will fit  the first and second  phase . It minimizes a loss function obtained from R^2 values.

More details on different parameters can be found in the code.