# Neural Architecture Search For Image Classification
Implementing Neural Architecture Search for CNN

```latex
\begin{algorithm}
\begin{algorithm}
\caption{Input: (S size of the population), ($\lambda$ normalization parameter), (PR percentage of population selected by parent selection), (N number of epochs),(NPT max size of Phenotype) , and (G number of generations)}
\begin{algorithmic}[]
\STATE Initialize randomly a population PG of size S of Genotypes
\STATE Generate a population PF of Neural Network with the genotypes 
\STATE Evaluate them with the objective function F

\FOR{i = 0 to G}

\IF {Parent Selection == True}
\STATE compute the size of the output individuals'list Sn = S * PR
\STATE randomly select Sn individuals from PG and store them in a new list PG1
\STATE F1 = corresponding F's part
\RETURN PG1, F1
\ENDIF

\IF {Recombination Step == True}
\STATE initialize an empty list of candidate solutions CS
\FOR{j to (S//2)}
\STATE  randomly sample x1, x2 from PG1
\STATE  randomly sample index I from 0 to len(x1)
\STATE x1 = concatenate(x1[:I], x2[I:])
\STATE x2 = concatenate(x2[:I], x1[I:])
\STATE append x1,x2 to CS
\ENDFOR
\RETURN CS
\STATE randomly select Sn individuals from PG and store them in a new list PG1
\RETURN PG1
\ENDIF

\IF {Mutation Step == True}
\STATE initialize an empty list of candidate solutions MCS
\FOR{x in PG1)}
\STATE  randomly sample index I from 0 to len(x)
\STATE access to genetype model space MG of Ith gene of x 
\STATE randomly sample R from M
\STATE x[I] = R
\STATE append x to MCS
\ENDFOR
\RETURN MCS
\ENDIF

\IF{Encoding Step == True}
\STATE initialize an empty list NNP
\FOR{x in MCS}
\STATE check if MCS are valid genotypes
\STATE get the phenotype model space MF
\STATE save a dictionary CF of configuration indexing MF with x
\STATE Initialize a CNN with configuration CF
\STATE append CNN to NNP
\ENDFOR
\RETURN NNP
\ENDIF
\STATE initialize a list FV
\FOR{m in NNP}
\STATE train m for N epochs
\STATE NP = number of parameters
\STATE access to the last validation result V
\STATE $F(m) = V + \lambda * (NP/NPT) $
\STATE append F(m) to FV
\ENDFOR

\IF{Survivor Selection == True}
\STATE SP = concatenate(PG1, MCS)
\STATE FP = concatenate(F1, FV)
\STATE sort AP based on FP
\STATE Select S best elements from SP and their value from FP, delete the rest
\RETURN SP1, FP1
\ENDIF

\ENDFOR
\RETURN MCS, FV

\end{algorithmic}
\end{algorithm}
\end{algorithm}
