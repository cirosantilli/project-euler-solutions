import solvers.«84.py.lean.d».StationaryRow
import ProjectEulerSolutions.Lib.MonopolyMatrixLayout

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode)
set_option Elab.async false

export ProjectEulerSolutions.MonopolyMatrixLayout (rowReferences matrixHeap matrix_length references_next matrix_next matrix_found matrix_row_found)

end solvers.«84».stationary_weights
