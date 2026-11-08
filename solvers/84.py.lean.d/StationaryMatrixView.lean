import solvers.«84.py.lean.d».StationaryNormalization
import ProjectEulerSolutions.Lib.MonopolyMatrixLayout

namespace solvers.«84».stationary_weights
open FormalSW.PyAst ProjectEulerSolutions
open TriangleBufferCalls (encode)
set_option Elab.async false

export ProjectEulerSolutions.MonopolyMatrixLayout (augmentedRow augmentedRow_length augmentedRow_coefficient augmentedRow_constant augmentedRow_ordinary augmentedRow_normalization MatrixView prepared_view)

end solvers.«84».stationary_weights
