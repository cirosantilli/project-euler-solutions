import solvers.«84.py.lean.d».StationaryInvariant
import ProjectEulerSolutions.Lib.MutableIntegerMatrix
import ProjectEulerSolutions.Lib.MonopolyMatrixLayout

namespace solvers.«84».stationary_weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

export ProjectEulerSolutions.MonopolyMatrixLayout (rowAddresses rowAddresses_distinct rowAddresses_after prepared_references augmentedRow_ofFn prepared_mutable_view)

end solvers.«84».stationary_weights
