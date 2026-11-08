import solvers.«84.py.lean.d».Encoding
import solvers.«84.py.lean.d».LandingCommunity
import ProjectEulerSolutions.Lib.DiceSums
import ProjectEulerSolutions.Lib.MonopolyStationary

namespace solvers.«84».encode_state
open FormalSW.PyAst HeapCalls

theorem «Returns» (square doubles : Int) :
    (HeapCalls.call program "encode_state"
      [.scalar (.int square), .scalar (.int doubles)] []).Returns :=
  (result square doubles []).returns

theorem «Supported» (square doubles : Int) :
    (HeapCalls.call program "encode_state"
      [.scalar (.int square), .scalar (.int doubles)] []).Supported :=
  («Returns» square doubles).supported

end solvers.«84».encode_state
