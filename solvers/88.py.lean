import solvers.«88.py.lean.d».Execution

namespace solvers.«88».solve
open FormalSW.PyAst ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

theorem «Returns» (maxSize : Int) :
    (SetCalls.call program "solve" [integer maxSize] []).Returns := by
  intro choices
  obtain ⟨heap, returned⟩ := result maxSize []
  exact ⟨number (answer maxSize.toNat), [], heap, 0, returned.finishes choices⟩

theorem «Supported» (maxSize : Int) :
    (SetCalls.call program "solve" [integer maxSize] []).Supported := by
  intro choices
  exact («Returns» maxSize choices).supported

end solvers.«88».solve
