import solvers.«83.py.lean.d».FreshState

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

theorem LoopLocals.pop {env : HeapCalls.Env}
    {matrixAddress distanceAddress visitedAddress queueAddress height width : Nat}
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress height width)
    (distance : Int) (index : Nat) :
    LoopLocals (assignNames env ["d", "u"] [number distance, number index])
      matrixAddress distanceAddress visitedAddress queueAddress height width := by
  apply locals.frame
  intro name notDistance notIndex _ _ _ _
  simp [assignNames, setLocal, notDistance, notIndex]

theorem LoopLocals.coordinates {env : HeapCalls.Env}
    {matrixAddress distanceAddress visitedAddress queueAddress height width : Nat}
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress height width) (index : Nat) :
    LoopLocals (preparedEnv env index width) matrixAddress distanceAddress visitedAddress queueAddress height width := by
  apply locals.frame
  intro name _ _ notRow notColumn _ _
  simp [preparedEnv, assignNames, setLocal, notRow, notColumn]

theorem LoopLocals.neighbor_locals {env : HeapCalls.Env}
    {matrixAddress distanceAddress visitedAddress queueAddress height width : Nat}
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress height width)
    (index : Nat) (distance : Int) (localDistance : env "d" = some (number distance)) :
    NeighborLocals (preparedEnv env index width) matrixAddress distanceAddress queueAddress
      (index / width) (index % width) height width distance := by
  refine ⟨?_, ?_, ?_, rfl, rfl, ?_, ?_, ?_⟩
  · simpa [preparedEnv, assignNames, setLocal] using locals.matrix
  · simpa [preparedEnv, assignNames, setLocal] using locals.buffer
  · simpa [preparedEnv, assignNames, setLocal] using locals.queue
  · simpa [preparedEnv, assignNames, setLocal] using locals.height
  · simpa [preparedEnv, assignNames, setLocal] using locals.width
  · simpa [preparedEnv, assignNames, setLocal] using localDistance

end solvers.«83»
