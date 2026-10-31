import solvers.«83.py.lean.d».Coordinates

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

def freshPrefix : List Stmt :=
  [.ifThen visitedExpressionAt [.continueLoop] [],
   .assign [visitedExpressionAt] (.constant (.bool true)),
   .ifThen (.compare (.name "u") [(.eq, .name "goal")]) [.return (.name "d")] [],
   coordinatesAssignment]

theorem fresh_source : loopBody.drop 1 = freshPrefix ++ neighborsCode := rfl

/-- A fresh non-goal pop reaches the neighbor branches after the actual visited write and divmod allocation. -/
theorem fresh_prefix (heap : Mutable.Heap) (env : HeapCalls.Env) (address index goal width : Nat)
    (flags : List Bool) (found : heap[address]? = some (.list (encodeVisited flags)))
    (inside : index < flags.length) (waiting : flags[index] = false) (notGoal : index ≠ goal)
    (positive : 0 < width)
    (localVisited : env "visited" = some (.ref address)) (localIndex : env "u" = some (number index))
    (localGoal : env "goal" = some (number goal)) (localWidth : env "m" = some (number width)) :
    BlockContinues program env freshPrefix heap
      (assignNames env ["r", "c"] [number (index / width : Nat), number (index % width : Nat)])
      (heap.set address (.list (encodeVisited (flags.set index true))) ++
        [.tuple [number (index / width : Nat), number (index % width : Nat)]]) := by
  have read := visited_read heap env address index flags found inside localVisited localIndex
  rw [waiting] at read
  have skipped : BlockContinues program env [.ifThen visitedExpressionAt [.continueLoop] []] heap env heap :=
    BlockContinues.branch (flag := false) read rfl BlockContinues.nil
  have marked := mark_visited heap env address index flags found inside localVisited localIndex
  let middle := heap.set address (.list (encodeVisited (flags.set index true)))
  have tested : Evaluates program env (.compare (.name "u") [(.eq, .name "goal")])
      middle (.scalar (.bool false)) middle := by
    apply Evaluates.compare (Evaluates.lookup localIndex) (Evaluates.lookup localGoal)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, notGoal]
  have skippedGoal : BlockContinues program env
      [.ifThen (.compare (.name "u") [(.eq, .name "goal")]) [.return (.name "d")] []] middle env middle :=
    BlockContinues.branch (flag := false) tested rfl BlockContinues.nil
  exact skipped.append (marked.append (skippedGoal.append
    (coordinates_result middle env index width positive localIndex localWidth)))

theorem fresh_branch_of_next (heap after : Mutable.Heap) (env : HeapCalls.Env)
    (address index goal width : Nat) (flags : List Bool) (answer : Mutable.Value)
    (found : heap[address]? = some (.list (encodeVisited flags)))
    (inside : index < flags.length) (waiting : flags[index] = false) (notGoal : index ≠ goal)
    (positive : 0 < width)
    (localVisited : env "visited" = some (.ref address)) (localIndex : env "u" = some (number index))
    (localGoal : env "goal" = some (number goal)) (localWidth : env "m" = some (number width))
    (next : BlockReturns program
      (assignNames env ["r", "c"] [number (index / width : Nat), number (index % width : Nat)])
      ((neighborsCode ++ [loopResume]) ++ afterLoop)
      (heap.set address (.list (encodeVisited (flags.set index true))) ++
        [.tuple [number (index / width : Nat), number (index % width : Nat)]]) answer after) :
    BlockReturns program env ((loopBody.drop 1 ++ [loopResume]) ++ afterLoop) heap answer after := by
  rw [fresh_source, List.append_assoc, List.append_assoc]
  exact (fresh_prefix heap env address index goal width flags found inside waiting notGoal positive
    localVisited localIndex localGoal localWidth).returns (by simpa only [List.append_assoc] using next)

end solvers.«83»
