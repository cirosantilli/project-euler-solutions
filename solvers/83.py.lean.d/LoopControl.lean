import solvers.«83.py.lean.d».Pop

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

def loopBody : List Stmt := match loopTail with
  | .whileDo _ body _ :: _ => body
  | _ => []
def afterLoop : List Stmt := loopTail.drop 1
abbrev loopStatement : Stmt := .whileDo (.name "heap") loopBody []
abbrev loopResume : Stmt := Calls.loopContinuation loopBody loopStatement
abbrev visitedExpressionAt : Expr := .subscript (.name "visited") (.name "u")
def encodeVisited (values : List Bool) : List Mutable.Value := values.map fun b => .scalar (.bool b)

theorem loop_source : loopTail = loopStatement :: afterLoop := rfl
theorem loop_body_source : loopBody = popAssignment ::
    .ifThen visitedExpressionAt [.continueLoop] [] :: loopBody.drop 2 := rfl

theorem visited_read (heap : Mutable.Heap) (env : HeapCalls.Env) (address index : Nat) (values : List Bool)
    (found : heap[address]? = some (.list (encodeVisited values))) (inside : index < values.length)
    (localVisited : env "visited" = some (.ref address)) (localIndex : env "u" = some (number index)) :
    Evaluates program env visitedExpressionAt heap (.scalar (.bool values[index])) heap := by
  apply Evaluates.subscript (by intros a b c impossible; cases impossible)
    (Evaluates.lookup localVisited) (Evaluates.lookup localIndex)
  simpa only [encodeVisited, List.getElem_map] using
    Mutable.getItem_list heap address index (encodeVisited values) found (by simpa [encodeVisited] using inside)

/-- A stale entry executes the real `continue`, skipping the mark, goal test, and all neighbor updates. -/
theorem stale_branch_of_next (heap after : Mutable.Heap) (env : HeapCalls.Env)
    (address index : Nat) (values : List Bool) (answer : Mutable.Value)
    (found : heap[address]? = some (.list (encodeVisited values))) (inside : index < values.length)
    (done : values[index] = true)
    (localVisited : env "visited" = some (.ref address)) (localIndex : env "u" = some (number index))
    (next : BlockReturns program env loopTail heap answer after) :
    BlockReturns program env ((loopBody.drop 1 ++ [loopResume]) ++ afterLoop) heap answer after := by
  have read := visited_read heap env address index values found inside localVisited localIndex
  rw [done] at read
  change BlockReturns program env
    (.ifThen visitedExpressionAt [.continueLoop] [] :: (loopBody.drop 2 ++ [loopResume]) ++ afterLoop)
    heap answer after
  apply BlockReturns.branch (flag := true) read rfl
  apply BlockReturns.continue_loop (next := loopStatement) (following := afterLoop) (by rfl)
  simpa only [loop_source] using next

theorem loop_enter (heap after : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (root : Mutable.Value) (rest : List Mutable.Value) (answer : Mutable.Value)
    (localHeap : env "heap" = some (.ref address))
    (found : heap[address]? = some (.list (root :: rest)))
    (iteration : BlockReturns program env ((loopBody ++ [loopResume]) ++ afterLoop) heap answer after) :
    BlockReturns program env loopTail heap answer after := by
  rw [loop_source]
  apply BlockReturns.while_loop (flag := true) (Evaluates.lookup localHeap)
    (by simp [Mutable.truth, found])
  exact iteration

/-- Settling a fresh vertex writes only its visited-list cell. -/
theorem mark_visited (heap : Mutable.Heap) (env : HeapCalls.Env) (address index : Nat) (values : List Bool)
    (found : heap[address]? = some (.list (encodeVisited values))) (inside : index < values.length)
    (localVisited : env "visited" = some (.ref address)) (localIndex : env "u" = some (number index)) :
    BlockContinues program env [.assign [visitedExpressionAt] (.constant (.bool true))] heap env
      (heap.set address (.list (encodeVisited (values.set index true)))) := by
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible) Evaluates.constant
    (.cons (Evaluates.lookup localVisited) (.cons (Evaluates.lookup localIndex) .nil))
  have nonnegative : ¬ (index : Int) < 0 := by omega
  have bounded : ¬ values.length ≤ index := by omega
  simp [Mutable.setItem, Mutable.index, found, encodeVisited, nonnegative, bounded, Except.map, List.map_set]

end solvers.«83»
