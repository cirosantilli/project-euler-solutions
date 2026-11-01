import solvers.«79.topo.initial.source»
import ProjectEulerStatements.Lib.HeapqSelection
import ProjectEulerStatements.Lib.HeapqPush

/-! Source loop updates preserve the priority queue and the separate output and count objects. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap
set_option Elab.async false

def workspace (base : Mutable.Heap) (queue output : List String)
    (degrees : List (Constant × Mutable.Value)) : Mutable.Heap :=
  base ++ [.list (values queue), .list (values output), .dict degrees]

def popBody : List Stmt :=
  [.assign [.name "u"] (.builtinCall .heapqPop [.name "heap"]),
   .expr (.call (.attribute (.name "out") "append") [.name "u"])]

def successorBody : List Stmt :=
  [.augAssign (.subscript (.name "indeg2") (.name "v")) .sub (.constant (.int 1)),
   .ifThen (.compare (.subscript (.name "indeg2") (.name "v"))
     [(.eq, .constant (.int 0))])
     [.expr (.builtinCall .heapqPush [.name "heap", .name "v"])] []]

def loopBody : List Stmt :=
  popBody ++ [.forIn (.name "v") (.subscript (.name "adj") (.name "u")) successorBody []]

theorem loop_source : topoCode[4]? = some (.whileDo (.name "heap") loopBody []) := rfl

/-- The actual pop and append statements emit the least eligible vertex. -/
theorem pop_emit (base : Mutable.Heap) (env : HeapCalls.Env)
    (nodes : List String) (edges : List (String × String)) (vertex : String)
    (rest output : List String) (degrees : List (Constant × Mutable.Value))
    (getQueue : env "heap" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (contents : (vertex :: rest).Perm
      (ProjectEulerStatements.TopologicalOrder.eligible nodes edges))
    (ordered : Ordered (vertex :: rest)) :
    ∃ queue, BlockContinues program env popBody
        (workspace base (vertex :: rest) output degrees) (setLocal env "u" (text vertex))
        (workspace base queue (output ++ [vertex]) degrees) ∧
      ProjectEulerStatements.TopologicalOrder.choose nodes edges = some vertex ∧
      queue.Perm rest ∧ Ordered queue := by
  obtain ⟨queue, popped, chosen, permutation, orderedQueue⟩ :=
    heappop_greedy_ordered text strings_encodeOrder nodes edges vertex rest
      (workspace base (vertex :: rest) output degrees) base.length contents ordered
      (by simp [workspace])
  have first : BlockContinues program env [popBody[0]]
      (workspace base (vertex :: rest) output degrees) (setLocal env "u" (text vertex))
      (workspace base queue output degrees) := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons (Evaluates.lookup getQueue) .nil)
    rw [primitive_heapq_pop]
    simpa [workspace, values] using popped
  have second : BlockContinues program (setLocal env "u" (text vertex)) [popBody[1]]
      (workspace base queue output degrees) (setLocal env "u" (text vertex))
      (workspace base queue (output ++ [vertex]) degrees) := by
    apply BlockContinues.discard
    have appended := Evaluates.append (module := program) (globals := [])
      (env := setLocal env "u" (text vertex)) (base := .name "out")
      (expression := .name "u") (heap := workspace base queue output degrees)
      (current := values output)
      (Evaluates.lookup (by simpa [setLocal] using getOutput))
      (show (workspace base queue output degrees)[base.length + 1]? =
        some (.list (values output)) by simp [workspace])
      (Evaluates.lookup (value := text vertex) (by simp [setLocal]))
      (by simp [workspace])
    simpa [workspace, values] using appended
  exact ⟨queue, first.append second, chosen, permutation, orderedQueue⟩

def decrease (degrees : List (Constant × Mutable.Value)) (vertex : String) (count : Int) :
    List (Constant × Mutable.Value) :=
  ScalarDict.insert degrees (.string vertex) (number (count - 1))

/-- A successor is enqueued exactly at its zero crossing; Python subtraction stays integral. -/
theorem discharge_one (base : Mutable.Heap) (env : HeapCalls.Env)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (vertex : String) (count : Int)
    (getQueue : env "heap" = some (.ref base.length))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getVertex : env "v" = some (text vertex))
    (lookup : ScalarDict.lookup degrees (.string vertex) = some (number count))
    (ordered : Ordered queue) :
    ∃ nextQueue, BlockContinues program env successorBody (workspace base queue output degrees)
        env (workspace base nextQueue output (decrease degrees vertex count)) ∧
      nextQueue.Perm (queue ++ if count = 1 then [vertex] else []) ∧ Ordered nextQueue := by
  let changed := decrease degrees vertex count
  have found : (workspace base queue output degrees)[base.length + 2]? =
      some (.dict degrees) := by simp [workspace]
  have decrement : BlockContinues program env [successorBody[0]]
      (workspace base queue output degrees) env (workspace base queue output changed) := by
    apply BlockContinues.augItem (updated := number (count - 1))
      (afterOperation := workspace base queue output degrees)
      (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getDegrees) (Evaluates.lookup getVertex)
      (Mutable.getItem_dictionary _ _ degrees found (.string vertex) lookup) Evaluates.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · simpa [workspace, changed, decrease] using Mutable.setItem_dictionary
        (workspace base queue output degrees) (base.length + 2) degrees found
        (.string vertex) (number (count - 1))
  have read : Evaluates program env (.subscript (.name "indeg2") (.name "v"))
      (workspace base queue output changed) (number (count - 1))
      (workspace base queue output changed) := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getDegrees) (Evaluates.lookup getVertex)
    exact Mutable.getItem_dictionary _ _ changed (by simp [workspace])
      (.string vertex) (by simp [changed, decrease, ScalarDict.lookup_insert_self])
  have tested : Evaluates program env
      (.compare (.subscript (.name "indeg2") (.name "v")) [(.eq, .constant (.int 0))])
      (workspace base queue output changed) (.scalar (.bool (decide (count = 1))))
      (workspace base queue output changed) := by
    apply Evaluates.compare read Evaluates.constant
    have zero : count - 1 = 0 ↔ count = 1 := by omega
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]
  by_cases crossing : count = 1
  · obtain ⟨nextQueue, pushed, permutation, orderedQueue⟩ :=
      heappush_ordered text strings_encodeOrder (workspace base queue output changed)
        base.length queue vertex ordered (by simp [workspace])
    refine ⟨nextQueue, decrement.append ?_, by simpa [crossing] using permutation, orderedQueue⟩
    apply BlockContinues.branch (flag := true) (by simpa [crossing] using tested) rfl
    apply BlockContinues.discard
    apply Evaluates.builtin
      (.cons (Evaluates.lookup getQueue) (.cons (Evaluates.lookup getVertex) .nil))
    rw [primitive_heapq_push]
    simpa [workspace, values] using pushed
  · refine ⟨queue, decrement.append ?_, by simp [crossing], ordered⟩
    exact BlockContinues.branch (flag := false) (by simpa [crossing] using tested) rfl .nil

end solvers.«79».topo
