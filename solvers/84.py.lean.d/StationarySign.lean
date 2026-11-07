import solvers.«84.py.lean.d».StationaryBackLoop

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def negateWeights : Expr := .listComp (.unaryOp .uSub (.name "weight")) [("weight", .name "weights", [])]

theorem negate_scan (base : Mutable.Heap) (weights : Nat) (before remaining output : List Int)
    (found : base[weights]? = some (.list (TriangleBufferCalls.encode (before ++ remaining))))
    (env : HeapCalls.Env) (globals : Mutable.Globals) :
    ∃ afterEnv, BlockContinues program env
      [.sequenceStepLoop "weight" (.ref weights) before.length [.comprehensionAppend base.length (.unaryOp .uSub (.name "weight"))] []]
      (TriangleBufferCalls.working base output) afterEnv (TriangleBufferCalls.working base (output ++ remaining.map Neg.neg)) globals := by
  induction remaining generalizing before output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.map_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := TriangleBufferCalls.encode before)
      · have kept := TriangleBufferCalls.prefix_found base output weights _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp [TriangleBufferCalls.encode]
      · exact .nil
  | cons head tail ih =>
      let boundEnv := setLocal env "weight" (.scalar (.int head))
      have emitted := BlockContinues.comprehension_append (module := program) (globals := globals)
        (Evaluates.unary (op := .uSub) (Evaluates.lookup (env := boundEnv) (name := "weight") rfl) rfl)
        (TriangleBufferCalls.found base output)
      have write : (TriangleBufferCalls.working base output).set base.length
          (.list (TriangleBufferCalls.encode output ++ [.scalar (.int (-head))])) =
          TriangleBufferCalls.working base (output ++ [-head]) := by
        simp [TriangleBufferCalls.working, TriangleBufferCalls.encode]
      rw [write] at emitted
      obtain ⟨afterEnv, rest⟩ := ih (before ++ [head]) (output ++ [-head])
        (by simpa only [List.append_assoc, List.singleton_append] using found) boundEnv
      refine ⟨afterEnv, ?_⟩
      apply BlockContinues.sequence_step (items := TriangleBufferCalls.encode (before ++ head :: tail)) (value := .scalar (.int head))
      · have kept := TriangleBufferCalls.prefix_found base output weights _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp [TriangleBufferCalls.encode]
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true,
          if_false, List.map_cons, List.append_assoc, List.singleton_append, List.length_append, List.length_singleton] using emitted.append rest

private theorem sequence_list (heap : Mutable.Heap) (address : Nat) (items : List Mutable.Value)
    (found : heap[address]? = some (.list items)) :
    Mutable.sequence (.ref address) heap = some (.ok items, heap) := by
  simp [Mutable.sequence, found]

theorem negate_result (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getWeights : env "weights" = some (.ref weights)) :
    Evaluates program env negateWeights heap (.ref heap.length)
      (heap ++ [.list (List.ofFn fun index => .scalar (.int (-values index)))]) globals := by
  obtain ⟨afterEnv, ran⟩ := negate_scan heap weights [] (List.ofFn values) []
    (by simpa only [List.nil_append, TriangleBufferCalls.encode, List.map_ofFn] using view.weightsFound)
    (comprehensionEnv env [("weight", .name "weights", [])]) globals
  apply Evaluates.list_comp_sequence (finalEnv := afterEnv) (items := List.ofFn fun index => .scalar (.int (values index)))
    (by intros; simp) (.lookup getWeights) (sequence_list heap weights _ view.weightsFound)
  simpa only [TriangleBufferCalls.working, TriangleBufferCalls.encode, List.nil_append, List.map_ofFn] using ran

def signedWeights (scale : Int) (values : Fin 120 → Int) : Fin 120 → Int :=
  fun index => if scale < 0 then -values index else values index

theorem sign_result (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int) (scale : Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView matrix values heap root addresses weights) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getWeights : env "weights" = some (.ref weights)) (getScale : env "scale" = some (.scalar (.int scale))) :
    ∃ afterHeap address extra,
      BlockReturns program env ((executableBody definition.2).drop 9) heap (.ref address) afterHeap globals ∧
      afterHeap = heap ++ extra ∧
      afterHeap[address]? = some (.list (List.ofFn fun index => .scalar (.int (signedWeights scale values index)))) := by
  have tested : Evaluates program env (.compare (.name "scale") [(.lt, .constant (.int 0))]) heap
      (.scalar (.bool (decide (scale < 0)))) heap globals :=
    Evaluates.compare (.lookup getScale) Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases negative : scale < 0
  · refine ⟨heap ++ [.list (List.ofFn fun index => .scalar (.int (-values index)))], heap.length,
      [.list (List.ofFn fun index => .scalar (.int (-values index)))], ?_, rfl, ?_⟩
    · apply BlockReturns.branch (flag := true) (by simpa only [negative, decide_true] using tested) rfl
      exact (BlockContinues.assign (name := "weights") (negate_result matrix values heap root weights addresses view env globals getWeights)).returns
        (BlockReturns.return_value (.lookup rfl))
    · rw [List.getElem?_append_right (Nat.le_refl heap.length)]
      simp only [signedWeights, if_pos negative, Nat.sub_self, List.getElem?_cons_zero]
  · refine ⟨heap, weights, [], ?_, by simp, ?_⟩
    · apply BlockReturns.branch (flag := false) (by simpa only [negative, decide_false] using tested) rfl
      exact BlockReturns.return_value (.lookup getWeights)
    · simpa only [signedWeights, if_neg negative] using view.weightsFound

end solvers.«84».stationary_weights
