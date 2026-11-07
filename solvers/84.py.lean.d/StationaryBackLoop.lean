import solvers.«84.py.lean.d».StationaryBackRow

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

/-- The descending source loop computes the mathematical reverse substitution with the same integer divisions. -/
theorem back_scan {sides : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) 119) (values : Fin 120 → Int)
    (count : Nat) (bound : count ≤ 120) (scale : Int)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView state.matrix values heap root addresses weights)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getSize : env "size" = some (.scalar (.int 120))) (getScale : env "scale" = some (.scalar (.int scale))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env [.rangeStepLoop "row" ((count : Int) - 1) (-1) (-1) backBody []] heap afterEnv afterHeap globals ∧
      BackView state.matrix (Bareiss.backwards state.matrix scale count bound values) afterHeap root addresses weights ∧
      (∀ name, name ≠ "row" → afterEnv name = env name) ∧
      ∀ address, weights ≠ address → afterHeap[address]? = heap[address]? := by
  induction count generalizing values heap env with
  | zero =>
      exact ⟨env, heap, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, view,
        fun _ _ => rfl, fun _ _ => rfl⟩
  | succ count ih =>
      let row : Fin 120 := ⟨count, by omega⟩
      let boundEnv := setLocal env "row" (.scalar (.int count))
      have boundMatrix : boundEnv "matrix" = some (.ref root) := by simpa [boundEnv, setLocal] using getMatrix
      have boundWeights : boundEnv "weights" = some (.ref weights) := by simpa [boundEnv, setLocal] using getWeights
      have boundSize : boundEnv "size" = some (.scalar (.int 120)) := by simpa [boundEnv, setLocal] using getSize
      have boundScale : boundEnv "scale" = some (.scalar (.int scale)) := by simpa [boundEnv, setLocal] using getScale
      obtain ⟨written, nextView⟩ := back_row_result state values heap root weights addresses view row scale boundEnv globals
        boundMatrix boundWeights rfl boundSize boundScale
      obtain ⟨afterEnv, afterHeap, rest, finalView, locals, objects⟩ := ih _ (by omega) _ nextView boundEnv boundMatrix boundWeights boundSize boundScale
      refine ⟨afterEnv, afterHeap, ?_, finalView, ?_, ?_⟩
      · simp only [Nat.cast_add, Nat.cast_one, Int.add_sub_cancel]
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, show Calls.loopControl backBody = false from rfl, Bool.false_eq_true, if_false,
          sub_eq_add_neg] using written.append rest
      · intro name notRow
        exact (locals name notRow).trans (by simp [boundEnv, setLocal, notRow])
      · intro address untouched
        exact (objects address untouched).trans (List.getElem?_set_ne untouched)

/-- All 120 source reverse steps recover the proved integer solution at the final determinant scale. -/
theorem back_loop_result {sides : Nat} (state : Bareiss.State (Monopoly.augmentedMatrix sides) 119)
    (heap : Mutable.Heap) (root weights : Nat) (addresses : Fin 120 → Nat)
    (view : BackView state.matrix (fun _ => 0) heap root addresses weights)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref root)) (getWeights : env "weights" = some (.ref weights))
    (getSize : env "size" = some (.scalar (.int 120)))
    (getScale : env "scale" = some (.scalar (.int (state.matrix 119 119)))) :
    ∃ afterEnv afterHeap,
      BlockContinues program env ((executableBody definition.2).drop 8 |>.take 1) heap afterEnv afterHeap globals ∧
      BackView state.matrix state.solution afterHeap root addresses weights ∧
      (∀ name, name ≠ "row" → afterEnv name = env name) ∧
      ∀ address, weights ≠ address → afterHeap[address]? = heap[address]? := by
  obtain ⟨afterEnv, afterHeap, ran, finalView, locals, objects⟩ := back_scan state (fun _ => 0) 120 (Nat.le_refl _)
    (state.matrix 119 119) heap root weights addresses view env globals getMatrix getWeights getSize getScale
  refine ⟨afterEnv, afterHeap, ?_, finalView, locals, objects⟩
  exact BlockContinues.for_range
    (.cons (Evaluates.binary (op := .sub) (.lookup getSize) Evaluates.constant rfl)
      (.cons (Evaluates.unary Evaluates.constant rfl) (.cons (Evaluates.unary Evaluates.constant rfl) .nil))) rfl ran

end solvers.«84».stationary_weights
