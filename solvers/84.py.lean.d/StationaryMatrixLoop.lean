import solvers.«84.py.lean.d».StationaryMatrixHeap

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

theorem matrix_scan (sides : Nat) (base : Mutable.Heap) (value : Mutable.Value) (view : transitions.View sides base value)
    (count cursor stop : Nat) (length : cursor + count = stop) (bound : stop ≤ 119)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getTransitions : env "transitions" = some value)
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides))))
    (getSize : env "size" = some (.scalar (.int 120))) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "target" cursor stop 1 [.comprehensionAppend base.length rowExpression] []]
      (matrixHeap base sides cursor) finalEnv (matrixHeap base sides stop) globals := by
  induction count generalizing cursor env with
  | zero =>
      have done : cursor = stop := by omega
      subst cursor
      exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil⟩
  | succ count ih =>
      have inside : cursor < 120 := by omega
      have ordinary : Fin.ofNat 120 cursor ≠ (119 : Fin 120) := by
        intro same
        have equal := congrArg Fin.val same
        simp only [Fin.val_ofNat, Nat.mod_eq_of_lt inside] at equal
        omega
      let entered := setLocal env "target" (.scalar (.int cursor))
      have evaluated := row_result sides (Fin.ofNat 120 cursor) ordinary (matrixHeap base sides cursor) entered globals
        value (view.append _) (by simpa [entered, setLocal] using getTransitions)
        (by simp [entered, setLocal, Fin.ofNat, Nat.mod_eq_of_lt inside])
        (by simpa [entered, setLocal] using getDenominator) (by simpa [entered, setLocal] using getSize)
      have found := HeapCollection.prefix_found (matrixHeap base sides cursor) (rowObjects sides (Fin.ofNat 120 cursor))
        base.length _ (matrix_found base sides cursor)
      have emitted := BlockContinues.comprehension_append evaluated found
      rw [matrix_next] at emitted
      obtain ⟨finalEnv, continued⟩ := ih (cursor + 1) (by omega) entered
        (by simpa [entered, setLocal] using getTransitions) (by simpa [entered, setLocal] using getDenominator)
        (by simpa [entered, setLocal] using getSize)
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true,
        if_false, Int.natCast_add, Int.natCast_one] using emitted.append continued

/-- The outer comprehension constructs all 119 stationary rows with their zero right-hand sides. -/
theorem matrix_result (sides : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value)
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides))))
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates program env matrixExpression heap (.ref heap.length) (matrixHeap heap sides 119) globals := by
  obtain ⟨finalEnv, ran⟩ := matrix_scan sides heap value view 119 0 119 rfl (by decide)
    (comprehensionEnv env [("target", .call (.name "range") [.binOp (.name "size") .sub (.constant (.int 1))], [])]) globals
    (by simpa [comprehensionEnv] using getTransitions) (by simpa [comprehensionEnv] using getDenominator)
    (by simpa [comprehensionEnv] using getSize)
  apply Evaluates.list_comp_range
    (.cons (Evaluates.binary (Evaluates.lookup getSize) Evaluates.constant rfl) .nil) rfl
  simpa [matrixHeap, HeapCollection.working, rowReferences, comprehensionFilters, comprehensionBody] using ran

end solvers.«84».stationary_weights
