import solvers.«84.py.lean.d».StationaryEntry
import ProjectEulerSolutions.Lib.TriangleBufferCalls

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode working)
set_option Elab.async false

def columnValues (sides : Nat) (target : Fin 120) (first count : Nat) : List Int :=
  (List.range' first count).map fun source => Monopoly.coefficients sides target (Fin.ofNat 120 source)

theorem columns_scan (sides : Nat) (target : Fin 120) (ordinary : target ≠ 119)
    (base : Mutable.Heap) (value : Mutable.Value) (view : transitions.View sides base value)
    (count first stop : Nat) (length : first + count = stop) (bound : stop ≤ 120)
    (output : List Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getTransitions : env "transitions" = some value) (getTarget : env "target" = some (.scalar (.int target.val)))
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides)))) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "source" first stop 1 [.comprehensionAppend base.length coefficientExpression] []]
      (working base output) finalEnv (working base (output ++ columnValues sides target first count)) globals := by
  induction count generalizing first output env with
  | zero =>
      refine ⟨env, ?_⟩
      simp only [columnValues, List.range'_zero, List.map_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      have inside : first < 120 := by omega
      let entered := setLocal env "source" (.scalar (.int first))
      have evaluated := coefficient_result sides (Fin.ofNat 120 first) target ordinary (working base output) entered globals
        value (view.append _) (by simpa [entered, setLocal] using getTransitions)
        (by simp [entered, setLocal, Fin.ofNat, Nat.mod_eq_of_lt inside])
        (by simpa [entered, setLocal] using getTarget) (by simpa [entered, setLocal] using getDenominator)
      have emitted := BlockContinues.comprehension_append evaluated (TriangleBufferCalls.found base output)
      have written : (working base output).set base.length
          (.list (encode output ++ [.scalar (.int (Monopoly.coefficients sides target (Fin.ofNat 120 first)))])) =
          working base (output ++ [Monopoly.coefficients sides target (Fin.ofNat 120 first)]) := by
        simp [working, encode]
      rw [written] at emitted
      obtain ⟨finalEnv, continued⟩ := ih (first + 1) (by omega)
        (output ++ [Monopoly.coefficients sides target (Fin.ofNat 120 first)]) entered
        (by simpa [entered, setLocal] using getTransitions) (by simpa [entered, setLocal] using getTarget)
        (by simpa [entered, setLocal] using getDenominator)
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true,
        if_false, columnValues, List.range'_succ, List.map_cons, List.append_assoc, List.singleton_append,
        Int.natCast_add, Int.natCast_one] using emitted.append continued

theorem columns_result (sides : Nat) (target : Fin 120) (ordinary : target ≠ 119)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value) (getTarget : env "target" = some (.scalar (.int target.val)))
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides))))
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates program env (.listComp coefficientExpression [("source", .call (.name "range") [.name "size"], [])])
      heap (.ref heap.length) (working heap (columnValues sides target 0 120)) globals := by
  obtain ⟨finalEnv, ran⟩ := columns_scan sides target ordinary heap value view 120 0 120 rfl (by decide) []
    (comprehensionEnv env [("source", .call (.name "range") [.name "size"], [])]) globals
    (by simpa [comprehensionEnv] using getTransitions) (by simpa [comprehensionEnv] using getTarget)
    (by simpa [comprehensionEnv] using getDenominator)
  apply Evaluates.list_comp_range (.cons (Evaluates.lookup getSize) .nil) rfl
  simpa [working, encode, comprehensionFilters, comprehensionBody] using ran

end solvers.«84».stationary_weights
