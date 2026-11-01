import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.IndexedFilter

/-! Range comprehensions with a pure filter retain shared element references. -/
namespace ProjectEulerSolutions.FilteredRangeCollection
open FormalSW.PyAst HeapCalls HeapCollection
set_option Elab.async false
variable {globals : Mutable.Globals}

theorem collect_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element condition : Expr) (values output : List Mutable.Value) (keep : Nat → Bool)
    (cursor : Nat) (bounded : cursor ≤ values.length)
    (evaluate : ∀ (collected : List Mutable.Value) (index : Nat) (inside : index < values.length),
      Evaluates program (setLocal env name (.scalar (.int index))) element (working base collected [])
        values[index] (working base collected []) globals)
    (test : ∀ (collected : List Mutable.Value) (index : Nat), index < values.length →
      Evaluates program (setLocal env name (.scalar (.int index))) condition (working base collected [])
        (.scalar (.bool (keep index))) (working base collected []) globals) :
    ∃ after, BlockContinues program env
      [.rangeStepLoop name cursor values.length 1 [.ifThen condition [.comprehensionAppend base.length element] []] []]
      (working base output []) after (working base (output ++ IndexedFilter.select keep cursor (values.drop cursor)) []) globals := by
  by_cases active : cursor < values.length
  · let entered := setLocal env name (.scalar (.int cursor))
    let nextOutput := if keep cursor then output ++ [values[cursor]] else output
    have body : BlockContinues program entered [.ifThen condition [.comprehensionAppend base.length element] []]
        (working base output []) entered (working base nextOutput []) globals := by
      apply BlockContinues.branch (flag := keep cursor) (test output cursor active)
        (by simp [Mutable.truth, scalarTruth])
      cases selected : keep cursor with
      | false => simpa [nextOutput, selected] using (BlockContinues.nil (module := program) (env := entered)
          (heap := working base output []) (globals := globals))
      | true =>
        have emitted := BlockContinues.comprehension_append (evaluate output cursor active) (found base output [])
        simpa [nextOutput, selected, store] using emitted
    have overwrite (index : Nat) :
        setLocal entered name (.scalar (.int index)) = setLocal env name (.scalar (.int index)) := by
      funext query
      by_cases equal : query = name <;> simp [entered, setLocal, equal]
    obtain ⟨after, continued⟩ := collect_loop program base entered name element condition values nextOutput keep
      (cursor + 1) (by omega)
      (by intro collected index inside; rw [overwrite]; exact evaluate collected index inside)
      (by intro collected index inside; rw [overwrite]; exact test collected index inside)
    refine ⟨after, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    cases selected : keep cursor <;>
      simpa only [List.drop_eq_getElem_cons active, IndexedFilter.select, nextOutput, selected,
        Bool.false_eq_true, if_false, if_true, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_or, List.append_assoc, List.singleton_append, Int.natCast_add, Int.natCast_one] using body.append continued
  · have done : cursor = values.length := by omega
    subst cursor
    refine ⟨env, ?_⟩
    simp only [List.drop_length, IndexedFilter.select, List.append_nil]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil
termination_by values.length - cursor
decreasing_by omega

theorem collect (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element condition : Expr) (values : List Mutable.Value) (keep : Nat → Bool)
    (expressions : List Expr) (arguments : List Mutable.Value)
    (evaluated : Evaluates.Arguments program env expressions base arguments base globals)
    (bounds : rangeBounds arguments = .ok (0, values.length, 1))
    (evaluate : ∀ (collected : List Mutable.Value) (index : Nat) (inside : index < values.length),
      Evaluates program
        (setLocal (comprehensionEnv env [(name, .call (.name "range") expressions, [condition])]) name (.scalar (.int index)))
        element (working base collected []) values[index] (working base collected []) globals)
    (test : ∀ (collected : List Mutable.Value) (index : Nat), index < values.length →
      Evaluates program
        (setLocal (comprehensionEnv env [(name, .call (.name "range") expressions, [condition])]) name (.scalar (.int index)))
        condition (working base collected []) (.scalar (.bool (keep index))) (working base collected []) globals) :
    Evaluates program env (.listComp element [(name, .call (.name "range") expressions, [condition])]) base
      (.ref base.length) (working base (IndexedFilter.select keep 0 values) []) globals := by
  obtain ⟨after, executed⟩ := collect_loop program base
    (comprehensionEnv env [(name, .call (.name "range") expressions, [condition])])
    name element condition values [] keep 0 (by omega) evaluate test
  apply Evaluates.list_comp_range evaluated bounds
  simpa [working, comprehensionFilters, comprehensionBody] using executed

end ProjectEulerSolutions.FilteredRangeCollection
