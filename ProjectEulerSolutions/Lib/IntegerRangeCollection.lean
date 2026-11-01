import ProjectEulerSolutions.Lib.TriangleBufferCalls

/-! Pure integer range comprehensions allocate one collector and preserve captured objects. -/
namespace ProjectEulerSolutions.IntegerRangeCollection
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.TriangleBufferCalls
set_option Elab.async false

theorem collect_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values output : List Int) (cursor : Nat)
    (bounded : cursor ≤ values.length)
    (evaluate : ∀ (collected : List Int) (index : Nat) (inside : index < values.length),
      Evaluates program (setLocal env name (number index)) element (working base collected)
        (number values[index]) (working base collected)) :
    ∃ after, BlockContinues program env
      [.rangeStepLoop name cursor values.length 1 [.comprehensionAppend base.length element] []]
      (working base output) after (working base (output ++ values.drop cursor)) := by
  by_cases active : cursor < values.length
  · let entered := setLocal env name (number cursor)
    have appended := BlockContinues.comprehension_append (evaluate output cursor active) (found base output)
    have written : (working base output).set base.length (.list (encode output ++ [number values[cursor]])) =
        working base (output ++ [values[cursor]]) := by simp [working, encode]
    rw [written] at appended
    obtain ⟨after, continued⟩ := collect_loop program base entered name element values (output ++ [values[cursor]])
      (cursor + 1) (by omega) (by
        intro collected index inside
        have same : setLocal entered name (number index) = setLocal env name (number index) := by
          funext query
          by_cases equal : query = name <;> simp [entered, setLocal, equal]
        rw [same]
        exact evaluate collected index inside)
    refine ⟨after, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    simpa only [List.drop_eq_getElem_cons active, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
      Bool.false_or, Bool.false_eq_true, if_false, List.append_assoc, List.singleton_append,
      Int.natCast_add, Int.natCast_one] using appended.append continued
  · have done : cursor = values.length := by omega
    subst cursor
    refine ⟨env, ?_⟩
    simp only [List.drop_length, List.append_nil]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
termination_by values.length - cursor
decreasing_by omega

theorem collect (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values : List Int) (expressions : List Expr) (arguments : List Mutable.Value)
    (evaluated : Evaluates.Arguments program env expressions base arguments base)
    (bounds : rangeBounds arguments = .ok (0, values.length, 1))
    (evaluate : ∀ (collected : List Int) (index : Nat) (inside : index < values.length),
      Evaluates program (setLocal (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name (number index))
        element (working base collected) (number values[index]) (working base collected)) :
    Evaluates program env (.listComp element [(name, .call (.name "range") expressions, [])]) base
      (.ref base.length) (working base values) := by
  obtain ⟨after, loop⟩ := collect_loop program base
    (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name element values [] 0 (by omega) evaluate
  apply Evaluates.list_comp_range evaluated bounds
  simpa [working, encode, comprehensionFilters, comprehensionBody] using loop

end ProjectEulerSolutions.IntegerRangeCollection
