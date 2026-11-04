import ProjectEulerSolutions.Lib.IntegerRangeCollection

/-! Integer comprehensions whose element calls allocate private objects. -/
namespace ProjectEulerSolutions.AllocatingIntegerRangeCollection
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.TriangleBufferCalls
set_option Elab.async false

def state (base : Mutable.Heap) (output : List Int) (privateHeap : Mutable.Heap) : Mutable.Heap :=
  working base output ++ privateHeap

theorem prefix_found (base privateHeap : Mutable.Heap) (output : List Int) (address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) : (state base output privateHeap)[address]? = some object := by
  unfold state working
  rw [List.append_assoc, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem collect_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values output : List Int) (privateHeap : Mutable.Heap) (cursor : Nat)
    (bounded : cursor ≤ values.length)
    (evaluate : ∀ (collected : List Int) (objects : Mutable.Heap) (index : Nat) (inside : index < values.length),
      ∃ extra, Evaluates program (setLocal env name (number index)) element (state base collected objects)
        (number values[index]) ((state base collected objects) ++ extra)) :
    ∃ after extra, BlockContinues program env
      [.rangeStepLoop name cursor values.length 1 [.comprehensionAppend base.length element] []]
      (state base output privateHeap) after (state base (output ++ values.drop cursor) (privateHeap ++ extra)) := by
  by_cases active : cursor < values.length
  · let entered := setLocal env name (number cursor)
    obtain ⟨allocated, evaluated⟩ := evaluate output privateHeap cursor active
    have collector : ((state base output privateHeap) ++ allocated)[base.length]? = some (.list (encode output)) := by
      simp [state, working, List.append_assoc]
    have emitted := BlockContinues.comprehension_append evaluated collector
    have written : ((state base output privateHeap) ++ allocated).set base.length
        (.list (encode output ++ [number values[cursor]])) =
        state base (output ++ [values[cursor]]) (privateHeap ++ allocated) := by
      simp [state, working, encode, List.append_assoc]
    rw [written] at emitted
    obtain ⟨after, more, continued⟩ := collect_loop program base entered name element values (output ++ [values[cursor]])
      (privateHeap ++ allocated) (cursor + 1) (by omega) (by
        intro collected objects index inside
        have same : setLocal entered name (number index) = setLocal env name (number index) := by
          funext query
          by_cases equal : query = name <;> simp [entered, setLocal, equal]
        rw [same]
        exact evaluate collected objects index inside)
    refine ⟨after, allocated ++ more, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    simpa only [List.drop_eq_getElem_cons active, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
      Bool.false_or, Bool.false_eq_true, if_false, List.append_assoc, List.singleton_append,
      Int.natCast_add, Int.natCast_one] using emitted.append continued
  · have done : cursor = values.length := by omega
    subst cursor
    refine ⟨env, [], ?_⟩
    simp only [List.drop_length, List.append_nil]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
termination_by values.length - cursor
decreasing_by omega

theorem collect (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (element : Expr) (values : List Int) (expressions : List Expr) (args : List Mutable.Value)
    (evaluated : Evaluates.Arguments program env expressions base args base)
    (bounds : rangeBounds args = .ok (0, values.length, 1))
    (evaluate : ∀ (collected : List Int) (objects : Mutable.Heap) (index : Nat) (inside : index < values.length),
      ∃ extra, Evaluates program
        (setLocal (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name (number index))
        element (state base collected objects) (number values[index]) ((state base collected objects) ++ extra)) :
    ∃ extra, Evaluates program env (.listComp element [(name, .call (.name "range") expressions, [])]) base
      (.ref base.length) (working base values ++ extra) := by
  obtain ⟨after, extra, loop⟩ := collect_loop program base
    (comprehensionEnv env [(name, .call (.name "range") expressions, [])]) name element values [] [] 0 (by omega) evaluate
  refine ⟨extra, ?_⟩
  apply Evaluates.list_comp_range evaluated bounds
  simpa [state, working, encode, comprehensionFilters, comprehensionBody] using loop

end ProjectEulerSolutions.AllocatingIntegerRangeCollection
