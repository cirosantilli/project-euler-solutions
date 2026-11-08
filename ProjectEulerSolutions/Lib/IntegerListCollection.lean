import ProjectEulerSolutions.Lib.HeapRangeFold
import ProjectEulerSolutions.Lib.TriangleBufferCalls

/-! Pure integer maps over existing Python lists preserve their captured source. -/
namespace ProjectEulerSolutions.IntegerListCollection
open FormalSW.PyAst HeapCalls TriangleBufferCalls
set_option Elab.async false

theorem collect_loop (module : Module) (name : String) (element : Expr) (transform : Int → Int)
    (base : Mutable.Heap) (address : Nat) (outer : HeapCalls.Env) (globals : Mutable.Globals)
    (evaluate : ∀ output locals value, HeapRangeFold.Frame [name] outer locals →
      locals name = some (number value) →
      Evaluates module locals element (working base output) (number (transform value)) (working base output) globals)
    (before remaining output : List Int) (env : HeapCalls.Env)
    (safe : HeapRangeFold.Frame [name] outer env)
    (found : base[address]? = some (.list (encode (before ++ remaining)))) :
    ∃ afterEnv, BlockContinues module env
      [.sequenceStepLoop name (.ref address) before.length [.comprehensionAppend base.length element] []]
      (working base output) afterEnv (working base (output ++ remaining.map transform)) globals := by
  induction remaining generalizing before output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.map_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := encode before)
      · have kept := prefix_found base output address _ found
        simp [sequenceValue, Mutable.sequence, kept, encode]
      · simp [encode]
      · exact .nil
  | cons value rest ih =>
      let ready := setLocal env name (number value)
      have readySafe := safe.trans (HeapRangeFold.Frame.updated [name] env name (number value) (by simp))
      have appended := BlockContinues.comprehension_append (evaluate output ready value readySafe (by simp [ready, setLocal])) (TriangleBufferCalls.found base output)
      have emitted : BlockContinues module ready [.comprehensionAppend base.length element]
          (working base output) ready (working base (output ++ [transform value])) globals := by
        simpa [working, encode] using appended
      obtain ⟨afterEnv, continued⟩ := ih (before ++ [value]) (output ++ [transform value]) ready readySafe
        (by simpa only [List.append_assoc, List.singleton_append] using found)
      refine ⟨afterEnv, ?_⟩
      apply BlockContinues.sequence_step (items := encode (before ++ value :: rest)) (value := number value)
      · have kept := prefix_found base output address _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp [encode]
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, List.map_cons, List.append_assoc, List.singleton_append,
          List.length_append, List.length_singleton] using emitted.append continued

/-- The iterable is evaluated in the enclosing scope before masking the comprehension's local name. -/
theorem collect (module : Module) (name : String) (element : Expr) (transform : Int → Int)
    (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals) (values : List Int)
    (found : heap[address]? = some (.list (encode values)))
    (iterable : Expr) (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates module env iterable heap (.ref address) heap globals)
    (evaluate : ∀ output locals value, HeapRangeFold.Frame [name] env locals →
      locals name = some (number value) →
      Evaluates module locals element (working heap output) (number (transform value)) (working heap output) globals) :
    Evaluates module env (.listComp element [(name, iterable, [])]) heap (.ref heap.length)
      (working heap (values.map transform)) globals := by
  let masked := comprehensionEnv env [(name, iterable, [])]
  have safe : HeapRangeFold.Frame [name] env masked := by
    intro query outside
    simp only [List.mem_singleton] at outside
    simp [masked, comprehensionEnv, Ne.symm outside]
  obtain ⟨afterEnv, executed⟩ := collect_loop module name element transform heap address env globals evaluate
    [] values [] masked safe (by simpa using found)
  apply Evaluates.list_comp_sequence (finalEnv := afterEnv) (items := encode values) ordinary evaluated
  · simp [Mutable.sequence, found]
  · exact executed

end ProjectEulerSolutions.IntegerListCollection
