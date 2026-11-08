import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.TuplePool

/-! List collectors whose callbacks allocate or reuse immutable cached tuples. -/
namespace ProjectEulerSolutions.MemoCollection
open FormalSW.PyAst HeapMemoCalls
set_option Elab.async false

variable {globals : Mutable.Globals}

theorem nil (holds : post (.scalar .none) heap) :
    BlockReturnsRel module spec env [] heap post globals := by
  intro anchors stack
  exact RunsRel.next (by trivial) rfl (RunsRel.refl ⟨_, _, rfl, holds⟩)

theorem emit
    (evaluated : EvaluatesRel module spec env expression heap expressionPost globals)
    (rest : ∀ value middle, expressionPost value middle → ∃ items,
      middle[address]? = some (.list items) ∧
      Cache.Preserves spec middle (middle.set address (.list (items ++ [value]))) ∧
      BlockReturnsRel module spec env body (middle.set address (.list (items ++ [value]))) post globals) :
    BlockReturnsRel module spec env (.comprehensionAppend address expression :: body) heap post globals := by
  intro anchors stack
  apply RunsRel.next (by trivial) rfl
  apply (evaluated anchors _).bind
  rintro state ⟨value, middle, rfl, holds⟩
  obtain ⟨items, found, preserved, continued⟩ := rest value middle holds
  apply RunsRel.next (by trivial) rfl
  change RunsRel module spec anchors
    ⟨.operation (.append address) [value], .discard body env :: stack, middle⟩ _ globals
  apply RunsRel.next (following := ⟨.value (.scalar .none), .discard body env :: stack,
    middle.set address (.list (items ++ [value]))⟩) (by trivial)
    (by simp [HeapCalls.step, HeapCalls.result, HeapCalls.run, HeapCalls.primitive, Mutable.appendItem, found, Except.map])
    _ preserved
  exact RunsRel.next (by trivial) rfl (continued anchors stack)

theorem list_comp_range
    (arguments : Evaluates.Arguments module spec env expressions heap values middle globals)
    (bounds : HeapCalls.rangeBounds values = .ok (start, stop, stride))
    (preserved : Cache.Preserves spec middle (middle ++ [.list []]))
    (executed : BlockReturnsRel module spec
      (HeapCalls.comprehensionEnv env ((name, .call (.name "range") expressions, filters) :: generators))
      [.rangeStepLoop name start stop stride
        (HeapCalls.comprehensionFilters filters (HeapCalls.comprehensionBody middle.length element generators)) []]
      (middle ++ [.list []]) (fun _ after => post (.ref middle.length) after) globals) :
    EvaluatesRel module spec env
      (.listComp element ((name, .call (.name "range") expressions, filters) :: generators)) heap post globals := by
  intro anchors stack
  apply RunsRel.next (by trivial) rfl
  apply (Evaluates.arguments_reach arguments _ [] anchors stack).bind
  rintro state rfl
  apply RunsRel.next (following := ⟨.block [.rangeStepLoop name start stop stride
      (HeapCalls.comprehensionFilters filters (HeapCalls.comprehensionBody middle.length element generators)) []]
      (HeapCalls.comprehensionEnv env ((name, .call (.name "range") expressions, filters) :: generators)),
      .comprehensionResult middle.length :: stack, middle ++ [.list []]⟩)
    (by trivial) (by simp [HeapCalls.step, HeapCalls.comprehensionStart, bounds]) _ preserved
  apply (executed anchors (.comprehensionResult middle.length :: stack)).bind
  rintro state ⟨value, after, rfl, holds⟩
  exact RunsRel.next (by trivial) rfl (RunsRel.refl ⟨_, _, rfl, holds⟩)

theorem tuple_frame (heap : Mutable.Heap) (root : Nat) (old replacement : List Mutable.Value)
    (found : heap[root]? = some (.list old)) (address : Nat) (items : List Mutable.Value)
    (tupleFound : heap[address]? = some (.tuple items)) :
    (heap.set root (.list replacement))[address]? = some (.tuple items) := by
  have different : root ≠ address := by
    intro same
    rw [same, tupleFound] at found
    cases found
  exact (List.getElem?_set_ne different).trans tupleFound

theorem dictionary_frame (heap : Mutable.Heap) (root : Nat) (old replacement : List Mutable.Value)
    (found : heap[root]? = some (.list old)) (address : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (dictionaryFound : heap[address]? = some (.keyDict entries)) :
    (heap.set root (.list replacement))[address]? = some (.keyDict entries) := by
  have different : root ≠ address := by
    intro same
    rw [same, dictionaryFound] at found
    cases found
  exact (List.getElem?_set_ne different).trans dictionaryFound

end ProjectEulerSolutions.MemoCollection
