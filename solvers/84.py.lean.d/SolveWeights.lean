import solvers.«84.py.lean.d».SolveOutput

namespace solvers.«84».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def WeightsPost (sides : Nat) (before : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ (weights : Fin 120 → Int) (address : Nat), value = .ref address ∧
    after[address]? = some (.list (MonopolyOutput.encode weights)) ∧ Monopoly.IntegerWeights sides weights ∧
    ∀ index, index < before.length → after[index]? = before[index]?

/-- The nested calls build the transition table before solving its exact stationary equations. -/
theorem weights_result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (.scalar (.int sides))) :
    HeapMemoCalls.EvaluatesRel program resolve_landing.pool.spec env weightsExpression heap (WeightsPost sides heap) globals := by
  have transition : HeapMemoCalls.EvaluatesRel program resolve_landing.pool.spec env
      (.call (.name "build_transition") [.name "dice_sides"]) heap (transitions.Post sides heap) globals := by
    apply HeapMemoCalls.EvaluatesRel.call
      (HeapMemoCalls.ArgumentsRel.fixed (.cons (HeapMemoCalls.Evaluates.lookup getSides) .nil))
    rintro _ after ⟨rfl, same⟩
    subst after
    exact build_transition.result sides positive heap globals
  apply HeapMemoCalls.EvaluatesRel.call (argumentPost := fun args after => ∃ value,
    args = [value, .scalar (.int (256 * (sides : Int) * sides))] ∧ transitions.View sides after value ∧
      ∀ index, index < heap.length → after[index]? = heap[index]?)
  · apply HeapMemoCalls.ArgumentsRel.cons transition
    rintro value after ⟨extra, rfl, view⟩
    have evaluated : Evaluates program env denominator (heap ++ extra)
        (.scalar (.int (256 * (sides : Int) * sides))) (heap ++ extra) globals :=
      Evaluates.binary (Evaluates.binary Evaluates.constant (.lookup getSides) rfl) (.lookup getSides) rfl
    have lifted := HeapMemoCalls.Evaluates.from_heap (spec := resolve_landing.pool.spec)
      (Extends.refl program) resolve_landing.pool.disjoint evaluated (by intro cache valid; exact valid)
    apply HeapMemoCalls.ArgumentsRel.cons lifted.rel
    rintro _ _ ⟨rfl, rfl⟩
    exact HeapMemoCalls.ArgumentsRel.nil ⟨value, rfl, view, fun index inside => List.getElem?_append_left inside⟩
  · rintro _ after ⟨value, rfl, view, preserved⟩
    obtain ⟨weights, final, address, ran, found, valid, kept⟩ := stationary_weights.result sides positive after value view globals
    have cacheKept := TuplePool.preserves_frame resolve_landing.pool.catalog after final
      (fun index items old => (kept index (List.getElem?_eq_some_iff.mp old).1).trans old)
    apply (HeapMemoCalls.ReturnsValue.from_heap (Extends.refl program) resolve_landing.pool.disjoint ran cacheKept).rel.consequence
    rintro _ _ ⟨rfl, rfl⟩
    refine ⟨weights, address, rfl, found, valid, ?_⟩
    intro index inside
    have old := (preserved index inside).trans (List.getElem?_eq_getElem inside)
    exact (kept index (List.getElem?_eq_some_iff.mp old).1).trans (preserved index inside)

end solvers.«84».solve
