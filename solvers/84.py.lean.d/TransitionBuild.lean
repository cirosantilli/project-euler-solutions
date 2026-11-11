import solvers.«84.py.lean.d».TransitionCollector

namespace solvers.«84».build_transition
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

theorem source : executableBody outcomes.definition.2 = outcomes.initialCode ++
    (.assign [.name "landings"] landings.expression :: transitions.code) := rfl

/-- For any positive die size the complete Python body constructs the exact independent-card transition table. -/
theorem body_result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (.scalar (.int sides))) :
    HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env (executableBody outcomes.definition.2) heap
      (transitions.Post sides heap) globals := by
  rw [source]
  apply outcomes.prefix_continue sides positive heap env globals getSides
  intro outcomeEnv _ getOutcomes
  apply landings.assign_continue
  intro after landingPost
  have table := transition_row.Table.of_post landingPost
  obtain ⟨landingExtra, layout⟩ := landingPost.prefix
  have diceView : outcomes.View after heap.length (outcomes.all sides) := by
    rw [layout]
    exact (outcomes.heap_view heap (outcomes.all sides)).append landingExtra
  apply (transitions.collect_result sides positive after _ globals heap.length
      (outcomes.heap heap (outcomes.all sides)).length diceView table
      (by simpa [setLocal] using getOutcomes) (by simp [setLocal])).consequence
  rintro value finalHeap ⟨extra, finalLayout, view⟩
  refine ⟨(.list (outcomes.references heap (outcomes.all sides)) ::
      (outcomes.all sides).map (fun outcome => .tuple outcome.items)) ++ landingExtra ++ extra, ?_, view⟩
  rw [finalLayout, layout]
  simp only [outcomes.heap, List.append_assoc]

theorem result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program resolve_landing.pool.spec "build_transition" [.scalar (.int sides)] heap
      (transitions.Post sides heap) globals := by
  apply HeapMemoCalls.BlockReturnsRel.function (by decide)
    (args := outcomes.definition.1) (body := outcomes.definition.2) rfl rfl
  apply body_result sides positive heap _ globals
  cases globals <;> rfl

end solvers.«84».build_transition
