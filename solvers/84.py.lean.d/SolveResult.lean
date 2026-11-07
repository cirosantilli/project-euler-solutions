import solvers.«84.py.lean.d».SolveWeights

namespace solvers.«84».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def Post (sides : Nat) (before : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ weights, Monopoly.IntegerWeights sides weights ∧ value = .scalar (.string (MonopolyOutput.answer weights)) ∧
    ∀ index, index < before.length → after[index]? = before[index]?

/-- Complete optimized source execution preserves caller objects and returns the exact modal string. -/
theorem body_result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (.scalar (.int sides))) :
    HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env (executableBody definition.2) heap
      (Post sides heap) globals := by
  rw [source]
  apply HeapMemoCalls.BlockReturnsRel.assign (weights_result sides positive heap env globals getSides)
  rintro _ middle ⟨weights, address, rfl, found, valid, kept⟩
  obtain ⟨extra, output⟩ := output_result weights middle address found
    (setLocal env "weights" (.ref address)) globals (by simp [setLocal])
  rw [source] at output
  intro anchors stack
  have lifted := HeapMemoCalls.Runs.from_heap (Extends.refl program) resolve_landing.pool.disjoint
    (output anchors stack) (TuplePool.preserves_append resolve_landing.pool.catalog middle extra)
  apply lifted.consequence
  intro final same
  subst final
  refine ⟨.scalar (.string (MonopolyOutput.answer weights)), middle ++ extra, rfl, weights, valid, rfl, ?_⟩
  intro index inside
  have old := (kept index inside).trans (List.getElem?_eq_getElem inside)
  exact (HeapCollection.prefix_found middle extra index _ old).trans (List.getElem?_eq_getElem inside).symm

theorem result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program resolve_landing.pool.spec "solve" [.scalar (.int sides)] heap (Post sides heap) globals := by
  apply HeapMemoCalls.BlockReturnsRel.function (by decide) (args := definition.1) (body := definition.2) rfl rfl
  apply body_result sides positive heap _ globals
  cases globals <;> rfl

end solvers.«84».solve
