import solvers.«74.equivalence.source»

/-! Public source contracts include initialization and every trailing-default call shape. -/
namespace solvers.«74»
open FormalSW.PyAst
set_option Elab.async false

theorem solve_source_equivalence (limit target : Int) (caller : Mutable.Heap) :
    caller.Valid → source_equivalence.Verified limit target caller :=
  source_equivalence.verified limit target caller

theorem solve_defaults_source_equivalence (limit : Int) (caller : Mutable.Heap) :
    caller.Valid →
      source_equivalence.VerifiedCall [solve.integer limit] limit 60 caller ∧
      source_equivalence.VerifiedCall [] 1_000_000 60 caller := by
  intro valid
  exact ⟨source_equivalence.limit_default_verified limit caller valid,
    source_equivalence.all_defaults_verified caller valid⟩

end solvers.«74»
