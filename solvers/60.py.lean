import data.«project-euler-statements».data.py.project_euler_statements.«p60.py»
import solvers.«60.py.lean.d».Execution
import ProjectEulerSolutions.Lib.PrimePairLarge.Certificate

/-! Complete initialized source equivalence, using the separately checked source and finite search proofs. -/
set_option Elab.async false

namespace solvers.«60».certificate
open ProjectEulerSolutions

theorem large_model_checked : solve_search.answerModel 26033 = ⟨26033, competitive.witness⟩ :=
  large_model_of_budget_lookup_certificate PrimePairLarge.graph PrimePairLarge.graph_covers
    PrimePairLarge.graph_coverage PrimePairLarge.large_search_checked

theorem models_eq_checked (limit : Int) :
    solve_search.answerModel limit =
      data.«project-euler-statements».data.py.project_euler_statements.p60.naive.answerModel limit :=
  competitive.models_eq_of_search_certificates small_model_checked large_model_checked limit

end solvers.«60».certificate

namespace solvers.«60».source_equivalence
open FormalSW.PyAst

theorem verified (limit : Int) (caller : Mutable.Heap) (valid : caller.Valid) : Verified limit caller :=
  verified_of_models limit caller valid (certificate.models_eq_checked limit)

end solvers.«60».source_equivalence

namespace solvers.«60»
open FormalSW.PyAst

theorem solve_source_equivalence (limit : Int) (caller : Mutable.Heap) :
    caller.Valid → source_equivalence.Verified limit caller :=
  source_equivalence.verified limit caller

end solvers.«60»
