import solvers.«84.py.lean.d».OutcomeModel
import solvers.«84.py.lean.d».LandingPool

namespace solvers.«84».outcomes
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def definition := (findFunctionIn program.body "build_transition").getD ([], [])
def rejected : List Stmt := match (executableBody definition.2)[(0 : Nat)]? with
  | some (.ifThen _ yes _) => yes
  | _ => []
def initialCode : List Stmt := (executableBody definition.2).take 3

theorem prefix_source : initialCode = [.ifThen (.compare (.name "dice_sides") [(.lt, .constant (.int 1))]) rejected [],
    .assign [.name "outcomes"] (.list []),
    .forIn (.name "move") (.call (.name "range") [.constant (.int 2),
      .binOp (.binOp (.constant (.int 2)) .mult (.name "dice_sides")) .add (.constant (.int 1))]) code []] := rfl

/-- The complete source prefix validates the die size and constructs every weighted dice category. -/
theorem prefix_result (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (number sides)) :
    ∃ finalEnv, BlockContinues program env initialCode base finalEnv (heap base (all sides)) globals ∧
      finalEnv "dice_sides" = some (number sides) ∧ finalEnv "outcomes" = some (.ref base.length) := by
  have tested : Evaluates program env (.compare (.name "dice_sides") [(.lt, .constant (.int 1))]) base
      (.scalar (.bool false)) base globals := by
    apply Evaluates.compare (Evaluates.lookup getSides) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  have guard : BlockContinues program env [.ifThen (.compare (.name "dice_sides") [(.lt, .constant (.int 1))]) rejected []]
      base env base globals := BlockContinues.branch (flag := false) tested rfl BlockContinues.nil
  obtain ⟨finalEnv, collected, finalSides, finalOutcomes⟩ := collect sides positive base env globals getSides
  refine ⟨finalEnv, ?_, finalSides, finalOutcomes⟩
  rw [prefix_source]
  exact guard.append collected

theorem prefix_continue (sides : Nat) (positive : 0 < sides) (base : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (number sides)) (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : ∀ finalEnv, finalEnv "dice_sides" = some (number sides) → finalEnv "outcomes" = some (.ref base.length) →
      HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec finalEnv body (heap base (all sides)) post globals) :
    HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env (initialCode ++ body) base post globals := by
  obtain ⟨finalEnv, executed, finalSides, finalOutcomes⟩ := prefix_result sides positive base env globals getSides
  intro anchors stack
  have run := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) resolve_landing.pool.disjoint
    (executed body anchors stack) (TuplePool.preserves_append resolve_landing.pool.catalog base _)
  apply run.bind
  rintro state rfl
  exact rest finalEnv finalSides finalOutcomes anchors stack

end solvers.«84».outcomes
