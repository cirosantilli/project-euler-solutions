import solvers.«84.py.lean.d».LandingTableLoop
import solvers.«84.py.lean.d».OutcomeSource

namespace solvers.«84».landings
open FormalSW.PyAst ProjectEulerSolutions resolve_landing
set_option Elab.async false

def expression : Expr := match (executableBody outcomes.definition.2)[(3 : Nat)]? with
  | some (.assign _ expression) => expression
  | _ => .constant .none

theorem source : expression = .listComp element [("pos", .call (.name "range") [.constant (.int 40)], [])] := rfl

/-- The actual comprehension produces the model landing dictionary for every board square. -/
theorem result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program pool.spec env expression heap (Post heap (List.range' 0 40)) globals := by
  rw [source]
  apply MemoCollection.list_comp_range
    (HeapMemoCalls.Evaluates.Arguments.cons HeapMemoCalls.Evaluates.constant .nil)
    (start := 0) (stop := 40) (stride := 1) rfl
    (TuplePool.preserves_append pool.catalog heap [.list []])
  change HeapMemoCalls.BlockReturnsRel program pool.spec _
    [.rangeStepLoop "pos" 0 40 1 [.comprehensionAppend heap.length element] []]
    (HeapCollection.working heap (references []) []) (fun _ after => Post heap (List.range' 0 40) (.ref heap.length) after) globals
  apply loop 40 0 40 rfl heap [] [] [] _ globals (.nil)
  intro dictionaries garbage views
  exact ⟨dictionaries, garbage, rfl, rfl, views⟩

theorem assign_continue (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : ∀ after, Post heap (List.range' 0 40) (.ref heap.length) after →
      HeapMemoCalls.BlockReturnsRel program pool.spec (HeapCalls.setLocal env "landings" (.ref heap.length)) body after post globals) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env (.assign [.name "landings"] expression :: body) heap post globals := by
  apply HeapMemoCalls.BlockReturnsRel.assign (result env heap globals)
  intro value after holds
  obtain ⟨dictionaries, garbage, rfl, same, views⟩ := holds
  exact rest after ⟨dictionaries, garbage, rfl, same, views⟩

end solvers.«84».landings
