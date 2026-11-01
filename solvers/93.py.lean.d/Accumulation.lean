import solvers.«93.py.lean.d».Cache
import ProjectEulerSolutions.Lib.RationalSetUpdates

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev recursiveStateExpr : Expr :=
  .builtinCall .tuple [.builtinCall .sorted [.binOp (.name "rest") .add (.list [.name "value"])]]
abbrev recursiveResultExpr : Expr := .call (.name recName) [recursiveStateExpr, .name cellName]
abbrev updateExpr : Expr := .call (.attribute (.name "results") "update") [recursiveResultExpr]

private def loopBody : Option Stmt → List Stmt
  | some (.forIn _ _ body _) => body
  | _ => []

/-- The innermost body is the actual recursive union statement in the lifted Python helper. -/
theorem source_update :
    loopBody ((loopBody ((loopBody recBody[3]?)[0]?))[1]?) =
      [.expr updateExpr] := rfl

/-- Bind the collector before evaluating the recursive argument, then union the returned set. -/
theorem update_result (heap middle : Mutable.Heap) (address source : Nat)
    (before current incoming : List Heap.SetEntry) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (original : Heap.SetAt heap address before) (getResult : env "results" = some (.ref address))
    (recursive : Evaluates program env recursiveResultExpr heap (.ref source) middle globals)
    (collector : Heap.SetAt middle address current) (input : Heap.SetAt middle source incoming) :
    ∃ after final,
      BlockContinues program env [.expr updateExpr] heap env after globals ∧
      Heap.SetAt after address final ∧
      (∀ pair, pair ∈ final.map Prod.fst ↔ pair ∈ current.map Prod.fst ∨ pair ∈ incoming.map Prod.fst) ∧
      (∀ other, other ≠ address → after[other]? = middle[other]?) ∧ after.length = middle.length := by
  obtain ⟨after, final, operation, found, members, unchanged, size⟩ := collector.update_values input
  refine ⟨after, final, ?_, found, members, unchanged, size⟩
  apply BlockContinues.discard
  apply Evaluates.updateSet (Evaluates.lookup getResult) _ original.keyEntries
    (.cons recursive .nil) operation
  intro name equal
  cases equal
  simp [getResult]

/-- A recursive memo store preserves the caller's collector and its shared rational tuples. -/
theorem set_after_memo_write (heap : Mutable.Heap) (memo address : Nat)
    (entries replacement : List (HashDict.Entry Mutable.Value)) (pairs : List Heap.SetEntry)
    (memoFound : MemoAt heap memo entries) (found : Heap.SetAt heap address pairs) :
    Heap.SetAt (heap.set memo (.keyDict replacement)) address pairs := by
  constructor
  · rcases found.stored with stored | ⟨empty, stored⟩
    · exact .inl (memoFound.other address _ stored rfl replacement)
    · exact .inr ⟨empty, memoFound.other address _ stored rfl replacement⟩
  · intro entry member
    exact memoFound.other entry.2 _ (found.rows entry member) rfl replacement

end solvers.«93».all_results
