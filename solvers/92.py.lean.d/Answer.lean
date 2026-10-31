import solvers.«92.py.lean.d».Sum
import solvers.«92.py.lean.d».Initialization

namespace solvers.«92».answer
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := table.number
abbrev literal : Expr := .dict [(.constant (.int 0), .constant (.int 0)),
  (.constant (.int 1), .constant (.int 1)), (.constant (.int 89), .constant (.int 89))]
abbrev body : List Stmt := [.assign [.name "known"] literal, .return summation.expression]

theorem source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 = body := rfl

theorem disjoint : HeapMemoCalls.CacheDisjoint program program := by
  intro name impossible
  have empty : program.memoizedFunctions = [] := rfl
  rw [empty] at impossible
  cases impossible

/-- Create the endpoint cache and execute the actual final sum without changing caller objects. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (bound address : Nat) (entries : table.Entries)
    (counts : before "counts" = some (.ref address))
    (found : table.Found heap address entries) (keys : table.Keys heap entries)
    (exact : Prefix.Exact bound (table.view entries)) :
    ∃ finalHeap,
      BlockReturns program before body heap (number (count (bound + 1 : Nat))) finalHeap globals ∧
      HeapMemoCalls.Cache.Preserves initialization.spec heap finalHeap := by
  let allocated := heap ++ [.dict (store.encode store.initial)]
  have initial : store.Found allocated heap.length store.initial := by simp [allocated, store.Found, NaturalDictionary.Found]
  obtain ⟨_, finalHeap, evaluated, _, _, frame⟩ :=
    summation.result allocated (setLocal before "known" (.ref heap.length)) globals bound address heap.length
      entries store.initial (by simpa [setLocal] using counts) (by simp [setLocal])
      (found.append _) (keys.append _) initial store.initial_valid
      (by have inside := found.inside; omega) exact
  refine ⟨finalHeap, ?_, ?_⟩
  · exact BlockReturns.assign (store.initial_result heap before globals)
      (BlockReturns.return_value evaluated)
  · have preserved := initialization.spec_preserves_frame allocated finalHeap heap.length
      (.dict (store.encode store.initial)) initial (by simp [table.tuple]) frame
    intro cache valid
    exact preserved cache (initialization.spec_preserves_append heap _ cache valid)

/-- The final source block also preserves the compiler's pooled initial tuple. -/
theorem verified (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (bound address : Nat) (entries : table.Entries)
    (counts : before "counts" = some (.ref address))
    (found : table.Found heap address entries) (keys : table.Keys heap entries)
    (exact : Prefix.Exact bound (table.view entries)) :
    HeapMemoCalls.BlockReturnsRel program initialization.spec before body heap
      (fun value _ => value = number (count (bound + 1 : Nat))) globals := by
  obtain ⟨finalHeap, ran, preserved⟩ := result heap before globals bound address entries counts found keys exact
  intro anchors stack
  exact (HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) disjoint
    (ran anchors stack) preserved).consequence (by intro state same; subst state; exact ⟨_, _, rfl, rfl⟩)

end solvers.«92».answer
