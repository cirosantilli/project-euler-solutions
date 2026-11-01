import solvers.«93.py.lean.d».RecursiveMiss

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

/-- Every canonical operand state terminates with exactly its arithmetic reduction results. -/
theorem recursive_verified (heap : Mutable.Heap) (cell memo state : Nat) (values : List Mutable.Value)
    (pairs : List Pair) (entries : List Memo.Entry) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (canonical : ∀ pair ∈ pairs, Canonical pair) :
    RecursiveVerified heap cell memo state pairs entries globals := by
  generalize size : pairs.length = count
  induction count using Nat.strong_induction_on generalizing heap cell memo state values pairs entries with
  | h count ih =>
    by_cases small : pairs.length ≤ 1
    · exact verified_terminal heap cell memo state values pairs entries globals cellFound valid stateFound rows canonical small
    cases lookup : Memo.lookup entries pairs with
    | some result =>
      exact verified_cache_hit heap cell memo state result entries values pairs globals cellFound valid stateFound rows lookup
    | none =>
      apply verified_recursive_miss heap cell memo state values pairs entries globals
        cellFound valid stateFound rows canonical (by omega) lookup
      intro child childCell childMemo childState childValues childPairs childEntries shorter childCellFound childValid childStateFound childRows childCanonical
      exact ih childPairs.length (by omega) child childCell childMemo childState childValues childPairs childEntries
        childCellFound childValid childStateFound childRows childCanonical rfl

/-- The recursive source contract quantifies over all canonical states and valid shared memos. -/
def RecursiveCallsVerified (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∀ (cell memo state : Nat) (values : List Mutable.Value) (pairs : List Pair) (entries : List Memo.Entry),
    heap[cell]? = some (.list [.ref memo]) → MemoValid heap memo entries →
    heap[state]? = some (.tuple values) → Heap.Rows heap values pairs →
    (∀ pair ∈ pairs, Canonical pair) → RecursiveVerified heap cell memo state pairs entries globals

theorem recursive_calls_verified (heap : Mutable.Heap) (globals : Mutable.Globals) : RecursiveCallsVerified heap globals :=
  fun cell memo state values pairs entries => recursive_verified heap cell memo state values pairs entries globals

end solvers.«93».all_results
