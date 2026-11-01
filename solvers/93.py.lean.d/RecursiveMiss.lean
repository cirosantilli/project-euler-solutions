import solvers.«93.py.lean.d».OuterPairs
import solvers.«93.py.lean.d».MemoBaseCases

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions
set_option Elab.async false

/-- A nonterminal cache miss collects every reduction result and stores the completed set. -/
theorem verified_recursive_miss (heap : Mutable.Heap) (cell memo state : Nat) (values : List Mutable.Value)
    (pairs : List Pair) (entries : List Memo.Entry) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (multiple : 1 < pairs.length)
    (missing : Memo.lookup entries pairs = none) (smaller : SmallerCalls pairs.length globals) :
    RecursiveVerified heap cell memo state pairs entries globals := by
  let env := functionEnv recArgs recBody [.ref state, .ref cell] globals
  have getCell : env cellName = some (.ref cell) := by cases globals <;> rfl
  have getState : env "state" = some (.ref state) := by cases globals <;> rfl
  have hash := Heap.state_hash heap state values pairs stateFound rows
  have tested := cached_test heap cell memo state (Memo.encode entries) (Heap.stateKey pairs) env globals
    cellFound valid.stored hash getCell getState
  have missed : HashDict.lookup (Memo.encode entries) (Heap.stateKey pairs) = none := by
    rw [Memo.lookup_encode, missing]
    rfl
  have size : Evaluates program env singletonGuard heap (.scalar (.bool false)) heap globals := by
    apply Evaluates.compare
      (Evaluates.builtin (.cons (Evaluates.lookup getState) .nil)
        (show primitive (.builtin .len) [.ref state] heap = some (.ok (.scalar (.int pairs.length)), heap) by
          simp [primitive, Mutable.length, stateFound, rows.length_eq])) Evaluates.constant
    have different : (pairs.length : Int) ≠ 1 := by omega
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, different]
  let initial := heap ++ [Mutable.Object.set []]
  let bound := setLocal env "results" (.ref heap.length)
  have initialCell := HeapCollection.prefix_found heap [.set []] cell _ cellFound
  have initialState := HeapCollection.prefix_found heap [.set []] state _ stateFound
  have initialMemo : MemoValid initial memo entries := valid.append [.set []]
  have initialRows : Heap.Rows initial values pairs := rows.append [.set []]
  have empty : Heap.SetAt initial heap.length [] := by
    refine ⟨.inr ⟨rfl, ?_⟩, ?_⟩
    · simp [initial]
    · intro entry member
      simp at member
  have boundCell : bound cellName = some (.ref cell) := by simpa [bound, setLocal, cellName] using getCell
  have boundState : bound "state" = some (.ref state) := by simpa [bound, setLocal] using getState
  have boundResult : bound "results" = some (.ref heap.length) := by simp [bound, setLocal]
  obtain ⟨middle, finalEnv, collected, middleEntries, loops, finalSet, members, middleMemo, origins, _, frame, locals⟩ :=
    pair_loops initial cell memo heap.length state values pairs canonical multiple entries [] bound globals smaller
      initialCell initialMemo initialState initialRows empty (valid.fresh_not_cached (Nat.le_refl _))
      boundCell boundState boundResult
  have complete : Heap.ResultAt middle heap.length pairs := by
    refine ⟨collected, finalSet, ?_⟩
    intro pair
    simpa using members pair
  have middleCell := frame.ordinary (Memo.encode entries) ([] : List (HashKey × Mutable.Value))
    initialMemo.stored empty.keyEntries cell _ initialCell rfl rfl
  have middleState := frame.ordinary (Memo.encode entries) ([] : List (HashKey × Mutable.Value))
    initialMemo.stored empty.keyEntries state _ initialState rfl rfl
  have middleRows := initialRows.preserved_outside frame (Memo.encode entries) [] initialMemo.stored empty.keyEntries
  have middleHash := Heap.state_hash middle state values pairs middleState middleRows
  have finalCell : finalEnv cellName = some (.ref cell) := by
    rw [locals cellName (by decide) (by decide) (by decide) (by decide)]
    exact boundCell
  have finalState : finalEnv "state" = some (.ref state) := by
    rw [locals "state" (by decide) (by decide) (by decide) (by decide)]
    exact boundState
  have finalResult : finalEnv "results" = some (.ref heap.length) := by
    rw [locals "results" (by decide) (by decide) (by decide) (by decide)]
    exact boundResult
  let after := middle.set memo (.keyDict (HashDict.insert (Memo.encode middleEntries) (Heap.stateKey pairs) (.ref state) (.ref heap.length)))
  let finalEntries := Memo.insert middleEntries pairs state heap.length
  have stored := store_result middle middle cell memo state (Memo.encode middleEntries) (Heap.stateKey pairs) (.ref heap.length)
    finalEnv globals (.name "results") (Evaluates.lookup finalResult) middleCell middleMemo.stored middleHash finalCell finalState
  have writeFrame : PreservesExcept memo middle after := PreservesExcept.write memo middle _
  have completed : Heap.ResultAt after heap.length pairs := complete.preserved writeFrame (Memo.encode middleEntries) middleMemo.stored
  have finalStored : MemoAt after memo (Memo.encode finalEntries) := by
    simp [after, finalEntries, Memo.insert_encode, MemoAt, Mutable.dictionaryEntries,
      List.getElem?_set_self middleMemo.stored.bound]
  have callerFrame : PreservesExcept memo heap middle :=
    frame.fresh_collector (PreservesExcept.append memo heap [.set []]) (Nat.le_refl _)
  have finalOrigins : Memo.Origins entries heap.length finalEntries := by
    apply ((Memo.Origins.refl entries heap.length).trans origins (by simp [initial])).insert
    exact .inr (Nat.le_refl _)
  refine ⟨after, heap.length, finalEntries, ?_,
    ⟨callerFrame.trans writeFrame, middleMemo.updated writeFrame finalStored completed, completed,
      Memo.lookup_insert .., finalOrigins⟩⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) tested (by simp [missed, Mutable.truth, scalarTruth])
  apply BlockReturns.branch (flag := false) size rfl
  apply BlockReturns.assign (Evaluates.builtin .nil
    (show primitive (.builtin .set) [] heap = some (.ok (.ref heap.length), initial) from rfl))
  exact loops.returns (stored.returns (BlockReturns.return_value (Evaluates.lookup finalResult)))

end solvers.«93».all_results
