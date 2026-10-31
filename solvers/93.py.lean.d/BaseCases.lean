import solvers.«93.py.lean.d».Cache
import ProjectEulerSolutions.Lib.RationalSets

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev singletonExpr : Expr := .builtinCall .setDisplay [.subscript (.name "state") (.constant (.int 0))]
abbrev singletonGuard : Expr := .compare (.builtinCall .len [.name "state"]) [(.eq, .constant (.int 1))]

theorem source_singleton : recBody[1]? = some (.ifThen singletonGuard
    [.assign [.subscript memoExpr (.name "state")] singletonExpr, .return cachedExpr] []) := rfl

def singletonHeap (heap : Mutable.Heap) (memo state operand : Nat) (pair : Pair)
    (entries : List (HashDict.Entry Mutable.Value)) : Mutable.Heap :=
  (heap ++ [Mutable.Object.keySet [(Heap.key pair, .ref operand)]]).set memo
    (.keyDict (HashDict.insert entries (Heap.stateKey [pair]) (.ref state) (.ref heap.length)))

theorem singleton_expression (heap : Mutable.Heap) (state operand : Nat) (pair : Pair)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (stateFound : heap[state]? = some (.tuple [.ref operand]))
    (operandFound : heap[operand]? = some (Heap.object pair))
    (getState : env "state" = some (.ref state)) :
    Evaluates program env singletonExpr heap (.ref heap.length)
      (heap ++ [.keySet [(Heap.key pair, .ref operand)]]) globals := by
  apply Evaluates.builtin (values := [.ref operand]) (middle := heap)
    (.cons (Evaluates.subscript (by intro a b c impossible; cases impossible)
      (Evaluates.lookup getState) Evaluates.constant
      (by simp [Mutable.getItem, stateFound, Mutable.index, Except.map])) .nil)
  exact Heap.singleton_set heap operand pair operandFound

/-- A one-fragment cache miss stores a singleton containing that same fragment tuple. -/
theorem singleton_miss (heap : Mutable.Heap) (cell memo state operand : Nat) (pair : Pair)
    (entries : List (HashDict.Entry Mutable.Value)) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (memoFound : MemoAt heap memo entries)
    (stateFound : heap[state]? = some (.tuple [.ref operand]))
    (operandFound : heap[operand]? = some (Heap.object pair))
    (missing : HashDict.lookup entries (Heap.stateKey [pair]) = none) :
    ReturnsValue program recName [.ref state, .ref cell] heap (.ref heap.length)
      (singletonHeap heap memo state operand pair entries) globals := by
  let env := functionEnv recArgs recBody [.ref state, .ref cell] globals
  have getCell : env cellName = some (.ref cell) := by cases globals <;> rfl
  have getState : env "state" = some (.ref state) := by cases globals <;> rfl
  have rows : Heap.Rows heap [.ref operand] [pair] := .cons ⟨operand, rfl, operandFound⟩ .nil
  have hash := Heap.state_hash heap state [.ref operand] [pair] stateFound rows
  have tested := cached_test heap cell memo state entries (Heap.stateKey [pair]) env globals
    cellFound memoFound hash getCell getState
  have size : Evaluates program env singletonGuard heap (.scalar (.bool true)) heap globals := by
    apply Evaluates.compare
      (Evaluates.builtin (.cons (Evaluates.lookup getState) .nil)
        (show primitive (.builtin .len) [.ref state] heap = some (.ok (.scalar (.int 1)), heap) by
          simp [primitive, Mutable.length, stateFound])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  let extra : Mutable.Heap := [.keySet [(Heap.key pair, .ref operand)]]
  let middle := heap ++ extra
  let updated := HashDict.insert entries (Heap.stateKey [pair]) (.ref state) (.ref heap.length)
  have middleCell := ProjectEulerSolutions.HeapCollection.prefix_found heap extra cell _ cellFound
  have middleState := ProjectEulerSolutions.HeapCollection.prefix_found heap extra state _ stateFound
  have middleOperand := ProjectEulerSolutions.HeapCollection.prefix_found heap extra operand _ operandFound
  have middleMemo : MemoAt middle memo entries := memoFound.append extra
  have middleHash := Heap.state_hash middle state [.ref operand] [pair] middleState (rows.append extra)
  have stored := store_result heap middle cell memo state entries (Heap.stateKey [pair]) (.ref heap.length)
    env globals singletonExpr (singleton_expression heap state operand pair env globals stateFound operandFound getState)
    middleCell middleMemo middleHash getCell getState
  have finalCell := middleMemo.other cell _ middleCell rfl updated
  have finalState := middleMemo.other state _ middleState rfl updated
  have finalOperand := middleMemo.other operand _ middleOperand rfl updated
  have finalMemo : MemoAt (middle.set memo (.keyDict updated)) memo updated := by
    simp [MemoAt, Mutable.dictionaryEntries, middleMemo.bound]
  have finalRows : Heap.Rows (middle.set memo (.keyDict updated)) [.ref operand] [pair] :=
    .cons ⟨operand, rfl, finalOperand⟩ .nil
  have finalHash := Heap.state_hash _ state [.ref operand] [pair] finalState finalRows
  have returned := read_cached _ cell memo state updated (Heap.stateKey [pair]) (.ref heap.length) env globals
    finalCell finalMemo finalHash (stored_lookup entries [pair] state (.ref heap.length)) getCell getState
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) tested (by simp [missing, Mutable.truth, scalarTruth])
  apply BlockReturns.branch (flag := true) size rfl
  exact stored.returns (BlockReturns.return_value returned)

/-- Storing the memo entry cannot overwrite its freshly allocated result set. -/
theorem singleton_found (heap : Mutable.Heap) (memo state operand : Nat) (pair : Pair)
    (entries : List (HashDict.Entry Mutable.Value)) (memoFound : MemoAt heap memo entries) :
    (singletonHeap heap memo state operand pair entries)[heap.length]? =
      some (.keySet [(Heap.key pair, .ref operand)]) := by
  have different : memo ≠ heap.length := Nat.ne_of_lt memoFound.bound
  simp [singletonHeap, different]

def emptyHeap (heap : Mutable.Heap) (memo state : Nat) (entries : List (HashDict.Entry Mutable.Value)) : Mutable.Heap :=
  (heap ++ [Mutable.Object.set []]).set memo
    (.keyDict (HashDict.insert entries (Heap.stateKey []) (.ref state) (.ref heap.length)))

/-- Empty states execute no pair loop and memoize an empty result set. -/
theorem empty_miss (heap : Mutable.Heap) (cell memo state : Nat)
    (entries : List (HashDict.Entry Mutable.Value)) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (memoFound : MemoAt heap memo entries)
    (stateFound : heap[state]? = some (.tuple []))
    (missing : HashDict.lookup entries (Heap.stateKey []) = none) :
    ReturnsValue program recName [.ref state, .ref cell] heap (.ref heap.length)
      (emptyHeap heap memo state entries) globals := by
  let env := functionEnv recArgs recBody [.ref state, .ref cell] globals
  have getCell : env cellName = some (.ref cell) := by cases globals <;> rfl
  have getState : env "state" = some (.ref state) := by cases globals <;> rfl
  have hash := Heap.state_hash heap state [] [] stateFound .nil
  have tested := cached_test heap cell memo state entries (Heap.stateKey []) env globals
    cellFound memoFound hash getCell getState
  have size : Evaluates program env singletonGuard heap (.scalar (.bool false)) heap globals := by
    apply Evaluates.compare
      (Evaluates.builtin (.cons (Evaluates.lookup getState) .nil)
        (show primitive (.builtin .len) [.ref state] heap = some (.ok (.scalar (.int 0)), heap) by
          simp [primitive, Mutable.length, stateFound])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  let middle := heap ++ [Mutable.Object.set []]
  let bound := setLocal env "results" (.ref heap.length)
  have middleCell := ProjectEulerSolutions.HeapCollection.prefix_found heap [.set []] cell _ cellFound
  have middleState := ProjectEulerSolutions.HeapCollection.prefix_found heap [.set []] state _ stateFound
  have middleMemo : MemoAt middle memo entries := memoFound.append [.set []]
  have middleHash := Heap.state_hash middle state [] [] middleState .nil
  have boundCell : bound cellName = some (.ref cell) := by simpa [bound, setLocal, cellName] using getCell
  have boundState : bound "state" = some (.ref state) := by simpa [bound, setLocal] using getState
  have boundResult : bound "results" = some (.ref heap.length) := by simp [bound, setLocal]
  have stored := store_result middle middle cell memo state entries (Heap.stateKey []) (.ref heap.length)
    bound globals (.name "results") (Evaluates.lookup boundResult) middleCell middleMemo middleHash boundCell boundState
  have lengthZero : Evaluates program bound (.builtinCall .len [.name "state"]) middle (.scalar (.int 0)) middle globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup boundState) .nil)
    simp [primitive, Mutable.length, middle, middleState]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) tested (by simp [missing, Mutable.truth, scalarTruth])
  apply BlockReturns.branch (flag := false) size rfl
  apply BlockReturns.assign (Evaluates.builtin .nil
    (show primitive (.builtin .set) [] heap = some (.ok (.ref heap.length), middle) from rfl))
  apply BlockReturns.for_range (.cons lengthZero .nil) rfl
  apply BlockReturns.range_done rfl
  exact stored.returns (BlockReturns.return_value (Evaluates.lookup boundResult))

theorem empty_found (heap : Mutable.Heap) (memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (memoFound : MemoAt heap memo entries) :
    (emptyHeap heap memo state entries)[heap.length]? = some (.set []) := by
  have different : memo ≠ heap.length := Nat.ne_of_lt memoFound.bound
  simp [emptyHeap, different]

/-- Complete source executions for cache hits and both terminal reduction-state sizes. -/
structure BaseCasesVerified (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop where
  cacheHit : ∀ (cell memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (values : List Mutable.Value) (pairs : List Pair) (cached : Mutable.Value),
    heap[cell]? = some (.list [.ref memo]) → MemoAt heap memo entries →
    heap[state]? = some (.tuple values) → Heap.Rows heap values pairs →
    HashDict.lookup entries (Heap.stateKey pairs) = some cached →
    ReturnsValue program recName [.ref state, .ref cell] heap cached heap globals
  singletonMiss : ∀ (cell memo state operand : Nat) (pair : Pair) (entries : List (HashDict.Entry Mutable.Value)),
    heap[cell]? = some (.list [.ref memo]) → MemoAt heap memo entries →
    heap[state]? = some (.tuple [.ref operand]) → heap[operand]? = some (Heap.object pair) →
    HashDict.lookup entries (Heap.stateKey [pair]) = none →
    ReturnsValue program recName [.ref state, .ref cell] heap (.ref heap.length)
      (singletonHeap heap memo state operand pair entries) globals
  emptyMiss : ∀ (cell memo state : Nat) (entries : List (HashDict.Entry Mutable.Value)),
    heap[cell]? = some (.list [.ref memo]) → MemoAt heap memo entries → heap[state]? = some (.tuple []) →
    HashDict.lookup entries (Heap.stateKey []) = none →
    ReturnsValue program recName [.ref state, .ref cell] heap (.ref heap.length)
      (emptyHeap heap memo state entries) globals

theorem base_cases_verified (heap : Mutable.Heap) (globals : Mutable.Globals) : BaseCasesVerified heap globals :=
  ⟨fun cell memo state entries values pairs cached => cache_hit heap cell memo state entries values pairs cached globals,
   fun cell memo state operand pair entries => singleton_miss heap cell memo state operand pair entries globals,
   fun cell memo state entries => empty_miss heap cell memo state entries globals⟩

end solvers.«93».all_results
