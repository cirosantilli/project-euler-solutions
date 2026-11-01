import solvers.«93.py.lean.d».BaseCases
import solvers.«93.py.lean.d».MemoInvariant

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions
set_option Elab.async false

theorem verified_cache_hit (heap : Mutable.Heap) (cell memo state result : Nat)
    (entries : List Memo.Entry) (values : List Mutable.Value) (pairs : List Pair) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (lookup : Memo.lookup entries pairs = some result) :
    RecursiveVerified heap cell memo state pairs entries globals := by
  have encoded : HashDict.lookup (Memo.encode entries) (Heap.stateKey pairs) = some (.ref result) := by
    rw [Memo.lookup_encode, lookup]
    rfl
  have executed := cache_hit heap cell memo state (Memo.encode entries) values pairs (.ref result) globals
    cellFound valid.stored stateFound rows encoded
  exact ⟨heap, result, entries, executed,
    ⟨PreservesExcept.refl memo heap, valid, valid.lookup lookup, lookup, Memo.Origins.refl entries heap.length⟩⟩

theorem verified_singleton_miss (heap : Mutable.Heap) (cell memo state operand : Nat) (pair : Pair)
    (entries : List Memo.Entry) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple [.ref operand])) (operandFound : heap[operand]? = some (Heap.object pair))
    (canonical : Canonical pair) (missing : Memo.lookup entries [pair] = none) :
    RecursiveVerified heap cell memo state [pair] entries globals := by
  let after := singletonHeap heap memo state operand pair (Memo.encode entries)
  let finalEntries := Memo.insert entries [pair] state heap.length
  have missed : HashDict.lookup (Memo.encode entries) (Heap.stateKey [pair]) = none := by
    rw [Memo.lookup_encode, missing]
    rfl
  have frame : PreservesExcept memo heap after :=
    (PreservesExcept.append memo heap [.keySet [(Heap.key pair, .ref operand)]]).trans (PreservesExcept.write memo _ _)
  have row := frame.non_dictionary (Memo.encode entries) valid.stored operand _ operandFound rfl
  have complete : Heap.ResultAt after heap.length [pair] :=
    Heap.ResultAt.singleton (singleton_found heap memo state operand pair (Memo.encode entries) valid.stored) row canonical
  have stored : MemoAt after memo (Memo.encode finalEntries) := by
    have bound : memo < (heap ++ [Mutable.Object.keySet [(Heap.key pair, .ref operand)]]).length := by
      have inside := valid.stored.bound
      simp only [List.length_append, List.length_singleton]
      omega
    simp [after, singletonHeap, MemoAt, Memo.insert_encode, finalEntries, Mutable.dictionaryEntries, List.getElem?_set_self bound]
  refine ⟨after, heap.length, finalEntries,
    singleton_miss heap cell memo state operand pair (Memo.encode entries) globals cellFound valid.stored stateFound operandFound missed,
    ⟨frame, valid.updated frame stored complete, complete, Memo.lookup_insert .., ?_⟩⟩
  exact (Memo.Origins.refl entries heap.length).insert [pair] state heap.length (.inr (Nat.le_refl _))

theorem verified_empty_miss (heap : Mutable.Heap) (cell memo state : Nat)
    (entries : List Memo.Entry) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple [])) (missing : Memo.lookup entries [] = none) :
    RecursiveVerified heap cell memo state [] entries globals := by
  let after := emptyHeap heap memo state (Memo.encode entries)
  let finalEntries := Memo.insert entries [] state heap.length
  have missed : HashDict.lookup (Memo.encode entries) (Heap.stateKey []) = none := by
    rw [Memo.lookup_encode, missing]
    rfl
  have frame : PreservesExcept memo heap after :=
    (PreservesExcept.append memo heap [.set []]).trans (PreservesExcept.write memo _ _)
  have complete : Heap.ResultAt after heap.length [] :=
    Heap.ResultAt.empty (empty_found heap memo state (Memo.encode entries) valid.stored)
  have stored : MemoAt after memo (Memo.encode finalEntries) := by
    have bound : memo < (heap ++ [Mutable.Object.set []]).length := by
      have inside := valid.stored.bound
      simp only [List.length_append, List.length_singleton]
      omega
    simp [after, emptyHeap, MemoAt, Memo.insert_encode, finalEntries, Mutable.dictionaryEntries, List.getElem?_set_self bound]
  refine ⟨after, heap.length, finalEntries,
    empty_miss heap cell memo state (Memo.encode entries) globals cellFound valid.stored stateFound missed,
    ⟨frame, valid.updated frame stored complete, complete, Memo.lookup_insert .., ?_⟩⟩
  exact (Memo.Origins.refl entries heap.length).insert [] state heap.length (.inr (Nat.le_refl _))

/-- All terminal-sized states satisfy the recursive invariant, whether the memo lookup hits or misses. -/
theorem verified_terminal (heap : Mutable.Heap) (cell memo state : Nat) (values : List Mutable.Value)
    (pairs : List Pair) (entries : List Memo.Entry) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (valid : MemoValid heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (canonical : ∀ pair ∈ pairs, Canonical pair) (small : pairs.length ≤ 1) :
    RecursiveVerified heap cell memo state pairs entries globals := by
  cases lookup : Memo.lookup entries pairs with
  | some result =>
    exact verified_cache_hit heap cell memo state result entries values pairs globals
      cellFound valid stateFound rows lookup
  | none =>
    cases pairs with
    | nil =>
      cases rows
      exact verified_empty_miss heap cell memo state entries globals cellFound valid stateFound lookup
    | cons pair rest =>
      cases rest with
      | nil =>
        cases rows with
        | cons row rest =>
          cases rest
          obtain ⟨operand, rfl, operandFound⟩ := row
          exact verified_singleton_miss heap cell memo state operand pair entries globals
            cellFound valid stateFound operandFound (canonical pair (by simp)) lookup
      | cons other rest => simp at small

end solvers.«93».all_results
