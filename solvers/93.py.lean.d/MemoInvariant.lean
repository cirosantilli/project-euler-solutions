import solvers.«93.py.lean.d».Cache
import ProjectEulerSolutions.Lib.RationalMemo
import ProjectEulerSolutions.Lib.RationalResultSpec

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions
set_option Elab.async false

structure MemoValid (heap : Mutable.Heap) (memo : Nat) (entries : List Memo.Entry) : Prop where
  stored : MemoAt heap memo (Memo.encode entries)
  results : Memo.Correct (fun state address => Heap.ResultAt heap address state) entries

theorem MemoValid.empty (heap : Mutable.Heap) (memo : Nat) (found : heap[memo]? = some (.dict [])) :
    MemoValid heap memo [] := by
  refine ⟨?_, Memo.correct_nil _⟩
  simp [MemoAt, found, Mutable.dictionaryEntries, Mutable.promoteDictionary, Memo.encode]

theorem MemoValid.lookup {heap : Mutable.Heap} {memo result : Nat} {entries : List Memo.Entry} {state : List Pair}
    (valid : MemoValid heap memo entries) (found : Memo.lookup entries state = some result) :
    Heap.ResultAt heap result state := Memo.lookup_correct valid.results found

theorem MemoValid.result_bound {heap : Mutable.Heap} {memo result : Nat} {entries : List Memo.Entry}
    (valid : MemoValid heap memo entries) (member : result ∈ entries.map Memo.Entry.result) : result < heap.length := by
  obtain ⟨entry, present, rfl⟩ := List.mem_map.mp member
  exact (valid.results entry present).bound

theorem MemoValid.fresh_not_cached {heap : Mutable.Heap} {memo address : Nat} {entries : List Memo.Entry}
    (valid : MemoValid heap memo entries) (fresh : heap.length ≤ address) : address ∉ entries.map Memo.Entry.result := by
  intro member
  have bound := valid.result_bound member
  omega

theorem MemoValid.updated {heap after : Mutable.Heap} {memo stateAddress result : Nat} {entries : List Memo.Entry}
    {state : List Pair} (valid : MemoValid heap memo entries) (frame : PreservesExcept memo heap after)
    (stored : MemoAt after memo (Memo.encode (Memo.insert entries state stateAddress result)))
    (complete : Heap.ResultAt after result state) :
    MemoValid after memo (Memo.insert entries state stateAddress result) := by
  refine ⟨stored, Memo.insert_correct ?_ complete⟩
  exact valid.results.mono (fun _ _ found => found.preserved frame (Memo.encode entries) valid.stored)

theorem MemoValid.append {heap : Mutable.Heap} {memo : Nat} {entries : List Memo.Entry}
    (valid : MemoValid heap memo entries) (extra : Mutable.Heap) : MemoValid (heap ++ extra) memo entries :=
  ⟨valid.stored.append extra, valid.results.mono (fun _ _ found => found.append extra)⟩

/-- A complete call preserves its caller and only adds result sets from old cache entries or fresh allocations. -/
structure RecursivePost (before after : Mutable.Heap) (memo : Nat) (old entries : List Memo.Entry)
    (state : List Pair) (result : Nat) : Prop where
  frame : PreservesExcept memo before after
  valid : MemoValid after memo entries
  complete : Heap.ResultAt after result state
  cached : Memo.lookup entries state = some result
  origins : Memo.Origins old before.length entries

theorem RecursivePost.resultOrigin {before after : Mutable.Heap} {memo result : Nat}
    {old entries : List Memo.Entry} {state : List Pair} (post : RecursivePost before after memo old entries state result) :
    result ∈ old.map Memo.Entry.result ∨ before.length ≤ result := Memo.lookup_correct post.origins post.cached

def RecursiveVerified (heap : Mutable.Heap) (cell memo state : Nat) (pairs : List Pair)
    (entries : List Memo.Entry) (globals : Mutable.Globals) : Prop :=
  ∃ after result finalEntries,
    ReturnsValue program recName [.ref state, .ref cell] heap (.ref result) after globals ∧
    RecursivePost heap after memo entries finalEntries pairs result

/-- The collector stays absent from the cache after a recursive child call. -/
theorem RecursivePost.collector_uncached {before after : Mutable.Heap} {memo address result : Nat}
    {old entries : List Memo.Entry} {state : List Pair} (post : RecursivePost before after memo old entries state result)
    (uncached : address ∉ old.map Memo.Entry.result) (inside : address < before.length) :
    address ∉ entries.map Memo.Entry.result := post.origins.avoids uncached inside

theorem RecursivePost.collector_distinct {before after : Mutable.Heap} {memo address result : Nat}
    {old entries : List Memo.Entry} {state : List Pair} (post : RecursivePost before after memo old entries state result)
    (uncached : address ∉ old.map Memo.Entry.result) (inside : address < before.length) : result ≠ address := by
  intro equal
  rcases post.resultOrigin with cached | fresh
  · exact uncached (equal ▸ cached)
  · omega

end solvers.«93».all_results
